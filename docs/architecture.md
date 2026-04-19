# factfull — 技術アーキテクチャ解説

## 概要

**factfull** は、ドキュメント（記事・議事録サマリーなど）に含まれる事実クレームを Truth ソース（トランスクリプト・原文など）と照合し、矛盾を自動修正する **LLM ベースのファクトチェックパイプライン**です。

スコアが閾値を超えるまでファクトチェック → 修正 を繰り返す「自己改善ループ」が特徴です。

---

## アーキテクチャ全体図

```
┌──────────────────────────────────────────────────────────────┐
│  refine_loop.py  (CLI エントリポイント)                        │
│                                                              │
│  ┌──────────────┐      ┌──────────────────────────────────┐  │
│  │ Truth ソース  │─────▶│ indexer.build_index()            │  │
│  │ (.txt/.md)   │      │  チャンク化 + BM25Okapi 構築      │  │
│  └──────────────┘      └──────────────┬───────────────────┘  │
│                                       │ (bm25, chunks)        │
│  ┌──────────────┐                     │                       │
│  │ 対象ドキュメント│      ┌────────────▼────────────────────┐  │
│  │ (.md)        │─────▶│ claim_extractor.extract()        │  │
│  └──────────────┘      │  セクションフィルタ + LLM でクレーム抽出 │  │
│                        └────────────┬────────────────────┘  │
│                                     │ [claim1, claim2, ...]  │
│                        ┌────────────▼────────────────────┐  │
│                        │ retriever.retrieve()             │  │
│                        │  BM25 でクレームに関連する証拠を検索  │  │
│                        └────────────┬────────────────────┘  │
│                                     │ [Chunk, Chunk, ...]    │
│                        ┌────────────▼────────────────────┐  │
│                        │ verifier.verify()                │  │
│                        │  LLM でクレームを証拠と照合・判定    │  │
│                        └────────────┬────────────────────┘  │
│                                     │ [VerificationResult]   │
│                    ┌────────────────▼────────────┐          │
│                    │  スコア計算 (reporter)         │          │
│                    │  score ≥ threshold? ─────YES─┼──▶ 完了 │
│                    │              │NO             │          │
│                    └──────────────┼───────────────┘          │
│                                   │                          │
│                        ┌──────────▼────────────────────┐    │
│                        │ corrector.correct()            │    │
│                        │  セクション特定 + LLM で外科的修正  │    │
│                        └──────────┬────────────────────┘    │
│                                   │ 修正済みドキュメント        │
│                                   └── ループ先頭へ戻る          │
│                                                              │
│  ループ完了後: editorial.append_editorial_note()  (--editorial) │
└──────────────────────────────────────────────────────────────┘
```

---

## モジュール詳解

### `factfull/llm.py` — LLM バックエンド抽象化

すべての LLM 呼び出しを一元管理する薄いラッパー。環境変数でバックエンドを切り替えられる。

```
FACTFULL_LLM_BACKEND=ollama     # デフォルト：ローカル Ollama
FACTFULL_LLM_BACKEND=anthropic  # Anthropic API
```

| 環境変数 | デフォルト値 | 説明 |
|---|---|---|
| `FACTFULL_LLM_BACKEND` | `ollama` | バックエンド選択 |
| `FACTFULL_OLLAMA_URL` | `http://localhost:11435/api/generate` | Ollama エンドポイント |
| `FACTFULL_OLLAMA_MODEL` | `glm-4.7-flash:latest` | Ollama モデル名 |
| `FACTFULL_ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Anthropic モデル名 |

**Ollama バックエンドの特徴:**
- ストリーミングレスポンス（`stream: true`）を逐次読み取り
- `temperature: 0.1` — 低温度で決定論的な出力
- タイムアウト時に最大 6 回リトライ（60秒待機）
- `TimeoutError`, `OSError`, `HTTPError`, `URLError` のみリトライ対象（それ以外は即時例外）

**Anthropic バックエンドの特徴:**
- `anthropic` SDK を使用（`import anthropic` は遅延インポート）
- `max_tokens: 4096`

---

### `factfull/indexer.py` — BM25 インデックス構築

Truth ソース（複数ファイル可）をチャンク化して BM25Okapi インデックスを構築する。

```python
def build_index(truth_paths: list[Path], chunk_size=400, overlap=80) -> (BM25Okapi, list[Chunk])
```

**チャンク化の仕組み:**
- 文字数 400 字ごとにスライディングウィンドウで分割（オーバーラップ 80 字）
- オーバーラップにより境界付近の情報損失を防ぐ

**トークナイザ `_tokenize(text)`:**
```
正規表現: [A-Za-z0-9]+|[^\s]
```
- 英数字連続 → 1トークン（小文字化）
- 日本語 → 1文字ずつトークン化
- BM25 のスコア計算はこのトークン列に基づく

**Chunk データクラス:**
```python
@dataclass
class Chunk:
    text: str    # チャンクのテキスト本文
    source: str  # 元ファイル名
    offset: int  # 元テキスト内の先頭位置
```

---

### `factfull/claim_extractor.py` — クレーム抽出

対象ドキュメントから「事実として検証可能なアトミックなクレーム」を抽出する。

```python
def extract(document: str, max_claims: int = 30) -> list[str]
```

**処理フロー:**
1. **セクションフィルタ** — `## 編集後記`, `## キーワード`, `## 動画` などの非事実セクションを除去
2. **チャンク分割** — ドキュメントを 4000 字ごとに分割（文末 `。` で区切る）
3. **LLM による抽出** — 各チャンクに対して JSON 配列形式でクレームを要求
4. **パースフォールバック** — JSON パースに失敗した場合は行単位でクレームを抽出

**抽出プロンプトの要点:**
- 数値・固有名詞・日時・金額を含むクレームを優先
- 意見・推測（「〜と思われる」）は除外
- 出力は JSON 配列のみ（前置きなし）

**ノイズフィルタ `_META_PATTERNS`:**
```
"以下は", "これらの", "上記は", "抽出しました", ... など
```
LLM がメタコメントをクレームとして出力してしまうケースをフィルタする。

---

### `factfull/retriever.py` — 証拠検索

各クレームに対して BM25 スコアで最も関連するチャンクを返す。

```python
def retrieve(claim: str, bm25: BM25Okapi, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]
```

- クレームを `indexer._tokenize()` でトークン化
- `bm25.get_scores(tokens)` で全チャンクにスコアを付与
- 上位 `top_k` 件を取得し、**スコア 0 のチャンクは除外**（完全に無関係なチャンクを排除）

---

### `factfull/verifier.py` — クレーム検証

クレームと証拠パッセージを LLM に渡して判定を取得する。

```python
def verify(claim: str, evidence_chunks: list[Chunk]) -> VerificationResult
```

**判定カテゴリ:**

| Verdict | 記号 | スコア寄与 | 意味 |
|---|---|---|---|
| `supported` | ✅ | 1.0 | 証拠によって明確に支持 |
| `partial` | ⚠️ | 0.5 | 部分的に正しい・誇張・誤記 |
| `contradicted` | ❌ | 0.0 | 証拠と明確に矛盾 |
| `unverifiable` | ❓ | 除外 | 証拠不足で判定不能 |

**スコア計算式:**
```
score = (sum(verdict_score for verifiable results) / len(verifiable_results)) * 100
```
`unverifiable` は分母・分子どちらにも含めない（判定できないものはスコアに影響しない）。

**JSON パースのフォールバック:**
LLM が純粋な JSON を返さない場合、レスポンステキストから `contradicted`, `partial`, `supported` のキーワードを順番に検索して verdict を推定する。

---

### `factfull/reporter.py` — レポート生成

各イテレーションの検証結果を Markdown レポートとして出力する。

```python
def generate_report(results: list[VerificationResult], target_name: str, truth_names: list[str]) -> str
```

レポートは `fact_check_iter01.md`, `fact_check_iter02.md` ... として保存される。

**出力優先順:**
1. ❌ CONTRADICTED（最優先）
2. ⚠️ PARTIAL
3. ✅ SUPPORTED
4. ❓ UNVERIFIABLE

各クレームについて判定・理由・証拠スニペット（最大 2件・200字）を出力する。

---

### `factfull/corrector.py` — 外科的修正

`CONTRADICTED` / `PARTIAL` なクレームを含むセクションを LLM で修正する。

```python
def correct(document: str, results: list[VerificationResult]) -> tuple[str, int]
# 返り値: (修正済みドキュメント, 修正したセクション数)
```

**修正アルゴリズム:**

1. **セクション分割** — `## / ###` 見出しでドキュメントをセクションに分解
   - YAML フロントマター（`--- ... ---`）はプリアンブルとして保持
   - `__preamble__` という仮想ヘッダで管理

2. **クレーム → セクション マッピング** — 各クレームを最も関連するセクションに割り当て
   - トークナイザ: 英数字は lowercase、日本語は **2-gram**（bigram）でセット化
   - Jaccard 係数（クレームトークン ∩ セクショントークン）÷ クレームトークン数
   - スコア 0.15 未満はマッチなし扱い（誤マッチ防止）

3. **セクション単位でまとめて LLM に渡す** — 同セクションに複数の問題がある場合はバンドル

4. **修正後の品質チェック:**
   - LLM が空文字を返した → 元の本文を維持
   - 修正後が元の 60% 未満に縮小 → セクション削除とみなして元の本文を維持

**修正対象外セクション:**
`動画`, `キーワード`, `チャンネル`, `YouTube`, `再生時間`, `外部で開く`, `編集後記`

---

### `factfull/editorial.py` — 編集後記生成

ファクトチェック完了後に「AI 編集後記」を記事末尾に追加する。

```python
def append_editorial_note(document: str, model: str | None = None) -> str
```

- `## 編集後記` セクションが既存なら生成をスキップ
- 記事が 12000 字超の場合は冒頭 12000 字のみをプロンプトに渡す
- `num_ctx=16384` — 長い記事に対応

**編集後記の設計思想:**
- 事実の再要約ではなく「気づき・問い・考察」を書く
- ファクトチェックの対象外（AI の解釈ゾーン）として完全分離
- ループ完了 *後* に生成することで、ファクトチェック対象に混入しない

---

### `refine_loop.py` — 自己改善ループ CLI

```
python refine_loop.py \
  --truth <transcript.txt or dir> \
  --target <summary_ja.md> \
  [--threshold 95] \
  [--max-iter 5] \
  [--editorial]
```

**ループ制御フロー:**
```
for iteration in 1..max_iter:
    results = factcheck(document)
    score = compute_score(results)
    save report as fact_check_iter{N}.md

    if score >= threshold:
        break  # 合格

    if iteration == max_iter:
        document = best_document  # ベストスコア版を採用して終了
        break

    corrected, n_fixed = correct(document, results)
    if n_fixed == 0:
        break  # 修正できないなら打ち切り

    save corrected as summary_ja_iter{N}.md
    document = corrected

if --editorial:
    document = append_editorial_note(document)

# スコアメタデータを更新して最終版を target に上書き
```

**ベストスコア追跡:**
各イテレーションでスコアが改善した場合のみ `best_document` を更新する。最終的にスコアが改善しないまま `max_iter` に達した場合、ベストスコアの版を採用する（修正が逆効果だった場合への保険）。

---

### `factfull/podcast/pipeline.py` — ポッドキャスト記事生成パイプライン

YouTube URL → ファクトチェック済み日本語記事 までをエンドツーエンドで実行する。
上位パイプライン（`homupe/pipelines/lex.py` など）から `PipelineResult` を受け取って
ブログ投稿・SNS 配信に連携できるよう設計されている。

```python
config = PipelineConfig(
    translate_model="translategemma:12b",
    analyze_model="gemma4:26b",
    factcheck_model="gemma4:e4b",
    editorial_model=None,   # None のとき factcheck_model を使用
    threshold=95.0,
    max_iter=5,
    editorial=True,
    blog_name="SoryuNews",
    reader_persona="英語圏情報にアクセスしたい日本語話者のエンジニア・研究者",
    n_questions=4,
)

result = run_pipeline(config, youtube_url, regen=False)
```

**`regen=True` のとき:**
同じ `video_id` を持つ既存エピソードディレクトリを探し、`section_summaries.json` があれば
Pass 1（チャンク別要点抽出）をスキップして Pass 2 以降から再実行する。

**`PipelineResult` の主要フィールド:**
| フィールド | 型 | 説明 |
|---|---|---|
| `video_id` | `str` | YouTube 動画 ID |
| `title` | `str` | YouTube タイトル（英語） |
| `channel` | `str` | チャンネル名 |
| `summary_path` | `Path` | 生成された `summary_ja.md` のパス |
| `episode_dir` | `Path` | エピソードディレクトリ |
| `score` | `float` | ファクトチェックスコア (0–100) |
| `metadata` | `dict` | `metadata.json` の全内容 |

---

### `factfull/podcast/archiver.py` — 記事生成エンジン

YouTube 動画から記事を生成する Map-Reduce パイプライン。

```
Step 1: メタデータ取得  (fetch_metadata)
Step 2: トランスクリプト取得  (fetch_transcript)
Step 3: 日本語翻訳  (translate_to_japanese)  ← translategemma:12b
Step 4: 日本語記事生成  (generate_summary)
  │
  ├── Pass 1:   チャンク別要点抽出（Map）      ← analyze_model
  ├── Pass 1.5: 英語引用抽出
  ├── Pass 2a:  論点生成（前半）               ← analyze_model
  ├── Pass 2b:  論点生成（後半）               ← analyze_model
  ├── Pass 2c:  概要・注目発言・キーワード       ← analyze_model
  └── Pass 2d:  「問いとして残るもの」生成      ← factcheck_model
```

`PipelineConfig` を渡すと、モデル名・チャンクサイズ・`blog_name`・`reader_persona`・
`n_questions` がすべてインスタンスに反映される。従来の引数指定（後方互換モード）も維持。

---

## データフロー サマリー

```
Truth ファイル群
  └─▶ [indexer] チャンク化 + BM25 構築
        └─▶ (bm25, chunks) ─────────────────────────┐
                                                     │
対象ドキュメント                                        │
  └─▶ [claim_extractor] クレーム抽出 (LLM)            │
        └─▶ [claim1, claim2, ...]                    │
              └─▶ for each claim:                    │
                    [retriever] BM25 検索 ◀──────────┘
                      └─▶ [evidence chunks]
                            └─▶ [verifier] LLM 照合
                                  └─▶ VerificationResult
                                        ├─▶ [reporter] レポート (.md)
                                        └─▶ [corrector] 外科的修正 (LLM)
                                              └─▶ 修正済みドキュメント
```

---

## 設計上の注意点

### LLM 呼び出し数
1回のイテレーションで LLM を呼ぶ回数:
- クレーム抽出: `ceil(doc_size / 4000)` 回
- クレーム検証: `max_claims` 回（デフォルト 50）
- セクション修正: 問題のあるセクション数分
- 合計: **最大数十回/イテレーション**

Ollama のリトライ設定（`max_retries=6`, `retry_wait=60`）により、1回のタイムアウトで最大 6 分待機する可能性がある。

### `num_ctx` の使い分け
| 処理 | `num_ctx` |
|---|---|
| クレーム抽出・検証 | 8192 |
| セクション修正 | 16384（長いセクションに対応） |
| 編集後記生成 | 16384 |

### スコアリングの特性
- `unverifiable` クレームはスコアに影響しない → Truth ソースに記載のない事実は評価対象外
- ドキュメントが Truth ソースの範囲外の情報を多く含む場合、スコアが過大評価される可能性がある

### セクション修正の限界
- クレームとセクションの Jaccard 係数が 0.15 未満のとき修正対象外（誤マッチ回避）
- 誤り箇所がプリアンブル（見出しなし冒頭部）にある場合は修正されない
