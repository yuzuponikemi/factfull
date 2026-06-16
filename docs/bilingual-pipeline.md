# 論文 英日対訳パイプライン & homupe 記事化

> 2026-06-16 記録。`factfull.bilingual`（論文 → 英日対訳 JSON）と、その出力を
> homupe（SoryuNews / MkDocs）の**対訳ビューア記事**として公開する一連の流れ・
> 設計判断・ハマりどころをまとめた作業メモ。denno プロジェクトの
> 「収集・検証・翻訳(factfull) → 公開(homupe)」フローの実体。

---

## 全体像

```
arXiv ID / URL / PDF
        │
        ▼  factfull（Knowledge 層）
  pipelines/bilingual.py
   extract → segment → translate → JSON
        │
        ▼  出力
  ~/papers/bilingual/<source_id>/
     ├ bilingual.json      対訳ドキュメント（EN/JA ブロック列）
     ├ assets/*.png        図表画像（原文位置を保持）
     └ extract_raw.json    （--dump-raw 時のデバッグ用）
        │
        ▼  homupe（公開層）
  scripts/create_paper_article.py   ← 決定論的レンダラ（LLM 不使用）
   bilingual.json → 対訳記事 Markdown ＋ 図表コピー
        │
        ▼
  docs/blog/posts/YYYY/MM/YYYY-MM-DD-<slug>.md（カテゴリ Research）
  docs/data/papers/<source_id>/*.png
        │
        ▼  main 直 push → Cloudflare Pages 自動デプロイ
  https://（SoryuNews）/blog/YYYY/MM/DD/<タイトル>/
```

---

## 1. factfull 側: 対訳 JSON の生成

### 実行

```bash
# factfull リポジトリで（venv は .venv、Python 3.11）
which python   # .venv を確認
python pipelines/bilingual.py 1706.03762          # arXiv ID
python pipelines/bilingual.py https://arxiv.org/abs/1706.03762
python pipelines/bilingual.py ~/papers/foo.pdf    # ローカル PDF
# 主なオプション: --model / --num-ctx / --keep-references / --skip-captions / --dump-raw
```

- 既定モデル `translategemma:12b`、Ollama は `http://localhost:11435`（**localhost**。
  Docker でない限り `host.docker.internal` は使わない）。
- 長時間実行: `head`/`tail` でパイプしない（SIGPIPE で死ぬ）。
  `... 2>&1 | tee logs/run-$(date +%s).log` を使う。

### パイプライン構成（`factfull/bilingual/`）

| 段階 | ファイル | 役割 |
|---|---|---|
| extract | `extract.py` | pymupdf でテキスト（フォントサイズ・太字つき）＋図表を読み順に抽出 |
| segment | `segment.py` | 見出し/段落/図表/キャプション/著者へ整形、section_path・ID 付与 |
| translate | `translate.py` | 段落バッチ翻訳（ja 充填）＋タイトル/アブストラクト翻訳 |
| 型 | `types.py` | `BilingualDoc` / `Block`（plain dataclass・to_dict/from_dict） |
| オーケストレータ | `pipeline.py` | 固定ステップで LLM を N 回呼ぶ（ReAct ループではない） |

### bilingual.json スキーマ

ルート `BilingualDoc`: `title_en/ja`・`authors`・`abstract_en/ja`・`arxiv_url`・
`source_id`・`model`・`metadata`・`blocks`。

`Block`（フラットに読み順で並ぶ）: `id`・`type`・`en`・`ja`・`level`・
`section_path`・`page`・`bbox`・`image_path`(assets 相対)・`label`・`skip_translate`。
`type` ∈ {title, heading, abstract, paragraph, caption, reference, figure, table}。

> 設計意図: homupe 側は blocks を走査し `type`/`level` だけでレイアウトを自由に
> 決められる（Scholaread 風の対訳レイアウト）。

### 抽出/セグメント層のロバストネス（2026-06-15 強化済み）

10 本の多様な論文で `extract → segment` を診断する回帰ハーネス
`scripts/segment_diag.py`（翻訳なしで故障モードを定量化）を整備。主な改善:

- **著者誤分類**: page1 の `@`/所属語 短行を heading でなく `skip_translate` 段落に分離。
- **図表ノイズ**: 装飾/反復ラスタ画像を `_keep_image` ＋ `_drop_repeated_images` で除去。
- **caption 誤判定**: 「Table 2 summarizes …」等の本文中参照を caption にしない（番号直後の区切り必須）。
- **読み順**: 全幅図割込時の二段組を帯分割で正す。
- **ベクター図捕捉**: ResNet 等 `figure=0` を解消（`get_drawings` 密集領域＋Figure キャプション隣接で図化）。
- **罫線無しテキスト表捕捉**: VGG 等の文字化けセル段落を一掃（Table キャプション直下の整列セルを表画像化）。

**既知の限界**: BERT 等の **2 段組テキスト主体表**（キャプションが表の下＋多列 gappy 行）は
VGG 型と判定原理が異なり未対応。ViT 等の図はラスタのサブパネルを忠実抽出するため粒度が細かい
（不具合ではない）。詳細・前後の数値は当時の調査ログ／コミット
（`fix(bilingual): …`, `feat(bilingual): ベクター図…`, `feat(bilingual): 罫線無しテキスト表…`）参照。

---

## 2. homupe 側: 対訳ビューア記事の生成

### スクリプト `homupe/scripts/create_paper_article.py`

bilingual.json を読み、**決定論的に**（LLM 呼び出しなし）対訳記事 Markdown を組み立てる。
`scripts/create_blog_posts.py` の組み立て方を踏襲。

```bash
# homupe リポジトリで
uv sync --group scripts                                  # mkdocs/factfull 等
uv run python scripts/create_paper_article.py            # 既定: ~/papers/bilingual/1706.03762/bilingual.json
uv run python scripts/create_paper_article.py ~/papers/bilingual/<id>/bilingual.json
uv run python scripts/create_paper_article.py <json> --date 2026-06-16 --category Research
```

生成物:
- 記事 `docs/blog/posts/YYYY/MM/YYYY-MM-DD-<英語タイトルslug>.md`（blog プラグインが
  自動でナビ・一覧に載せるので `mkdocs.yml` の nav 編集は**不要**）。
- 図表 `docs/data/papers/<source_id>/*.png`（assets からコピー）。

レンダリング規則（`build_body`）:
- `heading` → `##`/`###`（level+1, H1 は記事タイトル用に温存）、日本語＋EN 併記。
- `paragraph`/`abstract` → `<div class="bi"><p class="en">…</p><p class="ja">…</p></div>`。
- `figure`/`table` → **Markdown 画像** `![label](rel){ .bi-fig }`（後述の理由で生 HTML img は使わない）。
- `caption` → `<p class="bi-cap">` で図表キャプション（EN/JA）。
- `skip_translate`（著者ブロック等）・`reference` → 本文に出さない。
- 抜粋（`<!-- more -->` 前）は `abstract_ja` 冒頭 2 文。

### スタイル

- `docs/stylesheets/bilingual.css`（EN を控えめ・JA を主役、`img.bi-fig` 中央寄せ＋白背景）。
- `mkdocs.yml` の `extra_css` に `stylesheets/bilingual.css` を登録、
  `markdown_extensions` に **`attr_list`** を追加（画像へのクラス付与に必要）。

---

## 3. MkDocs ハマりどころ（重要）

1. **相対パス補正は Markdown 画像のみ**。MkDocs は `![](path)` の相対パスを
   ビルド先 URL 階層に**自動補正**するが、**生 HTML の `<img src>` は補正しない**。
   → 図表は必ず Markdown 画像構文で出す（`{ .bi-fig }` は `attr_list` でクラス付与）。
2. **ブログ記事の URL 階層は深い**: `blog/YYYY/MM/DD/<日本語タイトル>/`。
   ソースの `../../../../data/...`（docs まで 4 つ上）がビルド後は 5 つ上に補正される。
   生 HTML で固定したい場合は 5 つ上（`../../../../../data/...`）が必要。
3. **カテゴリは `mkdocs.yml` の `categories_allowed` 限定**:
   `Podcast / 日記 / 開発記 / Synthesis / Book Guide / Research`。論文記事は **Research**。
   （`docs/AGENTS.md`・`docs/howto.md` のカテゴリ一覧は古いので mkdocs.yml が正）。
4. **公開前チェック必須**: `uv run mkdocs build --strict`（warning 1 件でも push しない）。
   フロントマターに `<!-- more -->` 必須。
5. **画像置き場**: `docs/data/` 配下（記事から相対参照）。論文図表は `docs/data/papers/<id>/`。
6. **既知の地雷**: 未追跡のナイトリー生成物
   `docs/blog/posts/2026/06/2026-06-12-arxiv-digest.md` は tags が `- #具現化AI` と
   書かれ、YAML で `#` がコメント扱い→ None になり **ローカル `--strict` を止める**。
   未追跡なので CI/本番には無影響。直すなら tags の `#` を外す。

### デプロイ

- homupe は **main 直 push**（PR 不要）。push 後 Cloudflare Pages が 1〜2 分でデプロイ。
- CI（`.github/workflows/docs.yml`）は main push で `mkdocs build --strict` を実行。

---

## 4. 新しい論文を記事化する手順（runbook）

```bash
# 1) factfull で対訳 JSON を生成（ローカル: arxiv 到達 ＋ Ollama 起動が必要）
cd ~/source/personal/factfull
which python
python pipelines/bilingual.py <arxiv_id> 2>&1 | tee logs/run-$(date +%s).log

# 2) homupe で記事化
cd ~/source/personal/homupe
uv sync --group scripts
uv run python scripts/create_paper_article.py ~/papers/bilingual/<id>/bilingual.json

# 3) ビルド検証（壊れた未追跡 arxiv-digest がある場合は一時退避してから）
uv run mkdocs build --strict
uv run mkdocs serve   # 目視（任意）

# 4) 公開（main 直 push）
git add scripts/ docs/stylesheets/bilingual.css mkdocs.yml \
        docs/blog/posts/<Y>/<M>/<file>.md docs/data/papers/<id>/
git commit -m "post: 論文対訳「…」"
git push origin main   # behind なら git pull --rebase origin main 後に push
```

---

## 関連ファイル

- factfull: `pipelines/bilingual.py`, `factfull/bilingual/{extract,segment,translate,pipeline,types}.py`,
  `scripts/segment_diag.py`（回帰診断）。
- homupe: `scripts/create_paper_article.py`, `docs/stylesheets/bilingual.css`,
  `mkdocs.yml`（extra_css / attr_list / categories_allowed）。
- 初公開記事: `homupe/docs/blog/posts/2026/06/2026-06-16-attention-is-all-you-need.md`
  （「注意機構が全てである。」/ Attention Is All You Need）。
