"""
PodcastArchiver.py
==================
YouTube ポッドキャストの中間生成物をすべて保存するパイプライン。

保存物:
  transcript_en_timestamped.json  英語字幕（タイムスタンプ付き）
  transcript_en.txt               英語字幕（プレーンテキスト）
  transcript_ja.txt               日本語訳（全文）
  section_summaries.json          チャンク別要点メモ（Map フェーズ中間物）
  summary_ja.md                   日本語詳細解説記事（YouTube 埋め込み・タイムスタンプ付き）
  fact_check.md                   ファクトチェック結果（Evaluator フェーズ）
  comments_raw.json               コメント生データ（上位50件）
  comments_summary_ja.md          コメント日本語まとめ
  metadata.json                   動画メタデータ

パイプライン:
  Step 1: メタデータ取得
  Step 2: 英語トランスクリプト取得（タイムスタンプ付き）
  Step 3: 日本語翻訳（チャンク分割）
  Step 4: 日本語要約生成（Map-Reduce 2パス）
    └ Step 4b: ファクトチェック（Generator-Evaluator）
  Step 5: コメント取得
  Step 6: コメントまとめ生成
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


class PodcastArchiver:
    OLLAMA_URL = os.environ.get(
        "OLLAMA_URL",
        "http://host.docker.internal:11435/api/generate",
    )

    # モデル用途別設定
    # - 翻訳: translategemma:12b（翻訳特化、8GB、高速）
    # - 要約・分析: glm-4.7-flash:latest（汎用、高品質）
    TRANSLATE_MODEL = "translategemma:12b"
    ANALYZE_MODEL   = "gemma4:e4b"

    OUTPUT_BASE = Path(
        os.environ.get("PODCAST_OUTPUT_DIR", str(Path.home() / "podcasts"))
    )

    # 翻訳チャンクサイズ（文字数）
    CHUNK_SIZE = 6000

    # 要約用チャンクサイズ（文字数）- 日本語トランスクリプト分割用
    SUMMARY_CHUNK_SIZE = 5000

    def __init__(self, youtube_url: str, model: str = None,
                 translate_model: str = None, analyze_model: str = None):
        self.youtube_url = youtube_url
        self.video_id = self._extract_video_id(youtube_url)
        # 個別指定 > クラスデフォルト。--model で両方まとめて上書きも可
        self.translate_model = translate_model or (model if model else self.TRANSLATE_MODEL)
        self.analyze_model   = analyze_model   or (model if model else self.ANALYZE_MODEL)
        self.model = self.analyze_model  # 後方互換
        self.metadata: dict = {}
        self.transcript_raw: list[dict] = []   # [{text, start, duration}, ...]
        self.transcript_en: str = ""
        self.transcript_ja: str = ""
        self.summary_ja: str = ""
        self.comments_raw: list[dict] = []
        self.comments_summary_ja: str = ""

        # 出力ディレクトリ: output/podcasts/{video_id}_{YYYYMMDD}/
        date_str = datetime.now().strftime("%Y%m%d")
        self.out_dir = self.OUTPUT_BASE / f"{self.video_id}_{date_str}"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_video_id(url: str) -> str:
        patterns = [
            r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
            r"(?:embed/)([A-Za-z0-9_-]{11})",
        ]
        for p in patterns:
            m = re.search(p, url)
            if m:
                return m.group(1)
        raise ValueError(f"Cannot extract video_id from URL: {url}")

    def _save_json(self, filename: str, data) -> Path:
        path = self.out_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  💾 {filename}")
        return path

    def _save_text(self, filename: str, text: str) -> Path:
        path = self.out_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  💾 {filename}")
        return path

    def _ollama(self, prompt: str, num_ctx: int = 8192, model: str = None,
                per_request_timeout: int = 600, max_retries: int = 6,
                retry_wait: int = 45) -> str:
        """
        Ollama API を呼び出す。タイムアウト時は retry_wait 秒待ってリトライする。
        nanoclaw などの並行ジョブに割り込まれても自動回復できる。
        streaming モードで接続を維持するためプロキシ (OLLAMA_URL) を常に使用する。
        """
        url = self.OLLAMA_URL
        payload = json.dumps({
            "model": model or self.analyze_model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.3, "num_ctx": num_ctx},
        }).encode("utf-8")

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                chunks = []
                with urllib.request.urlopen(req, timeout=per_request_timeout) as resp:
                    for line in resp:
                        if not line.strip():
                            continue
                        data = json.loads(line.decode("utf-8"))
                        chunks.append(data.get("response", ""))
                        if data.get("done"):
                            break
                return "".join(chunks).strip()
            except (TimeoutError, OSError, urllib.error.HTTPError, urllib.error.URLError) as e:
                last_err = e
                if attempt < max_retries:
                    print(f"\n  ⏳ タイムアウト (attempt {attempt}/{max_retries})、{retry_wait}秒後にリトライ... [{type(e).__name__}: {e}]",
                          flush=True)
                    time.sleep(retry_wait)
                else:
                    raise RuntimeError(
                        f"Ollama が {max_retries} 回タイムアウトしました: {last_err}"
                    ) from last_err

    # ------------------------------------------------------------------ #
    #  Timestamp utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """秒数を H:MM:SS または MM:SS 形式に変換する"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _yt_link(self, seconds: float) -> str:
        """YouTube タイムスタンプ付きリンクを生成する"""
        return f"https://www.youtube.com/watch?v={self.video_id}&t={int(seconds)}s"

    def _get_total_duration(self) -> float:
        """transcript_en_timestamped.json から総再生時間（秒）を返す"""
        ts_path = self.out_dir / "transcript_en_timestamped.json"
        if not ts_path.exists() or not self.transcript_raw:
            # transcript_raw が既にメモリにある場合も使う
            if self.transcript_raw:
                last = self.transcript_raw[-1]
                return last["start"] + last["duration"]
            return 0.0
        try:
            segments = json.loads(ts_path.read_text(encoding="utf-8"))
            if segments:
                last = segments[-1]
                return last["start"] + last["duration"]
        except Exception:
            pass
        return 0.0

    def _chunk_time_range(self, chunk_index: int, total_chunks: int) -> tuple[float, float]:
        """チャンク番号から対応する動画時刻の範囲（秒）を推定する"""
        total_dur = self._get_total_duration()
        if total_dur == 0:
            return (0.0, 0.0)
        start = (chunk_index / total_chunks) * total_dur
        end = ((chunk_index + 1) / total_chunks) * total_dur
        return (start, end)

    # ------------------------------------------------------------------ #
    #  Step 1: Metadata
    # ------------------------------------------------------------------ #

    def fetch_metadata(self) -> dict:
        print("\n📋 Step 1: メタデータ取得...")
        url = f"https://www.youtube.com/watch?v={self.video_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r:
            html = r.read().decode("utf-8")

        title_m = re.search(r'"title":"([^"]+)"', html)
        channel_m = re.search(r'"channelName":"([^"]+)"', html)
        date_m = re.search(r'"publishDate":"([^"]+)"', html)

        self.metadata = {
            "video_id": self.video_id,
            "url": self.youtube_url,
            "title": title_m.group(1) if title_m else "Unknown",
            "channel": channel_m.group(1) if channel_m else "Unknown",
            "publish_date": date_m.group(1) if date_m else "Unknown",
            "archived_at": datetime.now().isoformat(),
        }
        print(f"  Title  : {self.metadata['title']}")
        print(f"  Channel: {self.metadata['channel']}")
        self._save_json("metadata.json", self.metadata)
        return self.metadata

    # ------------------------------------------------------------------ #
    #  Step 2: Transcript (EN, with timestamps)
    # ------------------------------------------------------------------ #

    def fetch_transcript(self) -> str:
        print("\n📝 Step 2: 英語トランスクリプト取得...")
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        transcripts = api.list(self.video_id)

        # 手動字幕優先、なければ自動字幕
        chosen = None
        for t in transcripts:
            if t.language_code == "en" and not t.is_generated:
                chosen = t
                break
        if chosen is None:
            for t in transcripts:
                if t.language_code.startswith("en"):
                    chosen = t
                    break

        if chosen is None:
            raise RuntimeError("英語字幕が見つかりません")

        fetched = chosen.fetch()
        # [{text, start, duration}, ...]
        self.transcript_raw = [
            {"text": s.text, "start": round(s.start, 2), "duration": round(s.duration, 2)}
            for s in fetched
        ]
        self.transcript_en = " ".join(s["text"] for s in self.transcript_raw)

        print(f"  取得セグメント数: {len(self.transcript_raw)}")
        print(f"  総文字数: {len(self.transcript_en):,}")
        total_dur = self._get_total_duration()
        print(f"  総再生時間: {self._fmt_time(total_dur)}")

        self._save_json("transcript_en_timestamped.json", self.transcript_raw)
        self._save_text("transcript_en.txt", self.transcript_en)
        return self.transcript_en

    # ------------------------------------------------------------------ #
    #  Step 3: Japanese Translation (chunked)
    # ------------------------------------------------------------------ #

    def translate_to_japanese(self) -> str:
        print("\n🌐 Step 3: 日本語翻訳（チャンク分割）...")
        text = self.transcript_en
        chunks = [text[i: i + self.CHUNK_SIZE] for i in range(0, len(text), self.CHUNK_SIZE)]
        print(f"  チャンク数: {len(chunks)}")

        print(f"  翻訳モデル: {self.translate_model}")
        translated_parts = []
        for i, chunk in enumerate(chunks, 1):
            print(f"  翻訳中 {i}/{len(chunks)}...", end="\r", flush=True)
            prompt = (
                "Translate the following English text into natural Japanese.\n"
                "Keep conversational tone. Keep proper nouns (names, products) in original.\n"
                "Output only the translation, no comments.\n\n"
                f"{chunk}"
            )
            translated_parts.append(self._ollama(prompt, model=self.translate_model))

        self.transcript_ja = "\n\n".join(translated_parts)
        print(f"\n  翻訳完了: {len(self.transcript_ja):,}文字")
        self._save_text("transcript_ja.txt", self.transcript_ja)
        return self.transcript_ja

    # ------------------------------------------------------------------ #
    #  Step 4: Japanese Summary (Map-Reduce 2-pass)
    # ------------------------------------------------------------------ #

    def _generate_chunk_summaries(self, text: str) -> list[str]:
        """
        transcript_ja を SUMMARY_CHUNK_SIZE 文字ごとに分割し、
        各チャンクの要点・タイムスタンプ・発言者情報を抽出する（Map フェーズ）。
        """
        chunks = [text[i: i + self.SUMMARY_CHUNK_SIZE]
                  for i in range(0, len(text), self.SUMMARY_CHUNK_SIZE)]
        total = len(chunks)
        print(f"  チャンク数: {total}")

        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  チャンク要約 {i+1}/{total}...", end="\r", flush=True)

            # このチャンクに対応する動画時刻を推定
            t_start, t_end = self._chunk_time_range(i, total)
            time_range = ""
            if t_start > 0 or t_end > 0:
                time_range = (
                    f"この区間の動画時刻の目安: "
                    f"{self._fmt_time(t_start)} 〜 {self._fmt_time(t_end)}\n\n"
                )

            prompt = (
                f"{time_range}"
                "以下はポッドキャストの一部の日本語トランスクリプトです。\n"
                "この区間で語られた内容を、以下の形式でまとめてください：\n\n"
                "**[時刻目安]** この区間の開始時刻（例: 00:00）\n"
                "**[話題]** この区間のメインテーマ（1行）\n"
                "**[主な論点]** 箇条書き3〜8点。各点は「誰が（ゲスト/インタビュアー/固有名詞）」「何を主張したか」を含める\n"
                "**[注目発言]** 印象的な発言を1〜3件。形式：「発言者名: 発言内容（日本語）」\n"
                "**[英語引用]** この区間で最も印象的・キャッチーな英語の発言を原文のまま1〜2件。形式：「発言者名: 'quote'」\n"
                "**[具体的な数字・固有名詞]** この区間に登場した数値・確率・期間・製品名・人名など\n\n"
                "⚠️ 厳守事項：\n"
                "- 数値・確率・期間・固有名詞はトランスクリプトから一字一句変えずに抜き出すこと\n"
                "- 同一トピックに複数の確率・時期が登場する場合は、条件（例:「10年以内」「1〜2年以内」）を維持したまま別々の箇条書きに分けること\n"
                "- 過去の発言（「当時は〜だった」）と現在の信念（「現在は〜と考える」）を混同しないこと\n"
                "- 発言者は必ず具体的に（「ゲスト」「インタビュアー」「Dario」など）。「AI」「Speaker」は使わないこと\n"
                "- 自分の解釈・推論を加えないこと。トランスクリプトの情報のみ記載する\n\n"
                "余分な前置きは不要。内容のみ出力。\n\n"
                f"---\n{chunk}"
            )
            summaries.append(self._ollama(prompt, num_ctx=8192, per_request_timeout=3600,
                                           model="gemma4:e4b"))

        print(f"\n  チャンク要約完了: {total}件 / 合計{sum(len(s) for s in summaries):,}文字")
        return summaries

    def _extract_english_quotes(self, transcript_en: str) -> list[str]:
        """
        英語トランスクリプトから印象的な発言を直接引用する（Pass 2 用）。
        前半・中盤・後半の 3 箇所をサンプリングして最大 8 件返す。
        """
        if not transcript_en:
            return []

        total = len(transcript_en)
        sections = [
            transcript_en[:4000],
            transcript_en[max(0, total // 2 - 2000): total // 2 + 2000],
            transcript_en[max(0, total - 4000):],
        ]

        all_quotes: list[str] = []
        for section in sections:
            prompt = (
                "Read the following podcast transcript and extract 1-3 of the most "
                "memorable, quotable statements — strong opinions, surprising claims, "
                "or vivid analogies.\n\n"
                "Rules:\n"
                "- Use the speaker's EXACT words (verbatim). Do NOT paraphrase.\n"
                "- Prefix each quote with the speaker name if identifiable "
                "  (e.g. 'Dario: ...' or 'Host: ...')\n"
                "- One quote per line. Skip greetings, filler, meta-commentary.\n"
                "- Output ONLY the quotes, nothing else.\n\n"
                f"Transcript:\n{section}"
            )
            raw = self._ollama(prompt, num_ctx=8192, per_request_timeout=3600,
                               model="gemma4:e4b")
            for line in raw.splitlines():
                line = line.strip().lstrip("0123456789.-•* ")
                if len(line) > 20:
                    all_quotes.append(line)

        return all_quotes[:8]

    def _build_youtube_header(self) -> str:
        """記事冒頭の YouTube 埋め込みセクションを生成する"""
        vid = self.video_id
        title = self.metadata.get("title", "")
        channel = self.metadata.get("channel", "")
        total_dur = self._get_total_duration()
        duration_str = self._fmt_time(total_dur) if total_dur > 0 else ""

        lines = [
            "## 動画",
            "",
            f'<iframe width="100%" height="400" '
            f'src="https://www.youtube.com/embed/{vid}" '
            f'title="{title}" frameborder="0" '
            f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
            f'gyroscope; picture-in-picture" allowfullscreen></iframe>',
            "",
            f"**チャンネル**: {channel}  ",
        ]
        if duration_str:
            lines.append(f"**再生時間**: {duration_str}  ")
        lines += [
            f"**YouTube**: [外部で開く](https://www.youtube.com/watch?v={vid})",
            "",
            "---",
            "",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Pass 2 分割生成ヘルパー
    # ------------------------------------------------------------------ #

    def _pass2_section_prompt_rules(self, n_points_min: int, n_points_max: int) -> str:
        """論点生成の共通ルールブロックを返す"""
        return f"""## 論点の書き方

- メモから {n_points_min}〜{n_points_max} 点の論点を抽出する
- 発言者（実名か役職）を主語にした文章で書く
- 箇条書きは最小限。基本的に段落文章で書く
- 数値・確率はメモの表現をそのまま使う（言い換え・まとめ禁止）
- メモにない情報・推測・補完は書かない

**論点の深度を意図的に変える（全論点を均等に書かない）：**
- コアとなる重要論点（2〜3点）：2,000〜3,000字で徹底的に深掘り（複数段落、具体例・数値・含意を網羅）
- 中程度の論点（{n_points_min - 3}〜{n_points_min - 1}点）：1,000〜1,500字で丁寧に展開
- 補足・転換となる論点（1〜2点）：600〜900字でコンパクトに

**⚠️ 文字数の下限を守ること：各論点は最低800字。このフェーズ全体の合計は6,000字以上にすること。**

**構成パターンを論点ごとに変える（同じパターンを連続させない）：**
- 予測型：「主張 → 根拠・数値 → 社会的含意」
- 問い型：「問題提起 → 現状の限界 → 開かれた可能性」
- 対比型：「従来の常識 → 転換点となる発言 → 新しい枠組み」
- 実例型：「具体的なプロジェクト・数値 → メカニズム → 一般化」
- 懐疑型：「大胆な主張 → 留保・反論の余地 → それでも残る問い」
- 時間軸型：「現在の状況 → 近い将来（3〜5年）→ 遠い将来（20〜30年）」

⚠️ 絶対ルール：
- `#`（H1 見出し）は一切使わない。`####` のみ使う
- 「ゲスト」という単語は使わない。発言者は実名か役職で書く
- 見出し（`## 主要論点` など）は出力しない。`####` 論点タイトルから直接始める
- 各論点は最低800字以上"""

    def _generate_article_pass2a(
        self,
        chunks: list[str],
        title: str,
        channel: str,
        duration_str: str,
    ) -> str:
        """Pass 2a: 前半チャンクから論点を生成する"""
        combined = "\n\n---\n\n".join(
            f"【区間{i+1}】\n{s}" for i, s in enumerate(chunks)
        )
        n = len(chunks)
        n_min, n_max = max(5, n), max(7, n + 2)
        rules = self._pass2_section_prompt_rules(n_min, n_max)

        prompt = f"""あなたは「SoryuNews」というブログのポッドキャスト記事ライターです。
読者は英語圏情報にアクセスしたい日本語話者のエンジニア・研究者です。

以下は「{title}」（{channel}、総再生時間: {duration_str}）の**前半区間**の詳細メモです。
この前半メモをもとに、論点セクション（前半）を Markdown で書いてください。

---

{rules}

---

前半区間の詳細メモ:
{combined}
"""
        return self._ollama(prompt, num_ctx=32768, per_request_timeout=3600)

    def _generate_article_pass2b(
        self,
        chunks: list[str],
        chunk_offset: int,
        title: str,
        channel: str,
        duration_str: str,
        pass2a_text: str,
    ) -> str:
        """Pass 2b: 後半チャンクから論点を生成する（2aと重複しない）"""
        import re as _re
        combined = "\n\n---\n\n".join(
            f"【区間{i + chunk_offset + 1}】\n{s}" for i, s in enumerate(chunks)
        )
        n = len(chunks)
        n_min, n_max = max(5, n), max(7, n + 2)
        rules = self._pass2_section_prompt_rules(n_min, n_max)

        covered = _re.findall(r'^####\s+(.+)$', pass2a_text, _re.MULTILINE)
        covered_block = "\n".join(f"- {t}" for t in covered) if covered else "（なし）"

        prompt = f"""あなたは「SoryuNews」というブログのポッドキャスト記事ライターです。
読者は英語圏情報にアクセスしたい日本語話者のエンジニア・研究者です。

以下は「{title}」（{channel}、総再生時間: {duration_str}）の**後半区間**の詳細メモです。
この後半メモをもとに、論点セクション（後半）を Markdown で書いてください。

---

## すでに前半で取り上げた論点（重複禁止）

{covered_block}

上記と同じテーマ・内容は扱わないこと。

---

{rules}

---

後半区間の詳細メモ:
{combined}
"""
        return self._ollama(prompt, num_ctx=32768, per_request_timeout=3600)

    def _generate_article_pass2c(
        self,
        all_ronten: str,
        en_quotes: list[str],
        title: str,
        channel: str,
    ) -> str:
        """Pass 2c: 概要・注目の発言・キーワードを生成する"""
        quotes_block = (
            "\n".join(f"- {q}" for q in en_quotes) if en_quotes else "（英語引用の抽出なし）"
        )
        # 長すぎる場合は先頭 16,000 字だけ渡す
        ronten_for_prompt = all_ronten[:16000]

        prompt = f"""あなたは「SoryuNews」というブログのポッドキャスト記事ライターです。

以下は「{title}」（{channel}）の記事本文（論点セクション）です。
この本文をもとに、記事の**冒頭部分と末尾部分**を Markdown で書いてください。

---

## 出力フォーマット（この順番で出力すること）

## 概要

（200〜400字。このポッドキャストの位置づけと核心メッセージを 2〜4 文で。箇条書き不要。）

## 注目の発言

（4〜7件。英語引用リストから優先的に選ぶ。）

> **"英語引用"**
> 「日本語訳または要約」
>
> ―― **発言者名** `MM:SS`

## キーワード

（重要な用語・固有名詞を 15〜20 個。バッククォートで列挙、スラッシュ区切り。）

---

⚠️ 絶対ルール：
- `#`（H1 見出し）は使わない
- 上記 3 セクションのみ出力する（論点本文を繰り返さない）
- 概要は論点の内容を踏まえた要約（コピー貼り付けではない）

---

英語の印象的な発言（引用候補）:
{quotes_block}

---

記事本文（論点セクション）:
{ronten_for_prompt}
"""
        return self._ollama(prompt, num_ctx=32768, per_request_timeout=1800)

    # ------------------------------------------------------------------ #
    #  Step 4: Summary generation (Map-Reduce 3-phase Pass 2)
    # ------------------------------------------------------------------ #

    def generate_summary(self) -> str:
        print("\n📊 Step 4: 日本語要約生成（Map-Reduce 3フェーズ Pass 2）...")

        # transcript_ja を優先、なければ transcript_en を使用
        source_text = self.transcript_ja if self.transcript_ja else self.transcript_en
        if not source_text:
            print("  ⚠️ トランスクリプトが空です")
            return ""

        title = self.metadata.get("title", "")
        channel = self.metadata.get("channel", "")
        total_dur = self._get_total_duration()
        duration_str = self._fmt_time(total_dur) if total_dur > 0 else "不明"

        # --- Pass 1: Map ---
        print("  [Pass 1] チャンク別要点抽出（タイムスタンプ・発言者付き）...")
        chunk_summaries = self._generate_chunk_summaries(source_text)

        # 中間結果を保存
        self._save_json("section_summaries.json",
                        [[f"chunk_{i+1}", s] for i, s in enumerate(chunk_summaries)])

        # --- Pass 1.5: 英語引用の直接抽出 ---
        print("  [Pass 1.5] 英語引用抽出...")
        en_quotes = self._extract_english_quotes(self.transcript_en)
        print(f"  英語引用: {len(en_quotes)}件")

        # --- Pass 2: 3フェーズ分割生成 ---
        mid = len(chunk_summaries) // 2
        chunks_first = chunk_summaries[:mid]
        chunks_second = chunk_summaries[mid:]

        print(f"  [Pass 2a] 前半論点生成（チャンク 1〜{mid}、計{len(chunks_first)}件）...")
        t2a = time.time()
        pass2a = self._generate_article_pass2a(chunks_first, title, channel, duration_str)
        print(f"  Pass 2a 完了: {len(pass2a):,}文字 ({int(time.time()-t2a)}秒)")

        print(f"  [Pass 2b] 後半論点生成（チャンク {mid+1}〜{len(chunk_summaries)}、計{len(chunks_second)}件）...")
        t2b = time.time()
        pass2b = self._generate_article_pass2b(
            chunks_second, mid, title, channel, duration_str, pass2a
        )
        print(f"  Pass 2b 完了: {len(pass2b):,}文字 ({int(time.time()-t2b)}秒)")

        all_ronten = f"## 主要論点\n\n{pass2a}\n\n{pass2b}"

        print("  [Pass 2c] 概要・引用・キーワード生成...")
        t2c = time.time()
        pass2c = self._generate_article_pass2c(all_ronten, en_quotes, title, channel)
        print(f"  Pass 2c 完了: {len(pass2c):,}文字 ({int(time.time()-t2c)}秒)")

        # 記事本文を組み立て（概要 → 主要論点 → 注目の発言・キーワード）
        article_body = f"{pass2c}\n\n{all_ronten}"

        # 生成条件メタデータ（factfull スコアは後で refine_loop が埋める）
        gen_model = self.analyze_model or "unknown"
        gen_meta = (
            "\n\n---\n\n"
            f"*生成条件: Pass 2 モデル `{gen_model}`（3フェーズ分割）"
            f" / factfull 検証 `gemma4:e4b` / スコア: TBD*\n"
        )

        # YouTube 埋め込みヘッダーを冒頭に付加
        youtube_header = self._build_youtube_header()
        self.summary_ja = f"{youtube_header}{article_body}{gen_meta}"

        print(f"  要約完了: {len(self.summary_ja):,}文字")
        self._save_text("summary_ja.md", self.summary_ja)
        return self.summary_ja

    # ------------------------------------------------------------------ #
    #  Step 4b: Fact-check (Evaluator)
    # ------------------------------------------------------------------ #

    def evaluate_summary(self) -> str:
        """
        Generator-Evaluator パターンの Evaluator フェーズ。
        生成済みの summary_ja.md を英語トランスクリプトと照合してファクトチェックする。
        結果は fact_check.md に保存する。
        """
        print("\n🔍 Step 4b: ファクトチェック（Evaluator）...")

        # summary を取得
        summary = self.summary_ja
        if not summary:
            summary_path = self.out_dir / "summary_ja.md"
            if summary_path.exists():
                summary = summary_path.read_text(encoding="utf-8")
        if not summary:
            print("  ⚠️ summary_ja.md が見つかりません")
            return ""

        # 英語トランスクリプトを取得（メモリになければファイルから）
        en = self.transcript_en
        if not en:
            en_path = self.out_dir / "transcript_en.txt"
            if en_path.exists():
                en = en_path.read_text(encoding="utf-8")
        if not en:
            print("  ⚠️ transcript_en.txt が見つかりません")
            return ""

        # トランスクリプトをサンプリング（前半・中盤・後半を均等に）
        total = len(en)
        if total > 6000:
            sample = (
                en[:2000]
                + "\n\n...[中略]...\n\n"
                + en[total // 2 - 1000: total // 2 + 1000]
                + "\n\n...[中略]...\n\n"
                + en[-2000:]
            )
        else:
            sample = en

        # 記事の主要部分（長すぎる場合は前半を使用）
        summary_sample = summary[:4000]

        title = self.metadata.get("title", "")

        prompt = f"""あなたはポッドキャスト記事のファクトチェッカーです。

「{title}」について生成された日本語記事を、元の英語トランスクリプトと照合してください。

## チェック項目
1. **翻訳の正確性**: 日本語の解釈が英語原文と一致しているか
2. **数字・固有名詞の正確性**: 数値、年号、人名、製品名・組織名に誤りがないか
3. **主張の誤読・誇張・矮小化**: 発言者の意図が正しく伝わっているか
4. **捏造・幻覚（Hallucination）**: 元のトランスクリプトにない情報が含まれていないか
5. **発言者の誤帰属**: 誰の発言かが正しく示されているか

## 出力形式（Markdown）

### ✅ 正確に伝えられている点（3〜5点）
箇条書きで具体的に記述。

### ⚠️ 要確認・要修正の箇所
各問題について：
- **記事の記述**: （記事からの引用）
- **問題点**: 何が不正確か・不明確か
- **原文の証拠**: （英語トランスクリプトの該当箇所）
- **修正案**: どう直すべきか

### 📊 総合評価
- **信頼度スコア**: X / 100
- **総評**: （2〜3文で全体的な品質を評価）
- **最重要修正点**: あれば1〜2点に絞って指摘

---
生成された記事（抜粋）:
{summary_sample}

---
英語トランスクリプト（抜粋）:
{sample}
"""
        result = self._ollama(prompt, num_ctx=32768, per_request_timeout=900)
        print(f"  ファクトチェック完了: {len(result):,}文字")
        self._save_text("fact_check.md", result)
        return result

    def regenerate_summary(self) -> str:
        """
        既存の transcript_ja.txt を読み込んで要約とファクトチェックを再生成する。
        podcast_archive 済みのエピソードに対して要約を改善したい場合に使う。
        """
        ja_path = self.out_dir / "transcript_ja.txt"
        en_path = self.out_dir / "transcript_en.txt"
        meta_path = self.out_dir / "metadata.json"
        ts_path = self.out_dir / "transcript_en_timestamped.json"

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        if ja_path.exists():
            with open(ja_path, "r", encoding="utf-8") as f:
                self.transcript_ja = f.read()
        if en_path.exists():
            with open(en_path, "r", encoding="utf-8") as f:
                self.transcript_en = f.read()
        if ts_path.exists():
            with open(ts_path, "r", encoding="utf-8") as f:
                self.transcript_raw = json.load(f)

        self.generate_summary()
        # self.evaluate_summary()  # TODO: factfull リポジトリで別途実装
        return self.summary_ja

    # ------------------------------------------------------------------ #
    #  Step 5: Comments
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_int(val) -> int:
        """'1.2K' '838' '' None などを int に正規化"""
        if val is None or val == "":
            return 0
        s = str(val).strip().upper().replace(",", "")
        if s.endswith("K"):
            return int(float(s[:-1]) * 1000)
        if s.endswith("M"):
            return int(float(s[:-1]) * 1_000_000)
        try:
            return int(float(s))
        except ValueError:
            return 0

    def fetch_comments(self, max_comments: int = 50) -> list[dict]:
        """
        コメントを取得し、複合スコアで優先順位付けして保存する。

        優先順位の考え方:
          - SORT_BY_POPULAR で YouTube 側が弾いた「いいね上位」を取得
          - さらに score = votes + replies * 10 で再ソート
            → 議論が活発（返信多い）スレッドを浮かび上がらせる
        """
        print(f"\n💬 Step 5: コメント取得（最大{max_comments}件）...")
        try:
            from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
            downloader = YoutubeCommentDownloader()
            gen = downloader.get_comments_from_url(
                f"https://www.youtube.com/watch?v={self.video_id}",
                sort_by=SORT_BY_POPULAR,
            )
            raw = []
            for comment in gen:
                votes = self._parse_int(comment.get("votes", 0))
                replies = self._parse_int(comment.get("replies", 0))
                raw.append({
                    "id": comment.get("cid", ""),
                    "author": comment.get("author", ""),
                    "text": comment.get("text", ""),
                    "votes": votes,
                    "reply_count": replies,
                    "score": votes + replies * 10,   # 複合スコア
                    "time": comment.get("time", ""),
                })
                if len(raw) >= max_comments:
                    break

            # 複合スコアで再ソート（議論が活発なものを上位に）
            self.comments_raw = sorted(raw, key=lambda x: x["score"], reverse=True)

            # サマリ表示
            print(f"  取得コメント数: {len(self.comments_raw)}")
            print(f"  Top3 (score順):")
            for c in self.comments_raw[:3]:
                print(f"    votes={c['votes']:>5}  replies={c['reply_count']:>3}"
                      f"  score={c['score']:>6}  {c['text'][:60]}")

        except Exception as e:
            print(f"  ⚠️ コメント取得失敗: {e}")
            self.comments_raw = []

        self._save_json("comments_raw.json", self.comments_raw)
        return self.comments_raw

    # ------------------------------------------------------------------ #
    #  Step 6: Comments Summary
    # ------------------------------------------------------------------ #

    def summarize_comments(self) -> str:
        print("\n📣 Step 6: コメントまとめ生成...")
        if not self.comments_raw:
            self.comments_summary_ja = "（コメントを取得できませんでした）"
            self._save_text("comments_summary_ja.md", self.comments_summary_ja)
            return self.comments_summary_ja

        # スコア順（既にソート済み）上位30件をテキスト化
        # votes=いいね数 / replies=返信数 を明示してLLMに文脈を伝える
        comments_text = "\n".join(
            f"[👍{c.get('votes',0)} 💬{c.get('reply_count',0)}replies] {c['text'][:300]}"
            for c in self.comments_raw[:30]
        )
        title = self.metadata.get("title", "")

        prompt = f"""以下は「{title}」のYouTubeコメント欄（高評価順）です。

視聴者の反応・議論を日本語でまとめてください。

## まとめの構成
1. **全体的な反応**: 視聴者の雰囲気（好意的/批判的/議論的 など）
2. **頻出テーマ**: コメントで繰り返し言及されているテーマ（3〜5点）
3. **注目コメント**: 特に示唆に富む・反応が多いコメントを3件（原文 + 日本語訳）
4. **批判・疑問**: 懐疑的・批判的なコメントがあれば要約

---
コメント一覧:
{comments_text}
"""
        self.comments_summary_ja = self._ollama(prompt)
        print(f"  コメントまとめ完了: {len(self.comments_summary_ja):,}文字")
        self._save_text("comments_summary_ja.md", self.comments_summary_ja)
        return self.comments_summary_ja

    # ------------------------------------------------------------------ #
    #  Run All
    # ------------------------------------------------------------------ #

    def run(self) -> str:
        print(f"\n🎙️ PodcastArchiver 開始")
        print(f"   URL: {self.youtube_url}")
        print(f"   出力先: {self.out_dir}")
        start = time.time()

        self.fetch_metadata()
        self.fetch_transcript()
        self.translate_to_japanese()
        self.generate_summary()
        # self.evaluate_summary()  # TODO: factfull リポジトリで別途実装
        self.fetch_comments()
        self.summarize_comments()

        elapsed = time.time() - start
        print(f"\n✅ 完了！ ({elapsed:.0f}秒)")
        print(f"   保存先: {self.out_dir}")
        print(f"   ファイル一覧:")
        for f in sorted(self.out_dir.iterdir()):
            size = f.stat().st_size
            print(f"     {f.name:<45} {size:>8,} bytes")
        return str(self.out_dir)
