"""
factfull/podcast/pipeline.py
==============================
ポッドキャスト処理パイプラインのオーケストレーター。

各ステップの実装は factfull/podcast/steps/ 以下に分離されている:
  steps/transcript.py  — Step 1-4: fetch / translate / summarize
  steps/factcheck.py   — Step 5-6: factcheck loop + critique + editorial
  steps/graph.py       — Step 7:   KG extract (Grounded) + Wikidata normalize

使い方:
    from factfull.podcast.pipeline import PipelineConfig, run_pipeline

    config = PipelineConfig(analyze_model="gemma4:26b", write_graph=True)
    result = run_pipeline(config, youtube_url)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ── 結果 ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    video_id: str
    title: str
    channel: str
    summary_path: Path
    episode_dir: Path
    score: float
    metadata: dict

    def to_processed_doc(self):
        """factfull.core.types.ProcessedDoc に変換する（後方互換）。"""
        from factfull.core.types import SourceDoc, ProcessedDoc

        transcript_path    = self.episode_dir / "transcript_en.txt"
        transcript_ja_path = self.episode_dir / "transcript_ja.txt"

        source = SourceDoc(
            source_type="podcast",
            source_id=self.video_id,
            title=self.title,
            text=transcript_path.read_text(encoding="utf-8") if transcript_path.exists() else "",
            text_ja=transcript_ja_path.read_text(encoding="utf-8") if transcript_ja_path.exists() else "",
            metadata={"channel": self.channel, "youtube_url": self.metadata.get("url", ""), **self.metadata},
        )
        summary = self.summary_path.read_text(encoding="utf-8") if self.summary_path.exists() else ""
        return ProcessedDoc(source=source, summary=summary, score=self.score, summary_path=self.summary_path)


# ── 設定 ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # モデル
    translate_model: str = "translategemma:12b"
    analyze_model: str   = "gemma4:26b"
    extract_model: str   = "gemma4:26b"   # KG抽出モデル（write_graph=True 時に使用）
    factcheck_model: str = "gemma4:e4b"
    editorial_model: str | None = None    # None のとき factcheck_model を使用

    # チャンクサイズ
    translate_chunk_size: int = 6000
    summary_chunk_size: int   = 5000

    # ファクトチェックループ
    threshold: float  = 95.0
    max_iter: int     = 5
    max_claims: int   = 50
    top_k: int        = 5

    # 機能フラグ
    critique: bool      = True
    editorial: bool     = True
    fetch_comments: bool = False
    write_graph: bool   = False   # True のとき完了後に Neo4j へ自動書き込み

    # 出力先
    output_base: Path = field(default_factory=lambda: Path.home() / "podcasts")

    # コンテンツ設定
    blog_name: str      = "SoryuNews"
    reader_persona: str = "英語圏情報にアクセスしたい日本語話者のエンジニア・研究者"
    n_questions: int    = 4


# ── エントリポイント ───────────────────────────────────────────────────────────

def run_pipeline(
    config: PipelineConfig,
    youtube_url: str,
    regen: bool = False,
) -> PipelineResult:
    """
    エンドツーエンドパイプライン:
      Step 1-4 : トランスクリプト取得・翻訳・サマリー生成
      Step 5-6 : ファクトチェック + 批評 + 編集後記
      Step 7   : ナレッジグラフ書き込み + Wikidata 正規化（write_graph=True 時）

    Returns:
        PipelineResult — ブログ投稿・SNS 配信などに必要な情報をすべて含む
    """
    from factfull.podcast import steps

    result = steps.fetch_episode(config, youtube_url, regen)
    result.score = steps.run_factcheck(result, config)

    if config.write_graph:
        steps.write_to_graph(result, config)

    return result
