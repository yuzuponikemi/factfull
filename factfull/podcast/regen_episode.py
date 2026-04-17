"""
汎用エピソード再生成スクリプト
Usage: python3 regen_episode.py <video_id>

環境変数:
  PODCAST_OUTPUT_DIR  エピソードディレクトリの親 (デフォルト: ~/podcasts)
  OLLAMA_URL          Ollama API エンドポイント

- section_summaries.json があれば Pass 1 をスキップ
- なければ Pass 1 からフルで実行
"""
import json, os, sys, time
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python3 regen_episode.py <video_id>")
    sys.exit(1)

VIDEO_ID = sys.argv[1]
EP_BASE = Path(os.environ.get("PODCAST_OUTPUT_DIR", str(Path.home() / "podcasts")))

from factfull.podcast.archiver import PodcastArchiver

# エピソードディレクトリを特定（transcript_en.txt を持つ最古のもの）
ep_dirs = sorted(EP_BASE.glob(f"{VIDEO_ID}_*"))
EP_DIR = None
for d in ep_dirs:
    if (d / "transcript_en.txt").exists():
        EP_DIR = d
        break
if EP_DIR is None:
    print(f"[ERROR] エピソードディレクトリが見つかりません: {VIDEO_ID}")
    sys.exit(1)

meta = json.loads((EP_DIR / "metadata.json").read_text())
transcript_en = (EP_DIR / "transcript_en.txt").read_text()
transcript_ja_path = EP_DIR / "transcript_ja.txt"
transcript_ja = transcript_ja_path.read_text() if transcript_ja_path.exists() else ""

print(f"タイトル: {meta.get('title', '?')}")
print(f"ディレクトリ: {EP_DIR.name}")

arch = PodcastArchiver(youtube_url=f"https://www.youtube.com/watch?v={meta['video_id']}")
arch.analyze_model = "gemma4:26b"
arch.out_dir = EP_DIR
arch.metadata = meta
arch.transcript_en = transcript_en
arch.transcript_ja = transcript_ja

sec_path = EP_DIR / "section_summaries.json"

if sec_path.exists():
    # ── REGEN モード: Pass 1 スキップ ──────────────────────────────────
    print("\n[モード] REGEN（section_summaries.json を再利用）")
    section_data = json.loads(sec_path.read_text())
    chunk_summaries = [s for _, s in section_data]

    title = meta.get("title", "")
    channel = meta.get("channel", "Unknown")
    total_dur = arch._get_total_duration()
    duration_str = arch._fmt_time(total_dur) if total_dur > 0 else "不明"

    print(f"チャンク数: {len(chunk_summaries)}")

    print("\n[Pass 1.5] 英語引用抽出...")
    t = time.time()
    en_quotes = arch._extract_english_quotes(transcript_en)
    print(f"  完了: {len(en_quotes)}件 ({int(time.time()-t)}秒)")

    mid = len(chunk_summaries) // 2
    print(f"\n[Pass 2a] 前半論点生成（チャンク 1〜{mid}）...")
    t = time.time()
    pass2a = arch._generate_article_pass2a(chunk_summaries[:mid], title, channel, duration_str)
    e = int(time.time()-t)
    print(f"  完了: {len(pass2a):,}文字 ({e}秒 / {e//60}分{e%60}秒)")

    print(f"\n[Pass 2b] 後半論点生成（チャンク {mid+1}〜{len(chunk_summaries)}）...")
    t = time.time()
    pass2b = arch._generate_article_pass2b(chunk_summaries[mid:], mid, title, channel, duration_str, pass2a)
    e = int(time.time()-t)
    print(f"  完了: {len(pass2b):,}文字 ({e}秒 / {e//60}分{e%60}秒)")

    all_ronten = f"## 主要論点\n\n{pass2a}\n\n{pass2b}"

    print("\n[Pass 2c] 概要・引用・キーワード生成...")
    t = time.time()
    pass2c = arch._generate_article_pass2c(all_ronten, en_quotes, title, channel)
    print(f"  完了: {len(pass2c):,}文字 ({int(time.time()-t)}秒)")

    article_body = f"{pass2c}\n\n{all_ronten}"

    print("\n[Pass 2d] 哲学的な問い生成（前半・後半 各2問）...")
    t = time.time()
    questions_a = arch._generate_questions_from_ronten(pass2a, n=2)
    questions_b = arch._generate_questions_from_ronten(pass2b, n=2)
    questions_text = (questions_a + "\n\n" + questions_b).strip()
    e = int(time.time()-t)
    print(f"  完了: {len(questions_text):,}文字 ({e}秒)")
    questions_section = f"\n\n## 問いとして残るもの\n\n{questions_text}" if questions_text else ""

    gen_meta = (
        "\n\n---\n\n"
        "*生成条件: Pass 2 モデル `gemma4:26b`（3フェーズ分割）"
        " / factfull 検証 `gemma4:e4b` / スコア: TBD*\n"
    )
    youtube_header = arch._build_youtube_header()
    summary = f"{youtube_header}{article_body}{questions_section}{gen_meta}"
    (EP_DIR / "summary_ja.md").write_text(summary, encoding="utf-8")
    print(f"\n✅ 保存完了: {EP_DIR / 'summary_ja.md'}")
    print(f"   合計文字数: {len(summary):,}文字")

else:
    # ── FULL モード: Pass 1 から実行 ──────────────────────────────────
    print("\n[モード] FULL（Pass 1 から実行）")
    arch.generate_summary()
