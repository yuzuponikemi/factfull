#!/usr/bin/env python3
"""
27個の Lex Fridman エピソード全てをNeo4jに一括登録する
"""
import sys
import json
from pathlib import Path

# Lex Fridman エピソード（YouTube ID）
EPISODES = [
    "-HzgcbRXUK8",  # Demis Hassabis
    "U1H1Ob7jk8Q",  # Jack Weatherford
    "y3yAVZk3tyA",  # Keyu Jin
    "jdCKiEJpwf4",  # Scott Horton
    "HsLgZzgpz9Y",  # Dave Plummer
    "-Qm1_On71Oo",  # Dave Hone
    "SvKv7D4pBjE",  # Norman Ohler
    "qjPH9njnaVU",  # Pavel Durov
    "7OLVwZeMCfY",  # Julia Shaw
    "o3gbXDjNWyI",  # Dan Houser
    "m_CFCyc2Shs",  # David Kirtley
    "Qp0rCU49lMs",  # Michael Levin
    "_bBRVNkAfkQ",  # Ancient Civilizations
    "14OPT6CcsH4",  # Infinity, Gödel
    "Z-FRe5AKmCU",  # Paul Rosolie
    "EV7WhVT270Q",  # State of AI
    "YFjfBk8HI5o",  # OpenClaw
    "KGVpKPNUdzA",  # Khabib
    "1SJiTwbSI58",  # Rick Beato
    "H9rF1CSSh-w",  # Jeff Kaplan
    "vif8NQcjVf0",  # Jensen Huang
    "iKx3gAODybU",  # Vikings
    "3W5FWUN5w2Q",  # Jeffrey Wasserstrom
    "A6m4iJIw_84",  # Janna Levin
    "9V6tWC4CdFQ",  # Sundar Pichai
    "HUkBz-cdB-k",  # Terence Tao
    "vagyIcmIGOQ",  # DHH
]

def main():
    from factfull.podcast.pipeline import PipelineConfig, PipelineResult
    from factfull.podcast.steps.graph import write_to_graph

    podcast_dir = Path.home() / "podcasts"

    print(f"📊 {len(EPISODES)}個のエピソードをKGに登録します")
    print(f"📁 処理ディレクトリ: {podcast_dir}\n")

    success = 0
    failed = 0
    skipped = 0

    config = PipelineConfig(write_graph=True)

    for yt_id in EPISODES:
        # 最新の処理済みディレクトリを見つける
        matching_dirs = sorted(podcast_dir.glob(f"{yt_id}_*"), reverse=True)

        if not matching_dirs:
            print(f"⚠️  {yt_id}: ディレクトリが見つかりません")
            skipped += 1
            continue

        episode_dir = matching_dirs[0]
        summary_file = episode_dir / "summary_ja.md"
        metadata_file = episode_dir / "metadata.json"

        if not summary_file.exists():
            print(f"⚠️  {yt_id}: summary_ja.md が見つかりません")
            skipped += 1
            continue

        try:
            # メタデータを読み込む
            metadata = {}
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

            # PipelineResult を再構築
            result = PipelineResult(
                video_id=yt_id,
                title=metadata.get("title", "Unknown"),
                channel="Lex Fridman",
                summary_path=summary_file,
                episode_dir=episode_dir,
                score=metadata.get("score", 0),
                metadata=metadata,
            )

            print(f"🔄 処理中: {yt_id}...", end=" ", flush=True)
            write_to_graph(result, config)
            print("✅")
            success += 1
        except Exception as e:
            print(f"❌ {e}")
            failed += 1

    print(f"\n📊 結果:")
    print(f"   ✅ 成功: {success}")
    print(f"   ❌ 失敗: {failed}")
    print(f"   ⚠️  スキップ: {skipped}")
    print(f"   合計: {len(EPISODES)}")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
