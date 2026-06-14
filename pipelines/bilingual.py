#!/usr/bin/env python3
"""
論文 英日対訳パイプライン
=========================
論文（arXiv ID / URL またはローカル PDF）を完全英日対訳の構造化 JSON にする。
Scholaread の代替（オフライン・クォータなし）。図表は原文位置を保持して
画像抽出し、見出しも抽出・翻訳する。

使い方:
  # arXiv ID から
  python pipelines/bilingual.py 2403.11996

  # arXiv URL から
  python pipelines/bilingual.py https://arxiv.org/abs/2403.11996

  # ローカル PDF から
  python pipelines/bilingual.py ~/papers/foo.pdf

  # 参考文献も JSON に残す（未翻訳）／キャプションを除外
  python pipelines/bilingual.py 2403.11996 --keep-references --skip-captions

環境変数:
  FACTFULL_OLLAMA_URL    Ollama エンドポイント（既定 http://localhost:11435/api/generate）
  FACTFULL_LLM_BACKEND   ollama（既定）/ anthropic

出力:
  {output-base}/{source_id}/bilingual.json   対訳ドキュメント
  {output-base}/{source_id}/assets/          抽出した図表画像
"""
import argparse
from pathlib import Path

from factfull.bilingual.pipeline import BilingualConfig, run_bilingual


def main() -> None:
    p = argparse.ArgumentParser(
        description="論文 → 英日対訳 JSON パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("source", help="arXiv ID / URL またはローカル PDF パス")
    p.add_argument("--model", default="translategemma:12b", help="翻訳モデル")
    p.add_argument("--batch-chars", type=int, default=3000,
                   help="1 バッチあたりの英語文字数")
    p.add_argument("--keep-references", action="store_true",
                   help="References を JSON に残す（未翻訳）")
    p.add_argument("--skip-captions", action="store_true",
                   help="図表キャプションを除外する")
    p.add_argument("--num-ctx", type=int, default=8192, help="Ollama num_ctx")
    p.add_argument("--output-base", type=Path,
                   default=Path.home() / "papers" / "bilingual",
                   help="出力ルートディレクトリ")
    p.add_argument("--dump-raw", action="store_true",
                   help="抽出した生テキストブロックを extract_raw.json に出力")
    args = p.parse_args()

    config = BilingualConfig(
        model=args.model,
        batch_chars=args.batch_chars,
        num_ctx=args.num_ctx,
        skip_references=not args.keep_references,
        skip_captions=args.skip_captions,
        output_base=args.output_base,
        dump_raw=args.dump_raw,
    )
    result = run_bilingual(config, args.source)
    print(f"\n✅ 対訳 JSON: {result.json_path}")
    print(f"   {result.title_en}")
    print(f"   → {result.title_ja}")
    print(f"   ブロック数: {result.n_blocks}  モデル: {result.model}")


if __name__ == "__main__":
    main()
