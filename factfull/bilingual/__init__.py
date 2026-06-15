"""
factfull.bilingual
==================
論文 1 本を英日対訳（段落単位・図表位置保持）の構造化 JSON に変換する
パイプライン。後段の homupe で記事化する際にレイアウトを自由に組み替えられる。

    from factfull.bilingual.pipeline import BilingualConfig, run_bilingual
    result = run_bilingual(BilingualConfig(), "2403.11996")
"""
from factfull.bilingual.pipeline import BilingualConfig, BilingualResult, run_bilingual
from factfull.bilingual.types import Block, BilingualDoc

__all__ = [
    "BilingualConfig",
    "BilingualResult",
    "run_bilingual",
    "BilingualDoc",
    "Block",
]
