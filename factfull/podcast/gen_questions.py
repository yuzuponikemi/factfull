"""
哲学的な問いを生成して summary_ja.md に挿入するスクリプト
Usage: python3 gen_questions.py <ep_dir>

環境変数:
  OLLAMA_URL  Ollama API エンドポイント

- summary_ja.md の「---\n\n*生成条件:」の直前に「## 問いとして残るもの」を挿入
- すでにセクションが存在する場合は上書きする
"""
import os, sys, json
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python3 gen_questions.py <ep_dir>")
    sys.exit(1)

EP_DIR = Path(sys.argv[1])

summary_path = EP_DIR / "summary_ja.md"
transcript_path = EP_DIR / "transcript_en.txt"

if not summary_path.exists():
    print(f"[ERROR] summary_ja.md が見つかりません: {EP_DIR}")
    sys.exit(1)

summary = summary_path.read_text(encoding="utf-8")
transcript_en = transcript_path.read_text(encoding="utf-8") if transcript_path.exists() else ""

# 既存の「問いとして残るもの」セクションがあれば除去
import re
summary = re.sub(
    r'\n## 問いとして残るもの\n[\s\S]*?(?=\n---\n)',
    '',
    summary
)

# 記事の論点部分を抽出（長すぎる場合は先頭16,000字）
article_sample = summary[:16000]

# トランスクリプトの前半・中盤・後半をサンプリング
if transcript_en:
    total = len(transcript_en)
    transcript_sample = (
        transcript_en[:3000]
        + "\n\n...\n\n"
        + transcript_en[max(0, total//2 - 1500): total//2 + 1500]
        + "\n\n...\n\n"
        + transcript_en[max(0, total - 3000):]
    )
else:
    transcript_sample = ""

prompt = f"""あなたは「SoryuNews」というブログの編集者です。

以下は英語ポッドキャストをもとに生成した日本語記事です。
この記事を読み、記事の論点が提起するが答えていない「哲学的な問い」を4つ生成してください。

## 問いの条件

- **記事の具体的な主張・発言・数値から直接引き出すこと**（「AIは安全か」のような汎用的な問いは禁止）
- 「この主張が正しいとすれば、なぜ〇〇なのか」「この構造では△△はどうなるか」という形式を意識する
- 答えを示唆しない。問いとして開いたままにする
- 各問いは **太字のタイトル（10〜20字）** + 本文（2〜3文）で構成する
- 本文は問いで終わること（「〜か。」「〜のか。」）

## 出力フォーマット（この形式を厳守）

**問いのタイトル1**
本文の説明文。問いで終わる文。

**問いのタイトル2**
本文の説明文。問いで終わる文。

（以下同様に4つ）

## 禁止事項
- 見出し（##）は使わない
- 番号付きリストは使わない
- 前置き・後書きは一切出力しない
- 記事に書いてある論点をそのまま繰り返さない

---

記事（抜粋）:
{article_sample}

---

英語トランスクリプト（参考・抜粋）:
{transcript_sample}
"""

# Ollama で生成
import urllib.request, time

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11435/api/generate")

print("哲学的な問いを生成中 (gemma4:26b)...")
t = time.time()

payload = json.dumps({
    "model": "gemma4:26b",
    "prompt": prompt,
    "stream": True,
    "options": {"temperature": 0.5, "num_ctx": 32768},
}).encode("utf-8")

chunks = []
req = urllib.request.Request(
    OLLAMA_URL,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=1800) as resp:
    for line in resp:
        if not line.strip():
            continue
        data = json.loads(line.decode("utf-8"))
        chunks.append(data.get("response", ""))
        if data.get("done"):
            break

questions_text = "".join(chunks).strip()
elapsed = int(time.time() - t)
print(f"完了: {len(questions_text):,}文字 ({elapsed}秒)")

# セクションを組み立て
section = f"\n## 問いとして残るもの\n\n{questions_text}\n"

# summary_ja.md の「---\n\n*生成条件:」の直前に挿入
marker = "\n---\n\n*生成条件:"
if marker in summary:
    summary = summary.replace(marker, section + marker, 1)
else:
    # マーカーが見つからない場合は末尾に追加
    summary = summary.rstrip() + "\n" + section

summary_path.write_text(summary, encoding="utf-8")
print(f"✅ 挿入完了: {summary_path}")
print(f"   セクション文字数: {len(questions_text):,}字")
