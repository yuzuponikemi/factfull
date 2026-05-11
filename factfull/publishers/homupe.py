"""
factfull/publishers/homupe.py
==============================
factfull の PipelineResult を homupe ブログ記事に変換・投稿する共通ライブラリ。

使い方:

    from factfull.podcast.pipeline import PipelineConfig, run_pipeline
    from factfull.publishers.homupe import BlogMetadata, generate_blog_metadata, create_blog_post

    result = run_pipeline(config, youtube_url)
    meta = generate_blog_metadata(result, model="gemma4:e4b")
    post_path = create_blog_post(result, meta, blog_dir=BLOG_DIR)
    post_tweet(result, meta)   # Selenium + Firefox（FIREFOX_PROFILE_PATH 要設定）
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from factfull.podcast.pipeline import PipelineResult


# ── BlogMetadata ───────────────────────────────────────────────────────────────

@dataclass
class BlogMetadata:
    title_ja: str       # 日本語タイトル
    excerpt: str        # 抜粋（2〜3文）
    tags: list[str]     # タグ（日本語）
    guest: str          # ゲスト紹介文（Markdown）
    slug: str           # URL スラッグ（例: 2026-04-19-jensen-nvidia）
    date: str           # 公開日（YYYY-MM-DD）


# ── メタデータ生成（LLM） ──────────────────────────────────────────────────────

_META_PROMPT = """\
あなたはポッドキャスト記事ブログの編集者です。
以下のファクトチェック済み日本語記事を読み、ブログ投稿に必要なメタデータを生成してください。

## 元の YouTube タイトル（英語）
{title_en}

## チャンネル名
{channel}
{search_section}
## 記事（抜粋）
{article}

---

## 出力形式（JSON のみ。前置き・説明・コードブロック記法は不要）

{{
  "title_ja": "日本語タイトル（最大50字。英語タイトルの和訳ではなく、記事の核心を伝える表現）",
  "excerpt": "2〜3文の日本語抜粋（ブログ一覧に表示される。記事の最重要論点と読む価値を伝える）",
  "tags": ["タグ1", "タグ2", "タグ3", "タグ4", "タグ5"],
  "guest": "ゲストの日本語紹介文（4〜6文。肩書き・所属・専門・代表的業績を含む。Web 検索結果があればそれを最優先で参照する。Markdown 太字可）",
  "slug": "URLスラッグ（英数字・ハイフンのみ、最大60字。例: jensen-huang-nvidia-ai-revolution）"
}}
"""


def generate_blog_metadata(
    result: PipelineResult,
    model: str = "gemma4:e4b",
    today: str | None = None,
    tavily_api_key: str | None = None,
) -> BlogMetadata:
    """
    PipelineResult から LLM を使ってブログメタデータを生成する。

    TAVILY_API_KEY が設定されている場合、ゲスト名を Tavily で検索して
    プロンプトに含めることでゲスト紹介文の精度を高める。

    tags は summary_ja.md の ## キーワード セクションから抽出を優先し、
    抽出できない場合のみ LLM 生成を使用する。

    Args:
        result: run_pipeline() の戻り値
        model: メタデータ生成に使う Ollama モデル
        today: 公開日（YYYY-MM-DD）。省略時は今日の日付
        tavily_api_key: Tavily API キー。省略時は TAVILY_API_KEY 環境変数を使用

    Returns:
        BlogMetadata
    """
    from factfull.llm import call

    os.environ["FACTFULL_LLM_BACKEND"] = "ollama"
    os.environ["FACTFULL_OLLAMA_MODEL"] = model

    summary = result.summary_path.read_text(encoding="utf-8")
    article_for_prompt = summary[:8000]

    # ## キーワード セクションからタグを直接抽出（LLM 呼び出し節約）
    existing_tags = _extract_keywords(summary)

    # ゲスト名を抽出して Tavily で検索
    search_section = ""
    guest_name = _extract_guest_name(result.title)
    if guest_name:
        api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY", "")
        if api_key:
            print(f"  🔍 ゲスト情報を検索中: {guest_name} ...", flush=True)
            search_text = _search_guest_info(guest_name, api_key)
            if search_text:
                search_section = (
                    f"\n## ゲスト Web 検索結果（{guest_name}）"
                    f"— ゲスト紹介文はこの情報を最優先で使用すること\n"
                    f"{search_text}\n"
                )
                print(f"  ✅ 検索完了: {len(search_text)}文字", flush=True)
        else:
            print("  ⚠️  TAVILY_API_KEY が未設定のためゲスト検索をスキップ", flush=True)

    prompt = _META_PROMPT.format(
        title_en=result.title,
        channel=result.channel,
        article=article_for_prompt,
        search_section=search_section,
    )
    raw = call(prompt, num_ctx=16384)
    data = _parse_json(raw)

    pub_date = today or date.today().strftime("%Y-%m-%d")
    slug_suffix = data.get("slug", _default_slug(result.title))
    slug = f"{pub_date}-{slug_suffix}"

    tags = existing_tags if existing_tags else data.get("tags", [])
    # チャンネル名をタグに追加（なければ）
    channel_tag = result.channel
    if channel_tag and channel_tag not in tags:
        tags.append(channel_tag)

    return BlogMetadata(
        title_ja=data.get("title_ja", result.title),
        excerpt=data.get("excerpt", ""),
        tags=tags,
        guest=data.get("guest", ""),
        slug=slug,
        date=pub_date,
    )


# ── ブログ記事作成 ─────────────────────────────────────────────────────────────

def create_blog_post(
    result: PipelineResult,
    meta: BlogMetadata,
    blog_dir: Path,
) -> Path:
    """
    MkDocs Material ブログ形式の Markdown ファイルを作成する。

    出力フォーマット:
        ---
        date: YYYY-MM-DD
        categories:
          - Podcast
        tags:
          - タグ1
        ---

        # 日本語タイトル

        抜粋テキスト

        <!-- more -->

        ## ゲスト
        ゲスト紹介文

        （summary_ja.md の本文）

    Args:
        result: run_pipeline() の戻り値
        meta: generate_blog_metadata() の戻り値
        blog_dir: ブログ記事の出力先ディレクトリ

    Returns:
        作成した .md ファイルのパス
    """
    out_path = blog_dir / f"{meta.slug}.md"

    if out_path.exists():
        print(f"  スキップ（既存）: {out_path.name}")
        return out_path

    blog_dir.mkdir(parents=True, exist_ok=True)

    summary = result.summary_path.read_text(encoding="utf-8")
    if meta.guest:
        summary = _insert_guest(summary, meta.guest)

    tags_yaml = "\n".join(f"  - {t}" for t in meta.tags)

    post = f"""---
date: {meta.date}
categories:
  - Podcast
tags:
{tags_yaml}
---

# {meta.title_ja}

{meta.excerpt}

<!-- more -->

## 概念グラフ
ポッドキャストの主要概念と関係をグラフで表示しています。ノードにカーソルを当てると詳細が表示されます。

<div class="kg-widget" data-src="/data/kg/{result.video_id}.json" data-height="630"></div>

{summary.strip()}
"""
    _export_kg_json(result.video_id, blog_dir)
    out_path.write_text(post, encoding="utf-8")
    print(f"  ✅ 作成: {out_path}")
    print(f"     タイトル: {meta.title_ja}")
    print(f"     スコア: {result.score:.0f}/100")

    return out_path


# ── Twitter/X 投稿（Selenium） ────────────────────────────────────────────────

def post_tweet(
    result: PipelineResult,
    meta: BlogMetadata,
    blog_url_base: str = "https://soryu.news",
    firefox_profile: str | None = None,
    headless: bool = True,
    dry_run: bool = False,
) -> None:
    """
    Selenium + Firefox でログイン済みプロファイルを使って X に投稿する。
    Twitter API 不要。

    必要な環境変数:
        FIREFOX_PROFILE_PATH  ログイン済み Firefox プロファイルのパス

    Args:
        result: PipelineResult（使用しないが将来の拡張用）
        meta: BlogMetadata（タイトル・抜粋・スラッグを使用）
        blog_url_base: ブログの Base URL
        firefox_profile: Firefox プロファイルパス。省略時は FIREFOX_PROFILE_PATH 環境変数
        headless: ヘッドレスモードで実行するか（デフォルト True）
        dry_run: True のとき投稿せずツイート内容だけ表示する
    """
    post_url = f"{blog_url_base}/blog/{meta.slug}/"
    tweet_text = _build_tweet_text(meta, post_url)

    print(f"\n🐦 X (Twitter) 投稿")
    print(f"  {tweet_text}")
    print(f"  ({len(tweet_text)} 文字)")

    if dry_run:
        print("  [dry_run] 投稿をスキップします。")
        return

    profile_path = firefox_profile or os.environ.get("FIREFOX_PROFILE_PATH", "")
    if not profile_path:
        print("  ⚠️  FIREFOX_PROFILE_PATH が未設定のため投稿をスキップ")
        return
    if not os.path.isdir(profile_path):
        print(f"  ⚠️  Firefox プロファイルが見つかりません: {profile_path}")
        return

    _post_via_selenium(tweet_text, profile_path, headless)


def _build_tweet_text(meta: BlogMetadata, post_url: str, max_len: int = 270) -> str:
    """タイトル・抜粋・URL からツイート本文を組み立てる。"""
    header = f"【新着】{meta.title_ja}\n\n"
    footer = f"\n\n{post_url}"
    budget = max_len - len(header) - len(footer)
    excerpt = meta.excerpt[:budget].rstrip("。、") + "…" if len(meta.excerpt) > budget else meta.excerpt
    return header + excerpt + footer


def _post_via_selenium(tweet_text: str, profile_path: str, headless: bool) -> None:
    """Selenium Firefox でツイートを投稿する。"""
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from webdriver_manager.firefox import GeckoDriverManager  # type: ignore

    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("-profile")
    options.add_argument(profile_path)

    service = Service(GeckoDriverManager().install())
    browser = webdriver.Firefox(service=service, options=options)
    wait = WebDriverWait(browser, 30)

    try:
        browser.get("https://x.com/compose/post")

        # テキストボックスを探して入力
        textbox = None
        for selector in [
            (By.CSS_SELECTOR, "div[data-testid='tweetTextarea_0'][role='textbox']"),
            (By.XPATH, "//div[@data-testid='tweetTextarea_0']//div[@role='textbox']"),
            (By.XPATH, "//div[@role='textbox']"),
        ]:
            try:
                textbox = wait.until(EC.element_to_be_clickable(selector))
                textbox.click()
                textbox.send_keys(tweet_text)
                break
            except Exception:
                continue

        if textbox is None:
            raise RuntimeError("ツイートテキストボックスが見つかりません。X にログインされているか確認してください。")

        # 投稿ボタンをクリック
        posted = False
        for selector in [
            (By.XPATH, "//button[@data-testid='tweetButtonInline']"),
            (By.XPATH, "//button[@data-testid='tweetButton']"),
            (By.XPATH, "//span[text()='Post']/ancestor::button"),
        ]:
            try:
                btn = wait.until(EC.element_to_be_clickable(selector))
                btn.click()
                posted = True
                break
            except Exception:
                continue

        if not posted:
            raise RuntimeError("投稿ボタンが見つかりません。")

        import time
        time.sleep(2)
        print("  ✅ X への投稿完了", flush=True)

    finally:
        browser.quit()


# ── 内部ヘルパー ──────────────────────────────────────────────────────────────

def _extract_guest_name(title: str) -> str | None:
    """
    YouTube タイトルからゲスト名を抽出する。

    対応フォーマット:
      "Jensen Huang: NVIDIA - The $4 Trillion Company..."  → "Jensen Huang"
      "Andrej Karpathy — We're summoning ghosts..."        → "Andrej Karpathy"
      "State of AI in 2026: LLMs, Coding..."               → None（パネル・特集回）
    """
    for sep in (":", "—", " - "):
        if sep in title:
            candidate = title.split(sep)[0].strip()
            words = candidate.split()
            # 1〜4単語で、各単語が大文字始まりなら人名と判断
            if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                return candidate
    return None


def _search_guest_info(guest_name: str, api_key: str) -> str:
    """
    Tavily でゲストの経歴情報を検索して整形済みテキストを返す。

    2クエリ実行（英語 + 日本語）して結果をまとめる。
    TAVILY_API_KEY が必要。
    """
    from tavily import TavilyClient  # type: ignore

    client = TavilyClient(api_key=api_key)

    queries = [
        f"{guest_name} biography career background expertise",
        f"{guest_name} 経歴 専門",
    ]

    parts: list[str] = []
    seen_urls: set[str] = set()

    for query in queries:
        try:
            resp = client.search(
                query=query,
                max_results=3,
                search_depth="basic",
                include_answer=True,
            )
            # Tavily の AI answer（あれば優先）
            if resp.get("answer") and len(parts) == 0:
                parts.append(f"[Tavily Summary]\n{resp['answer']}")
            # 個別の検索結果
            for item in resp.get("results", []):
                url = item.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                title = item.get("title", "")
                content = item.get("content", "")[:600]
                parts.append(f"[{title}]\n{content}")
        except Exception as e:
            print(f"  ⚠️  Tavily 検索エラー ({query[:30]}...): {e}", flush=True)

    return "\n\n".join(parts)


def _extract_keywords(summary: str) -> list[str]:
    """summary_ja.md の ## キーワード セクションからタグを抽出する。"""
    m = re.search(r'## キーワード\n+(.+?)(?:\n\n|\n##|\Z)', summary, re.DOTALL)
    if not m:
        return []
    keywords_text = m.group(1).strip()
    # フォーマット: `AI` / `機械学習` / ...
    tags = [k.strip().strip('`') for k in re.split(r'\s*/\s*', keywords_text) if k.strip()]
    return tags[:10]  # 最大10個


def _insert_guest(summary: str, guest_text: str) -> str:
    """summary_ja.md の --- と ## 概要 の間に ## ゲスト セクションを挿入する。"""
    pattern = r'(\n---\n+)(## 概要)'
    replacement = r'\1## ゲスト\n\n' + guest_text + r'\n\n\2'
    result = re.sub(pattern, replacement, summary, count=1)
    if result == summary:
        # フォールバック: ## 概要 の直前
        result = summary.replace("## 概要", f"## ゲスト\n\n{guest_text}\n\n## 概要", 1)
    return result


def _parse_json(text: str) -> dict:
    """LLM 出力から JSON オブジェクトを抽出してパースする。"""
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _default_slug(title_en: str) -> str:
    """英語タイトルから URL スラッグを生成するフォールバック。"""
    slug = re.sub(r'[^a-z0-9]+', '-', title_en.lower()).strip('-')
    return slug[:60]


def _export_kg_json(source_id: str, blog_dir: Path) -> bool:
    """Neo4j から KG JSON を homupe/docs/data/kg/ に書き出す。接続不能時は黙って False を返す。"""
    try:
        from factfull.export.book_graph import export_book_graph
        from factfull.graph.neo4j import Neo4jClient

        # blog_dir = .../homupe/docs/blog/posts/YYYY/MM → parents[4] = homupe root
        homupe_root = blog_dir.resolve().parents[4]
        out_dir = homupe_root / "docs" / "data" / "kg"
        out_dir.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("NEO4J_PASSWORD", "factfull123")
        with Neo4jClient() as client:
            data = export_book_graph(source_id, client)

        out_path = out_dir / f"{source_id}.json"
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  ✅ KG JSON: {out_path.name}  nodes={len(data['nodes'])}  links={len(data['links'])}")
        return True
    except Exception as e:
        print(f"  ⚠️  KG JSON エクスポートをスキップ（{e}）")
        return False


# ── Book Guide Publisher ───────────────────────────────────────────────────────

_BOOK_META_PROMPT = """\
あなたは書籍ブログの編集者です。
以下のファクトチェック済み日本語ブックガイドを読み、ブログ投稿に必要なメタデータを生成してください。

## 著者
{author}

## 書籍タイトル
{book_title}

## 記事（抜粋）
{article}

---

## 出力形式（JSON のみ。前置き・説明・コードブロック記法は不要）

{{
  "title_ja": "日本語タイトル（最大50字。書名をそのまま使わず、読む価値・核心を伝える表現）",
  "excerpt": "2〜3文の日本語抜粋（ブログ一覧に表示。本書の最重要論点と読む価値を伝える）",
  "tags": ["タグ1", "タグ2", "タグ3", "タグ4", "タグ5"],
  "slug": "URLスラッグ（英数字・ハイフンのみ、最大60字。例: plato-republic-book-guide）"
}}
"""


@dataclass
class BookGuideMetadata:
    title_ja: str
    excerpt: str
    tags: list[str]
    slug: str
    date: str


def generate_book_metadata(
    result,
    model: str = "gemma4:e4b",
    today: str | None = None,
) -> BookGuideMetadata:
    """
    BookPipelineResult からブックガイド用ブログメタデータを生成する。

    Args:
        result: BookPipelineResult
        model: メタデータ生成に使う Ollama モデル
        today: 公開日（YYYY-MM-DD）。省略時は今日の日付
    """
    from factfull.llm import call

    os.environ["FACTFULL_LLM_BACKEND"] = "ollama"
    os.environ["FACTFULL_OLLAMA_MODEL"] = model

    guide_text = result.book_guide_path.read_text(encoding="utf-8")
    article_for_prompt = guide_text[:8000]

    existing_tags = _extract_keywords(guide_text)

    prompt = _BOOK_META_PROMPT.format(
        author=result.author,
        book_title=result.book_title,
        article=article_for_prompt,
    )
    raw = call(prompt, num_ctx=16384)
    data = _parse_json(raw)

    pub_date = today or date.today().strftime("%Y-%m-%d")
    slug_suffix = data.get("slug", f"{result.book_id.replace('_', '-')}-book-guide")
    slug = f"{pub_date}-{slug_suffix}"

    tags = existing_tags if existing_tags else data.get("tags", [])
    if "Book Guide" not in tags:
        tags.append("Book Guide")

    return BookGuideMetadata(
        title_ja=data.get("title_ja", f"{result.book_title} ブックガイド"),
        excerpt=data.get("excerpt", ""),
        tags=tags,
        slug=slug,
        date=pub_date,
    )


def create_book_guide_post(
    result,
    meta: BookGuideMetadata,
    blog_dir: Path,
) -> Path:
    """
    MkDocs Material ブログ形式のブックガイド記事 Markdown ファイルを作成する。

    Args:
        result: BookPipelineResult
        meta: generate_book_metadata() の戻り値
        blog_dir: ブログ記事の出力先ディレクトリ

    Returns:
        作成した .md ファイルのパス
    """
    out_path = blog_dir / f"{meta.slug}.md"

    if out_path.exists():
        print(f"  スキップ（既存）: {out_path.name}")
        return out_path

    blog_dir.mkdir(parents=True, exist_ok=True)

    guide_text = result.book_guide_path.read_text(encoding="utf-8")
    tags_yaml = "\n".join(f"  - {t}" for t in meta.tags)

    post = f"""---
title: {meta.title_ja}
date: {meta.date}
categories:
  - Book Guide
tags:
{tags_yaml}
---

# 読書メモ：{result.author}『{result.book_title}』ブックガイド

{meta.excerpt}

<!-- more -->

## 概念グラフ
本書の核心概念と概念間の依存・矛盾・発展関係をグラフで表示しています。ノードにカーソルを当てると詳細が表示されます。

<div class="kg-widget" data-src="/data/kg/{result.book_id}.json" data-height="630"></div>

{guide_text.strip()}
"""
    _export_kg_json(result.book_id, blog_dir)
    out_path.write_text(post, encoding="utf-8")
    print(f"  ✅ 作成: {out_path}")
    print(f"     タイトル: {meta.title_ja}")
    print(f"     スコア: {result.score:.0f}/100")

    return out_path


def create_local_podcast_post(
    result: PipelineResult,
    meta: BlogMetadata,
    blog_dir: Path,
) -> Path:
    """ローカル音声ポッドキャスト記事を homupe ブログに書き出す（YouTube なし版）。

    create_blog_post() との違い: YouTube 埋め込みなし。チャンネル名・エピソード情報を表示。

    Args:
        result: run_local_pipeline() の戻り値 (PipelineResult)
        meta: generate_blog_metadata() または手動で作った BlogMetadata
        blog_dir: ブログ記事の出力先ディレクトリ

    Returns:
        作成した .md ファイルのパス
    """
    out_path = blog_dir / f"{meta.slug}.md"

    if out_path.exists():
        print(f"  スキップ（既存）: {out_path.name}")
        return out_path

    blog_dir.mkdir(parents=True, exist_ok=True)

    summary = result.summary_path.read_text(encoding="utf-8")
    if meta.guest:
        summary = _insert_guest(summary, meta.guest)

    tags_yaml = "\n".join(f"  - {t}" for t in meta.tags)

    # エピソード情報
    duration = result.metadata.get("duration", "") if result.metadata else ""
    pub_date_ep = result.metadata.get("pub_date", "") if result.metadata else ""
    ep_info_lines = []
    if result.channel:
        ep_info_lines.append(f"**チャンネル**: {result.channel}")
    if pub_date_ep:
        ep_info_lines.append(f"**配信日**: {pub_date_ep}")
    if duration:
        ep_info_lines.append(f"**再生時間**: {duration}")
    ep_info = "  \n".join(ep_info_lines)

    post = f"""---
date: {meta.date}
categories:
  - Podcast
tags:
{tags_yaml}
---

# {meta.title_ja}

{meta.excerpt}

<!-- more -->

{ep_info}

## 概念グラフ
ポッドキャストの主要概念と関係をグラフで表示しています。ノードにカーソルを当てると詳細が表示されます。

<div class="kg-widget" data-src="/data/kg/{result.video_id}.json" data-height="630"></div>

{summary.strip()}
"""
    _export_kg_json(result.video_id, blog_dir)
    out_path.write_text(post, encoding="utf-8")
    print(f"  ✅ 作成: {out_path}")
    print(f"     タイトル: {meta.title_ja}")
    print(f"     スコア: {result.score:.0f}/100")

    return out_path


def create_arxiv_digest_post(
    digest,              # factfull.arxiv.digest.DigestResult
    blog_dir: Path,
) -> Path:
    """arXiv ダイジェスト記事を homupe ブログに書き出す。

    Args:
        digest: DigestResult (factfull.arxiv.digest)
        blog_dir: 書き出し先ディレクトリ（例: homupe/docs/blog/posts/2026/05/）

    Returns:
        書き出した .md ファイルのパス
    """
    from factfull.arxiv.digest import render_digest_markdown
    from factfull.export.arxiv_graph import export_arxiv_digest_graph
    from factfull.graph.neo4j import Neo4jClient

    blog_dir.mkdir(parents=True, exist_ok=True)

    # KG JSON を生成して homupe/docs/data/kg/ に書き出す
    kg_source_ids = [f"arxiv_{ps.entry.paper_id}" for ps in digest.papers]
    kg_json_name = digest.digest_id  # "arxiv_digest_20260507"

    homupe_root = blog_dir.resolve().parents[4]
    kg_dir = homupe_root / "docs" / "data" / "kg"
    kg_dir.mkdir(parents=True, exist_ok=True)
    kg_path = kg_dir / f"{kg_json_name}.json"

    os.environ.setdefault("NEO4J_PASSWORD", "factfull123")
    with Neo4jClient() as client:
        kg_data = export_arxiv_digest_graph(kg_source_ids, digest.digest_id, client)
    kg_path.write_text(
        json.dumps(kg_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  ✅ KG JSON: {kg_json_name}.json  nodes={len(kg_data['nodes'])}  links={len(kg_data['links'])}")

    # Markdown 記事を生成
    md = render_digest_markdown(digest, kg_src=kg_json_name)
    slug = f"{digest.date}-arxiv-digest"
    out_path = blog_dir / f"{slug}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"  ✅ 作成: {out_path}")

    return out_path


def default_blog_dir(homupe_root: Path | None = None) -> Path:
    """今月のブログ記事ディレクトリを返す。

    Args:
        homupe_root: homupe リポジトリのルートパス。
                     省略時は HOMUPE_ROOT 環境変数、それもなければ ~/source/personal/homupe を使用。
    """
    today = date.today()
    root = homupe_root or Path(
        os.environ.get("HOMUPE_ROOT", str(Path.home() / "source" / "personal" / "homupe"))
    )
    return (
        root
        / "docs" / "blog" / "posts"
        / str(today.year)
        / f"{today.month:02d}"
    )
