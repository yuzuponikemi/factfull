"""
factfull/book/chapter_resolver.py
===================================
書籍の正確な章立てを複数ソースから収集・検証して返す。

問題: LLM が book_guide.md を生成する際に章タイトルを幻覚することがある。
解決: Wikipedia（EN/RU/ZH/JA）・Open Library など複数ソースから章立てを取得し、
      ≥2 ソースで一致した章のみ正式な章立てとして採用する。

使い方:
    from factfull.book.chapter_resolver import ChapterResolver

    resolver = ChapterResolver(model="gemma4:e4b")
    result = resolver.resolve("La Société de consommation", "Jean Baudrillard")
    print(f"confidence={result.confidence:.2f}  sources={result.sources_agreed}")
    for ch in result.chapters:
        print(f"  {ch.num}. {ch.title}")
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher


# ── データ構造 ────────────────────────────────────────────────────────────────

@dataclass
class ChapterEntry:
    title: str
    num: str = ""
    part: str = ""

    def normalized(self) -> str:
        """ファジー比較用の正規化タイトル。"""
        t = self.title.lower()
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def __str__(self) -> str:
        prefix = f"{self.num}. " if self.num else ""
        suffix = f" [{self.part}]" if self.part else ""
        return f"{prefix}{self.title}{suffix}"


@dataclass
class ChapterStructure:
    book_title: str
    author: str
    chapters: list[ChapterEntry]
    confidence: float          # 0.0–1.0  (≥0.6 が目安)
    sources_consulted: list[str]
    sources_agreed: int        # ≥2 なら信頼できる
    notes: str = ""

    def is_reliable(self) -> bool:
        return self.confidence >= 0.5 and self.sources_agreed >= 2

    def to_markdown_table(self) -> str:
        rows = ["| # | 章タイトル | 部 |", "| :- | :- | :- |"]
        for ch in self.chapters:
            rows.append(f"| {ch.num} | {ch.title} | {ch.part} |")
        return "\n".join(rows)


# ── リゾルバー ────────────────────────────────────────────────────────────────

class ChapterResolver:
    """複数 Web ソースから書籍の章立てを取得・検証する。"""

    # 優先言語: フランス語は哲学書の原語として重要
    WIKIPEDIA_LANGS = ["fr", "en", "ru", "zh", "ja", "de"]

    def __init__(
        self,
        model: str = "gemma4:e4b",
        ollama_url: str = "http://localhost:11435/api/generate",
        timeout_sec: int = 20,
        wikidata_delay: float = 1.5,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout_sec = timeout_sec
        self.wikidata_delay = wikidata_delay
        self._last_wikidata_call: float = 0.0
        self._wikidata_cache: dict[str, dict[str, str]] = {}

    # ── HTTP ユーティリティ ───────────────────────────────────────────────────

    def _http_get_json(self, url: str, retries: int = 3, throttle: bool = False) -> dict:
        """HTTP GET → JSON（429 に対して指数バックオフでリトライ）。

        Args:
            throttle: True なら wikidata_delay を適用（Wikidata API 用）。
        """
        if throttle:
            elapsed = time.monotonic() - self._last_wikidata_call
            if elapsed < self.wikidata_delay:
                time.sleep(self.wikidata_delay - elapsed)

        req = urllib.request.Request(url, headers={"User-Agent": "factfull/1.0"})
        for attempt in range(retries):
            if throttle:
                self._last_wikidata_call = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < retries - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    return {}
            except Exception:
                return {}
        return {}

    # ── 公開 API ──────────────────────────────────────────────────────────────

    def resolve(
        self,
        book_title: str,
        author: str,
        extra_urls: list[str] | None = None,
        translate_to: str | None = None,
    ) -> ChapterStructure:
        """
        複数ソースから章立てを取得し、合意形成して返す。

        Args:
            book_title:   書籍タイトル（原題・翻訳どちらでも可）
            author:       著者名
            extra_urls:   追加で参照する URL リスト（例: 出版社サイト）
            translate_to: 翻訳先言語コード（例: "ja", "en", "fr"）。
                          None の場合は翻訳しない。

        Returns:
            ChapterStructure。confidence ≥ 0.6 かつ sources_agreed ≥ 2 なら信頼できる。
        """
        print(f"\n📖 章立て解決: 『{book_title}』 — {author}")
        extractions: list[tuple[str, list[ChapterEntry]]] = []

        # 1. Wikidata で各言語版記事タイトルを解決
        wikidata_titles = self._resolve_via_wikidata(book_title, author)
        if wikidata_titles:
            print(f"  [wikidata] {len(wikidata_titles)} 言語版を確認: "
                  f"{', '.join(wikidata_titles.keys())}")

        # 2. Wikipedia 多言語
        for lang in self.WIKIPEDIA_LANGS:
            article_title = wikidata_titles.get(lang, "")
            url, sections = self._fetch_wikipedia_sections(
                book_title, author, lang, article_title=article_title
            )
            if url and sections:
                toc = self._extract_toc_from_sections(sections, f"Wikipedia ({lang})")
                if toc:
                    print(f"  [chapters] Wikipedia ({lang}): {len(toc)} 章  {url}")
                    extractions.append((url, toc))
            elif url:
                # セクション API が空 → 本文から LLM 抽出
                text = self._fetch_wikipedia_text(url)
                if text:
                    toc = self._extract_toc_with_llm(text, f"Wikipedia ({lang}): {url}")
                    if toc:
                        print(f"  [chapters] Wikipedia ({lang}) LLM: {len(toc)} 章")
                        extractions.append((url, toc))

        # 2. Open Library
        ol_result = self._fetch_openlibrary(book_title, author)
        if ol_result:
            url, toc = ol_result
            print(f"  [chapters] Open Library: {len(toc)} 章  {url}")
            extractions.append((url, toc))

        # 3. 追加 URL
        for url in (extra_urls or []):
            text = self._fetch_url_text(url)
            if text:
                toc = self._extract_toc_with_llm(text, url)
                if toc:
                    extractions.append((url, toc))

        sources = [url for url, _ in extractions]
        print(f"  [chapters] 抽出完了: {len(extractions)} ソース")

        if not extractions:
            return ChapterStructure(
                book_title=book_title, author=author,
                chapters=[], confidence=0.0,
                sources_consulted=[], sources_agreed=0,
                notes="章立て情報が取得できませんでした",
            )

        chapters, confidence, agreed = self._consensus(extractions)
        print(f"  [chapters] 合意: {len(chapters)} 章  confidence={confidence:.2f}  agreed={agreed}")

        if translate_to and chapters:
            chapters = self._translate_chapters(chapters, translate_to)

        return ChapterStructure(
            book_title=book_title, author=author,
            chapters=chapters, confidence=confidence,
            sources_consulted=sources, sources_agreed=agreed,
        )

    # ── Wikipedia ─────────────────────────────────────────────────────────────

    def _resolve_via_wikidata(self, book_title: str, author: str) -> dict[str, str]:
        """Wikidata 経由で各言語版 Wikipedia の記事タイトルを取得する。

        Returns:
            {lang: article_title}  例: {"fr": "La Société de consommation (ouvrage)", "ru": "..."}
        """
        cache_key = f"{book_title}|||{author}"
        if cache_key in self._wikidata_cache:
            return self._wikidata_cache[cache_key]

        # 著者名の last name を使って候補を絞る
        author_last = author.strip().split()[-1].lower() if author.strip() else ""

        def _search_wikidata(query: str) -> list[dict]:
            url = (
                f"https://www.wikidata.org/w/api.php"
                f"?action=wbsearchentities&search={urllib.parse.quote(query)}"
                f"&language=en&limit=10&format=json"
            )
            return self._http_get_json(url, throttle=True).get("search", [])

        def _pick_qid(results: list[dict]) -> str:
            """著者名が説明文に含まれる結果を優先して QID を選ぶ。"""
            if not results:
                return ""
            if author_last:
                for item in results:
                    desc = (item.get("description") or "").lower()
                    label = (item.get("label") or "").lower()
                    if author_last in desc or author_last in label:
                        return item["id"]
            return results[0]["id"]

        def _get_sitelinks(qid: str) -> dict[str, str]:
            sl_url = (
                f"https://www.wikidata.org/w/api.php"
                f"?action=wbgetentities&ids={qid}&props=sitelinks&format=json"
            )
            sl_data = self._http_get_json(sl_url, throttle=True)
            sitelinks = sl_data.get("entities", {}).get(qid, {}).get("sitelinks", {})
            return {
                lang: sitelinks[f"{lang}wiki"]["title"]
                for lang in self.WIKIPEDIA_LANGS
                if f"{lang}wiki" in sitelinks
            }

        # 書籍以外のエンティティを示す説明文パターン（地名・政党など）
        non_book_desc = re.compile(
            r"\b(country|nation|state|city|capital|newspaper|magazine|"
            r"political party|television|radio|film|movie|band|musician|"
            r"politician|organization|company|corporation)\b",
            re.IGNORECASE,
        )

        def _is_book_candidate(item: dict) -> bool:
            desc = (item.get("description") or "").lower()
            if non_book_desc.search(desc):
                return False
            book_hints = re.compile(
                r"\b(book|novel|work|essay|treatise|dialogue|written|"
                r"philosophy|poem|play|written by)\b",
                re.IGNORECASE,
            )
            return bool(book_hints.search(desc)) or author_last in desc

        try:
            # 試行1: タイトルのみで検索
            results = _search_wikidata(book_title)
            qid = _pick_qid(results)

            # 試行2: 最良候補が書籍エンティティでなければ著者名付きで再検索
            best = next((r for r in results if r["id"] == qid), None)
            if not qid or (best and not _is_book_candidate(best)):
                results2 = _search_wikidata(f"{book_title} {author}")
                qid2 = _pick_qid(results2)
                if qid2:
                    qid = qid2

            result = _get_sitelinks(qid) if qid else {}
            if result:
                self._wikidata_cache[cache_key] = result
            return result

        except Exception:
            return {}

    def _find_wikipedia_title_opensearch(
        self, book_title: str, author: str, lang: str
    ) -> str:
        """OpenSearch で Wikipedia 記事タイトルを特定する（Wikidata フォールバック用）。"""
        query = f"{book_title} {author}"
        url = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=opensearch&search={urllib.parse.quote(query)}&limit=5&format=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "factfull/1.0"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    data = json.loads(resp.read())
                titles = data[1]
                return titles[0] if titles else ""
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    return ""
            except Exception:
                return ""
        return ""

    def _fetch_wikipedia_sections(
        self,
        book_title: str,
        author: str,
        lang: str,
        article_title: str = "",
    ) -> tuple[str, list[dict]]:
        """Wikipedia の sections API でセクション一覧を取得する。

        Args:
            article_title: Wikidata 経由で解決済みのタイトル（なければ OpenSearch で検索）
        """
        if not article_title:
            article_title = self._find_wikipedia_title_opensearch(book_title, author, lang)
        if not article_title:
            return "", []

        api_url = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=parse&page={urllib.parse.quote(article_title)}"
            f"&prop=sections&redirects=1&format=json"
        )
        page_url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(article_title)}"
        req = urllib.request.Request(api_url, headers={"User-Agent": "factfull/1.0"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    data = json.loads(resp.read())
                parse = data.get("parse", {})
                if parse.get("title"):
                    canonical = parse["title"]
                    page_url = (
                        f"https://{lang}.wikipedia.org/wiki/"
                        f"{urllib.parse.quote(canonical)}"
                    )
                sections = parse.get("sections", [])
                return page_url, sections
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    return page_url, []
            except Exception:
                return page_url, []
        return page_url, []

    def _extract_toc_from_sections(
        self, sections: list[dict], source_label: str
    ) -> list[ChapterEntry]:
        """Wikipedia sections API の結果から章立てを抽出する。

        記事の「Contents」セクション（References / See also 等）は除外し、
        書籍の章・部に相当するセクションのみを返す。
        """
        if not sections:
            return []

        # 書籍章立てではない典型的なセクション名（完全一致）
        skip_exact = re.compile(
            r"^(references?|see also|bibliography|notes?|further reading|"
            r"external links?|citations?|footnotes?|sources?|"
            r"文献|参考|脚注|外部リンク|関連項目|出典|注釈|参考文献|影響|概要|"
            r"библиография|примечания|источники|критика|влияние|издания|заключение|"
            r"参考资料|外部链接|参见|注释|目次|"
            r"literatur|weblinks?|einzelnachweise|"             # ドイツ語
            r"description|popularité|réception|médias?|"       # フランス語 記事構造
            r"culture populaire|dans la culture|adaptations?|"
            r"articles?\s+connexes?|annexes?)$",
            re.IGNORECASE | re.UNICODE,
        )
        # 部分一致で除外すべきパターン（記事の目次・注釈・参照）
        skip_partial = re.compile(
            r"(содержание|литератур"          # ロシア語: 目次・文献
            r"|notes?\s+et\s+référence"       # French: Notes et références
            r"|voir\s+aussi|liens?\s+externe" # French: Voir aussi / Liens externes
            r"|table\s+of\s+contents"         # English
            r"|ссылк)",                       # ロシア語: リンク
            re.IGNORECASE | re.UNICODE,
        )

        result = []
        for sec in sections:
            line = re.sub(r"<[^>]+>", "", sec.get("line", "")).strip()
            num = sec.get("number", "")
            if not line:
                continue
            if skip_exact.match(line) or skip_partial.search(line):
                continue
            # toclevel 1–3 のみ（深すぎるサブセクションは除外）
            if int(sec.get("toclevel", 1)) > 3:
                continue
            result.append(ChapterEntry(title=line, num=num))

        return result

    def _fetch_wikipedia_text(self, page_url: str) -> str:
        """Wikipedia ページの本文テキストを取得する（extracts API）。"""
        # URL から lang と title を取得
        m = re.match(r"https://(\w+)\.wikipedia\.org/wiki/(.+)", page_url)
        if not m:
            return ""
        lang, title = m.group(1), m.group(2)

        api_url = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=query&prop=extracts&exintro=false&format=json"
            f"&titles={title}"
        )
        try:
            with urllib.request.urlopen(
                urllib.request.Request(api_url, headers={"User-Agent": "factfull/1.0"}),
                timeout=self.timeout_sec,
            ) as resp:
                data = json.loads(resp.read())
            pages = data.get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})
            text = page.get("extract", "")
            # HTML タグを除去
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s{3,}", "\n\n", text)
            return text.strip()
        except Exception:
            return ""

    # ── Open Library ──────────────────────────────────────────────────────────

    def _fetch_openlibrary(
        self, book_title: str, author: str
    ) -> tuple[str, list[ChapterEntry]] | None:
        """Open Library の works / editions API から目次を取得する。"""
        search_url = (
            f"https://openlibrary.org/search.json"
            f"?title={urllib.parse.quote(book_title)}"
            f"&author={urllib.parse.quote(author)}&limit=5"
        )
        try:
            with urllib.request.urlopen(
                urllib.request.Request(search_url, headers={"User-Agent": "factfull/1.0"}),
                timeout=self.timeout_sec,
            ) as resp:
                data = json.loads(resp.read())

            for doc in data.get("docs", []):
                work_key = doc.get("key", "")
                if not work_key:
                    continue

                # works に直接 table_of_contents がある場合
                work_url = f"https://openlibrary.org{work_key}.json"
                try:
                    with urllib.request.urlopen(
                        urllib.request.Request(work_url, headers={"User-Agent": "factfull/1.0"}),
                        timeout=self.timeout_sec,
                    ) as resp:
                        work = json.loads(resp.read())

                    toc_raw = work.get("table_of_contents", [])
                    if toc_raw:
                        toc = [
                            ChapterEntry(
                                title=(
                                    e.get("title") or e.get("value") or ""
                                ).strip(),
                                num=str(e.get("level", "")),
                            )
                            for e in toc_raw
                            if (e.get("title") or e.get("value") or "").strip()
                        ]
                        if toc:
                            return work_url, toc
                except Exception:
                    continue

        except Exception:
            pass
        return None

    # ── 追加 URL ──────────────────────────────────────────────────────────────

    def _fetch_url_text(self, url: str) -> str:
        """任意の URL からテキストを取得する。"""
        try:
            from factfull.ingest.web import ingest_url
            doc = ingest_url(url)
            return doc.text[:10000]
        except Exception:
            return ""

    # ── LLM 抽出 ──────────────────────────────────────────────────────────────

    def _translate_chapters(
        self, chapters: list[ChapterEntry], target_lang: str
    ) -> list[ChapterEntry]:
        """Ollama で章タイトルを target_lang に翻訳して返す。

        番号（num）は保持し、title のみ翻訳する。
        翻訳失敗時は原文のまま返す。
        """
        LANG_NAMES = {
            "ja": "Japanese", "en": "English", "fr": "French",
            "de": "German", "zh": "Chinese", "ko": "Korean",
            "es": "Spanish", "it": "Italian", "pt": "Portuguese",
            "ru": "Russian",
        }
        lang_name = LANG_NAMES.get(target_lang, target_lang)

        # 番号 → タイトル の対応表を作る
        lines = "\n".join(
            f"{ch.num or i+1}: {ch.title}"
            for i, ch in enumerate(chapters)
        )
        prompt = (
            f"Translate the following book chapter titles to {lang_name}.\n"
            f"Rules:\n"
            f"- Keep each line's prefix (the number before the colon) unchanged.\n"
            f"- Translate only the title text after the colon.\n"
            f"- Output each line as: <number>: <translated title>\n"
            f"- Output only the translated lines, nothing else.\n\n"
            f"{lines}\n\n"
            f"Translation:"
        )
        try:
            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 800},
            }).encode()
            req = urllib.request.Request(
                self.ollama_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())

            raw = result.get("response", "").strip()
            # num → translated title mapping
            translations: dict[str, str] = {}
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                m = re.match(r"^([^:]+):\s*(.+)$", line)
                if m:
                    translations[m.group(1).strip()] = m.group(2).strip()

            if not translations:
                return chapters

            print(f"  [translate] → {lang_name}: {len(translations)} タイトル翻訳済み")
            result_chapters = []
            for i, ch in enumerate(chapters):
                key = ch.num or str(i + 1)
                translated = translations.get(key, ch.title)
                result_chapters.append(ChapterEntry(
                    title=translated,
                    num=ch.num,
                    part=ch.part,
                ))
            return result_chapters

        except Exception:
            return chapters

    def _extract_toc_with_llm(
        self, text: str, source_desc: str
    ) -> list[ChapterEntry]:
        """Ollama LLM でテキストから章立てを抽出する。"""
        excerpt = text[:8000]
        prompt = (
            f"Extract the table of contents (chapters/parts/sections) of the book from the following text.\n\n"
            f"Source: {source_desc}\n\n"
            f"Text:\n{excerpt}\n\n"
            f"---\n"
            f"Rules:\n"
            f"- Extract only chapter/part titles as they appear in the book itself\n"
            f"- Exclude: page numbers, author bio, publisher info, index, references, preface (unless it IS a chapter), appendices\n"
            f"- If you see 'Part I / Chapter 1 / Chapter 2 / Part II / ...' structure, capture the hierarchy in 'part'\n"
            f"- Return JSON array only, no explanation:\n"
            f'  [{{"num": "1", "title": "Chapter Title Here", "part": "Part I (if applicable)"}}]\n'
            f"- If no chapter structure found, return: []\n\n"
            f"JSON:"
        )

        try:
            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 1200},
            }).encode()

            req = urllib.request.Request(
                self.ollama_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read())

            raw = result.get("response", "").strip()
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            if not m:
                return []
            entries = json.loads(m.group(0))
            return [
                ChapterEntry(
                    title=str(e.get("title", "")).strip(),
                    num=str(e.get("num", "")).strip(),
                    part=str(e.get("part", "")).strip(),
                )
                for e in entries
                if str(e.get("title", "")).strip()
            ]
        except Exception:
            return []

    # ── 合意形成 ──────────────────────────────────────────────────────────────

    def _consensus(
        self, extractions: list[tuple[str, list[ChapterEntry]]]
    ) -> tuple[list[ChapterEntry], float, int]:
        """
        複数ソースの章立てを比較して合意形成する。

        アルゴリズム:
          1. 番号付き章: 番号でグループ化（"1", "2.3" など）
          2. 番号なし章: タイトルファジーマッチ（≥ 0.75）でグループ化
          3. ≥2 ソースで確認されたグループのみ採用
          4. confidence = 確認済み章数 / 最大章数 × ソース合意率

        Returns:
            (合意章リスト, confidence 0-1, 合意ソース数の最大)
        """
        if not extractions:
            return [], 0.0, 0

        if len(extractions) == 1:
            return extractions[0][1], 0.35, 1

        # フェーズ1: 番号付き章をグループ化
        # key = "1", "2", "1.2" など（数字のドット区切り）
        numbered: dict[str, list[tuple[int, ChapterEntry]]] = {}
        unnumbered: list[tuple[int, ChapterEntry]] = []

        for src_idx, (_, toc) in enumerate(extractions):
            for entry in toc:
                num_digits = re.findall(r"\d+", entry.num) if entry.num else []
                if num_digits:
                    key = ".".join(num_digits)
                    numbered.setdefault(key, []).append((src_idx, entry))
                else:
                    unnumbered.append((src_idx, entry))

        # フェーズ2: 番号なし章をタイトルでクラスタリング（厳しめのしきい値 0.75）
        title_clusters: dict[str, list[tuple[int, ChapterEntry]]] = {}
        for src_idx, entry in unnumbered:
            norm = entry.normalized()
            if not norm:
                continue
            matched_key = None
            for key in title_clusters:
                if SequenceMatcher(None, norm, key).ratio() >= 0.75:
                    matched_key = key
                    break
            if matched_key:
                title_clusters[matched_key].append((src_idx, entry))
            else:
                title_clusters[norm] = [(src_idx, entry)]

        # フェーズ3: ≥2 ソースで確認されたグループを採用
        n_sources = len(extractions)
        threshold = min(2, n_sources)
        confirmed_items: list[tuple[ChapterEntry, int]] = []  # (entry, agreed_count)
        max_agreed = 0

        all_groups = list(numbered.values()) + list(title_clusters.values())
        for members in all_groups:
            src_set = {idx for idx, _ in members}
            if len(src_set) >= threshold:
                best = max((e for _, e in members), key=lambda e: len(e.title))
                confirmed_items.append((best, len(src_set)))
                max_agreed = max(max_agreed, len(src_set))

        if not confirmed_items:
            best_toc = max(extractions, key=lambda x: len(x[1]))[1]
            return best_toc, 0.2, 0

        # 番号順でソート
        def sort_key(item: tuple[ChapterEntry, int]):
            e, _ = item
            nums = re.findall(r"\d+", e.num) if e.num else []
            return ([int(n) for n in nums],) if nums else ([999999],)

        confirmed_items.sort(key=sort_key)
        confirmed = [e for e, _ in confirmed_items]

        # coverage は全ソースの中央値章数を分母にする（ひとつのソースだけ詳細な場合に公平）
        counts = sorted(len(toc) for _, toc in extractions)
        median_chapters = counts[len(counts) // 2]
        coverage = len(confirmed) / max(median_chapters, 1)
        source_agreement = max_agreed / n_sources
        confidence = round(min(1.0, (coverage * 0.6 + source_agreement * 0.4) * 1.2), 2)

        return confirmed, confidence, max_agreed
