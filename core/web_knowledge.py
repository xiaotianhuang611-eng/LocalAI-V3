from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib import robotparser
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass
class CrawledDocument:
    title: str
    url: str
    saved_path: Path
    text_chars: int


class WebKnowledgeError(RuntimeError):
    pass


class WebKnowledgeIngestor:
    """
    Web search + web crawling helper for LocalAI RAG.

    V4.8.2 changes:
    - More robust readable-text extraction.
    - Lower default minimum text threshold from 300 to 120 chars.
    - Extracts JSON-LD articleBody, meta description, OpenGraph, Twitter meta,
      article/main/body blocks, and normal HTML text.
    - Search results now keep snippets when available.
    - If search finds URLs but all crawls fail, it saves a search-summary
      Markdown document instead of failing completely. This keeps RAG usable
      with at least title/snippet/source metadata.

    Notes:
    - Static public pages and documentation pages work best.
    - JavaScript-only, login-only, protected, or very short pages may still not
      expose enough text to a simple crawler.
    - Direct URL crawl is still more stable than keyword search.
    """

    def __init__(
        self,
        root_dir: Path,
        user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        timeout_seconds: int = 20,
        max_page_bytes: int = 4_000_000,
        respect_robots: bool = True,
        min_text_chars: int = 120,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.user_agent = str(user_agent)
        self.timeout_seconds = int(timeout_seconds)
        self.max_page_bytes = int(max_page_bytes)
        self.respect_robots = bool(respect_robots)
        self.min_text_chars = int(min_text_chars)

        self.knowledge_dir = self.root_dir / "data" / "knowledge"
        self.web_dir = self.knowledge_dir / "web_crawled"
        self.web_dir.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        query = str(query or "").strip()
        max_results = max(1, min(int(max_results), 10))

        if not query:
            raise WebKnowledgeError("Search query is empty.")

        search_methods = [
            ("Bing RSS", self._search_bing_rss),
            ("DuckDuckGo Lite", self._search_duckduckgo_lite),
            ("DuckDuckGo HTML", self._search_duckduckgo_html),
            ("Bing HTML", self._search_bing_html),
        ]

        all_results: list[SearchResult] = []
        errors: list[str] = []

        for name, method in search_methods:
            try:
                results = method(query=query, max_results=max_results)
                all_results.extend(results)
            except Exception as exc:
                errors.append(f"{name}: {exc}")

            deduped = self._dedupe_results(all_results, max_results=max_results)
            if len(deduped) >= max_results:
                return deduped

        deduped = self._dedupe_results(all_results, max_results=max_results)
        if deduped:
            return deduped

        message = "No search results were extracted from the available search fallbacks."
        if errors:
            message += "\nSearch fallback errors:\n" + "\n".join(f"- {e}" for e in errors[:6])
        message += "\nUse Crawl URL with a direct article/documentation URL for the most stable result."
        raise WebKnowledgeError(message)

    def crawl_and_save(self, url: str) -> CrawledDocument:
        url = self._normalise_url(url)

        if not url:
            raise WebKnowledgeError("Invalid URL. Please use a normal http or https URL.")

        page = self._fetch_url(url, check_robots=self.respect_robots)
        title = self._extract_title(page) or self._title_from_url(url)
        readable_text = self._extract_best_readable_text(page)

        if len(readable_text) < self.min_text_chars:
            raise WebKnowledgeError(
                f"Extracted text is too short ({len(readable_text)} chars). "
                "This usually means the page is JavaScript-only, protected, login-only, "
                "or not article-like. Try a public documentation/article page, use another URL, "
                "or use Search so LocalAI can try several pages and save a search summary fallback."
            )

        saved_path = self._save_markdown(title=title, url=url, content=readable_text)

        return CrawledDocument(
            title=title,
            url=url,
            saved_path=saved_path,
            text_chars=len(readable_text),
        )

    def search_and_ingest(
        self,
        query: str,
        max_results: int = 3,
        delay_seconds: float = 1.0,
    ) -> tuple[list[CrawledDocument], list[str]]:
        query = str(query or "").strip()
        results = self.search(query=query, max_results=max_results)

        saved_documents: list[CrawledDocument] = []
        errors: list[str] = []

        for index, result in enumerate(results, start=1):
            try:
                document = self.crawl_and_save(result.url)
                saved_documents.append(document)
            except Exception as exc:
                errors.append(f"[{index}] {result.url} -> {exc}")

            if delay_seconds > 0:
                time.sleep(float(delay_seconds))

        if not saved_documents:
            summary_doc = self.save_search_summary(
                query=query,
                results=results,
                errors=errors,
            )
            saved_documents.append(summary_doc)
            errors.append(
                "All full-page crawls failed, so a search-summary Markdown file was saved instead. "
                "For better RAG quality, crawl a specific public article or documentation URL."
            )

        return saved_documents, errors

    def save_search_summary(
        self,
        query: str,
        results: list[SearchResult],
        errors: list[str] | None = None,
    ) -> CrawledDocument:
        query = str(query or "").strip()
        errors = errors or []
        now = datetime.now()
        safe_query = re.sub(r"[^A-Za-z0-9._-]+", "_", query).strip("_.-")[:70] or "web_search"
        path = self.web_dir / f"{now.strftime('%Y%m%d_%H%M%S')}_search_summary_{safe_query}.md"

        lines: list[str] = []
        lines.append("# Web Search Summary")
        lines.append("")
        lines.append("## Search Query")
        lines.append(query or "Untitled query")
        lines.append("")
        lines.append("## Created At")
        lines.append(now.isoformat(timespec="seconds"))
        lines.append("")
        lines.append("## Note")
        lines.append(
            "This document was automatically saved by LocalAI when direct crawling did not extract enough readable page text. "
            "It contains search result titles, snippets, and URLs, and can be indexed by the local RAG system. "
            "For higher-quality answers, crawl a specific static public article or documentation URL."
        )
        lines.append("")
        lines.append("## Search Results")

        if results:
            for index, result in enumerate(results, start=1):
                lines.append("")
                lines.append(f"### Result {index}: {result.title.strip() or 'Untitled'}")
                lines.append(f"URL: {result.url}")
                if result.snippet.strip():
                    lines.append("")
                    lines.append(result.snippet.strip())
        else:
            lines.append("")
            lines.append("No search results were available.")

        if errors:
            lines.append("")
            lines.append("## Crawl Errors")
            for error in errors[:12]:
                lines.append(f"- {error}")

        content = "\n".join(lines).strip() + "\n"
        path.write_text(content, encoding="utf-8")

        return CrawledDocument(
            title=f"Search summary: {query}",
            url="search://" + query,
            saved_path=path,
            text_chars=len(content),
        )

    def rebuild_rag_index(self) -> str:
        script_path = self.root_dir / "tools" / "build_rag_index.py"

        if not script_path.exists():
            raise FileNotFoundError(f"RAG builder not found: {script_path}")

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(self.root_dir),
            capture_output=True,
            text=True,
            check=False,
        )

        output = "\n".join(
            part.strip()
            for part in [completed.stdout, completed.stderr]
            if part and part.strip()
        ).strip()

        if completed.returncode != 0:
            raise WebKnowledgeError(
                f"RAG rebuild failed with exit code {completed.returncode}:\n{output}"
            )

        return output or "RAG index rebuilt successfully."

    def _search_bing_rss(self, query: str, max_results: int) -> list[SearchResult]:
        search_url = f"https://www.bing.com/search?q={quote_plus(query)}&format=rss"
        page = self._fetch_url(search_url, check_robots=False, allow_non_html=True)
        return self._parse_bing_rss_results(page)[:max_results]

    def _search_duckduckgo_lite(self, query: str, max_results: int) -> list[SearchResult]:
        search_url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        page = self._fetch_url(search_url, check_robots=False)
        return self._parse_duckduckgo_results(page)[:max_results]

    def _search_duckduckgo_html(self, query: str, max_results: int) -> list[SearchResult]:
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        page = self._fetch_url(search_url, check_robots=False)
        return self._parse_duckduckgo_results(page)[:max_results]

    def _search_bing_html(self, query: str, max_results: int) -> list[SearchResult]:
        search_url = f"https://www.bing.com/search?q={quote_plus(query)}&count={max_results + 5}&setlang=en-US"
        page = self._fetch_url(search_url, check_robots=False)
        return self._parse_bing_html_results(page)[:max_results]

    def _fetch_url(
        self,
        url: str,
        check_robots: bool = True,
        allow_non_html: bool = False,
    ) -> str:
        if check_robots and not self._can_fetch(url):
            raise WebKnowledgeError(f"robots.txt does not allow crawling this URL: {url}")

        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml,text/xml,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.7,zh;q=0.6",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
            method="GET",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                content_type = response.headers.get("Content-Type", "")
                content_type_lower = content_type.lower()

                if not allow_non_html:
                    allowed = (
                        "text/html" in content_type_lower
                        or "application/xhtml" in content_type_lower
                        or "text/plain" in content_type_lower
                    )
                    if content_type and not allowed:
                        raise WebKnowledgeError(
                            f"Unsupported content type: {content_type or 'unknown'}"
                        )

                raw = response.read(self.max_page_bytes + 1)
                if len(raw) > self.max_page_bytes:
                    raw = raw[: self.max_page_bytes]

                charset = self._charset_from_content_type(content_type) or "utf-8"
                return raw.decode(charset, errors="replace")

        except HTTPError as exc:
            raise WebKnowledgeError(f"HTTP error {exc.code} while fetching {url}") from exc
        except URLError as exc:
            raise WebKnowledgeError(f"Network error while fetching {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise WebKnowledgeError(f"Timeout while fetching {url}") from exc

    def _can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False

        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = robotparser.RobotFileParser()
        parser.set_url(robots_url)

        try:
            parser.read()
            return bool(parser.can_fetch(self.user_agent, url))
        except Exception:
            return True

    def _parse_bing_rss_results(self, page: str) -> list[SearchResult]:
        page = str(page or "")
        results: list[SearchResult] = []

        item_pattern = re.compile(r"<item\b[^>]*>(.*?)</item>", flags=re.IGNORECASE | re.DOTALL)
        for item in item_pattern.findall(page):
            title_match = re.search(r"<title\b[^>]*>(.*?)</title>", item, flags=re.IGNORECASE | re.DOTALL)
            link_match = re.search(r"<link\b[^>]*>(.*?)</link>", item, flags=re.IGNORECASE | re.DOTALL)
            desc_match = re.search(r"<description\b[^>]*>(.*?)</description>", item, flags=re.IGNORECASE | re.DOTALL)

            if not title_match or not link_match:
                continue

            title = self._strip_tags(title_match.group(1))
            url = html.unescape(link_match.group(1)).strip()
            url = self._normalise_url(url)
            snippet = self._strip_tags(desc_match.group(1)) if desc_match else ""

            if title and url:
                results.append(SearchResult(title=title, url=url, snippet=snippet))

        return results

    def _parse_bing_html_results(self, page: str) -> list[SearchResult]:
        page = str(page or "")
        results: list[SearchResult] = []

        li_pattern = re.compile(
            r'<li[^>]+class=["\'][^"\']*b_algo[^"\']*["\'][^>]*>(.*?)</li>',
            flags=re.IGNORECASE | re.DOTALL,
        )

        for block in li_pattern.findall(page):
            match = re.search(
                r'<h2[^>]*>.*?<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if not match:
                continue

            url = self._normalise_url(html.unescape(match.group(1)).strip())
            title = self._strip_tags(match.group(2))
            snippet = ""

            snippet_match = re.search(
                r'<p[^>]*>(.*?)</p>',
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if snippet_match:
                snippet = self._strip_tags(snippet_match.group(1))

            if title and url:
                results.append(SearchResult(title=title, url=url, snippet=snippet))

        if results:
            return results

        return self._parse_generic_external_links(page, blocked_domains={"bing.com", "microsoft.com"})

    def _parse_duckduckgo_results(self, page: str) -> list[SearchResult]:
        page = str(page or "")
        results: list[SearchResult] = []

        patterns = [
            re.compile(
                r'<a[^>]+class=["\'][^"\']*result__a[^"\']*["\'][^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                flags=re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r'<a[^>]+rel=["\']nofollow["\'][^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                flags=re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r'<a[^>]+href=["\']([^"\']+)["\'][^>]+class=["\'][^"\']*result-link[^"\']*["\'][^>]*>(.*?)</a>',
                flags=re.IGNORECASE | re.DOTALL,
            ),
        ]

        for pattern in patterns:
            for href, raw_title in pattern.findall(page):
                url = self._decode_duckduckgo_href(href)
                url = self._normalise_url(url)
                title = self._strip_tags(raw_title)

                if url and title:
                    results.append(SearchResult(title=title, url=url))

            if results:
                return results

        return self._parse_generic_external_links(page, blocked_domains={"duckduckgo.com", "ddg.gg"})

    def _parse_generic_external_links(
        self,
        page: str,
        blocked_domains: set[str] | None = None,
    ) -> list[SearchResult]:
        page = str(page or "")
        blocked_domains = blocked_domains or set()
        results: list[SearchResult] = []

        generic_pattern = re.compile(
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            flags=re.IGNORECASE | re.DOTALL,
        )

        for href, raw_title in generic_pattern.findall(page):
            url = self._decode_duckduckgo_href(href)
            url = self._normalise_url(url)
            title = self._strip_tags(raw_title)

            if not url or not title:
                continue

            parsed = urlparse(url)
            domain = parsed.netloc.lower().removeprefix("www.")

            if any(domain.endswith(blocked) for blocked in blocked_domains):
                continue

            if self._is_bad_result_url(url):
                continue

            if len(title) < 4 or len(title) > 220:
                continue

            results.append(SearchResult(title=title, url=url))

        return results

    def _dedupe_results(self, results: list[SearchResult], max_results: int) -> list[SearchResult]:
        deduped: list[SearchResult] = []
        seen: set[str] = set()

        for result in results:
            url = self._normalise_url(result.url)
            title = re.sub(r"\s+", " ", str(result.title or "")).strip()
            snippet = re.sub(r"\s+", " ", str(result.snippet or "")).strip()

            if not url or not title:
                continue

            if self._is_bad_result_url(url):
                continue

            key = self._canonical_url_key(url)
            if key in seen:
                continue

            seen.add(key)
            deduped.append(SearchResult(title=title[:180], url=url, snippet=snippet[:600]))

            if len(deduped) >= max_results:
                break

        return deduped

    def _decode_duckduckgo_href(self, href: str) -> str:
        href = html.unescape(str(href or "").strip())

        if href.startswith("//"):
            href = "https:" + href

        parsed = urlparse(href)

        if "duckduckgo.com" in parsed.netloc.lower() and parsed.path.startswith("/l/"):
            query = parse_qs(parsed.query)
            uddg = query.get("uddg", [""])[0]
            if uddg:
                return unquote(uddg)

        if href.startswith("/l/?"):
            query = parse_qs(urlparse("https://duckduckgo.com" + href).query)
            uddg = query.get("uddg", [""])[0]
            if uddg:
                return unquote(uddg)

        return href

    def _extract_title(self, page: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", page, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            og_title = self._extract_meta_content(page, ["og:title", "twitter:title"])
            return og_title[:180]

        title = self._strip_tags(match.group(1))
        title = re.sub(r"\s+", " ", title).strip()
        return title[:180]

    def _extract_best_readable_text(self, page: str) -> str:
        page = str(page or "")
        candidates: list[str] = []

        json_text = self._extract_json_ld_text(page)
        if json_text:
            candidates.append(json_text)

        for tag in ["article", "main", "section", "body"]:
            for block in self._extract_tag_blocks(page, tag):
                block_text = self._html_to_text(block)
                if block_text:
                    candidates.append(block_text)

        full_text = self._html_to_text(page)
        if full_text:
            candidates.append(full_text)

        meta_text = self._extract_meta_text(page)
        if meta_text:
            candidates.append(meta_text)

        cleaned_candidates = [self._post_clean_text(c) for c in candidates if self._post_clean_text(c)]
        if not cleaned_candidates:
            return ""

        # Prefer the longest content block, but do not allow extremely noisy duplicates.
        best = max(cleaned_candidates, key=len)
        return best.strip()

    def _extract_json_ld_text(self, page: str) -> str:
        texts: list[str] = []
        pattern = re.compile(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            flags=re.IGNORECASE | re.DOTALL,
        )

        for raw in pattern.findall(page):
            raw = html.unescape(raw).strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
                texts.extend(self._collect_json_text(data))
            except Exception:
                # Some websites contain invalid JSON-LD. Extract simple articleBody manually.
                match = re.search(r'"articleBody"\s*:\s*"(.*?)"', raw, flags=re.DOTALL)
                if match:
                    texts.append(match.group(1))

        return self._post_clean_text("\n\n".join(texts))

    def _collect_json_text(self, data) -> list[str]:
        texts: list[str] = []

        if isinstance(data, list):
            for item in data:
                texts.extend(self._collect_json_text(item))
            return texts

        if not isinstance(data, dict):
            return texts

        for key in ["headline", "name", "description", "articleBody", "text"]:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())

        for value in data.values():
            if isinstance(value, (list, dict)):
                texts.extend(self._collect_json_text(value))

        return texts

    def _extract_tag_blocks(self, page: str, tag: str) -> list[str]:
        pattern = re.compile(
            rf"<{tag}\b[^>]*>(.*?)</{tag}>",
            flags=re.IGNORECASE | re.DOTALL,
        )
        return pattern.findall(page)

    def _extract_meta_text(self, page: str) -> str:
        title = self._extract_title(page)
        meta = self._extract_meta_content(
            page,
            [
                "description",
                "og:description",
                "twitter:description",
                "article:tag",
            ],
        )

        pieces = [piece for piece in [title, meta] if piece.strip()]
        return self._post_clean_text("\n\n".join(pieces))

    def _extract_meta_content(self, page: str, names: list[str]) -> str:
        contents: list[str] = []
        for name in names:
            patterns = [
                rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]+content=["\']([^"\']+)["\'][^>]*>',
                rf'<meta[^>]+property=["\']{re.escape(name)}["\'][^>]+content=["\']([^"\']+)["\'][^>]*>',
                rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']{re.escape(name)}["\'][^>]*>',
                rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']{re.escape(name)}["\'][^>]*>',
            ]
            for pattern in patterns:
                for match in re.findall(pattern, page, flags=re.IGNORECASE | re.DOTALL):
                    text = html.unescape(match).strip()
                    if text:
                        contents.append(text)
        return self._post_clean_text("\n\n".join(contents))

    def _html_to_text(self, raw_html: str) -> str:
        text = str(raw_html or "")

        text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<noscript\b[^>]*>.*?</noscript>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<svg\b[^>]*>.*?</svg>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

        block_tags = [
            "article", "section", "main", "header", "footer", "aside", "nav", "div", "p",
            "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
            "tr", "td", "th", "blockquote", "pre", "code",
        ]

        for tag in block_tags:
            text = re.sub(rf"</?{tag}\b[^>]*>", "\n", text, flags=re.IGNORECASE)

        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = text.replace("\xa0", " ")
        return self._post_clean_text(text)

    def _post_clean_text(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("\\n", "\n")

        lines: list[str] = []
        for line in text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue

            lower = line.lower().strip()
            if lower in {
                "menu", "navigation", "skip to content", "advertisement",
                "privacy policy", "terms of service", "cookie policy",
                "cookies", "accept", "reject", "subscribe", "sign in",
                "log in", "copyright", "all rights reserved",
            }:
                continue

            if len(line) <= 2:
                continue

            # Drop very common navigation fragments.
            if len(line) < 25 and lower in {
                "home", "about", "contact", "products", "services", "blog",
                "pricing", "careers", "download", "documentation", "docs",
            }:
                continue

            lines.append(line)

        cleaned_lines: list[str] = []
        previous = ""
        seen_short: set[str] = set()

        for line in lines:
            if line == previous:
                continue
            previous = line

            key = line.lower()
            if len(line) < 100:
                if key in seen_short:
                    continue
                seen_short.add(key)

            cleaned_lines.append(line)

        content = "\n\n".join(cleaned_lines)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        return content

    def _save_markdown(self, title: str, url: str, content: str) -> Path:
        now = datetime.now()
        slug = self._slug_from_url(url)
        filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{slug}.md"
        path = self.web_dir / filename

        markdown = (
            "# Web Crawled Document\n\n"
            f"## Title\n{title.strip() or 'Untitled'}\n\n"
            f"## Source URL\n{url}\n\n"
            f"## Crawled At\n{now.isoformat(timespec='seconds')}\n\n"
            "## Content\n"
            f"{content.strip()}\n"
        )

        path.write_text(markdown, encoding="utf-8")
        return path

    def _slug_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        raw = f"{parsed.netloc}_{parsed.path}".strip("_/ ")
        raw = raw or "web_page"
        raw = unquote(raw)
        raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
        raw = re.sub(r"_+", "_", raw).strip("_.-")
        return (raw or "web_page")[:90]

    def _title_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        title = unquote(parsed.path.strip("/").split("/")[-1] or parsed.netloc)
        title = title.replace("-", " ").replace("_", " ").strip()
        return title or parsed.netloc or "Untitled Web Page"

    def _normalise_url(self, url: str) -> str:
        url = html.unescape(str(url or "").strip())

        if not url:
            return ""

        if url.startswith("//"):
            url = "https:" + url

        if not re.match(r"^https?://", url, flags=re.IGNORECASE):
            return ""

        parsed = urlparse(url)

        if parsed.scheme.lower() not in {"http", "https"}:
            return ""

        if not parsed.netloc:
            return ""

        return url

    def _canonical_url_key(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"

    def _is_bad_result_url(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().removeprefix("www.")
        path = parsed.path.lower()

        blocked_domains = {
            "bing.com", "microsoft.com", "duckduckgo.com", "google.com",
            "youtube.com", "youtu.be", "facebook.com", "instagram.com",
            "tiktok.com", "x.com", "twitter.com", "reddit.com",
        }

        if any(domain.endswith(blocked) for blocked in blocked_domains):
            return True

        blocked_extensions = (
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
            ".mp4", ".mp3", ".avi", ".mov", ".zip", ".rar", ".exe",
            ".dmg", ".iso",
        )

        if path.endswith(blocked_extensions):
            return True

        return False

    def _charset_from_content_type(self, content_type: str) -> str:
        match = re.search(r"charset=([^;]+)", str(content_type or ""), flags=re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).strip().strip('"')

    def _strip_tags(self, raw_html: str) -> str:
        text = re.sub(r"<[^>]+>", " ", str(raw_html or ""))
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
