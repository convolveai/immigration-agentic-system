#!/usr/bin/env python3
"""
Generate a sitemap.xml by crawling a starting URL.

Key behaviors:
- Restricts crawling to the same host as the start URL.
- By default, restricts discovered/enqueued links to the start URL's directory prefix.
  (Special case: always fetch the start URL even if it's a .html hub that sits "above" the prefix.)
- Respects robots.txt (basic).
- Canonicalizes URLs by dropping fragments + query strings.

Usage:
  uv run generate_sitemap.py --start <url> --out <file>

Example:
  uv run generate_sitemap.py \
    --start https://www.canada.ca/en/immigration-refugees-citizenship/corporate/publications-manuals/operational-bulletins-manuals.html \
    --out operational-bulletins-manuals.sitemap.xml
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import re
import sys
import time
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass
from typing import Iterable

import httpx
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class CrawlConfig:
    start_url: str
    out_path: str
    scope_prefix: str
    exclude_prefixes: tuple[str, ...]
    max_pages: int  # 0 means "no limit"
    timeout_s: float
    delay_s: float
    user_agent: str
    max_retries: int
    backoff_base_s: float


# Skip non-HTML resources (v1 choice). Add/remove as needed later.
_SKIP_EXT_RE = re.compile(
    r"\.(?:pdf|zip|rar|7z|tar|gz|tgz|bz2|xz|doc|docx|ppt|pptx|xls|xlsx|csv|"
    r"png|jpg|jpeg|gif|webp|svg|ico|mp3|mp4|mov|avi|mkv|wav|"
    r"css|js|json|xml|txt)$",
    re.IGNORECASE,
)


def _canonicalize(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    # Drop fragment and query for canonical sitemap URLs.
    parts = parts._replace(fragment="", query="")

    # Normalize path: remove duplicate slashes.
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    parts = parts._replace(path=path)

    canon = urllib.parse.urlunsplit(parts)

    # Strip trailing slash except for root.
    if canon.endswith("/") and urllib.parse.urlsplit(canon).path != "/":
        canon = canon[:-1]

    return canon


def _is_http_url(url: str) -> bool:
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    return scheme in {"http", "https"}


def _same_host(url: str, host: str) -> bool:
    return urllib.parse.urlsplit(url).netloc.lower() == host.lower()


def _in_scope(url: str, scope_prefix: str) -> bool:
    # Scope prefix is a full URL prefix (scheme+host+path prefix directory).
    prefix = scope_prefix.rstrip("/") + "/"
    candidate = _canonicalize(url).rstrip("/") + "/"
    return candidate.startswith(prefix)


def _is_excluded(url: str, exclude_prefixes: tuple[str, ...]) -> bool:
    if not exclude_prefixes:
        return False
    candidate = _canonicalize(url).rstrip("/") + "/"
    for raw_prefix in exclude_prefixes:
        prefix = _canonicalize(raw_prefix).rstrip("/") + "/"
        if candidate.startswith(prefix):
            return True
    return False


def _should_skip(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"}:
        return True
    if _SKIP_EXT_RE.search(parts.path or ""):
        return True
    return False


def _extract_links(base_url: str, html: str) -> Iterable[str]:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        yield urllib.parse.urljoin(base_url, href)


def _build_scope_prefix(start_url: str) -> str:
    start = urllib.parse.urlsplit(_canonicalize(start_url))
    path = start.path

    # If seed is an .html hub, scope to a same-named directory (common on canada.ca).
    if path.endswith(".html"):
        path = path[: -len(".html")].rstrip("/") + "/"
    elif not path.endswith("/"):
        path = path.rsplit("/", 1)[0] + "/"

    return urllib.parse.urlunsplit((start.scheme, start.netloc, path, "", ""))


def _robots_parser_for(
    start_url: str, client: httpx.Client, user_agent: str
) -> urllib.robotparser.RobotFileParser:
    parts = urllib.parse.urlsplit(start_url)
    robots_url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        resp = client.get(robots_url, headers={"User-Agent": user_agent})
        if resp.status_code >= 400:
            rp.parse([])  # allow all if robots.txt is missing/unreachable
            return rp
        rp.parse(resp.text.splitlines())
        return rp
    except Exception:
        rp.parse([])  # allow all on failure
        return rp


def _write_sitemap(urls: list[str], out_path: str) -> None:
    now = _dt.datetime.now(tz=_dt.timezone.utc).date().isoformat()

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
    for u in urls:
        lines.append("  <url>")
        lines.append(f"    <loc>{esc(u)}</loc>")
        # Crawl generation date (not true server lastmod).
        lines.append(f"    <lastmod>{now}</lastmod>")
        lines.append("  </url>")
    lines.append("</urlset>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _get_with_retries(client: httpx.Client, url: str, headers: dict[str, str], cfg: CrawlConfig) -> httpx.Response | None:
    # Basic backoff for rate limits / transient errors.
    for attempt in range(cfg.max_retries + 1):
        try:
            resp = client.get(url, headers=headers)
        except httpx.RequestError as e:
            if attempt >= cfg.max_retries:
                print(f"WARN request failed: {url} ({e})", file=sys.stderr)
                return None
            time.sleep(cfg.backoff_base_s * (2 ** attempt))
            continue

        if resp.status_code in {429, 500, 502, 503, 504}:
            if attempt >= cfg.max_retries:
                return resp
            # Respect Retry-After if present (seconds)
            ra = resp.headers.get("retry-after")
            if ra and ra.isdigit():
                time.sleep(float(ra))
            else:
                time.sleep(cfg.backoff_base_s * (2 ** attempt))
            continue

        return resp

    return None


def crawl(cfg: CrawlConfig) -> list[str]:
    start = _canonicalize(cfg.start_url)
    host = urllib.parse.urlsplit(start).netloc

    seen: set[str] = set()
    queued: collections.deque[str] = collections.deque([start])

    discovered: set[str] = set()

    print(f"Crawling start: {start}")
    print(f"Scope prefix: {cfg.scope_prefix}")
    print(f"Host: {host} (max_pages={cfg.max_pages or 'no limit'})")
    if cfg.exclude_prefixes:
        print("Excludes:")
        for x in cfg.exclude_prefixes:
            print(f"  - {x}")

    timeout = httpx.Timeout(cfg.timeout_s)
    headers = {"User-Agent": cfg.user_agent, "Accept": "text/html,application/xhtml+xml"}

    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        rp = _robots_parser_for(start, client, cfg.user_agent)

        try:
            while queued:
                # max_pages applies to discovered URLs in the output set
                if cfg.max_pages and len(discovered) >= cfg.max_pages:
                    print(f"Reached max_pages={cfg.max_pages}; stopping.")
                    break

                url = _canonicalize(queued.popleft())
                if url in seen:
                    continue
                seen.add(url)

                if _should_skip(url):
                    continue
                if not _is_http_url(url) or not _same_host(url, host):
                    continue
                if _is_excluded(url, cfg.exclude_prefixes):
                    continue

                is_seed = (url == start)

                # Always fetch seed even if it's not inside the scope prefix
                if (not is_seed) and (not _in_scope(url, cfg.scope_prefix)):
                    continue

                # robots.txt: use full URL (works better than just path)
                try:
                    allowed = rp.can_fetch(cfg.user_agent, url)
                except Exception:
                    allowed = True
                if not allowed:
                    continue

                if cfg.delay_s:
                    time.sleep(cfg.delay_s)

                resp = _get_with_retries(client, url, headers, cfg)
                if resp is None:
                    continue

                if resp.status_code >= 400:
                    continue

                ctype = (resp.headers.get("content-type") or "").lower()
                if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
                    continue

                # Include seed always; otherwise only include URLs that are in scope.
                if is_seed or _in_scope(url, cfg.scope_prefix):
                    discovered.add(url)

                if len(discovered) and (len(discovered) % 25 == 0):
                    print(f"... discovered {len(discovered)} urls")

                # Extract links; only enqueue those that are same-host + in-scope + not excluded.
                for link in _extract_links(url, resp.text):
                    link = _canonicalize(link)
                    if link in seen:
                        continue
                    if _should_skip(link):
                        continue
                    if not _same_host(link, host):
                        continue
                    if not _in_scope(link, cfg.scope_prefix):
                        continue
                    if _is_excluded(link, cfg.exclude_prefixes):
                        continue
                    queued.append(link)

        except KeyboardInterrupt:
            print("\nInterrupted; returning partial results...", file=sys.stderr)

    return sorted(discovered)



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start URL to crawl")
    p.add_argument("--out", required=True, help="Output sitemap xml path")
    p.add_argument(
        "--scope-prefix",
        default="",
        help="Optional full URL prefix to constrain crawling. Defaults to start URL's directory.",
    )
    p.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Full URL prefix to exclude. Can be provided multiple times.",
    )
    p.add_argument("--max-pages", type=int, default=500, help="Hard limit on sitemap URLs (0 = no limit)")
    p.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout seconds")
    p.add_argument("--delay", type=float, default=0.25, help="Delay between requests in seconds")
    p.add_argument("--retries", type=int, default=3, help="Retries for transient HTTP errors")
    p.add_argument("--backoff", type=float, default=0.75, help="Backoff base seconds for retries")
    p.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (compatible; SitemapGenerator/1.1; +https://example.invalid)",
        help="User-Agent header value",
    )

    args = p.parse_args(argv)

    scope = args.scope_prefix.strip() or _build_scope_prefix(args.start)
    scope = _canonicalize(scope).rstrip("/") + "/"

    # Default exclusion for IRCC manuals section: updates route.
    default_excludes: list[str] = []
    if scope.startswith(
        "https://www.canada.ca/en/immigration-refugees-citizenship/corporate/publications-manuals/operational-bulletins-manuals/"
    ):
        default_excludes.append(
            "https://www.canada.ca/en/immigration-refugees-citizenship/corporate/publications-manuals/operational-bulletins-manuals/updates/"
        )

    user_excludes = [x.strip() for x in (args.exclude_prefix or []) if x and x.strip()]
    exclude_prefixes = tuple(default_excludes + user_excludes)

    cfg = CrawlConfig(
        start_url=args.start,
        out_path=args.out,
        scope_prefix=scope,
        exclude_prefixes=exclude_prefixes,
        max_pages=max(0, int(args.max_pages)),
        timeout_s=float(args.timeout),
        delay_s=max(0.0, float(args.delay)),
        user_agent=str(args.user_agent),
        max_retries=max(0, int(args.retries)),
        backoff_base_s=max(0.0, float(args.backoff)),
    )

    urls = crawl(cfg)
    if not urls:
        print("No URLs discovered; not writing sitemap.", file=sys.stderr)
        return 2

    _write_sitemap(urls, cfg.out_path)
    print(f"Wrote sitemap: {cfg.out_path} ({len(urls)} urls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
