"""Helpers for writing portable, standalone HTML research reports.

The Phase 2 deliverables are often copied out of the repository for advisor
review.  Local image/CSS/JS references are therefore inlined at report-write
time so the copied HTML file does not depend on sibling asset folders.  External
CDN and map-tile URLs are intentionally preserved.
"""
from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlparse


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


def _is_external_or_inline(url: str) -> bool:
    stripped = url.strip()
    if not stripped:
        return True
    if stripped.startswith(("#", "//")):
        return True
    parsed = urlparse(stripped)
    return parsed.scheme in {"http", "https", "data", "mailto", "tel", "javascript"}


def _local_path(url: str, base_dir: Path) -> Path | None:
    if _is_external_or_inline(url):
        return None
    parsed = urlparse(url)
    if parsed.scheme or parsed.netloc:
        return None
    path_text = unquote(parsed.path)
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


def file_to_data_uri(path: str | Path) -> str:
    """Return a Base64 data URI for a local file."""
    file_path = Path(path)
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _embed_img_src(match: re.Match[str], base_dir: Path) -> str:
    prefix = match.group("prefix")
    quote = match.group("quote")
    url = match.group("url")
    path = _local_path(url, base_dir)
    if path and path.suffix.lower() in IMAGE_SUFFIXES and path.exists():
        return f"{prefix}{quote}{file_to_data_uri(path)}{quote}"
    return match.group(0)


def _embed_css_url(match: re.Match[str], base_dir: Path) -> str:
    quote = match.group("quote") or ""
    url = match.group("url")
    path = _local_path(url, base_dir)
    if path and path.suffix.lower() in IMAGE_SUFFIXES and path.exists():
        return f"url({quote}{file_to_data_uri(path)}{quote})"
    return match.group(0)


def _inline_local_stylesheet(match: re.Match[str], base_dir: Path) -> str:
    tag = match.group(0)
    href = match.group("href")
    path = _local_path(href, base_dir)
    if not path or path.suffix.lower() != ".css" or not path.exists():
        return tag
    css = path.read_text(encoding="utf-8")
    css = embed_local_html_assets(css, path.parent)
    return f"<style>\n{css}\n</style>"


def _inline_local_script(match: re.Match[str], base_dir: Path) -> str:
    src = match.group("src")
    path = _local_path(src, base_dir)
    if not path or path.suffix.lower() != ".js" or not path.exists():
        return match.group(0)
    script = path.read_text(encoding="utf-8")
    return f"<script>\n{script}\n</script>"


def embed_local_html_assets(document: str, base_dir: str | Path) -> str:
    """Inline local image/CSS/JS assets referenced by an HTML document.

    This intentionally does not rewrite ordinary ``<a href="...">`` report
    links or external CDN/tile URLs.  It only packages assets needed to render
    the current HTML page.
    """
    base = Path(base_dir)
    document = re.sub(
        r"(?P<prefix><img\b[^>]*?\bsrc\s*=\s*)(?P<quote>['\"])(?P<url>[^'\"]+)(?P=quote)",
        lambda match: _embed_img_src(match, base),
        document,
        flags=re.IGNORECASE,
    )
    document = re.sub(
        r"url\(\s*(?P<quote>['\"]?)(?P<url>[^)'\"\s]+)(?P=quote)\s*\)",
        lambda match: _embed_css_url(match, base),
        document,
        flags=re.IGNORECASE,
    )
    document = re.sub(
        r"<link\b(?=[^>]*\brel\s*=\s*['\"]stylesheet['\"])(?=[^>]*\bhref\s*=\s*['\"](?P<href>[^'\"]+)['\"])[^>]*>",
        lambda match: _inline_local_stylesheet(match, base),
        document,
        flags=re.IGNORECASE,
    )
    document = re.sub(
        r"<script\b(?=[^>]*\bsrc\s*=\s*['\"](?P<src>[^'\"]+)['\"])[^>]*>\s*</script>",
        lambda match: _inline_local_script(match, base),
        document,
        flags=re.IGNORECASE,
    )
    return document


def iframe_srcdoc_from_file(path: str | Path) -> str:
    """Return escaped ``srcdoc`` HTML for embedding a local HTML child page."""
    return html.escape(Path(path).read_text(encoding="utf-8"), quote=True)
