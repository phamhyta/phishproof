"""HTML preprocessing.

KnowPhish Sec. 5.2 removes JavaScript and CSS blocks and keeps the markup. PhishAgent App. C (HTML Processor) removes
CSS, JavaScript and tags with BeautifulSoup and keeps plain text.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup
from bs4.element import Comment

# KPD's HTML is cut at 12,000 characters because KnowPhish states no limit and GPT-3.5-turbo-instruct held about this much.
KPD_HTML_LIMIT = 12_000
# PhishAgent's text is cut at 7,000 characters, the paper's context limit for the panel, because the paper states no limit.
AGENT_TEXT_LIMIT = 7_000


def _clean_soup(html: str) -> BeautifulSoup:
    # the stdlib html.parser, and noscript removed with script and style as the paper's text extractor does.
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()
    return soup


def kpd_html(html: str, limit: int = KPD_HTML_LIMIT) -> str:
    """Return HTML with script and style blocks removed, keeping tags (KnowPhish Sec. 5.2)."""
    return re.sub(r"\s+", " ", str(_clean_soup(html))).strip()[:limit]


def plain_text(html: str, limit: int = AGENT_TEXT_LIMIT) -> str:
    """Return the page's plain text (PhishAgent App. C)."""
    return re.sub(r"\s+", " ", _clean_soup(html).get_text(" ", strip=True)).strip()[:limit]
