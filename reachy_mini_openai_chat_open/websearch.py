"""Internet lookup for the `web_search` tool.

The Realtime model can't search the web itself, so a search is one HTTPS call
to the OpenAI Responses API with its hosted `web_search` tool, under the same
API key. OpenAI does the browsing and hands back a short answer written to be
spoken; the query never goes to any other service, and `store: false` keeps
OpenAI from retaining the request.

Uses only the standard library and blocks, so callers must run it off the
asyncio loop (see realtime.SLOW_TOOLS).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .log import get_logger

logger = get_logger("websearch")

RESPONSES_URL = "https://api.openai.com/v1/responses"
TIMEOUT_S = 20.0
MAX_ANSWER_CHARS = 1200  # a spoken reply, not an article
MAX_SOURCES = 3

SEARCH_INSTRUCTIONS = (
    "Search the web and answer the question for a voice assistant that will "
    "read your answer out loud. Reply in two to four plain sentences: no "
    "markdown and no lists. Give the direct answer first, include dates or "
    "numbers when they matter, and say so plainly if the results don't settle "
    "the question."
)

# Reasoning models (gpt-5 family, o-series) default to medium effort, which
# roughly doubles a search's wait for no better answer; low halves it. Other
# models reject the parameter, and so do the gpt-5 "chat" variants.
_REASONING_MODEL = re.compile(r"^(gpt-5|o\d)(?!.*-chat)")


# Inline citations like "([bom.gov.au](https://...))": the sources are kept
# from the annotations instead, so the spoken text is left clean.
_INLINE_LINK = re.compile(r"\s*\(?\[[^\]]*\]\([^)]*\)\)?")
_HEADING = re.compile(r"^\s*#{1,6}\s+", re.M)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.M)


def _plain(text: str) -> str:
    """Strip markdown that some models (gpt-4.1-mini) add despite being told
    not to. They answer first and then append a headed block ("## Highlights",
    a market-data widget), so anything from the first heading after the answer
    is dropped; any other headings, bullets or bold markers are unwrapped."""
    first = _HEADING.search(text)
    if first and text[:first.start()].strip():
        text = text[:first.start()]
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = text.replace("**", "").replace("__", "")
    return re.sub(r"\n{3,}", "\n\n", text)


def _result(ok: bool, **fields) -> str:
    return json.dumps({"ok": ok, **fields}, ensure_ascii=False)


def _domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host[4:] if host.startswith("www.") else host


def _parse(body: dict) -> tuple[str, list[str]]:
    """Pull the answer text and its cited domains out of a Responses reply."""
    texts: list[str] = []
    sources: list[str] = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") != "output_text":
                continue
            texts.append(part.get("text") or "")
            for ann in part.get("annotations") or []:
                if ann.get("type") == "url_citation":
                    domain = _domain(ann.get("url") or "")
                    if domain and domain not in sources:
                        sources.append(domain)
    text = _plain(_INLINE_LINK.sub("", "\n".join(t for t in texts if t)))
    return text.strip(), sources[:MAX_SOURCES]


def search(query: str, api_key: str, model: str) -> str:
    """Look `query` up on the web. Returns a JSON string for the model.

    Never raises: a failure comes back as {"ok": false, "error": ...} so the
    robot can say it couldn't look it up instead of going quiet.
    """
    query = (query or "").strip()
    if not query:
        return _result(False, error="empty query")
    if not api_key:
        return _result(False, error="no API key")

    payload = {
        "model": model,
        "instructions": SEARCH_INSTRUCTIONS,
        "input": query,
        "tools": [{"type": "web_search"}],
        "store": False,
    }
    if _REASONING_MODEL.match(model):
        payload["reasoning"] = {"effort": "low"}
    req = urllib.request.Request(
        RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    logger.info("searching: %r (model=%s)", query, model)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            pass
        logger.warning("search failed: HTTP %s %s", e.code, detail)
        return _result(False, error=f"search failed (HTTP {e.code})")
    except (TimeoutError, urllib.error.URLError) as e:
        logger.warning("search failed: %s", e)
        return _result(False, error="search timed out or the internet is unreachable")
    except Exception:
        logger.exception("search failed")
        return _result(False, error="search failed")

    answer, sources = _parse(body)
    if not answer:
        logger.warning("search returned no answer (status=%s)", body.get("status"))
        return _result(False, error="no answer found")
    if len(answer) > MAX_ANSWER_CHARS:
        answer = answer[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0] + "…"
    logger.info("search answered from: %s", ", ".join(sources) or "(no citations)")
    return _result(True, answer=answer, sources=sources)
