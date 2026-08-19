#!/usr/bin/env python3
"""Official atomic company-news run on the locked 132-item set.

Fresh vendor calls only. Does not reuse dataset-building smokes.
Stores raw request/response + latency for every endpoint call.

  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/run_official_news_probe.py
  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/run_official_news_probe.py --resume DIR
  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/run_official_news_probe.py \\
    --endpoints linkup_fast,linkup_standard --attach DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from _shared import load_environment  # noqa: E402
from websearch.predictleads_news import (  # noqa: E402
    expand_categories,
    fetch_category_events,
    pick_categories,
    recipe_categories,
)
from websearch.run_atomic_news_probe import (  # noqa: E402
    MAX_RESULTS,
    _dedupe,
    _hit,
    _openai,
    _openai_usd,
    answer_in_excerpt,
    extract_answer,
    score_gold,
    source_metrics,
)

SAMPLES = ROOT / "data" / "company-news" / "samples.json"
OUT_ROOT = ROOT / "data" / "company-news" / "official-runs"
DEFAULT_MODEL = "gpt-5.6-terra"

# List prices as of 2026-08-16. Seltz and Brave Search are $5 / 1,000 requests.
# Tavily PAYG is $0.008/credit; ultra-fast is 1 credit, advanced is 2.
# RapidAPI google-search74 Pro overage is $0.003/request.
UNIT_COST_USD = {
    "parallel_turbo": 0.001,
    "parallel_basic": 0.005,
    "parallel_advanced": 0.005,
    "exa_instant": 0.007,
    "exa_deep": 0.012,
    "firecrawl": 0.005,
    "predictleads": 0.04,
    "seltz_news": 0.005,
    "seltz_companies": 0.005,
    "brave": 0.005,
    "tavily_ultrafast": 0.008,
    "tavily_advanced": 0.016,
    "serp": 0.003,
    "linkup_fast": 0.005,
    "linkup_standard": 0.005,
    "telnyx_web_search": 0.005,
    "openai_input_per_m": 0.40,
    "openai_output_per_m": 1.60,
}

ENDPOINTS = (
    "exa_instant",
    "exa_deep",
    "parallel_basic",
    "parallel_advanced",
    "firecrawl",
    "predictleads",
    "seltz_news",
    "brave",
    "tavily_advanced",
    "serp",
    "linkup_fast",
    "linkup_standard",
    "telnyx_web_search",
)

ENDPOINT_MAP = {
    "parallel_turbo": "POST https://api.parallel.ai/v1/search mode=turbo",
    "parallel_basic": "POST https://api.parallel.ai/v1/search mode=basic",
    "parallel_advanced": "POST https://api.parallel.ai/v1/search mode=advanced",
    "exa_instant": "POST https://api.exa.ai/search type=instant",
    "exa_deep": "POST https://api.exa.ai/search type=deep",
    "firecrawl": "POST https://api.firecrawl.dev/v2/search",
    "predictleads": "GET https://predictleads.com/api/v3/companies/{domain}/news_events categories[]",
    "seltz_news": "POST https://api.seltz.ai/v1/search scope=news max_results=10",
    "seltz_companies": "POST https://api.seltz.ai/v1/search scope=companies max_results=10",
    "brave": "GET https://api.search.brave.com/res/v1/web/search count=10 result_filter=web",
    "tavily_ultrafast": "POST https://api.tavily.com/search search_depth=ultra-fast max_results=10",
    "tavily_advanced": "POST https://api.tavily.com/search search_depth=advanced max_results=10",
    "serp": "GET https://google-search74.p.rapidapi.com/ query limit=10",
    "linkup_fast": "POST https://api.linkup.so/v1/search depth=fast outputType=searchResults maxResults=10",
    "linkup_standard": "POST https://api.linkup.so/v1/search depth=standard outputType=searchResults maxResults=10",
    "telnyx_web_search": "POST https://api.telnyx.com/v2/web_search count=10",
}

ENDPOINT_ENV = {
    "parallel_turbo": ("PARALLEL_API_KEY",),
    "parallel_basic": ("PARALLEL_API_KEY",),
    "parallel_advanced": ("PARALLEL_API_KEY",),
    "exa_instant": ("EXA_API_KEY",),
    "exa_deep": ("EXA_API_KEY",),
    "firecrawl": ("FIRECRAWL_API_KEY",),
    "predictleads": ("PREDICT_LEADS_API_KEY", "PREDICT_LEADS_API_TOKEN"),
    "seltz_news": ("SELTZ_API_KEY",),
    "seltz_companies": ("SELTZ_API_KEY",),
    "brave": ("BRAVE_SEARCH_API_KEY",),
    "tavily_ultrafast": ("TAVILY_API_KEY",),
    "tavily_advanced": ("TAVILY_API_KEY",),
    "serp": ("RAPIDAPI_KEY",),
    "linkup_fast": ("LINKUP_API_KEY",),
    "linkup_standard": ("LINKUP_API_KEY",),
    "telnyx_web_search": ("TELNYX_API_KEY",),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    out = {}
    for key, value in headers.items():
        lowered = key.lower()
        if any(tok in lowered for tok in ("key", "token", "authorization", "secret", "password")):
            out[key] = "***REDACTED***"
        else:
            out[key] = value
    return out


def _http(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | None,
    params: dict[str, Any] | None,
    timeout: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = _now()
    attempts: list[dict[str, Any]] = []
    last_status = None
    last_payload: Any = None
    error = None
    ok = False
    for attempt in range(4):
        attempt_started = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "method": method,
                "url": url,
                "headers": headers,
                "params": params,
                "timeout": timeout,
            }
            if body is not None:
                kwargs["json"] = body
            response = requests.request(**kwargs)
            elapsed_ms = round((time.perf_counter() - attempt_started) * 1000)
            last_status = response.status_code
            try:
                last_payload = response.json()
            except ValueError:
                last_payload = {"_raw": (response.text or "")[:8000]}
            attempts.append(
                {
                    "n": attempt + 1,
                    "status_code": response.status_code,
                    "latency_ms": elapsed_ms,
                    "ok": response.ok,
                }
            )
            if response.ok:
                ok = True
                error = None
                break
            error = f"HTTP {response.status_code} {url}: {str(last_payload)[:400]}"
            if response.status_code in {429, 500, 502, 503} and attempt < 3:
                time.sleep(8 * (attempt + 1))
                continue
            break
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = round((time.perf_counter() - attempt_started) * 1000)
            error = f"{type(exc).__name__}: {exc}"
            attempts.append({"n": attempt + 1, "status_code": None, "latency_ms": elapsed_ms, "ok": False, "error": error})
            if attempt < 3:
                time.sleep(4 * (attempt + 1))
                continue
            break
    return {
        "method": method,
        "url": url,
        "request": {
            "headers": _redact_headers(headers),
            "body": body,
            "params": params,
        },
        "ok": ok,
        "error": error,
        "status_code": last_status,
        "response": last_payload,
        "attempts": attempts,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "started_at": started_at,
        "ended_at": _now(),
    }


def _parse_exa(payload: Any) -> list[dict[str, Any]]:
    hits = []
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        highlights = item.get("highlights") or []
        snippet = " ".join(h for h in highlights if isinstance(h, str)) or item.get("text")
        hits.append(_hit(item["url"], item.get("title"), snippet))
    return _dedupe(hits)


def _parse_parallel(payload: Any) -> list[dict[str, Any]]:
    hits = []
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        excerpts = item.get("excerpts") or []
        hits.append(_hit(item["url"], item.get("title"), "\n".join(e for e in excerpts if isinstance(e, str))))
    return _dedupe(hits)


def _parse_firecrawl(payload: Any) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        rows = data.get("web") or data.get("results") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    hits = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link")
        if not url:
            continue
        hits.append(_hit(url, item.get("title"), item.get("description") or item.get("snippet")))
    return _dedupe(hits)


def _parse_brave(payload: Any) -> list[dict[str, Any]]:
    # Web Search only. Do not fold in news/videos/infobox as extra hit slots.
    web = payload.get("web") if isinstance(payload, dict) else None
    rows = (web or {}).get("results") or []
    hits = []
    for item in rows:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        parts = [item.get("description") or item.get("snippet")]
        hits.append(_hit(item["url"], item.get("title"), "\n".join(p for p in parts if isinstance(p, str))))
    return _dedupe(hits)


def _parse_tavily(payload: Any) -> list[dict[str, Any]]:
    hits = []
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        hits.append(_hit(item["url"], item.get("title"), item.get("content")))
    return _dedupe(hits)


def _parse_serp(payload: Any) -> list[dict[str, Any]]:
    hits = []
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link")
        if not url:
            continue
        hits.append(_hit(url, item.get("title"), item.get("description") or item.get("snippet")))
    return _dedupe(hits)


def _parse_linkup(payload: Any) -> list[dict[str, Any]]:
    hits = []
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        if item.get("type") == "image":
            continue
        hits.append(_hit(item["url"], item.get("name") or item.get("title"), item.get("content") or item.get("snippet")))
    return _dedupe(hits)


def _parse_telnyx(payload: Any) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    results = (data or {}).get("results") or {}
    rows = results.get("web") or []
    hits = []
    for item in rows:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        snippets = item.get("snippets") or []
        snippet = "\n".join(s for s in snippets if isinstance(s, str)) or item.get("description")
        hits.append(_hit(item["url"], item.get("title"), snippet))
    return _dedupe(hits)


def _parse_seltz(payload: Any) -> list[dict[str, Any]]:
    hits = []
    for item in (payload or {}).get("documents") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        hits.append(_hit(item["url"], item.get("title"), item.get("content") or item.get("snippet")))
    return _dedupe(hits)


def _parse_predictleads(payload: Any) -> list[dict[str, Any]]:
    included: dict[tuple[str, str], dict[str, Any]] = {}
    for inc in (payload or {}).get("included") or []:
        if isinstance(inc, dict) and inc.get("id"):
            included[(inc.get("type") or "", inc["id"])] = inc.get("attributes") or {}
    hits = []
    for ev in (payload or {}).get("data") or []:
        attrs = ev.get("attributes") or {}
        rel = ((ev.get("relationships") or {}).get("most_relevant_source") or {}).get("data") or {}
        art = included.get(("news_article", rel.get("id") or ""), {})
        url = art.get("url")
        if not url:
            continue
        snippet = attrs.get("article_sentence") or attrs.get("summary") or art.get("body")
        hits.append(_hit(url, art.get("title") or attrs.get("summary"), snippet))
    return _dedupe(hits)


def call_endpoint(name: str, question: str, case: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if name.startswith("parallel_"):
        mode = name.removeprefix("parallel_")
        if mode not in {"turbo", "basic", "advanced"}:
            raise KeyError(name)
        raw = _http(
            method="POST",
            url="https://api.parallel.ai/v1/search",
            headers={"x-api-key": os.environ["PARALLEL_API_KEY"], "Content-Type": "application/json"},
            body={
                "objective": question,
                "search_queries": [question],
                "mode": mode,
                "advanced_settings": {"max_results": MAX_RESULTS},
            },
            params=None,
            timeout=90 if mode == "advanced" else 60,
        )
        hits = _parse_parallel(raw.get("response")) if raw["ok"] else []
        return hits, raw
    if name.startswith("exa_"):
        exa_type = "instant" if name.endswith("instant") else "deep"
        raw = _http(
            method="POST",
            url="https://api.exa.ai/search",
            headers={"x-api-key": os.environ["EXA_API_KEY"], "Content-Type": "application/json"},
            body={"query": question, "type": exa_type, "numResults": MAX_RESULTS, "contents": {"highlights": True}},
            params=None,
            timeout=120 if exa_type == "deep" else 45,
        )
        hits = _parse_exa(raw.get("response")) if raw["ok"] else []
        return hits, raw
    if name == "firecrawl":
        raw = _http(
            method="POST",
            url="https://api.firecrawl.dev/v2/search",
            headers={"Authorization": f"Bearer {os.environ['FIRECRAWL_API_KEY']}", "Content-Type": "application/json"},
            body={"query": question, "limit": MAX_RESULTS},
            params=None,
            timeout=60,
        )
        hits = _parse_firecrawl(raw.get("response")) if raw["ok"] else []
        return hits, raw
    if name == "predictleads":
        domain = case.get("company_domain") or (case.get("gold") or {}).get("domain")
        recipe = case.get("recipe") or case.get("pattern")
        if not domain:
            raw = {
                "method": "GET",
                "url": "",
                "request": {"headers": {}, "body": None, "params": None},
                "ok": False,
                "error": "PredictLeads needs company_domain",
                "status_code": None,
                "response": None,
                "attempts": [],
                "latency_ms": 0,
                "started_at": _now(),
                "ended_at": _now(),
            }
            return [], raw
        categories = expand_categories(recipe_categories(recipe), recipe)
        try:
            picked, _, _ = pick_categories(_openai(), "gpt-4.1-mini", question, recipe)
            if picked:
                categories = picked
        except Exception:  # noqa: BLE001
            pass
        hits, raw = fetch_category_events(domain, categories)
        return hits, raw
    if name.startswith("seltz_"):
        scope = "news" if name.endswith("news") else "companies"
        raw = _http(
            method="POST",
            url="https://api.seltz.ai/v1/search",
            headers={"x-api-key": os.environ["SELTZ_API_KEY"], "Content-Type": "application/json"},
            body={"query": question, "max_results": MAX_RESULTS, "scope": scope},
            params=None,
            timeout=45,
        )
        hits = _parse_seltz(raw.get("response")) if raw["ok"] else []
        return hits, raw
    if name == "brave":
        # Same contract as the others: one NL query, one request, max 10 web
        # results. Web Search (not LLM Context / Answers / News). No freshness
        # window, goggles, pagination, extra_snippets, or rich callback.
        raw = _http(
            method="GET",
            url="https://api.search.brave.com/res/v1/web/search",
            headers={
                "X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"],
                "Accept": "application/json",
            },
            body=None,
            params={
                "q": question,
                "count": MAX_RESULTS,
                "result_filter": "web",
            },
            timeout=45,
        )
        hits = _parse_brave(raw.get("response")) if raw["ok"] else []
        return hits, raw
    if name.startswith("tavily_"):
        # Same contract: one NL query, one request, max 10 results. Search only
        # (not Extract / Research / include_answer / include_raw_content).
        # No topic=news, time_range, or auto_parameters.
        depth = "ultra-fast" if name.endswith("ultrafast") else "advanced"
        body: dict[str, Any] = {
            "query": question,
            "search_depth": depth,
            "max_results": MAX_RESULTS,
        }
        if depth == "advanced":
            body["chunks_per_source"] = 3
        raw = _http(
            method="POST",
            url="https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}",
                "Content-Type": "application/json",
            },
            body=body,
            params=None,
            timeout=90 if depth == "advanced" else 45,
        )
        hits = _parse_tavily(raw.get("response")) if raw["ok"] else []
        return hits, raw
    if name == "serp":
        # Same RapidAPI Google Search host as self-serve-backend llm/serp_search.py.
        # One NL query, one GET, max 10 organic results. No related_keywords,
        # no page scrape, no BrightData.
        raw = _http(
            method="GET",
            url="https://google-search74.p.rapidapi.com/",
            headers={
                "x-rapidapi-key": os.environ["RAPIDAPI_KEY"],
                "x-rapidapi-host": "google-search74.p.rapidapi.com",
            },
            body=None,
            params={"query": question, "limit": MAX_RESULTS},
            timeout=45,
        )
        hits = _parse_serp(raw.get("response")) if raw["ok"] else []
        return hits, raw
    if name.startswith("linkup_"):
        # Same contract: one NL query, one request, max 10 sources.
        # searchResults only — no sourcedAnswer / structured / deep.
        # fast is index-only; standard is Linkup's single-iteration agent mode.
        depth = "fast" if name.endswith("fast") else "standard"
        raw = _http(
            method="POST",
            url="https://api.linkup.so/v1/search",
            headers={
                "Authorization": f"Bearer {os.environ['LINKUP_API_KEY']}",
                "Content-Type": "application/json",
            },
            body={
                "q": question,
                "depth": depth,
                "outputType": "searchResults",
                "maxResults": MAX_RESULTS,
            },
            params=None,
            timeout=60 if depth == "standard" else 45,
        )
        hits = _parse_linkup(raw.get("response")) if raw["ok"] else []
        return hits, raw
    if name == "telnyx_web_search":
        raw = _http(
            method="POST",
            url="https://api.telnyx.com/v2/web_search",
            headers={
                "Authorization": f"Bearer {os.environ['TELNYX_API_KEY']}",
                "Content-Type": "application/json",
            },
            body={"query": question, "count": MAX_RESULTS},
            params=None,
            timeout=45,
        )
        hits = _parse_telnyx(raw.get("response")) if raw["ok"] else []
        return hits, raw
    raise KeyError(name)


def _write_raw(raw_dir: Path, case_id: str, endpoint: str, raw: dict[str, Any]) -> str:
    dest = raw_dir / case_id / f"{endpoint}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(raw, indent=2) + "\n")
    return str(dest.relative_to(raw_dir.parent))


def run_case(client: Any, model: str, case: dict[str, Any], endpoints: list[str], raw_dir: Path) -> dict[str, Any]:
    question = case["question"]
    gold_url = (case.get("gold") or {}).get("primary_url") or case.get("ground_truth_url") or ""
    cells = case.get("cells") or (case.get("gold") or {}).get("cells") or []
    expected = case.get("expected_answer") or case.get("ground_truth") or ""

    fetched: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        futs = {pool.submit(call_endpoint, name, question, case): name for name in endpoints}
        for fut in as_completed(futs):
            name = futs[fut]
            fetched[name] = fut.result()

    vendors_out: dict[str, Any] = {}
    gold_scores: dict[str, Any] = {}
    cost_events: list[dict[str, Any]] = []

    def _score_one(name: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        hits, raw = fetched[name]
        raw_path = _write_raw(raw_dir, case["id"], name, raw)
        events: list[dict[str, Any]] = []
        billed = raw["ok"] or (raw.get("status_code") not in {401, 402, 403} and raw.get("status_code") is not None)
        events.append(
            {
                "kind": "vendor_search",
                "vendor": name,
                "usd": UNIT_COST_USD.get(name, 0.0) if billed else 0.0,
                "ok": raw["ok"],
                "status_code": raw.get("status_code"),
                "latency_ms": raw.get("latency_ms"),
            }
        )
        extracted = {"answer": "", "cited_url": ""}
        extract_tokens = {"input_tokens": 0, "output_tokens": 0}
        if raw["ok"]:
            extracted, extract_tokens = extract_answer(client, model, question, hits)
        gold, score_tokens = score_gold(client, model, question, expected, cells, extracted["answer"])
        events.append(
            {
                "kind": "openai",
                "vendor": name,
                "usd": round(_openai_usd(extract_tokens) + _openai_usd(score_tokens), 6),
                "input_tokens": extract_tokens["input_tokens"] + score_tokens["input_tokens"],
                "output_tokens": extract_tokens["output_tokens"] + score_tokens["output_tokens"],
            }
        )
        block = {
            "endpoint": name,
            "latency_ms": raw.get("latency_ms"),
            "started_at": raw.get("started_at"),
            "ended_at": raw.get("ended_at"),
            "status_code": raw.get("status_code"),
            "attempts": raw.get("attempts"),
            "error": raw.get("error"),
            "raw_path": raw_path,
            "hits": hits,
            **extracted,
            "gold": gold,
            "source": source_metrics(hits, gold_url),
            "answer_in_excerpt": answer_in_excerpt(hits, cells),
        }
        return name, block, events

    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        for fut in as_completed([pool.submit(_score_one, name) for name in endpoints]):
            name, block, events = fut.result()
            vendors_out[name] = block
            gold_scores[name] = block["gold"]
            cost_events.extend(events)

    return {
        "id": case.get("id"),
        "domain": "company_news",
        "pattern": case.get("recipe") or case.get("pattern"),
        "company": case.get("company"),
        "company_domain": case.get("company_domain"),
        "question": question,
        "expected_answer": expected,
        "ground_truth": expected,
        "ground_truth_url": gold_url,
        "search_queries": [question],
        "vendors": vendors_out,
        "gold_scores": gold_scores,
        "cost_events": cost_events,
    }


def print_case(row: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(f"{row.get('id')}  [company_news/{row.get('pattern')}]")
    print(row["question"])
    for name, payload in row["vendors"].items():
        src = payload.get("source") or {}
        mark = "Y" if (payload.get("gold") or {}).get("correct") else "n"
        if payload.get("error"):
            print(f"  {name}  {payload.get('latency_ms')}ms  ERROR {str(payload['error'])[:140]}")
            continue
        print(
            f"  {name}  {payload.get('latency_ms')}ms  hits={len(payload.get('hits') or [])}  "
            f"gold={mark} r@5={src.get('recall_at_5')}  {(payload.get('answer') or '')[:110]}"
        )


def _dedupe_cost_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any]] = set()
    out: list[dict[str, Any]] = []
    for event in events:
        key = (event.get("kind"), event.get("vendor"))
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out


def _merge_row(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    vendors = dict(existing.get("vendors") or {})
    vendors.update(incoming.get("vendors") or {})
    gold = dict(existing.get("gold_scores") or {})
    gold.update(incoming.get("gold_scores") or {})
    events = list(existing.get("cost_events") or [])
    events.extend(incoming.get("cost_events") or [])
    merged["vendors"] = vendors
    merged["gold_scores"] = gold
    merged["cost_events"] = _dedupe_cost_events(events)
    return merged


def _parse_endpoints(raw: str) -> list[str]:
    if not raw.strip():
        return list(ENDPOINTS)
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in names if name not in ENDPOINTS]
    if unknown:
        raise SystemExit(f"unknown endpoints: {unknown}. choose from {list(ENDPOINTS)}")
    return names


def _required_env(endpoints: list[str]) -> list[str]:
    keys = ["OPENAI_API_KEY"]
    for name in endpoints:
        keys.extend(ENDPOINT_ENV[name])
    return sorted(set(keys))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vendors = [n for n in ENDPOINTS if any(n in (r.get("gold_scores") or {}) for r in rows)]
    by_recipe: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_recipe.setdefault(row.get("pattern") or "?", []).append(row)

    def _block(items: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(items), "vendors": {}}
        for name in vendors:
            correct = [bool(((r.get("gold_scores") or {}).get(name) or {}).get("correct")) for r in items]
            lat = [int(((r.get("vendors") or {}).get(name) or {}).get("latency_ms") or 0) for r in items]
            r5 = [bool(((r.get("vendors") or {}).get(name) or {}).get("source", {}).get("recall_at_5")) for r in items]
            search_usd = 0.0
            llm_usd = 0.0
            for row in items:
                for event in row.get("cost_events") or []:
                    if event.get("vendor") != name:
                        continue
                    if event.get("kind") == "vendor_search":
                        search_usd += float(event.get("usd") or 0)
                    elif event.get("kind") == "openai":
                        llm_usd += float(event.get("usd") or 0)
            n = len(items) or 1
            out["vendors"][name] = {
                "accuracy": round(sum(correct) / len(correct), 3) if correct else None,
                "latency_ms_mean": round(sum(lat) / len(lat), 1) if lat else None,
                "recall_at_5": round(sum(r5) / len(r5), 3) if r5 else None,
                "usd_search": round(search_usd, 4),
                "usd_openai": round(llm_usd, 4),
                "usd": round(search_usd + llm_usd, 4),
                "usd_per_query": round((search_usd + llm_usd) / n, 5),
            }
        return out

    return {
        "overall": _block(rows),
        "by_recipe": {name: _block(items) for name, items in sorted(by_recipe.items())},
    }


def _cost_rollups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    events = [e for row in rows for e in (row.get("cost_events") or [])]
    by_kind: dict[str, dict[str, Any]] = {}
    for event in events:
        key = event["vendor"] if event["kind"] == "vendor_search" else "openai"
        bucket = by_kind.setdefault(key, {"calls": 0, "usd": 0.0})
        if key == "openai":
            bucket.setdefault("input_tokens", 0)
            bucket.setdefault("output_tokens", 0)
            bucket["input_tokens"] += int(event.get("input_tokens") or 0)
            bucket["output_tokens"] += int(event.get("output_tokens") or 0)
        bucket["calls"] += 1
        bucket["usd"] = round(bucket["usd"] + float(event.get("usd") or 0), 6)
    return {
        "unit_cost_usd": UNIT_COST_USD,
        "by_endpoint": by_kind,
        "total_usd": round(sum(v["usd"] for v in by_kind.values()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=SAMPLES)
    parser.add_argument("--model", default=os.getenv("WEBSEARCH_PROBE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--endpoints", default="", help="Comma-separated endpoint names. Default: all.")
    parser.add_argument("--resume", type=Path, default=None, help="Existing official-run directory to resume")
    parser.add_argument(
        "--attach",
        type=Path,
        default=None,
        help="Existing official-run directory; only call requested endpoints missing on each case",
    )
    args = parser.parse_args()
    if args.resume and args.attach:
        raise SystemExit("use --resume or --attach, not both")

    load_environment()
    endpoints = _parse_endpoints(args.endpoints)
    missing = [k for k in _required_env(endpoints) if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"missing env: {missing}")

    cases = json.loads(args.samples.read_text())
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        cases = [c for c in cases if c.get("id") in wanted]
    if args.limit:
        cases = cases[: args.limit]

    attach_missing: dict[str, list[str]] = {}
    if args.attach:
        out_dir = args.attach
        if (out_dir / "run.json").exists():
            rows = json.loads((out_dir / "run.json").read_text()).get("cases") or []
        else:
            rows = []
        if (out_dir / "run.partial.json").exists():
            partial_rows = json.loads((out_dir / "run.partial.json").read_text()).get("cases") or []
            if not rows:
                rows = partial_rows
            else:
                by_partial = {r.get("id"): r for r in partial_rows}
                rows = [_merge_row(row, by_partial[row["id"]]) if row.get("id") in by_partial else row for row in rows]
                seen = {r.get("id") for r in rows}
                rows.extend(r for r in partial_rows if r.get("id") not in seen)
        by_id = {r.get("id"): r for r in rows}
        pending = []
        for case in cases:
            row = by_id.get(case.get("id"))
            have = set((row or {}).get("vendors") or {})
            needed = [name for name in endpoints if name not in have]
            if needed:
                pending.append(case)
                attach_missing[case["id"]] = needed
        cases = pending
        print(f"attach {out_dir} already={len(rows)} missing_cases={len(cases)} endpoints={endpoints}")
    elif args.resume:
        out_dir = args.resume
        rows = json.loads((out_dir / "run.partial.json").read_text()).get("cases") or []
        done = {r.get("id") for r in rows}
        cases = [c for c in cases if c.get("id") not in done]
        print(f"resume {out_dir} already={len(done)} remaining={len(cases)}")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = OUT_ROOT / stamp
        rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_calls"
    raw_dir.mkdir(parents=True, exist_ok=True)
    latest = OUT_ROOT / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_dir.name)

    print(
        f"official=true model={args.model} endpoints={endpoints} "
        f"cases={len(cases)} already={len(rows)} out={out_dir}"
    )

    client = _openai()
    partial = out_dir / "run.partial.json"

    def _write(path: Path, summary: dict[str, Any] | None) -> None:
        path.write_text(
            json.dumps(
                {
                    "created_at": _now(),
                    "protocol": "official_atomic_single_query",
                    "dataset": str(args.samples),
                    "model": args.model,
                    "endpoints": [n for n in ENDPOINTS if any(n in (r.get("gold_scores") or {}) for r in rows)],
                    "endpoint_map": ENDPOINT_MAP,
                    "summary": summary,
                    "cost": _cost_rollups(rows),
                    "cases": rows,
                },
                indent=2,
            )
            + "\n"
        )

    for i, case in enumerate(cases):
        if i:
            time.sleep(0.4)
        needed = attach_missing.get(case["id"], endpoints)
        row = run_case(client, args.model, case, needed, raw_dir)
        if args.attach:
            merged = False
            next_rows = []
            for existing in rows:
                if existing.get("id") == case["id"]:
                    next_rows.append(_merge_row(existing, row))
                    merged = True
                else:
                    next_rows.append(existing)
            if not merged:
                next_rows.append(row)
            rows = next_rows
        else:
            rows.append(row)
        print_case(row)
        sys.stdout.flush()
        if (i + 1) % 2 == 0 or i + 1 == len(cases):
            _write(partial, summarize(rows))

    summary = summarize(rows)
    cost = _cost_rollups(rows)
    cost["leaderboard"] = (summary.get("overall") or {}).get("vendors") or {}
    print("\n" + "=" * 78)
    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    print("COST")
    print(json.dumps(cost, indent=2))
    _write(out_dir / "run.json", summary)
    (out_dir / "cost.json").write_text(json.dumps(cost, indent=2) + "\n")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": _now(),
                "n": len(rows),
                "dataset": str(args.samples),
                "endpoints": [n for n in ENDPOINTS if any(n in (r.get("gold_scores") or {}) for r in rows)],
                "raw_calls_dir": "raw_calls",
                "note": "Official run. Fresh calls. Dataset-building smokes were not reused.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out_dir / 'run.json'}")


if __name__ == "__main__":
    main()
