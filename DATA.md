# Data contract

## Snapshot

`data/latest-websearch.json` is the public **company-news-public-119** snapshot: 129
locked company-news questions and 1,548 endpoint cells (12 public endpoints).
It is the file-backed mirror of the staging dataset loaded for
https://openbenchmarks.com/web-search.

Official run id: `20260816T020806Z`.

## Inputs

`data/company-news/samples.json` is the frozen scored set. Each row has the
question, company identity, gold cells, official URL, quote, and optional
hard negative. `ground_truth.json` is the compact gold projection.

`dropped-flagged-13.json` lists the 13 items removed after labeller flags.
They are not scored and are not in `samples.json`.

## Per-cell files

```
data/websearch-runs/company-news-public-119/<case_slug>/<endpoint>.json      slim
data/websearch-runs/company-news-public-119/<case_slug>/<endpoint>.raw.json  slim + HTTP + judge prompts
data/company-news/official-runs/20260816T020806Z/raw_calls/<case_slug>/<endpoint>.json
```

Slim files carry hits, the terra extract, the Opus accuracy verdict, and the
Opus AR@K verdict. Raw files add the redacted HTTP envelope and the judge
prompts reconstructed from the published scripts (structured outputs are the
stored verdicts; raw model token streams are not retained).

Auth headers are replaced with `***REDACTED***`.

## Public endpoints

- `exa_instant` → `exa-instant` (Exa instant)
- `exa_deep` → `exa-deep` (Exa deep)
- `parallel_basic` → `parallel-basic` (Parallel basic)
- `parallel_advanced` → `parallel-advanced` (Parallel advanced)
- `firecrawl` → `firecrawl-search` (Firecrawl)
- `predictleads` → `predictleads-news` (PredictLeads news events, category-filtered)
- `seltz_news` → `seltz-news` (Seltz news)
- `brave` → `brave-search` (Brave Search)
- `tavily_advanced` → `tavily-advanced` (Tavily advanced)
- `serp` → `serp-rapidapi` (SERP (RapidAPI))
- `linkup_fast` → `linkup-fast` (Linkup fast)
- `linkup_standard` → `linkup-standard` (Linkup standard)
- `telnyx_web_search` → `telnyx-web-search` (Telnyx Web Search)

Excluded from this snapshot: `parallel_turbo`, `tavily_ultrafast`, `seltz_companies`.

## Metrics

- **accuracy** — share of questions whose extract contains every gold cell
- **AR@K** — share of questions with at least one top-K snippet the AR judge
  labeled answer-bearing
- **latency_ms** — vendor HTTP wall time for that request
- **usd_search** — list-price search cost; extract/judge LLM cost is not the
  headline cost column

## Namesake attach (2026-08-17)

Ten domain-disambiguated namesake questions were added to the same official
run (`20260816T020806Z`). Each question names a leftover-famous company plus
its real domain so the leftover ticker / raise / CEO is a hard negative.
Scored n is now 129. Dataset slug stays `company-news-public-119`.

## PredictLeads category overlay (2026-08-19)

PredictLeads cells on this snapshot are the category-filtered harvest
`20260819T163248Z`, persisted as endpoint `predictleads` on the same gold
freeze (`20260816T020806Z`). The call is
`GET /api/v3/companies/{domain}/news_events?limit=10` with `categories[]`
from a closed-list pick plus sibling tags. The unfiltered dump (Acc 82,
AR@1 61) is no longer in this repo. There is no `rejudge-gpt56` file for
this overlay; accuracy and AR@K are Opus on the terra extract.

## Exa 402 retry

The first official call for `news-agent-energy-seed-amount` on `exa_instant`
and `exa_deep` returned HTTP 402 `NO_MORE_CREDITS` with empty hits. The
published cells are the credit-restored retries (HTTP 200). Original 402
envelopes: `data/company-news/official-runs/20260816T020806Z/rerun-exa-402/before/`.
