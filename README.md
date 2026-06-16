# LLM Cost-vs-Capability Map

Routing-decision support for **Intelligent Noise** and the broader Desai 2026 cohort. Maps the leading closed-lab and open-weight LLMs on **cost vs. capability** so teams can answer: *for this task, what's the cheapest model that's still good enough?*

This is not a leaderboard — it's a routing planner. The dashboard renders a single `models.json` produced by joining five sources.

---

## Architecture

```
Artificial Analysis  ──(live API)──►  intelligence + per-task sub-indices, speed, TTFT
Epoch AI             ──(monthly CSV)─►  country, open-weights, license, pub date
OpenRouter           ──(live API)────►  hosted price spread (min/med/max), cached-read,
                                          context, capability auto-detect (vision/tools/struct)
LMArena (HF dataset) ──(live API)────►  human-preference Elo (style-controlled), sc_delta, webdev coding
providers.json       ──(manual)──────►  direct + committed price, train/no-train, FT, multilingual
                                  │
                                  ▼
        build.py  (join on aliases.json)  ──►  models.json  ──►  index.html (dashboard)
                                  │
                                  └──►  data/radar.json  ──►  GitHub issue (new-model radar)
```

**Trust order for cost:** `providers.json` direct > OpenRouter spread > AA blended. AA's prices include markups; provider pages are the source of truth. OpenRouter shows what you'd actually pay across hosts (esp. for open-weight models).

**Trust order for capability:** AA for benchmarks (overall + per-task); LMArena for human preference. They sometimes disagree — that disagreement is a signal, not a problem.

---

## What each field comes from

| Field | Source |
|---|---|
| `intelligence_index`, `coding_index`, `output_tok_per_sec`, `ttft_seconds` | AA |
| `task_scores` (per-bucket 0–100) | AA sub-indices (gpqa, mmlu_pro, tau2, livecodebench, ifbench…) |
| `cost.direct_*` / `cost.committed_*` | providers.json (manual — truth) |
| `cost.aa_blended_per_1m` | AA (reference) |
| `openrouter.spread` (min/median/max + cheapest host), `cached_read_per_1m`, `context_length` | OpenRouter `/endpoints` |
| `capabilities.vision / function_calling / structured_output` | **auto-detected** from OpenRouter |
| `capabilities.fine_tuning / multilingual_tier` | providers.json (manual) |
| `provenance.country / open_weights / accessibility` | Epoch |
| `policy.train_policy / us_hosted_option` | providers.json (manual) |
| `arena.elo_overall_sc / elo_overall_raw / sc_delta / elo_coding / votes / rank` | LMArena (live) |

### Per-task quality (`task_scores`)

`build.py` maps Ryan's task buckets to AA sub-indices (`TASK_METRICS`), **weighted by RouterArena's Bloom-tier difficulty mix**. Harder benchmarks count more for buckets dominated by Analyze/Evaluate/Create-tier queries (research, coding, math); easier benchmarks count more for Remember/Understand-tier buckets (chat, writing). Weights live in `TASK_METRICS`.

| Bucket | AA metrics (weights) | Note |
|---|---|---|
| research | gpqa (1.5), mmlu_pro (1.0), hle (2.0) | knowledge/reasoning; hle dominates |
| data_synthesis | gpqa (1.0), mmlu_pro (1.0), ifbench (0.5) | |
| web_research | tau2 (1.5), terminalbench_hard (1.5) | **agentic / tool-use** |
| coding | coding_index (1.0), livecodebench (1.5) | contest-style weighted up |
| math | math_index (1.0), aime_25 (2.0) | AIME dominates; sparse for newest models |
| chat | ifbench (1.5), intelligence (0.5) | `needs_arena: true` — instruction-follow weighted up |
| writing_email | ifbench (1.5), intelligence (0.5) | `needs_arena: true` |

Buckets flagged `needs_arena` (chat, writing) use AA proxies for the score; LMArena human-preference rank is shown alongside them (cross-check, not blended).

### Cache-hit pricing (`effective()` cost)

Three sources in priority order: AA's `price_1m_cache_hit_tokens` (published rate, most authoritative when present) → OpenRouter endpoint `cached_read_per_1m` (real per-host) → manual `prompt_caching_discount_pct` in `providers.json` (fallback only). Open-weight Chinese models (DeepSeek, Kimi, Qwen, GLM) have no batch API — the batch slider correctly has no effect on them.

---

## Layout

```
.
├── aliases.json          # canonical slug -> source-specific names incl. openrouter_slug (hand-curated)
├── providers.json        # direct/committed price, train policy, FT, multilingual (manual)
├── build.py              # the merge script + new-model radar
├── models.json           # OUTPUT (committed; dashboard reads it)
├── index.html            # single-file dashboard (no build step)
├── data/                 # raw pulled artifacts (cached; gitignored) + radar.json (committed)
├── .github/workflows/
│   └── refresh-data.yml  # monthly cron → re-pull → commit models.json → radar issue
└── README.md
```

---

## Refresh model

| Source | Cadence | Mechanism |
|---|---|---|
| AA API | Daily (CI) | `AA_API_KEY` in GH secrets |
| Epoch CSV | Daily (CI) | Public URL — no auth |
| OpenRouter | Daily (CI) | Public API — no auth |
| LMArena | Daily (CI) | Public HF datasets-server — no auth |
| Provider price/policy | Quarterly + on launches | Human edits `providers.json` |
| `aliases.json` | On new model | Human; **new-model radar** opens a GitHub issue when AA shows an untracked model |

The GitHub Action runs **`cron: "0 6 * * *"`** (every day at 06:00 UTC), regenerates `models.json`, commits only if anything changed, and files a radar issue for new models. Failing pulls turn the Action red; locally, every fetch falls back to a cached file in `data/`. The daily cadence means the dashboard's "new model(s) flagged" banner and all pricing/score data are at most 24 hours behind reality.

### Setup secrets

```
GitHub repo → Settings → Secrets → Actions:
  AA_API_KEY = <Artificial Analysis API key>
  # (Epoch, OpenRouter, LMArena are public — no secrets needed)
```

### Run locally

```bash
export AA_API_KEY=...
python3 build.py            # writes models.json + data/radar.json
python3 -m http.server 8000 # then open http://localhost:8000
```

If your local Python lacks SSL certs, pre-populate `data/` with `curl` — `build.py` falls back to caches automatically.

---

## New-model radar

Each refresh, `build.py` scans AA's full catalog for models **above intelligence 45** from a tracked creator (OpenAI/Anthropic/Google/DeepSeek/Moonshot/Alibaba/Zhipu) that are **newer than our newest tracked model from that creator** and not in `aliases.json`. Hits are written to `data/radar.json` and the CI files a deduplicated GitHub issue. Tune the threshold via `RADAR_INTEL_THRESHOLD`.

---

## LMArena (live)

LMArena owns the **human-preference** axis the others lack — directly relevant to the `chat` / `writing_email` buckets that have no direct AA benchmark. Wired against the official **`lmarena-ai/leaderboard-dataset`** via the HuggingFace datasets-server (`/first-rows`, no auth):

- Configs pulled: **`text_style_control`** (overall style-controlled), **`text`** (raw, for `sc_delta`), **`webdev`** (coding cross-check).
- Per model: `elo_overall_sc`, `elo_overall_raw`, `elo_coding`, `sc_delta` (raw − SC, the "vibes vs. substance" signal), `votes`, `rank`, `low_confidence` (<5,000 votes).
- **Cross-check, don't blend:** Arena is shown alongside AA, never merged into the AA task score. Findings flag where AA's coding index and Arena's webdev Elo disagree.
- **Coverage:** `/first-rows` returns the top ~100 by rank, so models below rank 100 (currently Haiku 4.5, GPT-5.4 nano) come back `null` — handled gracefully. For full coverage, page the `/rows` endpoint.
- Names use Arena's inconsistent format (Anthropic dashes `claude-opus-4-8-thinking`, OpenAI/Google dots `gpt-5.5-high`, `gemini-3.5-flash`) — see `arena_name` in `aliases.json`, verified 2026-06-15.

---

## Adding a new model

1. `aliases.json` — add a row with canonical slug + names in each source (incl. `openrouter_slug`).
2. `providers.json` — add direct pricing + train policy + FT/multilingual + `last_verified`.
3. `python3 build.py` locally — confirm no warnings.
4. Commit. CI picks it up next refresh (or trigger manually via Actions tab).

---

## Project context

- **Owner:** Aaron (Desai Accelerator intern) · **Stakeholder:** Ryan Lynn (Intelligent Noise)
- **OKRs:** Product Development · Unit Economics
- **Scope (2026-06-09):** API/per-token *embedded* spend only. Enterprise/committed-use rates included for comparison. Excludes consumer subscriptions + internal tooling. No volume forecasting — *"how little intelligence can we get away with per task?"*
- **Thesis (data-validated):** GPT-5.4 ≈ frontier on most tasks at ~50% cost; Factory Router shows 96–99% of Opus 4.7 quality at 20–25% lower cost; Vercel production data shows teams already split premium-vs-cheap by task.
- **Compliance gotcha:** every best-value model is China-origin. Mitigation: run open weights via US hosts (Bedrock / Vertex / Together / Fireworks). Dashboard exposes a US-only / no-train toggle.
- **Confidentiality:** dashboard is public-data-only. NDA'd enterprise rates + sales/marketing-specific findings stay in private Notion / go to Ryan directly.
