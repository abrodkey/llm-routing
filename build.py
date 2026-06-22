#!/usr/bin/env python3
"""
build.py — merge AA + Epoch + OpenRouter + LMArena + provider overlay into models.json.

Sources (trust order for cost):
  1. providers.json (manual)  -> direct/committed price + train-policy [TRUTH for cost]
  2. Artificial Analysis API  -> intelligence + per-task sub-indices, speed, TTFT
  3. OpenRouter /models + /endpoints -> hosted price spread, cached-read, capability auto-detect
  4. Epoch AI CSV             -> country, open-weights, license, pub date
  5. LMArena leaderboard      -> human-preference Elo by category (stub; see README)

Also runs a NEW-MODEL RADAR: flags AA models >45 intelligence from tracked creators
that aren't in aliases.json, writing data/radar.json (CI turns this into a GitHub issue).

Run locally:        AA_API_KEY=... python3 build.py
                    (offline: pre-populate data/ caches via curl, script falls back to them)
Outputs:            models.json, data/radar.json
Sanity-checks fail loudly (non-zero exit) so upstream schema changes get noticed.
"""

import csv
import json
import os
import re
import statistics
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
(DATA / "or_endpoints").mkdir(exist_ok=True)

AA_URL        = "https://artificialanalysis.ai/api/v2/data/llms/models"
EPOCH_URL     = "https://epoch.ai/data/all_ai_models.csv"
OR_MODELS_URL = "https://openrouter.ai/api/v1/models"
OR_EP_URL     = "https://openrouter.ai/api/v1/models/{slug}/endpoints"
# LMArena via the official HF leaderboard-dataset (no auth). first-rows returns the top ~100 by rank.
ARENA_BASE    = "https://datasets-server.huggingface.co/first-rows?dataset=lmarena-ai/leaderboard-dataset&config={config}&split=latest"
ARENA_CONFIGS = {"overall_sc": "text_style_control", "overall_raw": "text", "coding": "webdev"}
# Per-category pull via the HF parquet CDN (NOT the rate-limited /rows API).
# `datasets-server/parquet` returns CDN URLs for the per-config parquet files; reading those is
# orders of magnitude faster than paginating /rows and isn't rate-limited.
ARENA_PARQUET_META = "https://datasets-server.huggingface.co/parquet?dataset=lmarena-ai/leaderboard-dataset"
# Per-task bucket mapping to Arena's category slug. Each bucket gets a human-preference signal
# alongside its AA-derived score. Buckets without a direct AA equivalent (legal, healthcare, business,
# software_engineering, expert) use the Arena rating AS the score.
ARENA_CATEGORY_FOR_TASK = {
    # Existing buckets: cross-check signal
    "chat":            "multi_turn",
    "writing_email":   "creative_writing",
    "research":        "hard_prompts_english",
    "data_synthesis":  "instruction_following",
    "coding":          "coding",
    "math":            "math",
    # New vertical buckets: Arena IS the primary score
    "legal":                 "industry_legal_and_government",
    "healthcare":            "industry_medicine_and_healthcare",
    "business":              "industry_business_and_management_and_financial_operations",
    "software_engineering":  "industry_software_and_it_services",
    "expert":                "expert",
    # NOTE: sales_coaching is NOT backed by Arena — it's backed by salesevals (special-cased in dashboard).
}
# Task buckets that have NO AA benchmark — Arena is the source of truth.
ARENA_ONLY_TASKS = {"legal", "healthcare", "business", "software_engineering", "expert"}
# Task buckets backed by salesevals (separate special-case path; $/call cost replaces $/1M in dashboard).
SALESEVALS_ONLY_TASKS = {"sales_coaching"}
# OpenRouter usage ordering (real-world adoption rank) + Vectara hallucination leaderboard (README table).
OR_USAGE_URL  = "https://openrouter.ai/api/v1/models?order=top-weekly"
HALLU_URL     = "https://raw.githubusercontent.com/vectara/hallucination-leaderboard/main/README.md"
# Explicit canonical -> Vectara model key (None where not in the leaderboard; no fuzzy matching).
HALLU_NAMES = {
    "claude-opus-4.8": None,
    "claude-sonnet-4.6-reasoning": "anthropic/claude-sonnet-4-6",
    "claude-sonnet-4.6-fast": "anthropic/claude-sonnet-4-6",
    "claude-haiku-4.5": "anthropic/claude-haiku-4-5-20251001",
    "gpt-5.5": "openai/gpt-5.5", "gpt-5.5-fast": "openai/gpt-5.5",
    "gpt-5.4": "openai/gpt-5.4-2026-03-05",
    "gpt-5.4-mini": "openai/gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano": "openai/gpt-5.4-nano-2026-03-17",
    "gemini-3.5-flash": None,
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "kimi-k2.6": "moonshotai/kimi-k2.6",
    "deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-v4-flash": None, "qwen-3.6-max": None, "glm-5.1": None,
}

ALIASES   = json.loads((ROOT / "aliases.json").read_text())
PROVIDERS = json.loads((ROOT / "providers.json").read_text())

# New-model radar config
TRACKED_CREATORS = {"OpenAI", "Anthropic", "Google", "DeepSeek", "Kimi", "Moonshot", "Alibaba", "Z AI", "Zhipu",
                    "Meta", "xAI", "MiniMax", "Xiaomi", "NVIDIA", "Mistral", "Cohere", "Amazon"}
RADAR_INTEL_THRESHOLD = 45
RADAR_WINDOW_DAYS = 14   # Only flag models released in the last N days — anything older isn't "new"

# Auto-promotion (Tier 1 — "PREVIEW"): see PROMOTION_POLICY.md
# When a new model appears in AA from a known creator, auto-include it in models.json
# with staging:true so it shows up in scatter/table within 24h. Excluded from recommender
# top-3 until a human graduates it to aliases.json.
PREVIEW_INTEL_THRESHOLD = 35   # below this it's not interesting enough to surface

# Creator name (as it appears in AA) -> base of providers.json key. We pick the most-recent
# providers.json entry matching this prefix as the metadata template for the preview model.
CREATOR_TO_PROVIDER_PREFIX = {
    "OpenAI":   "openai-",
    "Anthropic":"anthropic-",
    "Google":   "google-",
    "DeepSeek": "deepseek-",
    "Moonshot": "moonshot-",
    "Kimi":     "moonshot-",
    "Alibaba":  "alibaba-",
    "Z AI":     "zhipu-",
    "Zhipu":    "zhipu-",
    "Meta":     "meta-",
    "xAI":      "xai-",
    "MiniMax":  "minimax-",
    "Xiaomi":   "xiaomi-",
    "NVIDIA":   "nvidia-",
    "Mistral":  "mistral-",
    "Cohere":   "cohere-",
    "Amazon":   "amazon-",
}

# Maintainer veto list — canonicals that should NEVER auto-promote (typos, duplicates, abandoned releases)
_BLOCKLIST_PATH = ROOT / "radar_blocklist.json"
RADAR_BLOCKLIST = set(json.loads(_BLOCKLIST_PATH.read_text())) if _BLOCKLIST_PATH.exists() else set()

# Per-task quality mapping. Each task -> list of (aa_evaluation_key, scale_to_100).
# Index keys are already 0-100 (scale 1); raw benchmarks are 0-1 fractions (scale 100).
# needs_arena flags buckets AA can't measure well (chat/writing) -> LMArena fills these.
TASK_METRICS = {
    "research":       {"metrics": [("gpqa", 100), ("mmlu_pro", 100), ("hle", 100)], "needs_arena": False},
    "data_synthesis": {"metrics": [("gpqa", 100), ("mmlu_pro", 100), ("ifbench", 100)], "needs_arena": False},
    "web_research":   {"metrics": [("tau2", 100), ("terminalbench_hard", 100)], "needs_arena": False},
    "chat":           {"metrics": [("ifbench", 100), ("artificial_analysis_intelligence_index", 1)], "needs_arena": True},
    "coding":         {"metrics": [("artificial_analysis_coding_index", 1), ("livecodebench", 100)], "needs_arena": False},
    "math":           {"metrics": [("artificial_analysis_math_index", 1), ("aime_25", 100)], "needs_arena": False},
    "writing_email":  {"metrics": [("ifbench", 100), ("artificial_analysis_intelligence_index", 1)], "needs_arena": True},
    # New buckets driven by Intelligent Noise feature profiles (Source Relevance Routing, Pre-Approval Classifier, Follow-Up Gates, Conversation Compaction)
    "classification": {"metrics": [("ifbench", 100), ("artificial_analysis_intelligence_index", 1)], "needs_arena": False},
    "summarization":  {"metrics": [("ifbench", 100), ("artificial_analysis_intelligence_index", 1)], "needs_arena": True},
}

# ── Pull helpers (cache-first so local dev works without live SSL) ───────────────

def _get_json(url, dest, headers=None):
    """Fetch JSON to dest; on any failure fall back to an existing cache."""
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read().decode())
        dest.write_text(json.dumps(payload))
        return payload
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        if dest.exists():
            print(f"[fetch] {url} failed ({e}); using cache {dest.name}", file=sys.stderr)
            return json.loads(dest.read_text())
        raise

def fetch_aa():
    key = os.environ.get("AA_API_KEY")
    cached = DATA / "aa.json"
    if not key:
        if cached.exists():
            print("[aa] AA_API_KEY not set — using cached data/aa.json", file=sys.stderr)
            return json.loads(cached.read_text())["data"]
        sys.exit("ERROR: AA_API_KEY env var required (or pre-populate data/aa.json).")
    payload = _get_json(AA_URL, cached, headers={"x-api-key": key})
    return payload["data"]

def fetch_epoch():
    dest = DATA / "epoch.csv"
    try:
        urllib.request.urlretrieve(EPOCH_URL, dest)
    except (urllib.error.URLError, OSError) as e:
        if not dest.exists():
            sys.exit(f"ERROR: Epoch fetch failed and no cache: {e}")
        print(f"[epoch] fetch failed ({e}) — using cache", file=sys.stderr)
    with dest.open(newline="") as f:
        return list(csv.DictReader(f))

def fetch_openrouter_catalog():
    payload = _get_json(OR_MODELS_URL, DATA / "openrouter.json")
    return {m["id"]: m for m in payload.get("data", [])}

def fetch_or_endpoints(slug):
    """Per-provider price spread for one model. Cache-first; resilient to missing models."""
    dest = DATA / "or_endpoints" / (slug.replace("/", "__") + ".json")
    try:
        payload = _get_json(OR_EP_URL.format(slug=slug), dest)
    except Exception as e:
        print(f"[openrouter] endpoints fetch failed for {slug} ({e})", file=sys.stderr)
        return None
    eps = (payload.get("data") or {}).get("endpoints") or []
    prompts, completions, provs, uptimes = [], [], [], []
    for ep in eps:
        pr = ep.get("pricing", {})
        try:
            p = float(pr.get("prompt")) * 1e6
            c = float(pr.get("completion")) * 1e6
        except (TypeError, ValueError):
            continue
        if p <= 0 and c <= 0:
            continue
        up = ep.get("uptime_last_1d")
        up = round(up, 2) if isinstance(up, (int, float)) else None
        prompts.append(p); completions.append(c)
        provs.append({"provider": ep.get("provider_name"), "in": round(p, 4), "out": round(c, 4), "uptime": up})
        if up is not None:
            uptimes.append(up)
    if not prompts:
        return None
    cheapest = min(provs, key=lambda x: (x["in"] * 3 + x["out"]) / 4)
    return {
        "host_count": len(prompts),
        "input_min": round(min(prompts), 4), "input_median": round(statistics.median(prompts), 4), "input_max": round(max(prompts), 4),
        "output_min": round(min(completions), 4), "output_median": round(statistics.median(completions), 4), "output_max": round(max(completions), 4),
        "cheapest_provider": cheapest["provider"], "cheapest_in": cheapest["in"], "cheapest_out": cheapest["out"],
        "cheapest_uptime_1d": cheapest["uptime"], "best_uptime_1d": round(max(uptimes), 2) if uptimes else None,
    }

def fetch_or_usage():
    """OpenRouter top-weekly ordering -> {model_id: usage_rank}. Ordinal adoption signal (no numeric count)."""
    try:
        payload = _get_json(OR_USAGE_URL, DATA / "openrouter_usage.json")
    except Exception as e:
        print(f"[or-usage] fetch failed ({e})", file=sys.stderr)
        return {}
    return {m["id"]: i + 1 for i, m in enumerate(payload.get("data", []))}

def fetch_hallucination():
    """Parse the Vectara hallucination-leaderboard README table -> {model_key: {rates}}."""
    dest = DATA / "vectara.md"
    try:
        with urllib.request.urlopen(urllib.request.Request(HALLU_URL), timeout=40) as r:
            txt = r.read().decode()
        dest.write_text(txt)
    except Exception as e:
        if dest.exists():
            print(f"[hallu] fetch failed ({e}); using cache", file=sys.stderr); txt = dest.read_text()
        else:
            print(f"[hallu] fetch failed ({e})", file=sys.stderr); return {}
    def pct(s):
        try: return float(s.replace("%", "").strip())
        except (ValueError, AttributeError): return None
    out = {}
    for line in txt.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or "%" not in cells[1]:
            continue
        out[cells[0]] = {"hallucination_rate": pct(cells[1]), "factual_consistency": pct(cells[2]), "answer_rate": pct(cells[3])}
    return out

def fetch_arena():
    """Pull LMArena leaderboards (style-controlled overall, raw overall, webdev/coding) keyed by model_name."""
    data = {}
    for field, config in ARENA_CONFIGS.items():
        dest = DATA / ("arena_" + config + ".json")
        try:
            payload = _get_json(ARENA_BASE.format(config=config), dest)
        except Exception as e:
            print(f"[arena] fetch failed for {config} ({e})", file=sys.stderr)
            continue
        for r in payload.get("rows", []):
            row = r.get("row", {})
            name = row.get("model_name")
            if not name:
                continue
            entry = data.setdefault(name, {})
            entry[field] = row.get("rating")
            if field == "overall_sc":
                entry["votes"] = int(row.get("vote_count") or 0)
                entry["rank"] = row.get("rank")
                entry["publish_date"] = row.get("leaderboard_publish_date")
    return data

def fetch_salesevals():
    """Read the committed salesevals snapshot. Manual refresh via scripts/refresh_salesevals.py
    (quarterly + on-demand) — no scrape on every build because salesevals updates monthly at most."""
    snap = DATA / "salesevals_snapshot.json"
    if not snap.exists():
        print("[salesevals] no snapshot at data/salesevals_snapshot.json — sales_coaching task will be empty", file=sys.stderr)
        return None
    try:
        d = json.loads(snap.read_text())
        # Index by model name for fast join
        by_model = {row["model"]: row for row in d.get("ranked", [])}
        return {"meta": {k: d.get(k) for k in ("source", "snapshot_date", "data_date", "total_calls", "total_configs", "total_evaluations", "mean_score")},
                "by_model": by_model,
                "total_configs": len(by_model)}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[salesevals] parse failed ({e}) — task will be empty", file=sys.stderr)
        return None

def fetch_arena_categories():
    """Pull the full 'text' config parquet (single CDN download, NOT rate-limited) and bucket
    models per category for vertical-routing. Returns:
      ({model_name: {category_slug: {rating, rank, votes}}}, {category_slug: total_models})

    Parquet reading needs pyarrow (CI installs it; local dev may need `pip install pyarrow`).
    If pyarrow is missing AND no cache exists, returns empty dicts — vertical buckets show
    the friendly "data populating after next refresh" state."""
    cache = DATA / "arena_text_latest.parquet"
    # Refresh: fetch parquet metadata, pick the 'text/latest' URL, download
    try:
        with urllib.request.urlopen(ARENA_PARQUET_META, timeout=20) as r:
            meta = json.loads(r.read().decode())
        url = next((f["url"] for f in meta.get("parquet_files", []) if f.get("config") == "text" and f.get("split") == "latest"), None)
        if url:
            urllib.request.urlretrieve(url, cache)
    except Exception as e:
        if not cache.exists():
            print(f"[arena-cat] parquet fetch failed and no cache: {e}", file=sys.stderr)
            return {}, {}
        print(f"[arena-cat] parquet fetch failed ({e}) — using cache", file=sys.stderr)
    # Parse
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[arena-cat] pyarrow not installed; skipping per-category data (run `pip install pyarrow` to enable).", file=sys.stderr)
        return {}, {}
    table = pq.read_table(cache)
    rows = table.to_pylist()
    by_model = {}
    cat_sizes = {}
    for row in rows:
        name = row.get("model_name"); cat = row.get("category")
        if not name or not cat or cat == "exclude_ties":
            continue
        rating = row.get("rating")
        rank = row.get("rank")
        by_model.setdefault(name, {})[cat] = {
            "rating": round(rating, 1) if rating is not None else None,
            "rank": int(rank) if rank is not None else None,
            "votes": int(row.get("vote_count") or 0),
        }
        if rank is not None:
            cat_sizes[cat] = max(cat_sizes.get(cat, 0), int(rank))
    return by_model, cat_sizes

# ── Sanity checks ───────────────────────────────────────────────────────────────

def sanity(aa_rows, epoch_rows):
    if len(aa_rows) < 100:
        sys.exit(f"ERROR: AA returned {len(aa_rows)} rows; expected >100. Schema change?")
    if len(epoch_rows) < 1000:
        sys.exit(f"ERROR: Epoch returned {len(epoch_rows)} rows; expected >1000. Schema change?")
    if not {"name", "pricing", "evaluations"}.issubset(aa_rows[0].keys()):
        sys.exit(f"ERROR: AA missing fields. Got: {list(aa_rows[0].keys())}")

# ── Join helpers ──────────────────────────────────────────────────────────────

def find_aa(aa_rows, name):
    return next((r for r in aa_rows if r.get("name") == name), None)

def find_epoch(epoch_rows, needle):
    nl = needle.lower()
    return next((r for r in epoch_rows if nl in r.get("Model", "").lower()), None)

def base_name(name):
    return name.split("(")[0].strip()

def task_scores(ev):
    """Compute per-task 0-100 quality from available AA sub-indices."""
    out = {}
    for task, cfg in TASK_METRICS.items():
        vals, present = [], []
        for key, scale in cfg["metrics"]:
            v = ev.get(key)
            if v is not None:
                vals.append(v * scale)
                present.append(key)
        out[task] = {
            "score": round(sum(vals) / len(vals), 1) if vals else None,
            "metrics_used": present,
            "needs_arena": cfg["needs_arena"],
        }
    return out

def capabilities(or_entry, prov):
    """Vision/function-calling/structured-output auto-detected from OpenRouter;
    fine-tuning + multilingual_tier from manual providers.json."""
    arch = (or_entry or {}).get("architecture", {})
    params = set((or_entry or {}).get("supported_parameters", []))
    mods = arch.get("input_modalities", [])
    detected = or_entry is not None
    return {
        "vision": ("image" in mods) if detected else None,
        "function_calling": ("tools" in params or "tool_choice" in params) if detected else None,
        "structured_output": ("structured_outputs" in params or "response_format" in params) if detected else None,
        "fine_tuning": prov.get("supports_fine_tuning"),
        "multilingual_tier": prov.get("multilingual_tier"),
        "caps_source": "openrouter+manual" if detected else "manual-only (no OpenRouter match)",
    }

def merge_one(alias, aa_rows, epoch_rows, or_catalog, arena_data, or_usage, hallu_data, arena_cat_data, arena_cat_sizes, salesevals):
    aa    = find_aa(aa_rows, alias["aa_name"])
    epoch = find_epoch(epoch_rows, alias["epoch_name"])
    prov  = PROVIDERS.get(alias["provider_key"], {})
    or_entry = or_catalog.get(alias.get("openrouter_slug"))
    arena = arena_data.get(alias["arena_name"], {})

    out = {"canonical": alias["canonical"], "display": alias["display"], "warnings": []}
    if not aa:       out["warnings"].append(f"AA match not found for '{alias['aa_name']}'")
    if not epoch:    out["warnings"].append(f"Epoch match not found for '{alias['epoch_name']}'")
    if not prov:     out["warnings"].append(f"Provider row missing for '{alias['provider_key']}'")
    if not or_entry: out["warnings"].append(f"OpenRouter match not found for '{alias.get('openrouter_slug')}'")

    ev = (aa or {}).get("evaluations", {})
    out["intelligence_index"] = ev.get("artificial_analysis_intelligence_index")
    out["coding_index"]       = ev.get("artificial_analysis_coding_index")
    out["math_index"]         = ev.get("artificial_analysis_math_index")
    out["output_tok_per_sec"] = (aa or {}).get("median_output_tokens_per_second")
    out["ttft_seconds"]       = (aa or {}).get("median_time_to_first_token_seconds")
    out["task_scores"]        = task_scores(ev)

    # Cost: provider-direct = truth; committed (manual); AA blended + OpenRouter spread = reference
    di, do = prov.get("direct_input_per_1m"), prov.get("direct_output_per_1m")
    out["cost"] = {
        "direct_input_per_1m":  di,
        "direct_output_per_1m": do,
        "committed_input_per_1m":  prov.get("committed_input_per_1m"),
        "committed_output_per_1m": prov.get("committed_output_per_1m"),
        "committed_assumptions":   prov.get("committed_assumptions"),
        "aa_blended_per_1m":    (aa or {}).get("pricing", {}).get("price_1m_blended_3_to_1"),
        "prompt_caching_discount_pct": prov.get("prompt_caching_discount_pct"),
        "batch_api_discount_pct":      prov.get("batch_api_discount_pct"),
        "promo_note": prov.get("promo_note"),
    }
    if di is not None and do is not None:
        blended = (3 * di + do) / 4
        out["cost"]["direct_blended_3_to_1"] = round(blended, 4)
        if blended > 0 and out["intelligence_index"]:
            out["intel_per_dollar"] = round(out["intelligence_index"] / blended, 2)

    # OpenRouter: hosted price spread (open models) + cached-read price
    out["openrouter"] = None
    if or_entry:
        spread = fetch_or_endpoints(alias["openrouter_slug"])
        pr = or_entry.get("pricing", {})
        cached_read = None
        try:
            cached_read = round(float(pr.get("input_cache_read")) * 1e6, 4)
        except (TypeError, ValueError):
            pass
        out["openrouter"] = {
            "slug": alias["openrouter_slug"],
            "context_length": or_entry.get("context_length"),
            "max_output": (or_entry.get("top_provider") or {}).get("max_completion_tokens"),
            "cached_read_per_1m": cached_read,
            "spread": spread,
        }

    # Reliability (from OpenRouter endpoint uptime), real-world usage rank, hallucination/factuality
    sp = out["openrouter"]["spread"] if out["openrouter"] else None
    out["reliability"] = {"cheapest_uptime_1d": sp.get("cheapest_uptime_1d"), "best_uptime_1d": sp.get("best_uptime_1d")} if sp else None
    out["usage_rank"] = or_usage.get(alias.get("openrouter_slug"))
    hk = HALLU_NAMES.get(alias["canonical"])
    out["hallucination"] = hallu_data.get(hk) if hk else None

    out["capabilities"] = capabilities(or_entry, prov)
    out["provenance"] = {
        "provider":      prov.get("provider"),
        "country":       (epoch or {}).get("Country (of organization)"),
        "open_weights":  (epoch or {}).get("Open model weights?"),
        "accessibility": (epoch or {}).get("Model accessibility"),
        "release_date":  (aa or {}).get("release_date") or (epoch or {}).get("Publication date"),
    }
    out["policy"] = {
        "train_policy":     prov.get("train_policy"),
        "policy_note":      prov.get("policy_note"),
        "us_hosted_option": prov.get("us_hosted_option"),
        "context_tokens":   prov.get("context_tokens"),
    }
    out["arena"] = None
    if arena:
        sc, raw = arena.get("overall_sc"), arena.get("overall_raw")
        sc_delta = round(raw - sc, 1) if (sc is not None and raw is not None) else None
        out["arena"] = {
            "name": alias["arena_name"],
            "elo_overall_sc": round(sc, 1) if sc is not None else None,
            "elo_overall_raw": round(raw, 1) if raw is not None else None,
            "elo_coding": round(arena["coding"], 1) if arena.get("coding") is not None else None,
            "sc_delta": sc_delta,
            "votes": arena.get("votes"),
            "rank": arena.get("rank"),
            "low_confidence": (arena.get("votes") or 0) < 5000,
            "publish_date": arena.get("publish_date"),
        }

    # Per-category Arena data: rank in each use-case slice (legal, healthcare, business, multi_turn, …).
    # Keyed by Arena category slug. Empty dict if model not in category leaderboards yet.
    cat_data = arena_cat_data.get(alias.get("arena_name") or "", {})
    out["arena_categories"] = {
        cat: {**cat_data[cat], "total_in_cat": arena_cat_sizes.get(cat)}
        for cat in cat_data
    } if cat_data else {}

    # salesevals: sales-call coaching benchmark (manual snapshot). Score 0-100, cost in $/call (NOT $/1M).
    out["salesevals"] = None
    if salesevals and alias.get("salesevals_name"):
        row = salesevals["by_model"].get(alias["salesevals_name"])
        if row:
            out["salesevals"] = {
                "score": row["score"],
                "cost_per_call": row["cost_per_call"],
                "rank": row["rank"],
                "n": row.get("n"),
                "total_in_test": salesevals["total_configs"],
            }

    out["last_verified"] = prov.get("last_verified")
    return out

# ── New-model radar ──────────────────────────────────────────────────────────

def new_model_radar(aa_rows):
    """Flag models >threshold intel from tracked creators that are NEWER than our newest
    tracked model from that same creator (so we surface new drops, not back-catalog)."""
    aliases = ALIASES["models"]
    tracked_bases = {base_name(a["aa_name"]) for a in aliases}
    aa_by_name = {r.get("name"): r for r in aa_rows}

    # Newest release_date we currently track, per creator.
    tracked_newest = {}
    for a in aliases:
        r = aa_by_name.get(a["aa_name"])
        if not r:
            continue
        creator = (r.get("model_creator") or {}).get("name", "")
        d = r.get("release_date")
        if d and (creator not in tracked_newest or d > tracked_newest[creator]):
            tracked_newest[creator] = d

    # Cutoff for "new" — anything released before this is back-catalog noise, not a fresh drop
    from datetime import date, timedelta
    cutoff_str = (date.today() - timedelta(days=RADAR_WINDOW_DAYS)).isoformat()
    candidates = {}
    for r in aa_rows:
        intel = (r.get("evaluations") or {}).get("artificial_analysis_intelligence_index")
        creator = (r.get("model_creator") or {}).get("name", "")
        if intel is None or intel < RADAR_INTEL_THRESHOLD or creator not in TRACKED_CREATORS:
            continue
        b = base_name(r.get("name", ""))
        if b in tracked_bases:
            continue
        rel = r.get("release_date")
        # Skip undated models and anything older than the RADAR_WINDOW_DAYS cutoff
        if not rel or rel < cutoff_str:
            continue
        newest = tracked_newest.get(creator)
        # Only flag if newer than our newest tracked model from this creator (back-catalog = noise).
        if newest and rel and rel <= newest:
            continue
        prev = candidates.get(b)
        if not prev or intel > prev["max_intel"]:
            candidates[b] = {"base": b, "creator": creator, "max_intel": intel,
                             "release_date": rel, "newer_than_our": newest}
    ranked = sorted(candidates.values(), key=lambda x: (x["release_date"] or "", x["max_intel"]), reverse=True)
    (DATA / "radar.json").write_text(json.dumps({"threshold": RADAR_INTEL_THRESHOLD, "candidates": ranked}, indent=2))
    return ranked

def _canonical_from_name(name: str) -> str:
    """Turn an AA model name into a kebab-case canonical id (matches aliases.json convention)."""
    return re.sub(r"[^a-z0-9.-]+", "-", name.lower()).strip("-")

def _pick_template_provider_key(creator: str):
    """Find the most-recent providers.json key for this creator. We use it as the metadata
    template for the auto-promoted preview model (train policy, multilingual tier, etc.).
    Returns None if no template exists — we won't auto-promote in that case."""
    prefix = CREATOR_TO_PROVIDER_PREFIX.get(creator)
    if not prefix:
        return None
    candidates = [k for k in PROVIDERS if not k.startswith("_") and k.startswith(prefix)]
    if not candidates:
        return None
    # Prefer the lexically-greatest key (e.g. "openai-gpt-5.5" beats "openai-gpt-5.4") as a rough
    # proxy for "most-recent generation" — same train policy / fine-tuning support is more likely.
    return sorted(candidates)[-1]

def find_promotion_candidates(aa_rows):
    """Find ALL competitive untracked models from TRACKED_CREATORS (no date window).
    This is separate from new_model_radar() — radar shows truly-recent drops for the
    'N new flagged' notification, promotion backfills any model we should be covering
    regardless of release age. Returns the same shape as new_model_radar()."""
    tracked_bases = {base_name(a["aa_name"]) for a in ALIASES["models"]}
    candidates = {}
    for r in aa_rows:
        intel = (r.get("evaluations") or {}).get("artificial_analysis_intelligence_index")
        creator = (r.get("model_creator") or {}).get("name", "")
        if intel is None or intel < PREVIEW_INTEL_THRESHOLD or creator not in TRACKED_CREATORS:
            continue
        b = base_name(r.get("name", ""))
        if b in tracked_bases:
            continue
        prev = candidates.get(b)
        if not prev or intel > prev["max_intel"]:
            candidates[b] = {"base": b, "creator": creator, "max_intel": intel,
                             "release_date": r.get("release_date"), "newer_than_our": None}
    return sorted(candidates.values(), key=lambda x: (x["release_date"] or "", x["max_intel"]), reverse=True)

def promote_radar_to_preview(radar_candidates, aa_rows, arena_data, or_catalog, or_usage, hallu_data, arena_cat_data, arena_cat_sizes, salesevals, epoch_rows):
    """Auto-promote radar candidates to Tier 1 PREVIEW per PROMOTION_POLICY.md.
    Returns a list of synthetic model entries (each with staging=True) ready to merge into models.json."""
    aa_by_name = {r.get("name"): r for r in aa_rows}
    previews = []
    notes = []
    for c in radar_candidates:
        name = c["base"]
        canonical = _canonical_from_name(name)
        if canonical in RADAR_BLOCKLIST:
            notes.append(f"  • {name}: blocked (in radar_blocklist.json)")
            continue
        if (c.get("max_intel") or 0) < PREVIEW_INTEL_THRESHOLD:
            notes.append(f"  • {name}: intel {c.get('max_intel')} < threshold {PREVIEW_INTEL_THRESHOLD}")
            continue
        template_key = _pick_template_provider_key(c["creator"])
        if not template_key:
            notes.append(f"  • {name}: no providers.json template for creator '{c['creator']}'")
            continue
        # Find the AA row whose name matches this radar base (radar's `base` is base_name()-stripped)
        aa_row = None
        for r in aa_rows:
            if base_name(r.get("name","")) == name:
                aa_row = r
                break
        if not aa_row:
            notes.append(f"  • {name}: no AA row found")
            continue
        # Synthesize a minimal alias dict and let merge_one do the rest.
        alias = {
            "canonical": canonical,
            "display":   name,
            "aa_name":   aa_row["name"],
            "epoch_name": name,                  # likely miss; non-fatal
            "arena_name": canonical,             # best guess; non-fatal
            "openrouter_slug": None,             # no auto-mapping — OR data fills in later when human graduates
            "vectara_name":  None,
            "salesevals_name": None,
            "provider_key":  template_key,
        }
        entry = merge_one(alias, aa_rows, epoch_rows, or_catalog, arena_data, or_usage, hallu_data, arena_cat_data, arena_cat_sizes, salesevals)
        entry["staging"] = True
        entry["staging_provenance"] = {
            "promoted_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "creator": c["creator"],
            "template_provider": template_key,
            "release_date": c.get("release_date"),
            "max_intel": c.get("max_intel"),
            "data_coverage": [src for src,present in [
                ("aa", entry.get("intelligence_index") is not None),
                ("arena", entry.get("arena") is not None),
                ("openrouter", entry.get("openrouter") is not None),
            ] if present],
        }
        previews.append(entry)
        notes.append(f"  ✓ {name}: promoted as PREVIEW (template={template_key}, coverage={entry['staging_provenance']['data_coverage']})")
    # Also write radar_auto.json — machine-generated suggestions for the maintainer review queue.
    (ROOT / "radar_auto.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "promoted": [{"canonical": p["canonical"], "display": p["display"], **p["staging_provenance"]} for p in previews],
        "decisions": notes,
    }, indent=2))
    return previews, notes

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("[1/6] Artificial Analysis…")
    aa_rows = fetch_aa();        print(f"      -> {len(aa_rows)} models")
    print("[2/6] Epoch AI…")
    epoch_rows = fetch_epoch();  print(f"      -> {len(epoch_rows)} rows")
    print("[3/6] OpenRouter catalog + usage…")
    or_catalog = fetch_openrouter_catalog(); or_usage = fetch_or_usage()
    print(f"      -> {len(or_catalog)} models, {len(or_usage)} usage-ranked")
    print("[4/6] LMArena overall + style-controlled + webdev…")
    arena_data = fetch_arena();  print(f"      -> {len(arena_data)} models")
    print("[4b/6] LMArena per-category (vertical task buckets)…")
    arena_cat_data, arena_cat_sizes = fetch_arena_categories()
    print(f"      -> {len(arena_cat_data)} models across {len(arena_cat_sizes)} categories")
    print("[5/6] Vectara hallucination…")
    hallu_data = fetch_hallucination(); print(f"      -> {len(hallu_data)} models")
    print("[5b/6] salesevals.com snapshot…")
    salesevals = fetch_salesevals()
    print(f"      -> {salesevals['total_configs'] if salesevals else 0} configs (snapshot {salesevals['meta']['snapshot_date'] if salesevals else 'missing'})")

    sanity(aa_rows, epoch_rows)

    print("[6/6] Merging + per-provider price spreads…")
    merged = []
    for alias in ALIASES["models"]:
        row = merge_one(alias, aa_rows, epoch_rows, or_catalog, arena_data, or_usage, hallu_data, arena_cat_data, arena_cat_sizes, salesevals)
        merged.append(row)
        for w in row["warnings"]:
            print(f"      ⚠ {alias['canonical']}: {w}", file=sys.stderr)

    radar = new_model_radar(aa_rows)
    if radar:
        print(f"\n🛰  NEW-MODEL RADAR — {len(radar)} candidate(s) >{RADAR_INTEL_THRESHOLD} intel not in aliases.json:", file=sys.stderr)
        for c in radar:
            print(f"      • {c['base']} ({c['creator']}, intel {c['max_intel']}, {c['release_date']})", file=sys.stderr)

    # Tier 1 auto-promotion — see PROMOTION_POLICY.md
    # Uses find_promotion_candidates() (no date window) so we backfill any competitive
    # model from a tracked creator, not just recent drops.
    promotion_candidates = find_promotion_candidates(aa_rows)
    previews, preview_notes = promote_radar_to_preview(promotion_candidates, aa_rows, arena_data, or_catalog, or_usage, hallu_data, arena_cat_data, arena_cat_sizes, salesevals, epoch_rows)
    if preview_notes:
        print(f"\n📋  TIER 1 AUTO-PROMOTION:", file=sys.stderr)
        for n in preview_notes:
            print(n, file=sys.stderr)
    merged.extend(previews)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "aa":        {"url": AA_URL, "rows_pulled": len(aa_rows)},
            "epoch":     {"url": EPOCH_URL, "rows_pulled": len(epoch_rows)},
            "openrouter":{"url": OR_MODELS_URL, "rows_pulled": len(or_catalog), "usage_ranked": len(or_usage)},
            "arena":     {"source": "lmarena-ai/leaderboard-dataset", "rows_pulled": len(arena_data), "configured": len(arena_data) > 0},
            "arena_categories": {"models": len(arena_cat_data), "categories": list(arena_cat_sizes.keys())},
            "salesevals":   {"source": "salesevals.com", "configs": salesevals["total_configs"] if salesevals else 0, "snapshot_date": (salesevals or {}).get("meta", {}).get("snapshot_date"), "data_date": (salesevals or {}).get("meta", {}).get("data_date")},
            "hallucination": {"source": "vectara/hallucination-leaderboard", "rows_pulled": len(hallu_data)},
            "providers": {"file": "providers.json", "rows": len(PROVIDERS) - 1},
        },
        "radar": radar,
        "task_buckets": list(TASK_METRICS.keys()) + sorted(ARENA_ONLY_TASKS) + sorted(SALESEVALS_ONLY_TASKS),
        "arena_category_for_task": ARENA_CATEGORY_FOR_TASK,
        "arena_only_tasks": sorted(ARENA_ONLY_TASKS),
        "salesevals_only_tasks": sorted(SALESEVALS_ONLY_TASKS),
        "feature_profiles": json.loads((ROOT / "feature_profiles.json").read_text()) if (ROOT / "feature_profiles.json").exists() else None,
        "models": merged,
    }
    (ROOT / "models.json").write_text(json.dumps(out, indent=2))
    print(f"\n✓ wrote models.json ({len(merged)} models) + data/radar.json ({len(radar)} candidates)")

if __name__ == "__main__":
    main()
