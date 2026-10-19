"""
Pure analysis logic (no web framework). Everything the dashboard shows is computed
here from the Dataset, so nothing is hardcoded: change the data and the
"BioNexus leads on X of Y metrics" headline, winners and excels/lags lists update.
"""
from __future__ import annotations

import math
from typing import Any

from .data_source import Dataset
from .registry import (
    CATEGORIES, FOCUS_MODEL_ID, METRICS, METRIC_BY_ID, MetricDef, PAM50_CLASSES,
    MINORITY_CLASSES, RADAR_AXES,
)


class ComparisonError(ValueError):
    """Bad user input (unknown model id / category)."""


# ----------------------------------------------------------------- helpers
def _num(v: Any) -> float | None:
    if v is None:
        return None
    return 1.0 if v is True else 0.0 if v is False else float(v)


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def select_models(ds: Dataset, ids: list[str] | None, focus: str = FOCUS_MODEL_ID) -> list:
    """Filter models; the focus model is always kept so 'leads on' stays meaningful."""
    if focus not in ds.by_id:
        raise ComparisonError(f"focus model '{focus}' not found in results")
    if not ids:
        return list(ds.models)
    unknown = [i for i in ids if i not in ds.by_id]
    if unknown:
        raise ComparisonError(f"unknown model id(s): {unknown}")
    keep = set(ids) | {focus}
    return [m for m in ds.models if m.id in keep]


def _rel_gap(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-9)


# ----------------------------------------------------------------- ranking
def rank_metric(models: list, m: MetricDef) -> dict:
    """Competition ranking (ties share a rank). Models without a value are listed last, unranked."""
    measured = [(x.id, x.metrics.get(m.id)) for x in models if x.metrics.get(m.id) is not None]
    missing = [x.id for x in models if x.metrics.get(m.id) is None]

    measured.sort(key=lambda t: _num(t[1]), reverse=(m.direction == "higher"))
    entries, prev_val, prev_rank = [], None, 0
    for pos, (mid, val) in enumerate(measured, start=1):
        v = _num(val)
        rank = prev_rank if prev_val is not None and _same(v, prev_val) else pos
        entries.append({"model_id": mid, "value": val, "display": m.fmt(val), "rank": rank, "is_best": rank == 1})
        prev_val, prev_rank = v, rank
    entries += [{"model_id": mid, "value": None, "display": "n/a", "rank": None, "is_best": False} for mid in missing]

    distinct = {round(_num(e["value"]), 9) for e in entries if e["value"] is not None}
    return {
        "metric_id": m.id, "label": m.label, "category": m.category, "direction": m.direction,
        "unit": m.unit, "kind": m.kind, "description": m.description,
        # A metric where every model ties (or <2 have data) says nothing about who is better.
        "differentiating": len(measured) >= 2 and len(distinct) > 1,
        "winners": [e["model_id"] for e in entries if e["is_best"]],
        "entries": entries,
    }


def leaderboard(models: list, category: str | None = None) -> list[dict]:
    if category is not None and category not in {c["id"] for c in CATEGORIES}:
        raise ComparisonError(f"unknown category '{category}'")
    return [rank_metric(models, m) for m in METRICS if category in (None, m.category)]


def _focus_entry(board: dict, focus: str) -> dict | None:
    return next((e for e in board["entries"] if e["model_id"] == focus and e["rank"] is not None), None)


# ----------------------------------------------------------------- summary
def summarize(boards: list[dict], models: list, focus: str = FOCUS_MODEL_ID) -> dict:
    name = next(m.name for m in models if m.id == focus)
    considered = [b for b in boards if b["differentiating"] and _focus_entry(b, focus)]
    leads = [b for b in considered if _focus_entry(b, focus)["rank"] == 1]

    by_cat = []
    for c in CATEGORIES:
        cb = [b for b in considered if b["category"] == c["id"]]
        cl = [b for b in cb if _focus_entry(b, focus)["rank"] == 1]
        if cb:
            by_cat.append({"category": c["id"], "label": c["label"], "leads": len(cl), "total": len(cb),
                           "status": "leads" if len(cl) == len(cb) else "mixed" if cl else "trails"})

    ranks = [_focus_entry(b, focus)["rank"] for b in considered]
    return {
        "focus_model_id": focus, "focus_model_name": name, "compared_against": len(models) - 1,
        "metrics_considered": len(considered), "metrics_led": len(leads),
        "lead_ratio": round(len(leads) / len(considered), 3) if considered else None,
        "mean_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
        "by_category": by_cat,
        "headline": f"{name} leads on {len(leads)} of {len(considered)} comparable metrics "
                    f"across {len(models) - 1} alternative techniques.",
    }


# ----------------------------------------------------------------- insights
def insights(boards: list[dict], models: list, focus: str = FOCUS_MODEL_ID) -> dict:
    """Where the focus model excels vs where others win, with plain-English reasons."""
    names = {m.id: m.name for m in models}
    excels, lags = [], []

    for b in boards:
        fe = _focus_entry(b, focus)
        if not b["differentiating"] or fe is None:
            continue
        m = METRIC_BY_ID[b["metric_id"]]
        others = [e for e in b["entries"] if e["model_id"] != focus and e["rank"] is not None]
        if not others:
            continue
        best_other = min(others, key=lambda e: e["rank"])

        if fe["rank"] == 1:
            # strictly better than the best competitor?
            margin = None if m.kind == "boolean" else round(_rel_gap(_num(fe["value"]), _num(best_other["value"])) * 100, 1)
            if m.kind == "boolean":
                solo = all(_num(e["value"]) != _num(fe["value"]) for e in others)
                detail = (f"Only {names[focus]} offers this." if solo
                          else f"Shared with {sum(1 for e in others if e['rank'] == 1)} other model(s).")
            else:
                detail = f"{fe['display']} vs {best_other['display']} for the next best ({names[best_other['model_id']]})."
            excels.append({"metric_id": m.id, "label": m.label, "category": m.category,
                           "value": fe["value"], "display": fe["display"],
                           "runner_up": names[best_other["model_id"]], "margin_pct": margin,
                           "detail": detail, "reason": m.excels_reason})
        else:
            best_val = [e for e in b["entries"] if e["rank"] == 1]
            lags.append({"metric_id": m.id, "label": m.label, "category": m.category,
                         "value": fe["value"], "display": fe["display"], "rank": fe["rank"],
                         "of": len(others) + 1,
                         "best_models": [names[e["model_id"]] for e in best_val],
                         "best_display": best_val[0]["display"],
                         "gap_pct": None if m.kind == "boolean" else
                         round(_rel_gap(_num(fe["value"]), _num(best_val[0]["value"])) * 100, 1),
                         "detail": f"{fe['display']} (rank {fe['rank']} of {len(others) + 1}); "
                                   f"best is {best_val[0]['display']} ({', '.join(names[e['model_id']] for e in best_val)}).",
                         "reason": m.lags_reason})

    excels.sort(key=lambda x: -(x["margin_pct"] if x["margin_pct"] is not None else 1e9))
    lags.sort(key=lambda x: -(x["gap_pct"] if x["gap_pct"] is not None else 1e9))
    return {"excels": excels, "lags": lags}


# ----------------------------------------------------------------- radar
def _normalise(values: dict[str, float], lower_better: bool, log: bool) -> dict[str, float]:
    if log:
        values = {k: math.log10(v) for k, v in values.items() if v > 0}
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if _same(lo, hi):
        return {k: 100.0 for k in values}
    out = {k: (v - lo) / (hi - lo) * 100 for k, v in values.items()}
    return {k: 100 - s for k, s in out.items()} if lower_better else out


def radar(models: list) -> dict:
    """0-100 score per model per axis (100 = best in the selected set; inverted for lower-is-better)."""
    scores: dict[str, dict[str, list[float]]] = {m.id: {a["id"]: [] for a in RADAR_AXES} for m in models}
    for axis in RADAR_AXES:
        for metric_id in axis["metrics"]:
            mdef = METRIC_BY_ID[metric_id]
            vals = {m.id: _num(m.metrics.get(metric_id)) for m in models}
            norm = _normalise({k: v for k, v in vals.items() if v is not None},
                              mdef.direction == "lower", axis["log"])
            for mid, s in norm.items():
                scores[mid][axis["id"]].append(s)
    series = [{"model_id": m.id, "name": m.name,
               "values": {a["id"]: (round(sum(scores[m.id][a["id"]]) / len(scores[m.id][a["id"]]), 1)
                                    if scores[m.id][a["id"]] else None) for a in RADAR_AXES}}
              for m in models]
    return {"axes": [{"id": a["id"], "label": a["label"]} for a in RADAR_AXES], "series": series}


# ----------------------------------------------------------------- per-class
def per_class(models: list, focus: str = FOCUS_MODEL_ID) -> dict:
    rows = [{"model_id": m.id, "name": m.name, "f1": {c: m.per_class_f1.get(c) for c in PAM50_CLASSES}} for m in models]
    classes = []
    for c in PAM50_CLASSES:
        vals = {r["model_id"]: r["f1"][c] for r in rows if r["f1"][c] is not None}
        if not vals:
            continue
        best_id = max(vals, key=vals.get)
        others = [v for k, v in vals.items() if k != focus]
        classes.append({
            "class": c, "is_minority": c in MINORITY_CLASSES, "best_model_id": best_id,
            "focus_f1": vals.get(focus),
            "focus_advantage_pp": round((vals[focus] - max(others)) * 100, 1) if focus in vals and others else None,
            "focus_vs_mean_others_pp": round((vals[focus] - sum(others) / len(others)) * 100, 1)
            if focus in vals and others else None,
        })
    return {"classes": [c["class"] for c in classes], "rows": rows, "summary": classes}


# ----------------------------------------------------------------- bundle
def build_dashboard(ds: Dataset, model_ids: list[str] | None = None, focus: str = FOCUS_MODEL_ID) -> dict:
    models = select_models(ds, model_ids, focus)
    boards = leaderboard(models)
    return {
        "meta": {"data_status": ds.data_status, "notes": ds.notes, "focus_model_id": focus,
                 "models": [{"id": m.id, "name": m.name, "family": m.family,
                             "metrics": m.metrics, "per_class_f1": m.per_class_f1} for m in models]},
        "summary": summarize(boards, models, focus),
        "leaderboard": boards,
        "insights": insights(boards, models, focus),
        "radar": radar(models),
        "per_class": per_class(models, focus),
    }
