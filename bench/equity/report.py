"""Generate a Markdown report from the Phase-4 benchmark JSON.

Reads  : ../results/equity-benchmark.json
Writes : ../results/equity-benchmark-report.md

The report answers three questions:
    1. Does the fusion model beat the individual models?
    2. At what training N does fusion's advantage emerge?
    3. Is there a tradeable edge (Sharpe > 1.0)?
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_IN = RESULTS_DIR / "equity-benchmark.json"
DEFAULT_OUT = RESULTS_DIR / "equity-benchmark-report.md"

MODEL_ORDER = ["xgboost", "lewm_standalone", "awm_standalone", "fusion"]
MODEL_LABEL = {
    "xgboost": "XGBoost",
    "lewm_standalone": "Le-WM",
    "awm_standalone": "AWM (rule regime)",
    "fusion": "Fusion (Le-WM + AWM)",
}


def _fmt(v: Any, kind: str = "float") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if kind == "auc":
        return f"{float(v):.4f}"
    if kind == "pct":
        return f"{float(v) * 100:.2f}%"
    if kind == "sharpe":
        return f"{float(v):+.2f}"
    if kind == "int":
        return str(int(v))
    return f"{float(v):.4f}"


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _metrics(run: dict, model: str) -> dict:
    return _get(run, "models", model, "metrics", default={}) or {}


def _backtest(run: dict, model: str) -> dict:
    return _get(run, "models", model, "backtest", default={}) or {}


def summary_table(results: dict) -> str:
    """Model × Threshold grid with AUC + Sharpe side by side."""
    thresholds = [r["threshold"] for r in results["runs"]]

    lines: list[str] = []
    lines.append("### Summary — AUC and Sharpe by Threshold\n")
    header = "| Threshold |" + "".join(
        f" {MODEL_LABEL[m]} AUC | {MODEL_LABEL[m]} Sharpe |" for m in MODEL_ORDER
    )
    sep = "|---|" + "".join(["---|---|" for _ in MODEL_ORDER])
    lines.append(header)
    lines.append(sep)

    for run in results["runs"]:
        th = run["threshold"]
        row = [f"| {th} |"]
        for m in MODEL_ORDER:
            auc = _metrics(run, m).get("auc")
            sh = _backtest(run, m).get("sharpe_ratio")
            row.append(f" {_fmt(auc, 'auc')} | {_fmt(sh, 'sharpe')} |")
        lines.append("".join(row))
    return "\n".join(lines) + "\n"


def detailed_metrics(results: dict) -> str:
    """Second table with Brier / accuracy / trade count / drawdown."""
    lines: list[str] = []
    lines.append("### Detailed Metrics\n")
    lines.append("| Threshold | Model | AUC | Brier | Acc | P@90R | Trades | Tot Ret | Ann Ret | Sharpe | Max DD | Win % |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run in results["runs"]:
        th = run["threshold"]
        for m in MODEL_ORDER:
            met = _metrics(run, m)
            bt = _backtest(run, m)
            lines.append(
                "| {th} | {m} | {auc} | {br} | {acc} | {p90} | {tr} | {tret} | {aret} | {sh} | {dd} | {wr} |".format(
                    th=th,
                    m=MODEL_LABEL[m],
                    auc=_fmt(met.get("auc"), "auc"),
                    br=_fmt(met.get("brier"), "float"),
                    acc=_fmt(met.get("accuracy"), "pct"),
                    p90=_fmt(met.get("p_at_90_recall"), "auc"),
                    tr=_fmt(bt.get("trade_count"), "int"),
                    tret=_fmt(bt.get("total_return"), "pct"),
                    aret=_fmt(bt.get("annualized_return"), "pct"),
                    sh=_fmt(bt.get("sharpe_ratio"), "sharpe"),
                    dd=_fmt(bt.get("max_drawdown"), "pct"),
                    wr=_fmt(bt.get("win_rate"), "pct"),
                )
            )
    return "\n".join(lines) + "\n"


def per_ticker_table(results: dict, model: str) -> str:
    """Per-ticker AUC breakdown for the best model at its best threshold."""
    # pick (model, threshold) with max AUC
    best = None
    for run in results["runs"]:
        auc = _metrics(run, model).get("auc")
        if auc is None or math.isnan(auc):
            continue
        if best is None or auc > best[0]:
            best = (auc, run["threshold"], run)
    if best is None:
        return ""
    _, th, run = best
    per_t: dict = _get(run, "models", model, "per_ticker", default={}) or {}
    rows = [(t, d) for t, d in per_t.items() if isinstance(d, dict)]
    rows.sort(key=lambda x: -(x[1].get("auc") or 0))
    lines: list[str] = []
    lines.append(
        f"### Per-Ticker Performance — {MODEL_LABEL[model]} (threshold={th})\n"
    )
    lines.append("| Ticker | Train N | Test N | AUC | Brier | Acc | Pos Rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for t, d in rows:
        lines.append(
            f"| {t} | {_fmt(d.get('train_n'), 'int')} | {_fmt(d.get('test_n'), 'int')} | "
            f"{_fmt(d.get('auc'), 'auc')} | {_fmt(d.get('brier'), 'float')} | "
            f"{_fmt(d.get('accuracy'), 'pct')} | {_fmt(d.get('pos_rate'), 'pct')} |"
        )
    return "\n".join(lines) + "\n"


def answer_questions(results: dict) -> str:
    """Three key questions + quantitative answers."""
    # Build a quick lookup: (model, threshold) -> (auc, sharpe)
    grid: dict[str, dict[int, dict[str, float]]] = {m: {} for m in MODEL_ORDER}
    for run in results["runs"]:
        th = run["threshold"]
        for m in MODEL_ORDER:
            grid[m][th] = {
                "auc": _metrics(run, m).get("auc") or float("nan"),
                "sharpe": _backtest(run, m).get("sharpe_ratio") or float("nan"),
                "total_return": _backtest(run, m).get("total_return") or float("nan"),
            }

    lines: list[str] = []
    lines.append("### Key Findings\n")

    # Q1: fusion vs. individuals (by AUC, averaged across thresholds)
    def _avg(model: str, key: str) -> float:
        vals = [v[key] for v in grid[model].values() if not math.isnan(v[key])]
        return float(np.mean(vals)) if vals else float("nan")

    fusion_auc = _avg("fusion", "auc")
    fusion_sharpe = _avg("fusion", "sharpe")
    best_indiv_auc_model = max(
        ["xgboost", "lewm_standalone", "awm_standalone"],
        key=lambda m: _avg(m, "auc") if not math.isnan(_avg(m, "auc")) else -1,
    )
    best_indiv_auc = _avg(best_indiv_auc_model, "auc")
    best_indiv_sharpe_model = max(
        ["xgboost", "lewm_standalone", "awm_standalone"],
        key=lambda m: _avg(m, "sharpe") if not math.isnan(_avg(m, "sharpe")) else -99,
    )
    best_indiv_sharpe = _avg(best_indiv_sharpe_model, "sharpe")

    q1 = (
        f"**Q1 — Does fusion beat the individuals?**  "
        f"Fusion avg AUC = {fusion_auc:.4f}, Sharpe = {fusion_sharpe:+.2f}. "
        f"Best individual by AUC: **{MODEL_LABEL[best_indiv_auc_model]}** "
        f"({best_indiv_auc:.4f}). "
        f"Best individual by Sharpe: **{MODEL_LABEL[best_indiv_sharpe_model]}** "
        f"({best_indiv_sharpe:+.2f}).  "
    )
    if not math.isnan(fusion_auc) and not math.isnan(best_indiv_auc):
        delta = fusion_auc - best_indiv_auc
        verdict = "✅ Fusion wins on AUC." if delta > 0.005 else (
            "❌ Fusion loses on AUC." if delta < -0.005 else "⚖️ Tied on AUC."
        )
        q1 += f"Δ(AUC) = {delta:+.4f} → {verdict}"
    lines.append(q1 + "\n")

    # Q2: threshold where fusion first beats both individuals
    thresholds_sorted = sorted(grid["fusion"].keys())
    first_win: int | None = None
    for th in thresholds_sorted:
        f_auc = grid["fusion"][th]["auc"]
        best_other = max(
            grid[m][th]["auc"] for m in ["xgboost", "lewm_standalone", "awm_standalone"]
            if not math.isnan(grid[m][th]["auc"])
        )
        if not math.isnan(f_auc) and f_auc > best_other + 0.005:
            first_win = th
            break
    q2 = (
        f"**Q2 — At what training N does fusion's advantage emerge?**  "
        + (
            f"Fusion first exceeds best individual AUC at **threshold = {first_win}**."
            if first_win is not None
            else "Fusion does not clearly beat the best individual at any threshold tested."
        )
    )
    lines.append(q2 + "\n")

    # Q3: tradeable edge (Sharpe > 1.0)
    candidates: list[tuple[str, int, float, float]] = []
    for m in MODEL_ORDER:
        for th, v in grid[m].items():
            if not math.isnan(v["sharpe"]) and v["sharpe"] > 1.0:
                candidates.append((m, th, v["sharpe"], v["total_return"]))
    candidates.sort(key=lambda c: -c[2])
    if candidates:
        top = candidates[0]
        lines.append(
            f"**Q3 — Is there a tradeable edge (Sharpe > 1.0)?**  "
            f"✅ Yes. Best: **{MODEL_LABEL[top[0]]}** at threshold={top[1]}, "
            f"Sharpe={top[2]:+.2f}, total return={top[3] * 100:.2f}%. "
            f"{len(candidates)} (model, threshold) combinations cleared the bar.\n"
        )
    else:
        lines.append(
            "**Q3 — Is there a tradeable edge (Sharpe > 1.0)?**  "
            "❌ No model × threshold combination produces Sharpe > 1.0 after costs.\n"
        )

    return "\n".join(lines) + "\n"


def recommendations(results: dict) -> str:
    lines = ["### Recommendations\n"]
    # Highest AUC run
    best = None
    for run in results["runs"]:
        for m in MODEL_ORDER:
            auc = _metrics(run, m).get("auc")
            if auc is None or math.isnan(auc):
                continue
            if best is None or auc > best[0]:
                best = (auc, m, run["threshold"])
    if best is not None:
        lines.append(
            f"- **Highest AUC overall**: {MODEL_LABEL[best[1]]} at threshold={best[2]} "
            f"(AUC={best[0]:.4f}). Use this combination if calibration / ranking is the "
            f"primary goal.\n"
        )

    # Highest Sharpe run
    best_s = None
    for run in results["runs"]:
        for m in MODEL_ORDER:
            sh = _backtest(run, m).get("sharpe_ratio")
            if sh is None or math.isnan(sh):
                continue
            if best_s is None or sh > best_s[0]:
                best_s = (sh, m, run["threshold"])
    if best_s is not None:
        lines.append(
            f"- **Highest Sharpe**: {MODEL_LABEL[best_s[1]]} at threshold={best_s[2]} "
            f"(Sharpe={best_s[0]:+.2f}). Use this for live trading allocation.\n"
        )

    lines.append(
        "- Validate the fusion result on held-out 2025-2026 data before sizing positions.\n"
    )
    lines.append(
        "- Re-run with `cost_bps=25` to stress-test edge resilience under realistic "
        "transaction costs.\n"
    )
    return "\n".join(lines) + "\n"


def generate(results: dict) -> str:
    meta = results.get("meta", {})
    lines: list[str] = []
    lines.append("# Phase 4 — Equity Benchmark Report\n")
    lines.append(
        f"- **Universe**: {len(meta.get('tickers', []))} tickers "
        f"({', '.join(meta.get('tickers', [])[:8])}{'…' if len(meta.get('tickers', [])) > 8 else ''})\n"
    )
    lines.append(f"- **Thresholds swept**: {meta.get('thresholds')}\n")
    lines.append(f"- **Fusion clusters**: {meta.get('fusion_clusters')}\n")
    bt_cfg = meta.get("backtest_cfg", {})
    lines.append(
        f"- **Backtest**: entry>{bt_cfg.get('entry_threshold', 0.55)}, "
        f"cost={bt_cfg.get('cost_bps', 10)} bps, hold={bt_cfg.get('hold_days', 5)}d\n"
    )
    lines.append("\n")

    lines.append(summary_table(results))
    lines.append("\n")
    lines.append(answer_questions(results))
    lines.append(recommendations(results))
    lines.append("\n")
    lines.append(detailed_metrics(results))
    lines.append("\n")

    # Best model per-ticker table (by AUC)
    best_model = None
    best_auc = -1.0
    for run in results["runs"]:
        for m in MODEL_ORDER:
            auc = _metrics(run, m).get("auc")
            if auc is None or math.isnan(auc):
                continue
            if auc > best_auc:
                best_auc, best_model = auc, m
    if best_model:
        lines.append(per_ticker_table(results, best_model))

    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Phase-4 benchmark report")
    ap.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    with open(args.in_path) as f:
        results = json.load(f)

    md = generate(results)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"[report] wrote {out_path} ({len(md)} chars)", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
