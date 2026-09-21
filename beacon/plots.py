from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication-grade aesthetic styling
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#ebebeb",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

COLORS = {
    "ag40": "#2E86AB",       # Classic Deep Blue
    "ag60": "#A23B72",       # Raspberry Purple
    "ag80": "#F18F01",       # Warm Amber
    "best": "#C73E1D",       # Brick Crimson
    "calm": "#7F7F7F",       # Neutral Slate Gray
    "random": "#555555",     # Charcoal
}


def load_experiment_results(runs_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    """Loads all finished run results from runs directory."""
    root = Path(runs_dir)
    results = {}
    if not root.exists():
        return results

    for p in sorted(root.glob("*/results.json")) + sorted(root.glob("*/kfold_summary.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            name = data.get("name") or data.get("experiment") or p.parent.name
            results[name] = data
        except Exception:
            pass
    return results


def plot_recall_bars(
    runs: Dict[str, Dict[str, Any]],
    output_path: str | Path,
    ks: Tuple[int, ...] = (1, 5, 10, 50, 100),
    title: str = "BEACON Screening Retrieval Recall Across Antigen Clusters"
) -> Path:
    """
    Plots grouped bar charts comparing Recall@K across ag40, ag60, ag80
    with 95% Confidence Interval error bars and random baseline markers.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cluster_keys = [k for k in ("ag40", "ag60", "ag80") if k in runs]
    if not cluster_keys:
        cluster_keys = list(runs.keys())[:3]

    if not cluster_keys:
        print("[!] No runs found to plot.")
        return out_path

    n_ks = len(ks)
    fig, axes = plt.subplots(1, n_ks, figsize=(3.8 * n_ks, 4.2), sharey=False)
    if n_ks == 1:
        axes = [axes]

    bar_width = 0.55

    for idx, k in enumerate(ks):
        ax = axes[idx]
        x_positions = np.arange(len(cluster_keys))
        means = []
        errors = []
        best_vals = []

        for name in cluster_keys:
            r = runs[name]
            summary = r.get("summary", {}).get("mean", {})
            k_str = str(k)
            k_data = summary.get(k_str, {})

            val = k_data.get("mean", 0.0) * 100.0
            err = k_data.get("ci95", 0.0) * 100.0
            means.append(val)
            errors.append(err)

            # Check if single best fold exists
            best_info = r.get("best_fold", {}).get("test_scores", {}).get("mean", {})
            best_val = best_info.get(k_str if k_str in best_info else k, 0.0) * 100.0
            best_vals.append(best_val)

        # Plot 5-Fold Mean with 95% CI
        bars = ax.bar(
            x_positions,
            means,
            yerr=errors,
            capsize=4.0,
            width=bar_width,
            color=[COLORS.get(name, "#2E86AB") for name in cluster_keys],
            edgecolor="#222222",
            linewidth=0.8,
            label="5-Fold Mean ± 95% CI",
            zorder=3
        )

        # Plot Single Best Fold Marker
        if any(b > 0 for b in best_vals):
            ax.scatter(
                x_positions,
                best_vals,
                color=COLORS["best"],
                s=45,
                marker="D",
                edgecolors="black",
                linewidth=0.8,
                label="Single Best Fold",
                zorder=4
            )

        # Random baseline (k / 256 for CALM benchmark pool)
        random_pct = (float(k) / 256.0) * 100.0
        ax.axhline(random_pct, color=COLORS["random"], linestyle="--", linewidth=1.0, zorder=2)
        ax.text(
            len(cluster_keys) - 0.5, random_pct,
            f"Random ({random_pct:.1f}%)",
            fontsize=7.5, color=COLORS["random"], ha="right", va="bottom"
        )

        # Annotate bar values
        for i, (m, err) in enumerate(zip(means, errors)):
            top = m + err
            ax.text(
                i, top + (top * 0.03 + 0.5),
                f"{m:.2f}%",
                ha="center", va="bottom", fontsize=8, fontweight="medium", color="#111111"
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([name.upper() for name in cluster_keys], fontsize=9)
        ax.set_title(f"Recall@{k}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Antigen Cluster Split", fontsize=9, labelpad=6)
        if idx == 0:
            ax.set_ylabel("Retrieval Recall (%)", fontsize=10, labelpad=6)

        ax.grid(axis="y", zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        max_val = max(max(means) + max(errors), max(best_vals) if best_vals else 0, random_pct)
        ax.set_ylim(0, max_val * 1.25)

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.07)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved Recall Bar Plot to: {out_path}")
    return out_path


def plot_recall_curves(
    runs: Dict[str, Dict[str, Any]],
    output_path: str | Path,
    ks: Tuple[int, ...] = (1, 5, 10, 50, 100),
    title: str = "BEACON Cumulative Recall Curve Across Top-K"
) -> Path:
    """
    Plots continuous cumulative retrieval recall curves over K (log-spaced or discrete)
    comparing different cluster splits.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    cluster_keys = [k for k in ("ag40", "ag60", "ag80") if k in runs]
    if not cluster_keys:
        cluster_keys = list(runs.keys())

    x_vals = np.array(ks)

    for name in cluster_keys:
        r = runs[name]
        summary = r.get("summary", {}).get("mean", {})
        means = []
        errors = []
        for k in ks:
            k_data = summary.get(str(k), {})
            means.append(k_data.get("mean", 0.0) * 100.0)
            errors.append(k_data.get("ci95", 0.0) * 100.0)

        means = np.array(means)
        errors = np.array(errors)
        col = COLORS.get(name, "#2E86AB")

        ax.plot(x_vals, means, marker="o", linewidth=2.0, color=col, label=f"{name.upper()} (Mean)")
        ax.fill_between(x_vals, np.clip(means - errors, 0, 100), np.clip(means + errors, 0, 100), color=col, alpha=0.18)

    # Random baseline curve
    rand_vals = np.array([(float(k) / 256.0) * 100.0 for k in ks])
    ax.plot(x_vals, rand_vals, linestyle="--", color=COLORS["random"], label="Random Baseline", linewidth=1.2)

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Top-K Candidates Retrieved", fontsize=10, labelpad=6)
    ax.set_ylabel("Retrieval Recall (%)", fontsize=10, labelpad=6)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved Recall Curve Plot to: {out_path}")
    return out_path


def plot_calm_vs_aalite_comparison(
    aalite_runs: Dict[str, Dict[str, Any]],
    calm_runs: Dict[str, Dict[str, Any]],
    output_path: str | Path,
    target_k: int = 10,
    title: str = "Model Architecture Comparison: BEACON vs CALM (Recall@10)"
) -> Path:
    """
    Plots direct head-to-head comparison between BEACON and CALM baseline across antigen splits.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    splits = [s for s in ("ag40", "ag60", "ag80") if s in calm_runs or s in aalite_runs]
    if not splits:
        print("[!] Insufficient data for head-to-head comparison.")
        return out_path

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(splits))
    width = 0.35

    calm_means, calm_cis = [], []
    aalite_means, aalite_cis = [], []

    for s in splits:
        c_item = calm_runs.get(s, {}).get("summary", {}).get("mean", {}).get(str(target_k), {})
        calm_means.append(c_item.get("mean", 0.0) * 100.0)
        calm_cis.append(c_item.get("ci95", 0.0) * 100.0)

        a_item = aalite_runs.get(s, {}).get("summary", {}).get("mean", {}).get(str(target_k), {})
        aalite_means.append(a_item.get("mean", 0.0) * 100.0)
        aalite_cis.append(a_item.get("ci95", 0.0) * 100.0)

    # Bars for CALM
    rects1 = ax.bar(
        x - width/2, calm_means, width, yerr=calm_cis, capsize=3.5,
        label="CALM (MLP + Uniform Mean)", color="#9E9E9E", edgecolor="#333333", linewidth=0.8
    )

    # Bars for BEACON
    rects2 = ax.bar(
        x + width/2, aalite_means, width, yerr=aalite_cis, capsize=3.5,
        label="BEACON (Conv1D + Gated Attention)", color="#1F77B4", edgecolor="#111111", linewidth=0.8
    )

    # Annotations
    for rect in rects1:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    for rect in rects2:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_ylabel(f"Recall@{target_k} (%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in splits], fontsize=9.5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved Head-to-Head Comparison to: {out_path}")
    return out_path


def generate_publication_markdown_table(runs: Dict[str, Dict[str, Any]], ks: Tuple[int, ...] = (1, 5, 10, 50, 100)) -> str:
    """Generates standard publication markdown table ready for reports/papers."""
    header = ["Split", "Setting", "Folds"] + [f"Recall@{k} (Mean±CI)" for k in ks] + ["Best Fold Recall@10"]
    sep = ["---"] * len(header)
    rows = [f"| {' | '.join(header)} |", f"| {' | '.join(sep)} |"]

    for name in ("ag40", "ag60", "ag80", "pa_ep40", "pa_ep60", "pa_ep80"):
        if name not in runs:
            continue
        r = runs[name]
        label = r.get("label", name)
        raw_folds = r.get("folds", 5)
        n_folds = len(raw_folds) if isinstance(raw_folds, list) else raw_folds
        summary = r.get("summary", {}).get("mean", {})
        
        cells = [name.upper(), label, f"{n_folds}-Fold"]
        for k in ks:
            kd = summary.get(str(k), {})
            m = kd.get("mean", 0.0) * 100.0
            ci = kd.get("ci95", 0.0) * 100.0
            cells.append(f"{m:5.2f} ± {ci:4.2f}%")

        best_score = r.get("best_fold", {}).get("test_scores", {}).get("mean", {}).get("10", 0.0) * 100.0
        best_fold = r.get("best_fold", {}).get("fold", 0)
        cells.append(f"{best_score:.2f}% (Fold {best_fold})")
        rows.append(f"| {' | '.join(cells)} |")

    return "\n".join(rows) + "\n"


def plot_full_vs_masked_comparison(
    runs: Dict[str, Dict[str, Any]],
    output_path: str | Path,
    ks: Tuple[int, ...] | int = (1, 5, 10),
    target_k: Optional[int] = None,
    title: str = "Bidirectional Retrieval Recall: Full-Length Ab-Ag vs Masked Paratope-Epitope"
) -> Path:
    """
    Plots paired comparison between Full-Length (Ab-Ag) and Masked (Pa-Ep) controls,
    strictly mirroring CALM Figure 3 across 40%, 60%, 80% antigen sequence identity clusters
    with 95% CI error bars and random baseline line (k/256).
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if target_k is not None:
        eval_ks = (target_k,)
    elif isinstance(ks, int):
        eval_ks = (ks,)
    else:
        eval_ks = tuple(ks)

    unmasked_color = "#2f6fb0"  # CALM steel blue
    masked_color = "#e07a2f"    # CALM deep orange

    rows_spec = [
        ("40%", "ag40", "pa_ep40"),
        ("60%", "ag60", "pa_ep60"),
        ("80%", "ag80", "pa_ep80"),
    ]
    if "rs_full" in runs or "rs_masked" in runs:
        rows_spec.append(("RS", "rs_full", "rs_masked"))

    n_ks = len(eval_ks)
    fig, axes = plt.subplots(1, n_ks, figsize=(4.2 * n_ks, 4.0), sharex=True)
    if n_ks == 1:
        axes = [axes]

    width = 0.36

    for col, k in enumerate(eval_ks):
        ax = axes[col]
        labels = []
        drawn = 0
        baseline_val = None
        max_y = 0.0
        annotations = []

        for pos, (label, unmasked_name, masked_name) in enumerate(rows_spec):
            labels.append(label)
            for offset, (name, color) in enumerate(((unmasked_name, unmasked_color), (masked_name, masked_color))):
                payload = runs.get(name)
                if not payload:
                    continue
                summary = payload.get("summary", {}).get("mean", {})
                item = summary.get(str(int(k)))
                if not item:
                    continue

                baseline = (payload.get("random_baseline") or {}).get(str(int(k)))
                if baseline is not None:
                    baseline_val = float(baseline) * 100.0
                else:
                    baseline_val = (float(k) / 256.0) * 100.0

                centre = pos + (offset - 0.5) * width
                val = float(item.get("mean", 0.0)) * 100.0
                err = float(item.get("ci95", 0.0)) * 100.0

                ax.bar(
                    centre, val, width=width, color=color,
                    edgecolor="black", linewidth=0.7,
                    yerr=err, capsize=3.0, zorder=3
                )
                drawn += 1
                top = val + err
                if top > max_y:
                    max_y = top
                annotations.append((centre, top, val))

        if baseline_val is not None:
            ax.axhline(baseline_val, linestyle="--", color="#555555", linewidth=1.0, zorder=2)
            ax.text(
                len(rows_spec) - 0.5, baseline_val,
                f"k/256 = {baseline_val:.2f}%",
                fontsize=7, color="#555555", ha="right", va="bottom"
            )
            if baseline_val > max_y:
                max_y = baseline_val

        if drawn > 0 and max_y > 0:
            ax.set_ylim(bottom=0, top=max_y * 1.22)
            y_offset = max_y * 0.025
            for centre, top, val in annotations:
                ax.text(centre, top + y_offset, f"{val:.2f}%", ha="center", va="bottom", fontsize=6.5, color="#222222")

        ax.set_xticks(range(len(rows_spec)))
        ax.set_xticklabels(labels)
        ax.set_title(f"Recall@{k}", fontsize=11, fontweight="bold")
        ax.set_xlabel("antigen sequence identity threshold", fontsize=9)
        if col == 0:
            ax.set_ylabel("Recall (%)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=unmasked_color, edgecolor="black", label="full-length Ab-Ag"),
        Patch(facecolor=masked_color, edgecolor="black", label="masked paratope-epitope"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved CALM-style Full vs Masked Figure to: {out_path}")
    return out_path
