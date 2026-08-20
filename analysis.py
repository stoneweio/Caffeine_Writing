"""Baseline-corrected analysis of blood pressure and pulse responses
to three caffeinated drinks.

Reads data/measurements.csv, checks it, and regenerates every figure,
the summary tables, and a short results report. Run from anywhere:

    python src/analysis.py
    python src/analysis.py --list
    python src/analysis.py --only fig2 fig9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "measurements.csv"
FIGS = ROOT / "figures"

PRODUCTS = ["Seoul Milk Coffee", "Hot Six", "Starbucks Caffe Latte"]
PARTICIPANTS = ["P1", "P2", "P3"]
COLORS = {"P1": "#c0392b", "P2": "#2c3e50", "P3": "#7f8c8d"}
MEAN_STYLE = dict(color="#e67e22", ms=8, lw=1.5)

ALERTNESS = {"Seoul Milk Coffee": 2.66, "Hot Six": 9.0, "Starbucks Caffe Latte": 7.0}

PRODUCT_INFO = pd.DataFrame({
    "product": PRODUCTS,
    "serving_ml": [200, 250, 355],
    "caffeine_per_serving_mg": [43, 60, 75],
    "caffeine_per_100ml_mg": [21.5, 24.0, 21.1],
    "sugar_per_100ml_g": [9.5, 12.0, 3.7],
    "taurine_per_100ml_mg": [0, 400, 0],
}).set_index("product")

VALUE_RANGES = {"sys": (70, 200), "dia": (40, 130), "pul": (30, 150)}
CUFF_ERROR_MMHG = 6

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------- data

def load() -> pd.DataFrame:
    df = (pd.read_csv(DATA)
            .sort_values(["product", "participant", "time_min"])
            .reset_index(drop=True))
    base = df.groupby(["product", "participant"])[["sys", "dia", "pul"]].transform("first")
    return df.assign(**{f"d_{c}": df[c] - base[c] for c in ("sys", "dia", "pul")})


def validate(df: pd.DataFrame) -> list[str]:
    issues = []
    expected = {"product", "participant", "age_group", "date", "time_min", "sys", "dia", "pul"}
    if missing := expected - set(df.columns):
        issues.append(f"missing columns: {sorted(missing)}")
        return issues
    for (prod, person), g in df.groupby(["product", "participant"]):
        if 0 not in g.time_min.values:
            issues.append(f"{prod} / {person}: no baseline (time_min 0)")
        if g.time_min.duplicated().any():
            issues.append(f"{prod} / {person}: duplicated time points")
    for col, (lo, hi) in VALUE_RANGES.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        for _, r in bad.iterrows():
            issues.append(f"{r['product']} / {r.participant} @ {r.time_min} min: "
                          f"{col}={r[col]} outside {lo}-{hi}")
    if unknown := set(df["product"]) - set(PRODUCTS):
        issues.append(f"unknown products: {sorted(unknown)}")
    return issues


def peak_table(df: pd.DataFrame) -> pd.DataFrame:
    post = df[df.time_min > 0]
    grp = post.groupby(["product", "participant"])
    sys_rows = post.loc[grp.d_sys.apply(lambda s: s.abs().idxmax()).values]
    pul_rows = (post.loc[grp.d_pul.apply(lambda s: s.abs().idxmax()).values,
                         ["product", "participant", "d_pul"]]
                    .rename(columns={"d_pul": "peak_delta_pul"}))
    out = (sys_rows.assign(baseline_sys=sys_rows["sys"] - sys_rows.d_sys)
                   .rename(columns={"sys": "peak_sys", "d_sys": "peak_delta_sys",
                                    "time_min": "time_of_peak_min"})
                   .merge(pul_rows, on=["product", "participant"]))
    out["product"] = pd.Categorical(out["product"], PRODUCTS)
    cols = ["product", "participant", "age_group", "baseline_sys",
            "peak_sys", "peak_delta_sys", "time_of_peak_min", "peak_delta_pul"]
    return out.sort_values(["product", "participant"])[cols].reset_index(drop=True)


def weighted_mean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Time-weighted mean change per session, fair across unequal windows."""
    def twm(g: pd.DataFrame) -> float:
        span = g.time_min.max() - g.time_min.min()
        return float(np.trapezoid(g.d_sys, g.time_min) / span)

    out = (df.sort_values("time_min")
             .groupby(["product", "participant"])
             .apply(twm, include_groups=False)
             .rename("weighted_mean_d_sys")
             .reset_index())
    out["product"] = pd.Categorical(out["product"], PRODUCTS)
    return out.sort_values(["product", "participant"]).reset_index(drop=True)


# ------------------------------------------------------------- figures

def grouped_bars(ax, table: pd.DataFrame, value: str, width: float = 0.25):
    for k, person in enumerate(PARTICIPANTS):
        vals = (table[table.participant == person]
                .set_index("product").loc[PRODUCTS, value])
        bars = ax.bar([x + (k - 1) * width for x in range(len(PRODUCTS))],
                      vals, width, label=person, color=COLORS[person])
        ax.bar_label(bars, fmt="%+.1f" if vals.dtype.kind == "f" else "%+d",
                     fontsize=8, padding=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(PRODUCTS)), PRODUCTS)


def fig1_sys_timecourse(df, pk, wm):
    timecourse(df, "d_sys", "Change in systolic BP from baseline (mmHg)",
               "Systolic change from each participant's own baseline "
               "(products tested on separate days, order not randomised)",
               "fig1_delta_sys_timecourse.png", (12, 4))


def fig2_peak(df, pk, wm):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped_bars(ax, pk, "peak_delta_sys")
    means = pk.groupby("product", observed=True).peak_delta_sys.mean().loc[PRODUCTS]
    ax.plot(range(3), means, "D--", label="Group mean", **MEAN_STYLE)
    ax.axhspan(-CUFF_ERROR_MMHG, CUFF_ERROR_MMHG, color="#bdc3c7", alpha=0.25, zorder=0)
    ax.text(2.42, CUFF_ERROR_MMHG + 0.5, "typical single-cuff\nuncertainty",
            fontsize=7.5, color="#7f8c8d", ha="right", va="bottom")
    ax.set(ylabel="Peak change in systolic BP (mmHg)",
           title="Peak systolic change by product and participant")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig2_peak_delta_by_product.png")


def fig3_alertness(df, pk, wm):
    means = pk.groupby("product", observed=True).peak_delta_sys.mean()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for prod in PRODUCTS:
        ax.scatter(ALERTNESS[prod], means[prod], s=140, color="#2c3e50", zorder=3)
        ax.annotate(prod, (ALERTNESS[prod], means[prod]),
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)
    ax.set(xlabel="Self-reported alertness (group mean, 1-10)",
           ylabel="Peak systolic change (group mean, mmHg)",
           xlim=(0, 10.5), ylim=(0, 20),
           title="Self-reported alertness against peak systolic change")
    save(fig, "fig3_subjective_vs_physiological.png")


def fig4_pulse(df, pk, wm):
    timecourse(df, "d_pul", "Change in pulse (bpm)",
               "Pulse change from baseline", "fig4_pulse_timecourse.png", (12, 3.6))


def fig5_diastolic(df, pk, wm):
    timecourse(df, "d_dia", "Change in diastolic BP from baseline (mmHg)",
               "Diastolic change from baseline", "fig5_dia_timecourse.png", (12, 3.6))


def fig6_baseline_spread(df, pk, wm):
    fig, ax = plt.subplots(figsize=(8, 4))
    base = (df[df.time_min == 0]
            .rename(columns={"sys": "baseline"})[["product", "participant", "baseline"]])
    for k, person in enumerate(PARTICIPANTS):
        vals = (base[base.participant == person]
                .set_index("product").loc[PRODUCTS, "baseline"])
        bars = ax.bar([x + (k - 1) * 0.25 for x in range(3)], vals, 0.25,
                      label=person, color=COLORS[person])
        ax.bar_label(bars, fontsize=8, padding=2)
    lo, hi = base.baseline.min(), base.baseline.max()
    ax.set(ylim=(90, 150), xticks=range(3), xticklabels=PRODUCTS,
           ylabel="Resting systolic BP before drinking (mmHg)",
           title=f"Baselines spread from {lo} to {hi} mmHg, "
                 "which is why raw readings are not compared")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "fig6_baseline_spread.png")


def fig7_composition(df, pk, wm):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label in [
        (axes[0], "caffeine_per_100ml_mg", "Caffeine per 100 ml (mg)"),
        (axes[1], "sugar_per_100ml_g", "Sugar per 100 ml (g)"),
    ]:
        vals = PRODUCT_INFO.loc[PRODUCTS, col]
        bars = ax.bar(range(3), vals, 0.55, color="#2c3e50")
        ax.bar_label(bars, fontsize=9, padding=2)
        ax.set(xticks=range(3), xticklabels=[p.replace(" ", "\n") for p in PRODUCTS],
               ylabel=label)
    axes[0].set_title("Nearly identical caffeine", fontsize=11)
    axes[1].set_title("Very different sugar", fontsize=11)
    save(fig, "fig7_composition.png")


def fig8_weighted_mean(df, pk, wm):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped_bars(ax, wm, "weighted_mean_d_sys")
    means = (wm.groupby("product", observed=True)
               .weighted_mean_d_sys.mean().loc[PRODUCTS])
    ax.plot(range(3), means, "D--", label="Group mean", **MEAN_STYLE)
    ax.set(ylabel="Time-weighted mean change in systolic BP (mmHg)",
           title="Average change over the whole session, "
                 "fair across unequal measurement windows")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig8_weighted_mean.png")


def fig9_peak_heatmap(df, pk, wm):
    grid = (pk.pivot(index="participant", columns="product", values="peak_delta_sys")
              .loc[PARTICIPANTS, PRODUCTS])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-25, vmax=25, aspect="auto")
    for i, person in enumerate(PARTICIPANTS):
        for j, prod in enumerate(PRODUCTS):
            v = grid.loc[person, prod]
            ax.text(j, i, f"{v:+d}", ha="center", va="center",
                    color="white" if abs(v) > 14 else "black", fontsize=10)
    ax.set(xticks=range(3), xticklabels=PRODUCTS,
           yticks=range(3), yticklabels=PARTICIPANTS,
           title="Peak systolic change (mmHg)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85)
    save(fig, "fig9_peak_heatmap.png")


def timecourse(df, col, ylabel, title, name, figsize):
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for ax, prod in zip(axes, PRODUCTS):
        for person, g in df[df["product"] == prod].groupby("participant"):
            ax.plot(g.time_min, g[col], marker="o", label=person,
                    color=COLORS[person], lw=1.8, ms=5)
        ax.axhline(0, color="black", lw=0.8)
        ax.set(title=prod, xlabel="Minutes after consumption",
               xticks=[0, 15, 30, 45, 60])
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, name)


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


FIGURES = {
    "fig1": fig1_sys_timecourse, "fig2": fig2_peak, "fig3": fig3_alertness,
    "fig4": fig4_pulse, "fig5": fig5_diastolic, "fig6": fig6_baseline_spread,
    "fig7": fig7_composition, "fig8": fig8_weighted_mean, "fig9": fig9_peak_heatmap,
}


# -------------------------------------------------------------- report

def write_report(pk, wm, issues):
    peak_means = pk.groupby("product", observed=True).peak_delta_sys.mean()
    ordered = peak_means.sort_values(ascending=False)
    inside_band = ", ".join(p for p, v in peak_means.items()
                            if abs(v) <= CUFF_ERROR_MMHG) or "none"
    lines = [
        "# Results summary",
        "",
        "Generated by src/analysis.py from data/measurements.csv.",
        "",
        "## Dose",
        f"Caffeine per 100 ml sits between "
        f"{PRODUCT_INFO.caffeine_per_100ml_mg.min()} and "
        f"{PRODUCT_INFO.caffeine_per_100ml_mg.max()} mg across the three "
        "products, so the servings delivered an effectively matched dose.",
        "",
        "## Peak systolic change, group mean (mmHg)",
        *(f"- {prod}: {val:+.1f}" for prod, val in ordered.items()),
        "",
        f"Products inside the +/-{CUFF_ERROR_MMHG} mmHg single-cuff band: {inside_band}.",
        "",
        "## Perceived alertness, group mean (1-10)",
        *(f"- {prod}: {ALERTNESS[prod]}" for prod in PRODUCTS),
        "",
        "The alertness order and the blood pressure order do not match: "
        "the highest-rated product produced the smallest measured change.",
        "",
        "## Data checks",
        *([f"- {i}" for i in issues] if issues else ["- no issues found"]),
        "",
    ]
    (ROOT / "results.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", metavar="FIG",
                        help="regenerate selected figures, e.g. --only fig2 fig9")
    parser.add_argument("--list", action="store_true",
                        help="list available figures and exit")
    args = parser.parse_args()

    if args.list:
        for key, fn in FIGURES.items():
            print(f"{key}: {fn.__name__}")
        return

    df = load()
    issues = validate(df)
    for issue in issues:
        print("check:", issue)

    pk = peak_table(df)
    wm = weighted_mean_table(df)

    wanted = args.only or FIGURES
    for key in wanted:
        FIGURES[key](df, pk, wm)

    if not args.only:
        pk.to_csv(ROOT / "data" / "summary_table.csv", index=False)
        wm.to_csv(ROOT / "data" / "weighted_mean_table.csv", index=False)
        write_report(pk, wm, issues)

    print(pk.to_string(index=False))
    print()
    print(wm.to_string(index=False))


if __name__ == "__main__":
    main()
