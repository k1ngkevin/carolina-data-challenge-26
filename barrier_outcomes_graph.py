"""Compare median outcomes at matched NC schools by ODIS barrier group.

Run ``python barrier_outcomes_graph.py`` to refresh the presentation figure.
The notebook calls ``make_comparison_chart(analysis_df)`` directly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTCOMES = [
    ("graduation", "Four-year graduation"),
    ("english_ii", "English II proficiency"),
    ("math_1", "NC Math 1 proficiency"),
]
GROUPS = [
    ("Other matched schools", False, "#315C79"),
    ("High-barrier schools", True, "#C96B39"),
]


def reported_bounds(schools: pd.DataFrame, prefix: str) -> tuple[pd.Series, pd.Series]:
    """Use DPI's interval labels without imputing an exact masked rate."""
    exact = pd.to_numeric(schools[f"{prefix}_pct"], errors="coerce")
    labels = schools[f"{prefix}_display"].astype("string")
    lower, upper = exact.copy(), exact.copy()
    above_95 = labels.eq(">95%").fillna(False)
    below_5 = labels.eq("<5%").fillna(False)
    lower.loc[above_95], upper.loc[above_95] = 95, 100
    lower.loc[below_5], upper.loc[below_5] = 0, 5
    return lower, upper


def summarize_outcomes(data: pd.DataFrame) -> pd.DataFrame:
    """Return equally weighted school medians for the two matched groups."""
    matched = data.loc[data["agency_code"].notna()].copy()
    if not matched["agency_code"].is_unique:
        raise ValueError("Expected one row per matched school")
    if "high_barrier" not in matched:
        raise ValueError("Missing high_barrier column")

    rows = []
    for prefix, label in OUTCOMES:
        for group_name, is_high_barrier, _ in GROUPS:
            schools = matched.loc[matched["high_barrier"].eq(is_high_barrier)]
            lower, upper = reported_bounds(schools, prefix)
            rows.append({
                "outcome": label,
                "group": group_name,
                "schools_in_group": len(schools),
                "schools_reporting": int(lower.notna().sum()),
                "median_lower": float(lower.median()),
                "median_upper": float(upper.median()),
            })
    summary = pd.DataFrame(rows)
    if summary["median_lower"].isna().any():
        raise ValueError("An outcome has no reported rates")
    if not np.allclose(summary["median_lower"], summary["median_upper"]):
        raise ValueError("A median is only bounded; show its interval before plotting")
    return summary


def _rate_label(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def make_comparison_chart(
    data: pd.DataFrame,
    output_dir: Path = Path("figures"),
) -> tuple[pd.DataFrame, plt.Figure]:
    """Save a chart and its underlying medians; return both for the notebook."""
    summary = summarize_outcomes(data)
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y_positions = np.arange(len(OUTCOMES))

    for group_index, (group_name, _, color) in enumerate(GROUPS):
        group_rows = (summary.loc[summary["group"].eq(group_name)]
                      .set_index("outcome")
                      .loc[[label for _, label in OUTCOMES]])
        y = y_positions + (-0.17 if group_index == 0 else 0.17)
        values = group_rows["median_lower"].to_numpy(dtype=float)
        bars = ax.barh(y, values, height=0.29, color=color, label=group_name, zorder=3)
        for bar, value, count in zip(bars, values, group_rows["schools_reporting"]):
            ax.text(
                value - 1.2, bar.get_y() + bar.get_height() / 2,
                f"{_rate_label(value)}%",
                ha="right", va="center", color="white", fontsize=10.5,
                fontweight="bold", zorder=4,
            )

    other = summary.loc[summary["group"].eq(GROUPS[0][0])].set_index("outcome")
    high = summary.loc[summary["group"].eq(GROUPS[1][0])].set_index("outcome")
    for y, (_, label) in zip(y_positions, OUTCOMES):
        gap = high.loc[label, "median_lower"] - other.loc[label, "median_lower"]
        gap_label = ("+" if gap >= 0 else "−") + _rate_label(abs(gap)) + " %"
        ax.text(102, y, gap_label, ha="left", va="center", fontsize=11,
                color="#273C4A", fontweight="bold")

    ax.set_yticks(y_positions, [label for _, label in OUTCOMES], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 119)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Median school-level rate (%)", fontsize=11, labelpad=10)
    ax.grid(axis="x", color="#DEE6EB", linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#B8C6CE")
    ax.tick_params(axis="y", length=0, pad=12)
    ax.tick_params(axis="x", colors="#425969")
    ax.legend(ncol=2, frameon=False, loc="lower left",
              bbox_to_anchor=(0, 1.02), fontsize=11)
    fig.suptitle("High-barrier schools have lower typical outcomes",
                 x=0.04, y=0.99, ha="left", fontsize=20, fontweight="bold",
                 color="#152D3B")
    fig.text(0.04, 0.89,
             "Matched non-charter NC schools · NC DPI 2024–25 · High barrier: ODIS percentile ≥80",
             fontsize=11, color="#4D6573")
    fig.text(0.04, 0.055,
             "Gap = high-barrier minus other matched schools, in percentage points. Each school has equal weight.\n"
             "Masked >95% and <5% results enter as bounds; all six medians are exact. Provisional ODIS–DPI matches.",
             fontsize=9.5, color="#4D6573", linespacing=1.5)
    fig.subplots_adjust(left=0.25, right=0.96, top=0.75, bottom=0.24)

    output_dir.mkdir(exist_ok=True)
    summary.to_csv(output_dir / "barrier_outcomes_comparison.csv", index=False)
    fig.savefig(output_dir / "barrier_outcomes_comparison.png", dpi=220,
                bbox_inches="tight")
    fig.savefig(output_dir / "barrier_outcomes_comparison.pdf", bbox_inches="tight")
    return summary, fig


if __name__ == "__main__":
    schools_df = pd.read_csv(Path(__file__).with_name("map_data.csv"))
    summary_df, figure = make_comparison_chart(
        schools_df, Path(__file__).with_name("figures")
    )
    print(summary_df.to_string(index=False))
    plt.close(figure)
