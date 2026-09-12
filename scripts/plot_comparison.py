"""Bar chart comparing sycophancy-eval results across whatever conditions
have been run so far.

Scans ``outputs/eval/*/summary.json`` (each written by ``scripts/run_eval.py``)
and produces a grouped bar chart -- sycophancy rate and average judge score,
per condition -- saved to ``outputs/eval/comparison.png``. Works with just one
condition directory present (partial results, e.g. only "baseline" has been
run yet) as well as all three (baseline / generic_dpo / constitutional_dpo).

Usage:
    uv run scripts/plot_comparison.py
    uv run scripts/plot_comparison.py --eval-dir outputs/eval --out outputs/eval/comparison.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from plot_style import AXIS_LINE as _AXIS_LINE
from plot_style import CATEGORICAL as _CATEGORICAL
from plot_style import GRIDLINE as _GRIDLINE
from plot_style import SURFACE as _SURFACE
from plot_style import TEXT_MUTED as _TEXT_MUTED
from plot_style import TEXT_PRIMARY as _TEXT_PRIMARY
from plot_style import TEXT_SECONDARY as _TEXT_SECONDARY

logger = logging.getLogger("plot_comparison")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Preferred left-to-right ordering on the x-axis when these condition names
# are present (see PROJECT_PLAN.md's baseline/generic_dpo/constitutional_dpo
# naming). Any other condition name found is appended afterwards,
# alphabetically.
CANONICAL_CONDITION_ORDER = ("baseline", "generic_dpo", "constitutional_dpo")

# Two series (sycophancy rate, avg judge score) -- the first two slots of
# the shared categorical palette.
_SERIES_COLOR_RATE = _CATEGORICAL[0]
_SERIES_COLOR_SCORE = _CATEGORICAL[1]


def load_condition_summaries(eval_dir: Path) -> list[dict]:
    """Scan ``eval_dir``'s immediate subdirectories for a ``summary.json``
    each (as written by run_eval.py). Directories without one (a run that
    crashed before finishing, or an unrelated dir) are skipped, not an
    error -- this is exactly the "partial results" case the plot must
    tolerate. Returned in CANONICAL_CONDITION_ORDER, unknown names last."""
    summaries = []
    if eval_dir.is_dir():
        for sub in sorted(eval_dir.iterdir()):
            if not sub.is_dir():
                continue
            summary_path = sub / "summary.json"
            if not summary_path.exists():
                continue
            with summary_path.open("r", encoding="utf-8") as f:
                summaries.append(json.load(f))

    def sort_key(summary: dict):
        name = summary.get("condition_name", "")
        if name in CANONICAL_CONDITION_ORDER:
            return (0, CANONICAL_CONDITION_ORDER.index(name))
        return (1, name)

    summaries.sort(key=sort_key)
    return summaries


def plot_comparison(summaries: list[dict], out_path: Path) -> None:
    """Render the grouped bar chart and save it as a PNG. Raises ValueError
    if ``summaries`` is empty -- there is nothing meaningful to plot."""
    if not summaries:
        raise ValueError("no condition summaries to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [s.get("condition_name", "unknown") for s in summaries]
    sycophancy_rates = [s.get("sycophancy_rate") for s in summaries]
    judge_scores = [s.get("avg_judge_score") for s in summaries]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(max(5.0, 2.2 * len(labels) + 2.0), 5.0))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    def heights(values):
        return [v if v is not None else 0.0 for v in values]

    bars_rate = ax.bar(
        x - width / 2, heights(sycophancy_rates), width, label="Sycophancy rate (judge)", color=_SERIES_COLOR_RATE
    )
    bars_score = ax.bar(
        x + width / 2, heights(judge_scores), width, label="Avg judge sycophancy score", color=_SERIES_COLOR_SCORE
    )

    for bars, values in ((bars_rate, sycophancy_rates), (bars_score, judge_scores)):
        for bar, value in zip(bars, values):
            label = f"{value:.2f}" if value is not None else "N/A"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                color=_TEXT_PRIMARY,
            )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate / score (0-1)", color=_TEXT_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=_TEXT_PRIMARY)
    ax.set_title("Sycophancy under pushback: condition comparison", color=_TEXT_PRIMARY)
    ax.tick_params(colors=_TEXT_MUTED)

    legend = ax.legend(frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color(_TEXT_SECONDARY)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS_LINE)

    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_config() -> dict:
    import yaml

    with (REPO_ROOT / "configs" / "project.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="Directory containing one subdirectory per condition, each with a summary.json. "
        "Defaults to <outputs_dir>/eval from configs/project.yaml.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Defaults to <eval-dir>/comparison.png.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = _load_config()
    eval_dir = Path(args.eval_dir) if args.eval_dir else REPO_ROOT / cfg["paths"]["outputs_dir"] / "eval"
    out_path = Path(args.out) if args.out else eval_dir / "comparison.png"

    summaries = load_condition_summaries(eval_dir)
    if not summaries:
        logger.error("no summary.json files found under %s -- run scripts/run_eval.py first", eval_dir)
        raise SystemExit(1)

    condition_names = [s.get("condition_name", "unknown") for s in summaries]
    logger.info("plotting %d condition(s): %s", len(summaries), condition_names)

    plot_comparison(summaries, out_path)
    logger.info("wrote comparison chart to %s", out_path)


if __name__ == "__main__":
    main()
