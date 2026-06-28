"""Saving, plotting, and sensitivity helpers for simulation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


def _get_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install the packages in requirements.txt."
        ) from exc
    return plt


def save_probability_tables(
    stage_probabilities: pd.DataFrame, output_dir: str | Path
) -> dict[str, Path]:
    """Save the complete stage table and champion-only table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_path = output_dir / "stage_probabilities.csv"
    winner_path = output_dir / "winner_probabilities.csv"
    stage_probabilities.to_csv(stage_path)
    (
        stage_probabilities[["Champion"]]
        .rename(columns={"Champion": "win_probability"})
        .to_csv(winner_path)
    )
    return {"stages": stage_path, "winners": winner_path}


def plot_probability_bars(
    probabilities: pd.Series,
    title: str,
    output_path: str | Path,
    top_n: int | None = None,
) -> tuple[Any, Any]:
    """Create and save a sorted horizontal probability bar chart."""
    plt = _get_pyplot()
    values = probabilities.sort_values(ascending=True)
    if top_n is not None:
        values = values.tail(top_n)
    height = max(5.0, 0.30 * len(values))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(values.index, values.values, color="#3266a8")
    ax.set_xlabel("Probability")
    ax.set_title(title)
    ax.set_xlim(0, max(float(values.max()) * 1.12, 0.01))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig, ax


def plot_stage_heatmap(
    stage_probabilities: pd.DataFrame,
    output_path: str | Path,
    top_n: int | None = 20,
) -> tuple[Any, Any]:
    """Create a dependency-free matplotlib heatmap of stage probabilities."""
    plt = _get_pyplot()
    data = stage_probabilities.sort_values("Champion", ascending=False)
    if top_n is not None:
        data = data.head(top_n)
    fig_height = max(6.0, 0.34 * len(data))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    image = ax.imshow(data.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(data.columns)), labels=data.columns)
    ax.set_yticks(np.arange(len(data.index)), labels=data.index)
    for row in range(len(data.index)):
        for column in range(len(data.columns)):
            value = data.iat[row, column]
            ax.text(
                column,
                row,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=8,
            )
    ax.set_title("Probability of reaching each knockout stage")
    fig.colorbar(image, ax=ax, label="Probability", fraction=0.03, pad=0.02)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig, ax


def save_standard_plots(
    stage_probabilities: pd.DataFrame,
    plots_dir: str | Path,
) -> dict[str, Path]:
    """Save champion, finalist, and stage heatmap figures."""
    plt = _get_pyplot()
    plots_dir = Path(plots_dir)
    paths = {
        "winners": plots_dir / "winner_probabilities.png",
        "finalists": plots_dir / "final_probabilities.png",
        "heatmap": plots_dir / "stage_probability_heatmap.png",
    }
    figures = [
        plot_probability_bars(
            stage_probabilities["Champion"],
            "World Cup 2026 win probabilities",
            paths["winners"],
        )[0],
        plot_probability_bars(
            stage_probabilities["Final"],
            "World Cup 2026 final probabilities",
            paths["finalists"],
        )[0],
        plot_stage_heatmap(stage_probabilities, paths["heatmap"])[0],
    ]
    for figure in figures:
        plt.close(figure)
    return paths


def run_weight_sensitivity(
    group_weights: Iterable[float],
    simulation_factory: Callable[[float], object],
    n_simulations: int = 5_000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Run comparable simulations for several group-form weights.

    ``simulation_factory(weight)`` must return an object exposing
    ``simulate_many``. The same seed is intentionally reused across scenarios.
    """
    frames: list[pd.DataFrame] = []
    for group_weight in group_weights:
        if not 0 <= group_weight <= 1:
            raise ValueError("Every group weight must be between 0 and 1")
        simulator = simulation_factory(group_weight)
        result = simulator.simulate_many(
            n_simulations=n_simulations,
            seed=seed,
            show_progress=False,
        )
        champion = result[["Champion"]].rename(
            columns={"Champion": f"group_weight_{group_weight:.1f}"}
        )
        frames.append(champion)
    return pd.concat(frames, axis=1).fillna(0).sort_index()
