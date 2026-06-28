"""Sequential Elo ratings, group-stage form, and strength combination."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_TOURNAMENT_K_FACTORS = {
    "Friendly": 10.0,
    "UEFA Nations League": 20.0,
    "FIFA World Cup qualification": 25.0,
    "UEFA Euro qualification": 25.0,
    "Copa América": 35.0,
    "UEFA Euro": 35.0,
    "FIFA World Cup": 50.0,
}


@dataclass(frozen=True)
class EloConfig:
    """Parameters for chronological Elo updates."""

    initial_elo: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 100.0
    rating_scale: float = 400.0
    sort_by_date: bool = True
    tournament_k_factors: Mapping[str, float] = field(
        default_factory=lambda: DEFAULT_TOURNAMENT_K_FACTORS.copy()
    )


def _actual_score(home_score: int, away_score: int) -> float:
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5


def update_elo_ratings(
    matches: pd.DataFrame,
    initial_ratings: Mapping[str, float] | None = None,
    config: EloConfig | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Update Elo after every match and return final ratings plus an audit history.

    If ``neutral`` is absent, matches are treated as neutral. This is suitable for
    World Cup group matches; historical input should provide the column.
    """
    config = config or EloConfig()
    required = {"home_team", "away_team", "home_score", "away_score"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Elo input is missing columns: {sorted(missing)}")
    if config.k_factor <= 0 or config.rating_scale <= 0:
        raise ValueError("k_factor and rating_scale must be positive")
    invalid_k_factors = {
        tournament: factor
        for tournament, factor in config.tournament_k_factors.items()
        if factor <= 0
    }
    if invalid_k_factors:
        raise ValueError(
            f"Tournament K-factors must be positive: {invalid_k_factors}"
        )

    ratings = dict(initial_ratings or {})
    ordered = matches.copy()
    if config.sort_by_date:
        if "date" not in ordered:
            raise ValueError("date is required when EloConfig.sort_by_date is True")
        ordered = ordered.sort_values("date", kind="stable")

    history: list[dict[str, object]] = []
    for index, match in ordered.iterrows():
        home = str(match["home_team"]).strip()
        away = str(match["away_team"]).strip()
        if not home or not away:
            raise ValueError(f"Empty team name in Elo input at index {index}")
        if home == away:
            raise ValueError(f"A team cannot play itself in Elo input at index {index}")
        home_before = float(ratings.get(home, config.initial_elo))
        away_before = float(ratings.get(away, config.initial_elo))

        is_neutral = bool(match.get("neutral", True))
        advantage = 0.0 if is_neutral else config.home_advantage
        expected_home = 1.0 / (
            1.0 + 10.0 ** ((away_before - home_before - advantage) / config.rating_scale)
        )
        actual_home = _actual_score(int(match["home_score"]), int(match["away_score"]))
        tournament_value = match.get("tournament", pd.NA)
        tournament = (
            None if pd.isna(tournament_value) else str(tournament_value).strip()
        )
        match_k_factor = float(
            config.tournament_k_factors.get(tournament, config.k_factor)
        )
        change = match_k_factor * (actual_home - expected_home)
        home_after = home_before + change
        away_after = away_before - change
        ratings[home] = home_after
        ratings[away] = away_after

        history.append(
            {
                "source_index": index,
                "date": match.get("date", pd.NaT),
                "home_team": home,
                "away_team": away,
                "tournament": tournament,
                "k_factor": match_k_factor,
                "home_elo_before": home_before,
                "away_elo_before": away_before,
                "expected_home": expected_home,
                "actual_home": actual_home,
                "elo_change": change,
                "home_elo_after": home_after,
                "away_elo_after": away_after,
            }
        )

    return ratings, pd.DataFrame(history)


def update_elo_with_warmup(
    matches: pd.DataFrame,
    full_weight_start_date: str | pd.Timestamp = "2018-01-01",
    warmup_k_scale: float = 0.20,
    initial_ratings: Mapping[str, float] | None = None,
    config: EloConfig | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Calibrate Elo on older matches before applying full-weight updates.

    Matches before ``full_weight_start_date`` use every configured K-factor
    multiplied by ``warmup_k_scale``. Matches on or after the boundary use the
    normal configuration. The returned audit history labels each update as
    ``warmup`` or ``full_weight``.
    """
    if not 0 < warmup_k_scale <= 1:
        raise ValueError("warmup_k_scale must be greater than 0 and at most 1")
    if "date" not in matches:
        raise ValueError("date is required for warm-up Elo updates")

    boundary = pd.Timestamp(full_weight_start_date)
    if pd.isna(boundary):
        raise ValueError("full_weight_start_date must be a valid date")
    dates = pd.to_datetime(matches["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Elo warm-up input contains invalid dates")

    config = config or EloConfig()
    warmup_config = replace(
        config,
        k_factor=config.k_factor * warmup_k_scale,
        tournament_k_factors={
            tournament: factor * warmup_k_scale
            for tournament, factor in config.tournament_k_factors.items()
        },
    )
    warmup_matches = matches.loc[dates < boundary].copy()
    full_weight_matches = matches.loc[dates >= boundary].copy()

    warmup_ratings, warmup_history = update_elo_ratings(
        warmup_matches,
        initial_ratings=initial_ratings,
        config=warmup_config,
    )
    ratings, full_weight_history = update_elo_ratings(
        full_weight_matches,
        initial_ratings=warmup_ratings,
        config=config,
    )
    warmup_history["phase"] = "warmup"
    full_weight_history["phase"] = "full_weight"
    history = pd.concat(
        [warmup_history, full_weight_history],
        ignore_index=True,
    )
    return ratings, history


def compute_group_form(
    matches: pd.DataFrame,
    goal_difference_weight: float = 0.20,
    goals_scored_weight: float = 0.10,
    xg_difference_weight: float = 0.10,
) -> pd.DataFrame:
    """Aggregate group results and produce a per-match form score.

    Form starts with points per match, then adds weighted goal difference and
    goals scored per match. When both xG columns exist, xG difference is added.
    """
    required = {"home_team", "away_team", "home_score", "away_score"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Group-form input is missing columns: {sorted(missing)}")
    has_xg = {"home_xg", "away_xg"}.issubset(matches.columns)

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    columns = ["played", "points", "goals_for", "goals_against"]
    if has_xg:
        columns += ["xg_for", "xg_against"]
    totals = pd.DataFrame(0.0, index=teams, columns=columns)

    for match in matches.itertuples(index=False):
        home, away = match.home_team, match.away_team
        home_goals, away_goals = int(match.home_score), int(match.away_score)
        totals.loc[[home, away], "played"] += 1
        totals.loc[home, ["goals_for", "goals_against"]] += [home_goals, away_goals]
        totals.loc[away, ["goals_for", "goals_against"]] += [away_goals, home_goals]

        if home_goals > away_goals:
            totals.loc[home, "points"] += 3
        elif away_goals > home_goals:
            totals.loc[away, "points"] += 3
        else:
            totals.loc[[home, away], "points"] += 1

        if has_xg:
            totals.loc[home, ["xg_for", "xg_against"]] += [match.home_xg, match.away_xg]
            totals.loc[away, ["xg_for", "xg_against"]] += [match.away_xg, match.home_xg]

    totals["goal_difference"] = totals["goals_for"] - totals["goals_against"]
    totals["points_per_game"] = totals["points"] / totals["played"]
    totals["goals_for_per_game"] = totals["goals_for"] / totals["played"]
    totals["goal_difference_per_game"] = totals["goal_difference"] / totals["played"]
    totals["form_score"] = (
        totals["points_per_game"]
        + goal_difference_weight * totals["goal_difference_per_game"]
        + goals_scored_weight * totals["goals_for_per_game"]
    )
    if has_xg:
        totals["xg_difference"] = totals["xg_for"] - totals["xg_against"]
        totals["xg_difference_per_game"] = totals["xg_difference"] / totals["played"]
        totals["form_score"] += (
            xg_difference_weight * totals["xg_difference_per_game"]
        )

    totals.index.name = "team"
    totals["played"] = totals["played"].astype(int)
    return totals.sort_values("form_score", ascending=False)


def minmax_normalize(values: pd.Series) -> pd.Series:
    """Normalize finite values to [0, 1], using 0.5 for a constant series."""
    values = values.astype(float)
    if values.empty:
        return values
    if not np.isfinite(values).all():
        raise ValueError("Cannot normalize non-finite strength values")
    low, high = values.min(), values.max()
    if np.isclose(low, high):
        return pd.Series(0.5, index=values.index, dtype=float)
    return (values - low) / (high - low)


def combine_strengths(
    elo_ratings: Mapping[str, float],
    group_form: pd.DataFrame | pd.Series,
    teams: Sequence[str] | None = None,
    historical_weight: float = 0.7,
    group_weight: float = 0.3,
    missing_elo: float = 1500.0,
    minimum_strength: float = 0.05,
) -> pd.DataFrame:
    """Combine normalized Elo and group form into a positive model strength."""
    if historical_weight < 0 or group_weight < 0:
        raise ValueError("Strength weights cannot be negative")
    weight_total = historical_weight + group_weight
    if weight_total <= 0:
        raise ValueError("At least one strength weight must be positive")
    historical_weight /= weight_total
    group_weight /= weight_total

    form = (
        group_form["form_score"]
        if isinstance(group_form, pd.DataFrame)
        else group_form
    ).astype(float)
    if teams is not None:
        team_index = sorted(set(teams))
    else:
        team_index = sorted(set(elo_ratings.keys()) | set(form.index.astype(str)))
    if not team_index:
        raise ValueError("No teams were supplied for strength calculation")

    elo = pd.Series(
        {team: float(elo_ratings.get(team, missing_elo)) for team in team_index},
        dtype=float,
    )
    normalized_elo = minmax_normalize(elo)
    normalized_form = minmax_normalize(form)
    normalized_form = normalized_form.reindex(team_index).fillna(0.5)
    combined = historical_weight * normalized_elo + group_weight * normalized_form

    result = pd.DataFrame(
        {
            "elo": elo,
            "normalized_elo": normalized_elo,
            "form_score": form.reindex(team_index),
            "normalized_group_form": normalized_form,
            "final_strength": combined.clip(lower=minimum_strength),
        }
    )
    result.index.name = "team"
    return result.sort_values("final_strength", ascending=False)
