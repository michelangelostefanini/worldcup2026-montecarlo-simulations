"""Bracket-preserving World Cup knockout simulation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - used only in deliberately minimal installs
    def tqdm(iterable, **_: object):
        """Fall back to an unwrapped iterator when tqdm is unavailable."""
        return iterable

from .match_model import MatchResult, PoissonMatchModel


ROUND_ORDER = ("R32", "R16", "QF", "SF", "F")
ADVANCEMENT_STAGE = {
    "R32": "Round of 16",
    "R16": "Quarterfinal",
    "QF": "Semifinal",
    "SF": "Final",
    "F": "Champion",
}
STAGE_COLUMNS = ("Round of 16", "Quarterfinal", "Semifinal", "Final", "Champion")
EXPECTED_MATCH_COUNTS = {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "F": 1}


@dataclass(frozen=True)
class TournamentResult:
    winner: str
    stages_reached: dict[str, set[str]]
    matches: tuple[MatchResult, ...]


@dataclass(frozen=True)
class TournamentSimulationReport:
    """Stage and per-match probabilities from the same Monte Carlo run."""

    stage_probabilities: pd.DataFrame
    matchup_probabilities: pd.DataFrame
    most_likely_matchups: pd.DataFrame
    team_match_probabilities: pd.DataFrame


def validate_bracket(bracket: pd.DataFrame, require_full_r32: bool = True) -> None:
    """Validate IDs, round links, slots, and the standard 32-team shape."""
    required = {
        "match_id",
        "round",
        "team_a",
        "team_b",
        "next_match_id",
        "next_slot",
    }
    missing = required - set(bracket.columns)
    if missing:
        raise ValueError(f"Bracket is missing columns: {sorted(missing)}")
    if bracket["match_id"].isna().any() or bracket["match_id"].duplicated().any():
        raise ValueError("Bracket match_id values must be present and unique")

    invalid_rounds = sorted(set(bracket["round"]) - set(ROUND_ORDER))
    if invalid_rounds:
        raise ValueError(f"Bracket contains invalid rounds: {invalid_rounds}")

    if require_full_r32:
        actual_counts = bracket["round"].value_counts().to_dict()
        if actual_counts != EXPECTED_MATCH_COUNTS:
            raise ValueError(
                "A full R32 bracket must contain match counts "
                f"{EXPECTED_MATCH_COUNTS}; received {actual_counts}"
            )

    r32 = bracket.loc[bracket["round"] == "R32"]
    if r32[["team_a", "team_b"]].isna().any().any():
        raise ValueError("Every R32 match must have team_a and team_b")
    initial_teams = pd.concat([r32["team_a"], r32["team_b"]])
    if initial_teams.duplicated().any():
        duplicates = sorted(initial_teams[initial_teams.duplicated()].unique())
        raise ValueError(f"R32 teams may appear only once: {duplicates}")

    by_id = bracket.set_index("match_id")
    incoming: Counter[tuple[str, str]] = Counter()
    for row in bracket.itertuples(index=False):
        if row.round == "F":
            if pd.notna(row.next_match_id) or pd.notna(row.next_slot):
                raise ValueError("The final must not point to another match")
            continue
        if pd.isna(row.next_match_id) or pd.isna(row.next_slot):
            raise ValueError(f"Match {row.match_id} is missing its next match or slot")
        if row.next_match_id not in by_id.index:
            raise ValueError(
                f"Match {row.match_id} points to unknown match {row.next_match_id}"
            )
        if row.next_slot not in {"team_a", "team_b"}:
            raise ValueError(
                f"Match {row.match_id} has invalid next_slot {row.next_slot!r}"
            )
        source_round = ROUND_ORDER.index(row.round)
        target_round = ROUND_ORDER.index(by_id.loc[row.next_match_id, "round"])
        if target_round != source_round + 1:
            raise ValueError(
                f"Match {row.match_id} must point to the immediately following round"
            )
        incoming[(row.next_match_id, row.next_slot)] += 1

    for row in bracket.loc[bracket["round"] != "R32"].itertuples(index=False):
        for slot in ("team_a", "team_b"):
            if incoming[(row.match_id, slot)] != 1:
                raise ValueError(
                    f"Match {row.match_id} slot {slot} must have exactly one feeder"
                )


class TournamentSimulator:
    """Simulate a validated knockout bracket without changing its topology."""

    def __init__(
        self,
        bracket: pd.DataFrame,
        strengths: Mapping[str, float] | pd.Series,
        match_model: PoissonMatchModel | None = None,
        require_full_r32: bool = True,
        static_elo: bool = True,
    ) -> None:
        self.static_elo = static_elo
        # TODO: Implement dynamic Elo/strength updates after every simulated
        # knockout match, then compare them with the static-Elo simulations.
        if not self.static_elo:
            raise NotImplementedError(
                "Dynamic Elo updates are not implemented yet; use static_elo=True"
            )
        self.bracket = bracket.copy()
        for column in ("match_id", "round", "team_a", "team_b", "next_match_id", "next_slot"):
            if column in self.bracket:
                self.bracket[column] = self.bracket[column].where(
                    self.bracket[column].notna(), pd.NA
                )
        validate_bracket(self.bracket, require_full_r32=require_full_r32)
        self.strengths = {str(team): float(value) for team, value in dict(strengths).items()}
        self.match_model = match_model or PoissonMatchModel()
        self.teams = sorted(
            set(self.bracket.loc[self.bracket["round"] == "R32", "team_a"])
            | set(self.bracket.loc[self.bracket["round"] == "R32", "team_b"])
        )
        missing = sorted(set(self.teams) - set(self.strengths))
        if missing:
            raise ValueError(f"Missing final strengths for bracket teams: {missing}")
        nonpositive = [team for team in self.teams if self.strengths[team] <= 0]
        if nonpositive:
            raise ValueError(f"Strengths must be positive for: {nonpositive}")

        self._slot_templates: dict[str, tuple[str | None, str | None]] = {}
        self._matches_by_round: dict[
            str, tuple[tuple[str, str | None, int | None], ...]
        ] = {}
        for row in self.bracket.itertuples(index=False):
            team_a = None if pd.isna(row.team_a) else str(row.team_a)
            team_b = None if pd.isna(row.team_b) else str(row.team_b)
            self._slot_templates[row.match_id] = (team_a, team_b)
        for round_name in ROUND_ORDER:
            compiled: list[tuple[str, str | None, int | None]] = []
            for row in self.bracket.loc[
                self.bracket["round"] == round_name
            ].itertuples(index=False):
                next_id = None if pd.isna(row.next_match_id) else row.next_match_id
                next_slot = (
                    None
                    if pd.isna(row.next_slot)
                    else 0 if row.next_slot == "team_a" else 1
                )
                compiled.append((row.match_id, next_id, next_slot))
            self._matches_by_round[round_name] = tuple(compiled)
        self._match_sequence = tuple(
            (match_id, round_name)
            for round_name in ROUND_ORDER
            for match_id, _, _ in self._matches_by_round[round_name]
        )

    def simulate_once(self, rng: np.random.Generator) -> TournamentResult:
        slots = {
            match_id: [team_a, team_b]
            for match_id, (team_a, team_b) in self._slot_templates.items()
        }
        stages = {stage: set() for stage in STAGE_COLUMNS}
        match_results: list[MatchResult] = []
        winner: str | None = None

        for round_name in ROUND_ORDER:
            for match_id, next_match_id, next_slot in self._matches_by_round[round_name]:
                team_a, team_b = slots[match_id]
                if team_a is None or team_b is None:
                    raise RuntimeError(
                        f"Bracket propagation left match {match_id} incomplete"
                    )
                result = self.match_model.simulate(
                    team_a,
                    team_b,
                    self.strengths[team_a],
                    self.strengths[team_b],
                    rng,
                    knockout=True,
                )
                match_results.append(result)
                assert result.winner is not None
                stages[ADVANCEMENT_STAGE[round_name]].add(result.winner)

                if round_name == "F":
                    winner = result.winner
                else:
                    assert next_match_id is not None and next_slot is not None
                    slots[next_match_id][next_slot] = result.winner

        if winner is None:
            raise RuntimeError("Tournament finished without a champion")
        return TournamentResult(winner, stages, tuple(match_results))

    def simulate_many(
        self,
        n_simulations: int = 10_000,
        seed: int | None = 2026,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Return stage probabilities, indexed by every R32 team."""
        if n_simulations <= 0:
            raise ValueError("n_simulations must be a positive integer")
        rng = np.random.default_rng(seed)
        counts = {
            team: {stage: 0 for stage in STAGE_COLUMNS}
            for team in self.teams
        }
        iterator = range(n_simulations)
        if show_progress:
            iterator = tqdm(iterator, desc="Simulating tournaments", unit="sim")

        for _ in iterator:
            result = self.simulate_once(rng)
            for stage, reached in result.stages_reached.items():
                for team in reached:
                    counts[team][stage] += 1

        probabilities = pd.DataFrame.from_dict(counts, orient="index")
        probabilities.index.name = "team"
        probabilities = probabilities.loc[:, STAGE_COLUMNS] / n_simulations
        return probabilities.sort_values("Champion", ascending=False)

    def simulate_many_detailed(
        self,
        n_simulations: int = 10_000,
        seed: int | None = 2026,
        show_progress: bool = True,
    ) -> TournamentSimulationReport:
        """Return stage, matchup, and team-by-match probabilities.

        ``matchup_probability`` is unconditional: it estimates the chance that
        the two named teams occupy that bracket match. Winner probabilities are
        conditional on that matchup occurring.
        """
        if n_simulations <= 0:
            raise ValueError("n_simulations must be a positive integer")

        rng = np.random.default_rng(seed)
        stage_counts = {
            team: {stage: 0 for stage in STAGE_COLUMNS}
            for team in self.teams
        }
        matchup_counts: Counter[tuple[str, str, str]] = Counter()
        matchup_winner_counts: Counter[tuple[str, str, str, str]] = Counter()
        appearance_counts: Counter[tuple[str, str, str]] = Counter()
        match_winner_counts: Counter[tuple[str, str]] = Counter()

        iterator = range(n_simulations)
        if show_progress:
            iterator = tqdm(iterator, desc="Simulating tournaments", unit="sim")

        for _ in iterator:
            result = self.simulate_once(rng)
            for stage, reached in result.stages_reached.items():
                for team in reached:
                    stage_counts[team][stage] += 1

            for (match_id, _), match in zip(
                self._match_sequence, result.matches, strict=True
            ):
                matchup_key = (match_id, match.team_a, match.team_b)
                matchup_counts[matchup_key] += 1
                assert match.winner is not None
                matchup_winner_counts[(*matchup_key, match.winner)] += 1
                appearance_counts[(match_id, "team_a", match.team_a)] += 1
                appearance_counts[(match_id, "team_b", match.team_b)] += 1
                match_winner_counts[(match_id, match.winner)] += 1

        stage_probabilities = pd.DataFrame.from_dict(stage_counts, orient="index")
        stage_probabilities.index.name = "team"
        stage_probabilities = (
            stage_probabilities.loc[:, STAGE_COLUMNS] / n_simulations
        ).sort_values("Champion", ascending=False)

        round_by_match = dict(self._match_sequence)
        order_by_match = {
            match_id: order
            for order, (match_id, _) in enumerate(self._match_sequence)
        }
        matchup_rows: list[dict[str, object]] = []
        for (match_id, team_a, team_b), occurrences in matchup_counts.items():
            wins_a = matchup_winner_counts[(match_id, team_a, team_b, team_a)]
            wins_b = matchup_winner_counts[(match_id, team_a, team_b, team_b)]
            probability_a = wins_a / occurrences
            probability_b = wins_b / occurrences
            favorite = team_a if probability_a >= probability_b else team_b
            matchup_rows.append(
                {
                    "match_id": match_id,
                    "round": round_by_match[match_id],
                    "team_a": team_a,
                    "team_b": team_b,
                    "matchup_probability": occurrences / n_simulations,
                    "team_a_win_probability_given_matchup": probability_a,
                    "team_b_win_probability_given_matchup": probability_b,
                    "favorite": favorite,
                    "favorite_win_probability": max(probability_a, probability_b),
                    "simulated_occurrences": occurrences,
                    "_match_order": order_by_match[match_id],
                }
            )
        matchup_probabilities = pd.DataFrame(matchup_rows).sort_values(
            ["_match_order", "matchup_probability", "team_a", "team_b"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        most_likely_matchups = (
            matchup_probabilities.groupby("match_id", sort=False, as_index=False)
            .head(1)
            .copy()
        )
        matchup_probabilities = matchup_probabilities.drop(columns="_match_order")
        most_likely_matchups = most_likely_matchups.drop(columns="_match_order")

        appearance_rows: list[dict[str, object]] = []
        for (match_id, slot, team), appearances in appearance_counts.items():
            wins = match_winner_counts[(match_id, team)]
            appearance_rows.append(
                {
                    "match_id": match_id,
                    "round": round_by_match[match_id],
                    "slot": slot,
                    "team": team,
                    "appearance_probability": appearances / n_simulations,
                    "win_probability_given_appearance": wins / appearances,
                    "unconditional_match_win_probability": wins / n_simulations,
                    "simulated_appearances": appearances,
                    "_match_order": order_by_match[match_id],
                }
            )
        team_match_probabilities = (
            pd.DataFrame(appearance_rows)
            .sort_values(
                [
                    "_match_order",
                    "appearance_probability",
                    "unconditional_match_win_probability",
                    "team",
                ],
                ascending=[True, False, False, True],
                kind="stable",
            )
            .drop(columns="_match_order")
        )

        return TournamentSimulationReport(
            stage_probabilities=stage_probabilities,
            matchup_probabilities=matchup_probabilities.reset_index(drop=True),
            most_likely_matchups=most_likely_matchups.reset_index(drop=True),
            team_match_probabilities=team_match_probabilities.reset_index(drop=True),
        )
