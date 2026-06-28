"""Poisson-based football match simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchResult:
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    winner: str | None
    decided_by: str


@dataclass
class PoissonMatchModel:
    """Simulate scores from relative strengths.

    ``base_goals`` is the expected regulation-time goals for each team when
    strengths are equal. Team A and team B are neutral bracket slots: knockout
    simulations never apply home advantage. Expected goals are clipped to avoid
    extreme values.
    """

    base_goals: float = 1.35
    strength_exponent: float = 1.0
    minimum_lambda: float = 0.10
    maximum_lambda: float = 5.00
    extra_time_fraction: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        if self.base_goals <= 0:
            raise ValueError("base_goals must be positive")
        if self.minimum_lambda <= 0 or self.maximum_lambda < self.minimum_lambda:
            raise ValueError("Invalid Poisson lambda bounds")
        if not 0 < self.extra_time_fraction <= 1:
            raise ValueError("extra_time_fraction must be in (0, 1]")

    def expected_goals(
        self, strength_a: float, strength_b: float
    ) -> tuple[float, float]:
        if strength_a <= 0 or strength_b <= 0:
            raise ValueError("Team strengths must be positive")
        ratio = (strength_a / strength_b) ** self.strength_exponent
        lambda_a = float(
            np.clip(self.base_goals * ratio, self.minimum_lambda, self.maximum_lambda)
        )
        lambda_b = float(
            np.clip(self.base_goals / ratio, self.minimum_lambda, self.maximum_lambda)
        )
        return lambda_a, lambda_b

    def simulate(
        self,
        team_a: str,
        team_b: str,
        strength_a: float,
        strength_b: float,
        rng: np.random.Generator,
        knockout: bool = True,
    ) -> MatchResult:
        """Simulate regulation, then extra time and penalties when required."""
        lambda_a, lambda_b = self.expected_goals(strength_a, strength_b)
        goals_a = int(rng.poisson(lambda_a))
        goals_b = int(rng.poisson(lambda_b))

        if goals_a != goals_b:
            winner = team_a if goals_a > goals_b else team_b
            return MatchResult(team_a, team_b, goals_a, goals_b, winner, "regulation")
        if not knockout:
            return MatchResult(team_a, team_b, goals_a, goals_b, None, "draw")

        goals_a += int(rng.poisson(lambda_a * self.extra_time_fraction))
        goals_b += int(rng.poisson(lambda_b * self.extra_time_fraction))
        if goals_a != goals_b:
            winner = team_a if goals_a > goals_b else team_b
            return MatchResult(team_a, team_b, goals_a, goals_b, winner, "extra_time")

        penalty_probability_a = strength_a / (strength_a + strength_b)
        winner = team_a if rng.random() < penalty_probability_a else team_b
        return MatchResult(team_a, team_b, goals_a, goals_b, winner, "penalties")
