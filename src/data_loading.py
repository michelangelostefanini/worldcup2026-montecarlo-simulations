"""Input loading and validation for the World Cup simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


HISTORICAL_COLUMNS = (
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
)
GROUP_COLUMNS = HISTORICAL_COLUMNS
BRACKET_COLUMNS = {
    "match_id",
    "round",
    "team_a",
    "team_b",
    "next_match_id",
    "next_slot",
}
VALID_ROUNDS = {"R32", "R16", "QF", "SF", "F"}


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _clean_team_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    for column in columns:
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or frame[column].eq("").any():
            raise ValueError(f"{name}.{column} contains missing or empty team names")
    if {"home_team", "away_team"}.issubset(columns):
        same_team = frame["home_team"].eq(frame["away_team"])
        if same_team.any():
            rows = (frame.index[same_team] + 2).tolist()[:5]
            raise ValueError(
                f"{name} has a team playing itself at CSV row(s): {rows}"
            )


def _clean_scores(frame: pd.DataFrame, name: str) -> None:
    for column in ("home_score", "away_score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].isna() | (frame[column] < 0) | (frame[column] % 1 != 0)
        if invalid.any():
            rows = (frame.index[invalid] + 2).tolist()[:5]
            raise ValueError(
                f"{name}.{column} must contain non-negative integers; "
                f"invalid CSV row(s): {rows}"
            )
        frame[column] = frame[column].astype(int)


def _clean_dates(frame: pd.DataFrame, name: str) -> None:
    parsed = pd.to_datetime(frame["date"], errors="coerce")
    if parsed.isna().any():
        rows = (frame.index[parsed.isna()] + 2).tolist()[:5]
        raise ValueError(f"{name}.date has invalid values at CSV row(s): {rows}")
    frame["date"] = parsed


def _parse_boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


def load_historical_results(
    path: str | Path,
    start_date: str | pd.Timestamp = "2018-01-01",
    end_date: str | pd.Timestamp = "2026-06-11",
) -> pd.DataFrame:
    """Load the requested historical window for sequential Elo updates.

    Rows without a completed score are discarded. Only the seven columns used
    by the Elo model are retained, and both date boundaries are inclusive.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    _require_columns(frame, HISTORICAL_COLUMNS, path.name)
    frame = frame.loc[:, list(HISTORICAL_COLUMNS)].copy()
    _clean_dates(frame, path.name)

    frame = frame.dropna(subset=["home_score", "away_score"]).copy()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Historical start_date and end_date must be valid dates")
    if start > end:
        raise ValueError("Historical start_date cannot be after end_date")
    frame = frame.loc[frame["date"].between(start, end, inclusive="both")].copy()

    _clean_team_columns(frame, ("home_team", "away_team"), path.name)
    _clean_scores(frame, path.name)
    try:
        frame["neutral"] = frame["neutral"].map(_parse_boolean)
    except ValueError as exc:
        raise ValueError(f"{path.name}.neutral: {exc}") from exc
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def load_group_stage(path: str | Path) -> pd.DataFrame:
    """Load and validate 2026 group-stage results."""
    path = Path(path)
    frame = pd.read_csv(path)
    _require_columns(frame, GROUP_COLUMNS, path.name)
    _clean_dates(frame, path.name)
    _clean_team_columns(frame, ("home_team", "away_team"), path.name)
    _clean_scores(frame, path.name)
    frame["tournament"] = frame["tournament"].astype("string").str.strip()
    if frame["tournament"].isna().any() or frame["tournament"].eq("").any():
        raise ValueError(f"{path.name}.tournament contains missing or empty values")
    try:
        frame["neutral"] = frame["neutral"].map(_parse_boolean)
    except ValueError as exc:
        raise ValueError(f"{path.name}.neutral: {exc}") from exc

    optional_xg = {"home_xg", "away_xg"} & set(frame.columns)
    if optional_xg and optional_xg != {"home_xg", "away_xg"}:
        raise ValueError(
            f"{path.name} must provide both home_xg and away_xg, or neither"
        )
    for column in optional_xg:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or (frame[column] < 0).any():
            raise ValueError(f"{path.name}.{column} must contain non-negative numbers")

    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def load_bracket(path: str | Path) -> pd.DataFrame:
    """Load bracket links while preserving match IDs as strings."""
    path = Path(path)
    frame = pd.read_csv(
        path,
        dtype={
            "match_id": "string",
            "round": "string",
            "team_a": "string",
            "team_b": "string",
            "next_match_id": "string",
            "next_slot": "string",
        },
    )
    _require_columns(frame, BRACKET_COLUMNS, path.name)

    frame["match_id"] = frame["match_id"].str.strip()
    if frame["match_id"].isna().any() or frame["match_id"].eq("").any():
        raise ValueError(f"{path.name}.match_id contains missing values")
    if frame["match_id"].duplicated().any():
        duplicates = frame.loc[frame["match_id"].duplicated(), "match_id"].tolist()
        raise ValueError(f"{path.name}.match_id contains duplicates: {duplicates}")

    frame["round"] = frame["round"].str.strip().str.upper()
    if frame["round"].isna().any() or frame["round"].eq("").any():
        raise ValueError(f"{path.name}.round contains missing values")
    invalid_rounds = sorted(set(frame["round"].dropna()) - VALID_ROUNDS)
    if invalid_rounds:
        raise ValueError(f"{path.name}.round has invalid values: {invalid_rounds}")

    for column in ("team_a", "team_b", "next_match_id", "next_slot"):
        frame[column] = frame[column].str.strip().replace("", pd.NA)
    frame["next_slot"] = frame["next_slot"].str.lower()
    return frame.reset_index(drop=True)


def load_inputs(
    data_dir: str | Path,
    historical_start_date: str | pd.Timestamp = "2018-01-01",
    historical_end_date: str | pd.Timestamp = "2026-06-11",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three project inputs from a directory."""
    data_dir = Path(data_dir)
    return (
        load_historical_results(
            data_dir / "historical_results.csv",
            start_date=historical_start_date,
            end_date=historical_end_date,
        ),
        load_group_stage(data_dir / "group_stage_2026.csv"),
        load_bracket(data_dir / "bracket_2026.csv"),
    )
