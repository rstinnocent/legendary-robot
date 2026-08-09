"""
profiler.py — pure-pandas data profiling for the auto-analysis feature.

No LLM involved anywhere in this module. The chart-plan step (built on top of
this) asks a model to pick which of these facts are worth showing; it never
computes them. Keeping profiling deterministic means it's fully unit-testable
without an API key, and a bad chart plan still sits on top of a correct
understanding of the data.

Column role assignment
-----------------------
Five roles: identifier, dimension, measure, date, skip. The order the checks
run in matters — a naive "2-20 distinct values -> dimension" rule fires on
things it shouldn't:

  - `annual_ctc` is ~100% unique (every salary is close to distinct) but is
    obviously a measure, not an identifier — near-uniqueness is a strong
    identifier signal for strings (a repeated employee name would mean two
    employees share a name) but a weak one for continuous numbers, which are
    unique-ish by nature. So the near-unique check only fires on non-numeric
    columns; numeric columns are only identifiers if their name says so
    (`_id` suffix).
  - `working_days` in the attendance data only takes 3 distinct values
    (20/21/22) across 240 rows — well inside the 2-20 "dimension" window by
    cardinality alone, but it's a measure you'd sum or average, not something
    you'd group by. So numeric columns are classified as measures before the
    dimension cardinality rule is even considered; the dimension rule only
    applies to non-numeric (mostly string) columns. This does mean a
    genuinely categorical numeric column (a 1-5 rating stored as int) would
    be misread as a measure — not a concern for the shipped sample data, and
    telling "coded category" apart from "small-range measure" from values
    alone is a judgment call with no clean answer.
  - `date_of_joining` is also ~100% unique per-row, same as `annual_ctc`, but
    it's a date, not a measure — so date detection is tried first, before
    either the identifier or measure checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Column role assignment
# ---------------------------------------------------------------------------

ROLE_IDENTIFIER = "identifier"
ROLE_DIMENSION = "dimension"
ROLE_MEASURE = "measure"
ROLE_DATE = "date"
ROLE_SKIP = "skip"

MAX_NULL_RATE = 0.6
NEAR_UNIQUE_RATIO = 0.95
MIN_ROWS_FOR_NEAR_UNIQUE = 5  # below this, "100% unique" is not a meaningful signal
DIMENSION_MIN_DISTINCT = 2
DIMENSION_MAX_DISTINCT = 20

_ID_NAME_RE = re.compile(r"(^|_)id$", re.IGNORECASE)

# Tried in order; the first format that parses (almost) every non-null value
# without producing NaT wins. Covers ISO, both day/month orderings with '-'
# and '/', and a bare year-month for monthly-grain columns like `month`.
_DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y",
                  "%m/%d/%Y", "%Y-%m", "%Y/%m"]
_DATE_PARSE_MIN_SUCCESS = 0.95


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    role: str
    distinct_count: int
    null_rate: float
    min: Any = None
    max: Any = None
    sample_values: list = field(default_factory=list)
    date_format: str | None = None  # set only when role == "date"


@dataclass
class FrameProfile:
    name: str
    n_rows: int
    columns: dict[str, ColumnProfile]

    def columns_with_role(self, role: str) -> list[str]:
        return [c.name for c in self.columns.values() if c.role == role]


@dataclass
class JoinKey:
    left_frame: str
    left_column: str
    right_frame: str
    right_column: str
    overlap_count: int
    overlap_ratio: float  # intersection size / smaller side's distinct count


def _is_id_named(name: str) -> bool:
    return bool(_ID_NAME_RE.search(name)) or name.lower() == "id"


def _try_parse_dates(series: pd.Series) -> tuple[pd.Series, str] | tuple[None, None]:
    """Try each candidate format; return the parsed series for the first one
    that resolves almost all non-null values, or (None, None) if none do.

    Explicit `format=` parsing (rather than pandas' dayfirst-guessing
    to_datetime) is deterministic: a format either matches a value or
    produces NaT, so picking "whichever format has the fewest NaT" reliably
    separates DD-MM-YYYY from MM-DD-YYYY data instead of silently guessing
    per-row the way pandas' free-form parser does.
    """
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None, None
    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(non_null, format=fmt, errors="coerce")
        if parsed.notna().mean() >= _DATE_PARSE_MIN_SUCCESS:
            return pd.to_datetime(series.astype(str), format=fmt, errors="coerce"), fmt
    return None, None


def profile_column(name: str, series: pd.Series) -> ColumnProfile:
    n = len(series)
    null_rate = series.isna().mean() if n else 1.0
    non_null = series.dropna()
    distinct_count = non_null.nunique()
    dtype_str = str(series.dtype)
    sample_values = non_null.head(5).tolist()

    if n == 0 or null_rate > MAX_NULL_RATE or distinct_count <= 1:
        return ColumnProfile(name, dtype_str, ROLE_SKIP, distinct_count, null_rate,
                              sample_values=sample_values)

    is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    is_bool = pd.api.types.is_bool_dtype(series)

    if not is_numeric:
        parsed, fmt = _try_parse_dates(series)
        if parsed is not None:
            valid = parsed.dropna()
            return ColumnProfile(
                name, dtype_str, ROLE_DATE, distinct_count, null_rate,
                min=valid.min() if not valid.empty else None,
                max=valid.max() if not valid.empty else None,
                sample_values=sample_values, date_format=fmt,
            )

    if _is_id_named(name):
        return ColumnProfile(name, dtype_str, ROLE_IDENTIFIER, distinct_count, null_rate,
                              sample_values=sample_values)

    if not is_numeric and not is_bool and len(non_null) >= MIN_ROWS_FOR_NEAR_UNIQUE:
        if distinct_count / len(non_null) >= NEAR_UNIQUE_RATIO:
            return ColumnProfile(name, dtype_str, ROLE_IDENTIFIER, distinct_count, null_rate,
                                  sample_values=sample_values)

    if is_numeric:
        return ColumnProfile(
            name, dtype_str, ROLE_MEASURE, distinct_count, null_rate,
            min=non_null.min(), max=non_null.max(), sample_values=sample_values,
        )

    if DIMENSION_MIN_DISTINCT <= distinct_count <= DIMENSION_MAX_DISTINCT:
        return ColumnProfile(name, dtype_str, ROLE_DIMENSION, distinct_count, null_rate,
                              sample_values=sample_values)

    return ColumnProfile(name, dtype_str, ROLE_SKIP, distinct_count, null_rate,
                          sample_values=sample_values)


def profile_frame(name: str, df: pd.DataFrame) -> FrameProfile:
    columns = {col: profile_column(col, df[col]) for col in df.columns}
    return FrameProfile(name=name, n_rows=len(df), columns=columns)


def profile_frames(frames: dict[str, pd.DataFrame]) -> dict[str, FrameProfile]:
    return {name: profile_frame(name, df) for name, df in frames.items()}


# ---------------------------------------------------------------------------
# Cross-file join key detection
# ---------------------------------------------------------------------------

MIN_JOIN_OVERLAP_RATIO = 0.5
MIN_JOIN_OVERLAP_COUNT = 2


def _name_root(col_name: str) -> str:
    """Strip a trailing '_id'/'id' suffix: 'employee_id' -> 'employee'."""
    return _ID_NAME_RE.sub("", col_name).strip("_").lower()


def detect_join_keys(
    frames: dict[str, pd.DataFrame],
    profiles: dict[str, FrameProfile],
    min_overlap_ratio: float = MIN_JOIN_OVERLAP_RATIO,
    min_overlap_count: int = MIN_JOIN_OVERLAP_COUNT,
) -> list[JoinKey]:
    """Find columns across different files that share actual values.

    Restricted to columns already classified `identifier` — checking every
    column pair would flag things like two files both having a `gender`
    dimension with overlapping {Male, Female} values, which is a coincidence
    of small shared domains, not a join key.

    Value overlap alone is not sufficient, though: on the sample data,
    `attendance.record_id` (a row counter, 1-240) and `exits.exit_id` (also a
    row counter, 1-9) hit ratio=1.0 purely because both are gapless integer
    sequences starting at 1 — a coincidence of two unrelated surrogate keys,
    not a real relationship. `employee_id` in both files refers to the same
    real-world entity. Values alone can't tell these apart, so a second,
    weaker signal is required alongside overlap: the column names' semantic
    root (the name minus its `_id`/`id` suffix) must match, e.g. 'employee'
    == 'employee'. This is "not just matching names" — overlap still does
    the real filtering — names only break the tie between two candidates
    that both pass the value-overlap bar.
    """
    results: list[JoinKey] = []
    for frame_a, frame_b in combinations(frames.keys(), 2):
        id_cols_a = profiles[frame_a].columns_with_role(ROLE_IDENTIFIER)
        id_cols_b = profiles[frame_b].columns_with_role(ROLE_IDENTIFIER)
        for col_a in id_cols_a:
            values_a = set(frames[frame_a][col_a].dropna().tolist())
            if not values_a:
                continue
            root_a = _name_root(col_a)
            for col_b in id_cols_b:
                root_b = _name_root(col_b)
                if root_a and root_b and root_a != root_b:
                    continue
                values_b = set(frames[frame_b][col_b].dropna().tolist())
                if not values_b:
                    continue
                overlap = values_a & values_b
                smaller = min(len(values_a), len(values_b))
                ratio = len(overlap) / smaller if smaller else 0.0
                if len(overlap) >= min_overlap_count and ratio >= min_overlap_ratio:
                    results.append(JoinKey(
                        left_frame=frame_a, left_column=col_a,
                        right_frame=frame_b, right_column=col_b,
                        overlap_count=len(overlap), overlap_ratio=ratio,
                    ))
    return results
