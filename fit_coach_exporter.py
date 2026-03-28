#!/usr/bin/env python3
"""
fit_coach_exporter.py  — v3.3.3  (Production Grade / Zero Compromises)

Batch converter for cycling FIT files (Garmin Edge, Wahoo KICKR, MyWhoosh and
similar) into coach-friendly CSV datasets.

Outputs (stable schema, deterministic column order, schema_version field):
  activities.csv         — one row per workout / activity
  laps.csv               — one row per lap / interval / segment
  workout_steps.csv      — planned ERG / structured workout steps
  records_extracted.csv  — original device samples, verbatim FIT values
  records_1hz.csv        — resampled to 1 Hz with imputation and quality flags
  best_efforts.csv       — rolling best powers for common durations
  file_inventory.csv     — source files and parsing status

Architecture: streaming incremental export.
  Each FIT file is parsed and immediately appended to the output CSVs.
  No batch-level in-memory accumulation of ParsedFitFile objects.
  Large per-file collections (records_extracted, records_1hz) are streamed
  via append_dicts() to avoid materialising full intermediate lists.

  Empty-batch contract:
    When no .fit files are found the program exits with code 3 WITHOUT
    creating or touching any output CSV files.  BatchCsvWriter is only
    instantiated after confirming that at least one file exists.

  CSV header-validation contract:
    CsvStreamWriter validates the header of any pre-existing CSV before
    appending to it.  If the file exists but is empty, has a mismatched
    header, or cannot be read, the program raises CsvSchemaMismatchError.
    This prevents silently mixing data from different schema versions in
    the same output file.

Data layer naming:
  records_extracted   — verbatim FIT record messages; enhanced channels
                        stored separately, base channels unmodified.
  records_clean_input — working copy with enhanced channels merged into
                        base columns; used only inside parse_fit_file,
                        never exported directly.
  records_1hz         — 1 Hz resampled timeline with imputation flags,
                        quality flags, and derived coach columns.

─── FileInventoryRow.status values ──────────────────────────────────────────
  "ok"
      Full analytic success.  A 1 Hz timeline was built.  All applicable
      metrics have been computed (subject to power quality gating).

  "ok_no_timeline"
      The FIT file was read successfully and header/session data may be
      present, but NO 1 Hz timeline could be constructed.
      Possible causes (recorded in status_message):
        • no record messages exist in the file,
        • record messages have no timestamp field,
        • all record timestamps are invalid or unparseable,
        • the timeline is empty after deduplication / resampling.
      Consequence: records_1hz rows, best_efforts rows, and all power / HR /
      grade metrics derived from the 1 Hz timeline are absent.

  "error"
      Processing failed at a named pipeline stage.  error_stage and
      error_class identify where and what failed.  The batch continues.

─── Power metrics quality status (activities.csv) ───────────────────────────
  power_metrics_quality  — machine-readable categorical field.  Values:

    "full"
        Timeline is dense (≈1 Hz) OR sparse with adequate power coverage.
        All power metrics are computed with the standard algorithm.
        power_quality_note is None.

    "sparse_warning"
        Timeline is sparse AND power_channel_is_low_coverage is True AND
        valid sample count ≥ _SPARSE_POWER_MIN_VALID_SAMPLES.
        Metrics are computed with the sparse-mode NP algorithm (gaps skipped
        not zeroed); values may overestimate training load.
        power_quality_note carries an explicit warning.

    "withheld"
        Timeline is sparse AND power_channel_is_low_coverage is True AND
        valid sample count < _SPARSE_POWER_MIN_VALID_SAMPLES.
        NP, NP_moving, IF, TSS, work_kj, VI, z1–z7, per-row power_zone,
        and best_efforts are all None.
        power_quality_note explains what was withheld and why.

    None
        No power channel present in the activity; quality assessment does
        not apply.

  power_quality_note is always consistent with power_metrics_quality.

─── Channel coverage flag semantics ─────────────────────────────────────────
  power_channel_is_low_coverage / speed_channel_is_low_coverage:
    These are heuristic coverage signals, NOT objective measures of sensor
    quality.  They are True when the channel carries fewer than
    _CHANNEL_SPARSE_COVERAGE_THRESHOLD (50 %) non-NaN values in the raw
    pre-resampled records DataFrame.  This means the device either did not
    record this channel for more than half the activity, or smart-recording
    skipped many samples.
    False when coverage ≥ 50 % OR when the column is entirely absent
    (channel was never recorded — absence ≠ low coverage).

Dependencies:
    pip install pandas numpy fitdecode

Example:
    python fit_coach_exporter.py \\
        --input  ./fit_files \\
        --output ./coach_csv \\
        --athlete  "Przemek" \\
        --ftp      250 \\
        --resting-hr 50 \\
        --max-hr   185 \\
        --lthr     170
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import fitdecode
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Schema version ────────────────────────────────────────────────────────────
# Bump when any CSV column is added, removed, or renamed.
EXPORT_SCHEMA_VERSION = "3.3.3"

# ── Power metrics quality status values ───────────────────────────────────────
# These are the only valid values for FitActivity.power_metrics_quality.
_PMQ_FULL = "full"
_PMQ_SPARSE_WARNING = "sparse_warning"
_PMQ_WITHHELD = "withheld"

# ── Timeline density thresholds ───────────────────────────────────────────────
_DENSE_MEDIAN_GAP_THRESHOLD_S: float = 2.0
_DENSE_MAX_GAP_THRESHOLD_S: float = 10.0

# ── Effort channel dropout forgiveness ────────────────────────────────────────
_EFFORT_DROPOUT_FFILL_LIMIT_S: int = 2   # applies to power, cadence, speed

# ── HR imputation limit ────────────────────────────────────────────────────────
_HR_FFILL_LIMIT_S: int = 5

# ── Sparse-timeline power metric gate ─────────────────────────────────────────
# When power_channel_is_low_coverage is True, power metrics are only computed
# when the raw DataFrame has at least this many non-NaN power values.
# 300 ≈ 5 minutes of real measurements.
_SPARSE_POWER_MIN_VALID_SAMPLES: int = 300

# ── Channel coverage threshold ────────────────────────────────────────────────
# A channel is flagged as low-coverage when fewer than this fraction of raw
# records carry a non-NaN value.  This is a heuristic, not a sensor-quality
# measurement — see module docstring for full semantics.
_CHANNEL_SPARSE_COVERAGE_THRESHOLD: float = 0.50  # < 50 % → low coverage


# ---------------------------------------------------------------------------
# CSV schema mismatch error
# ---------------------------------------------------------------------------


class CsvSchemaMismatchError(RuntimeError):
    """
    Raised when an existing CSV file has a header that does not match the
    expected column list for the current schema version.

    This is a fatal condition: appending to a file with a different schema
    would silently corrupt the dataset by mixing incompatible rows.
    Callers must not continue writing to the file after this exception.
    """


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def to_iso(value: Any) -> Optional[str]:
    """Convert datetime or any value to an ISO-8601 string, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.isoformat()
    return str(value)


def safe_float(value: Any) -> Optional[float]:
    """
    Convert *value* to float, returning None for missing / unrepresentable data.

    Contract:
    - Always returns exactly float | None — never numpy scalar, never NaN.
    - Use ``is None`` to test absence; downstream code must never use ``or``
      to distinguish None from a legitimate 0.0.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        f = float(value)
        return None if math.isnan(f) else f
    try:
        s = str(value).strip()
        if s == "":
            return None
        f = float(s)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def safe_int(value: Any) -> Optional[int]:
    """Convert *value* to int via safe_float, or None."""
    out = safe_float(value)
    return None if out is None else int(round(out))


def meters_to_km(m: Optional[float]) -> Optional[float]:
    """Convert metres to kilometres, preserving None."""
    return None if m is None else m / 1000.0


def ms_to_kmh(v: Optional[float]) -> Optional[float]:
    """Convert m/s to km/h, preserving None."""
    return None if v is None else v * 3.6


def seconds_to_hms(seconds: Optional[float]) -> Optional[str]:
    """Format seconds as HH:MM:SS string, or None."""
    if seconds is None:
        return None
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def mean_ignore_none(values: Any) -> Optional[float]:
    """Arithmetic mean of a sequence, ignoring None and NaN."""
    vals: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(f):
            vals.append(f)
    return float(sum(vals) / len(vals)) if vals else None


# ---------------------------------------------------------------------------
# Timeline quality assessment
# ---------------------------------------------------------------------------


@dataclass
class TimelineQuality:
    """
    Quality assessment of a raw records timeline, computed before resampling.

    ── Overall timeline density ───────────────────────────────────────────────
    timeline_is_dense_1hz : bool
        True  — original records are natively dense (~1 Hz).  Zero-filling
                effort channels is semantically correct: gap = coasting.
        False — smart recording or lower sample rate.  Gaps represent unknown
                intervals, not confirmed zero-effort.

    ── Per-channel coverage flags (heuristic) ────────────────────────────────
    power_channel_is_low_coverage : bool
        True when the power column has fewer than
        _CHANNEL_SPARSE_COVERAGE_THRESHOLD (50 %) non-NaN values in the raw
        pre-resampled DataFrame.  This is a coverage heuristic, not a sensor
        quality measurement.  The channel may have been recorded at a lower
        rate, turned off mid-ride, or simply absent for parts of the activity.
        False when coverage ≥ 50 %, OR when the power column is entirely
        absent (absence ≠ low coverage; has_power flag handles the latter).

    speed_channel_is_low_coverage : bool
        Same heuristic applied to the speed column.  When True, moving_time_s
        is withheld because speed-based movement detection is unreliable.

    ── Gap statistics ────────────────────────────────────────────────────────
    median_gap_s : Optional[float]
        Median positive inter-sample gap in seconds.  None if < 2 valid samples.

    max_gap_s : Optional[float]
        Maximum positive inter-sample gap in seconds.  None if < 2 samples.

    ── Power sample count ────────────────────────────────────────────────────
    power_valid_sample_count : int
        Non-NaN power values in the raw pre-resampled DataFrame.  Used
        together with power_channel_is_low_coverage to decide which power
        metrics are computable on sparse timelines.  0 when absent.
    """
    timeline_is_dense_1hz: bool
    power_channel_is_low_coverage: bool
    speed_channel_is_low_coverage: bool
    median_gap_s: Optional[float]
    max_gap_s: Optional[float]
    power_valid_sample_count: int


def _channel_coverage_is_low(df: pd.DataFrame, col: str) -> bool:
    """
    Return True when *col* has fewer than _CHANNEL_SPARSE_COVERAGE_THRESHOLD
    non-NaN values as a fraction of total rows.

    This is a heuristic coverage signal only.  Returns False when the column
    is entirely absent — "not recorded" differs from "recorded sparsely".
    """
    if col not in df.columns:
        return False
    total = len(df)
    if total == 0:
        return False
    valid = int(df[col].notna().sum())
    return (valid / total) < _CHANNEL_SPARSE_COVERAGE_THRESHOLD


def assess_timeline_quality(df: pd.DataFrame) -> TimelineQuality:
    """
    Analyse inter-sample gaps and per-channel coverage in *df*.

    Must be called on the pre-resampled records_clean_input DataFrame so that
    gap statistics reflect the device's actual sample rate.
    """
    _default = TimelineQuality(
        timeline_is_dense_1hz=False,
        power_channel_is_low_coverage=True,
        speed_channel_is_low_coverage=True,
        median_gap_s=None,
        max_gap_s=None,
        power_valid_sample_count=0,
    )

    if "timestamp" not in df.columns or len(df) < 2:
        return _default

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return _default

    gaps = ts.diff().dropna().dt.total_seconds()
    gaps = gaps[gaps > 0]
    if gaps.empty:
        return _default

    median_gap = float(gaps.median())
    max_gap = float(gaps.max())

    is_dense = (
        median_gap <= _DENSE_MEDIAN_GAP_THRESHOLD_S
        and max_gap <= _DENSE_MAX_GAP_THRESHOLD_S
    )

    power_valid = int(df["power"].notna().sum()) if "power" in df.columns else 0

    return TimelineQuality(
        timeline_is_dense_1hz=is_dense,
        power_channel_is_low_coverage=_channel_coverage_is_low(df, "power"),
        speed_channel_is_low_coverage=_channel_coverage_is_low(df, "speed"),
        median_gap_s=median_gap,
        max_gap_s=max_gap,
        power_valid_sample_count=power_valid,
    )


# ---------------------------------------------------------------------------
# Power metrics quality — single source of truth
# ---------------------------------------------------------------------------


def _determine_power_metrics_quality(quality: TimelineQuality) -> Optional[str]:
    """
    Determine the power_metrics_quality category for an activity.

    This is the single source of truth for the three-way power quality decision.
    All downstream code (compute_power_metrics, best_efforts gating, notes,
    zone series) must derive their behaviour from this value.

    Returns one of _PMQ_FULL, _PMQ_SPARSE_WARNING, _PMQ_WITHHELD, or None.

    None:
        No power channel present in the activity (power_valid_sample_count == 0
        AND column absent).  Quality assessment does not apply.

    _PMQ_FULL ("full"):
        Dense timeline — standard algorithm, full reliability.
        OR sparse timeline with power_channel_is_low_coverage=False — adequate
        coverage despite smart recording gaps.

    _PMQ_SPARSE_WARNING ("sparse_warning"):
        Sparse timeline, low-coverage power channel, but ≥
        _SPARSE_POWER_MIN_VALID_SAMPLES valid samples.
        Metrics are computable but may overestimate load.

    _PMQ_WITHHELD ("withheld"):
        Sparse timeline, low-coverage power channel, and <
        _SPARSE_POWER_MIN_VALID_SAMPLES valid samples.
        Metrics cannot be reliably computed; withheld.
    """
    if quality.power_valid_sample_count == 0:
        return None

    if quality.timeline_is_dense_1hz or not quality.power_channel_is_low_coverage:
        return _PMQ_FULL

    if quality.power_valid_sample_count >= _SPARSE_POWER_MIN_VALID_SAMPLES:
        return _PMQ_SPARSE_WARNING

    return _PMQ_WITHHELD


# ---------------------------------------------------------------------------
# Cycling metric functions
# ---------------------------------------------------------------------------


def rolling_mean_best(values: List[Optional[float]], seconds: int) -> Optional[float]:
    """
    Highest mean power over any fully-populated window of *seconds* length.

    A window qualifies only when every slot is non-NaN (directly measured).
    """
    if seconds <= 0:
        return None
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    if len(arr) < seconds:
        return None
    valid = ~np.isnan(arr)
    filled = np.where(valid, arr, 0.0)
    csum = np.cumsum(np.insert(filled, 0, 0.0))
    vcum = np.cumsum(np.insert(valid.astype(int), 0, 0))
    wsum = csum[seconds:] - csum[:-seconds]
    wcnt = vcum[seconds:] - vcum[:-seconds]
    complete = wcnt == seconds
    if not complete.any():
        return None
    return float(np.max(wsum[complete] / seconds))


def normalized_power(
    power_1hz: List[Optional[float]],
    timeline_is_dense: bool,
) -> Optional[float]:
    """
    Coggan NP: 30 s rolling mean → 4th-power mean → 4th root.

    Dense (timeline_is_dense=True):  NaN → 0 W (standard Coggan).
    Sparse (timeline_is_dense=False): NaN skipped (gaps-skipped algorithm).
    Returns None when fewer than 30 valid samples are present.
    """
    arr = np.array([np.nan if v is None else float(v) for v in power_1hz], dtype=float)
    if int(np.sum(~np.isnan(arr))) < 30:
        return None
    if timeline_is_dense:
        arr = np.where(np.isnan(arr), 0.0, arr)
        roll = np.convolve(arr, np.ones(30) / 30, mode="valid")
        return float(np.mean(roll ** 4) ** 0.25)
    s = pd.Series(arr)
    roll = s.rolling(30, min_periods=25).mean().dropna()
    if len(roll) < 1:
        return None
    return float((roll ** 4).mean() ** 0.25)


def variability_index(np_w: Optional[float], avg_w: Optional[float]) -> Optional[float]:
    """NP / avg_power.  None when either is absent or avg_w is zero."""
    if np_w is None or avg_w is None or avg_w == 0:
        return None
    return float(np_w / avg_w)


def intensity_factor(np_w: Optional[float], ftp: Optional[float]) -> Optional[float]:
    """NP / FTP.  None when either is absent or ftp is zero."""
    if np_w is None or ftp is None or ftp == 0:
        return None
    return float(np_w / ftp)


def training_stress_score(
    duration_s: Optional[float],
    np_w: Optional[float],
    ftp: Optional[float],
) -> Optional[float]:
    """Coggan TSS.  None when any required input is absent or ftp is zero."""
    if duration_s is None or np_w is None or ftp is None or ftp == 0:
        return None
    return float((duration_s * np_w * (np_w / ftp)) / (ftp * 3600.0) * 100.0)


def trimp_score(
    hr_1hz: List[Optional[float]],
    resting_hr: Optional[float],
    max_hr: Optional[float],
    sex: str = "male",
) -> Optional[float]:
    """Banister TRIMP from a 1 Hz HR series.  None when HR bounds absent."""
    if resting_hr is None or max_hr is None:
        return None
    hrr_range = max_hr - resting_hr
    if hrr_range <= 0:
        return None
    k = 1.92 if sex != "female" else 1.67
    total = 0.0
    count = 0
    for v in hr_1hz:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f):
            continue
        hrr = max(0.0, min(1.0, (f - resting_hr) / hrr_range))
        total += hrr * math.exp(k * hrr)
        count += 1
    return float(total / 60.0) if count > 0 else None


def calculate_pw_hr_decoupling(df_1hz: pd.DataFrame) -> Optional[float]:
    """
    Simplified Pw:Hr decoupling — first-half vs second-half ratio.

    Limitations: 50/50 split by paired row count, not elapsed time.
    Warm-up / cool-down / intervals distort the boundary.
    Returns None when fewer than 60 valid paired samples exist.
    """
    if df_1hz.empty or "power" not in df_1hz.columns or "heart_rate" not in df_1hz.columns:
        return None
    df = df_1hz[["power", "heart_rate"]].dropna()
    df = df[(df["power"] > 0) & (df["heart_rate"] > 0)]
    if len(df) < 60:
        return None
    mid = len(df) // 2
    r1 = df.iloc[:mid]["power"].mean() / df.iloc[:mid]["heart_rate"].mean()
    r2 = df.iloc[mid:]["power"].mean() / df.iloc[mid:]["heart_rate"].mean()
    if r1 == 0:
        return None
    return float((r2 - r1) / r1 * 100.0)


def time_in_power_zones(
    df_1hz: pd.DataFrame,
    ftp: Optional[float],
) -> Dict[str, Optional[float]]:
    """Seconds in each Coggan 7-zone model.  None values when ftp absent."""
    names = [
        "z1_recovery_sec", "z2_endurance_sec", "z3_tempo_sec",
        "z4_threshold_sec", "z5_vo2_sec", "z6_anaerobic_sec",
        "z7_neuromuscular_sec",
    ]
    null_result: Dict[str, Optional[float]] = dict.fromkeys(names, None)
    if ftp is None or ftp == 0 or df_1hz.empty or "power" not in df_1hz.columns:
        return null_result
    power = df_1hz["power"].dropna()
    bounds = [0.0, 0.55, 0.75, 0.90, 1.05, 1.20, 1.50, float("inf")]
    ratios = power / ftp
    return {
        name: float(((ratios >= bounds[i]) & (ratios < bounds[i + 1])).sum())
        for i, name in enumerate(names)
    }


def time_in_hr_zones(
    df_1hz: pd.DataFrame,
    lthr: Optional[float],
    max_hr: Optional[float],
) -> Dict[str, Optional[float]]:
    """Seconds in each Friel 5-zone HR model.  None values when ref absent."""
    names = ["hr_z1_sec", "hr_z2_sec", "hr_z3_sec", "hr_z4_sec", "hr_z5_sec"]
    null_result: Dict[str, Optional[float]] = dict.fromkeys(names, None)
    ref = lthr if lthr is not None else max_hr
    if ref is None or ref == 0 or df_1hz.empty or "heart_rate" not in df_1hz.columns:
        return null_result
    hr = df_1hz["heart_rate"].dropna()
    bounds = [0.0, 0.81, 0.89, 0.94, 1.00, float("inf")]
    ratios = hr / ref
    return {
        name: float(((ratios >= bounds[i]) & (ratios < bounds[i + 1])).sum())
        for i, name in enumerate(names)
    }


def derive_grade_from_records(
    df: pd.DataFrame,
    is_indoor: Optional[bool],
    elevation_threshold_m: float = 0.5,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute (avg_grade_pct, max_grade_pct_p99, elevation_gain_m).

    Returns (None, None, None) unless is_indoor is explicitly False.
    """
    if is_indoor is not False:
        return None, None, None
    if not {"distance", "altitude"}.issubset(df.columns):
        return None, None, None
    x = df[["distance", "altitude"]].dropna().copy()
    if len(x) < 5:
        return None, None, None
    if (x["altitude"].max() - x["altitude"].min()) < 1.0:
        return None, None, None
    x["d_dist"] = x["distance"].diff()
    x["d_alt"] = x["altitude"].diff()
    x = x[
        (x["d_dist"] > 1.0) & (x["d_dist"] < 100.0) & (x["d_alt"].abs() < 20.0)
    ]
    if x.empty:
        return None, None, None
    x["grade_pct"] = x["d_alt"] / x["d_dist"] * 100.0
    x = x[np.isfinite(x["grade_pct"])]
    if x.empty:
        return None, None, None
    return (
        float(x["grade_pct"].mean()),
        float(x["grade_pct"].quantile(0.99)),
        float(x.loc[x["d_alt"] > elevation_threshold_m, "d_alt"].sum()),
    )


# ---------------------------------------------------------------------------
# Strongly typed domain models
# ---------------------------------------------------------------------------


@dataclass
class LapRow:
    """One lap (or interval / segment) from a FIT lap message."""
    source_file: str
    athlete: Optional[str]
    export_schema_version: str
    start_time: Optional[str]
    timestamp: Optional[str]
    event: Optional[str]
    event_type: Optional[str]
    lap_trigger: Optional[str]
    total_elapsed_time_s: Optional[float]
    total_timer_time_s: Optional[float]
    total_distance_m: Optional[float]
    total_distance_km: Optional[float]
    avg_speed_mps: Optional[float]
    avg_speed_kmh: Optional[float]
    max_speed_mps: Optional[float]
    max_speed_kmh: Optional[float]
    avg_heart_rate_bpm: Optional[float]
    max_heart_rate_bpm: Optional[float]
    avg_power_w: Optional[float]
    max_power_w: Optional[float]
    normalized_power_w: Optional[float]
    avg_cadence_rpm: Optional[float]
    max_cadence_rpm: Optional[float]
    total_ascent_m: Optional[float]
    total_descent_m: Optional[float]
    intensity: Optional[str]
    duration_hms: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class WorkoutStepRow:
    """One planned step from a FIT workout_step message."""
    source_file: str
    athlete: Optional[str]
    export_schema_version: str
    step_index: Optional[int]
    wkt_name: Optional[str]
    duration_type: Optional[str]
    duration_value: Optional[float]
    target_type: Optional[str]
    target_low: Optional[float]
    target_high: Optional[float]
    intensity: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class BestEffortRow:
    """
    Rolling best-power result for a given duration.

    value_w is None when:
    - no power data exists,
    - no fully-populated window of the required length is present,
    - or power_metrics_quality == "withheld".
    """
    source_file: str
    athlete: Optional[str]
    export_schema_version: str
    metric: str
    seconds: int
    value_w: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class FileInventoryRow:
    """
    Processing outcome for a single source file.

    status       : "ok" | "ok_no_timeline" | "error"
    error_class  : exception class name (only set when status == "error")
    error_stage  : pipeline stage name (only set when status == "error")
    status_message:
        None when status == "ok".
        Human-readable reason when status == "ok_no_timeline" or "error".
    """
    source_file: str
    export_schema_version: str
    status: str
    error_class: Optional[str]
    error_stage: Optional[str]
    status_message: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RecordExtractedRow:
    """
    One raw sample extracted verbatim from a FIT record message.

    Verbatim layer — values exactly as received from fitdecode.
    Enhanced channels stored as separate fields; NO merging here.
    """
    source_file: str
    athlete: Optional[str]
    export_schema_version: str
    timestamp: Optional[str]
    heart_rate: Optional[float]
    cadence: Optional[float]
    power: Optional[float]
    speed: Optional[float]
    distance: Optional[float]
    altitude: Optional[float]
    temperature: Optional[float]
    latitude_deg: Optional[float]
    longitude_deg: Optional[float]
    grade: Optional[float]
    vertical_speed: Optional[float]
    left_right_balance: Optional[float]
    fractional_cadence: Optional[float]
    enhanced_speed: Optional[float]
    enhanced_altitude: Optional[float]
    gps_accuracy: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Record1HzRow:
    """
    One 1 Hz resampled record with imputation flags and derived coach columns.

    ── Imputation policy ─────────────────────────────────────────────────────
    Dense (timeline_is_dense_1hz=True):
      power, cadence, speed : ffill ≤ 2 s → 0.0
      heart_rate            : ffill ≤ 5 s → NaN
      distance, altitude    : time-interpolated
      temperature           : ffill (no limit)

    Sparse (timeline_is_dense_1hz=False):
      power, cadence, speed : ffill ≤ 2 s → NaN (NOT zeroed)
      heart_rate            : ffill ≤ 5 s → NaN
      distance, altitude    : time-interpolated
      temperature           : ffill (no limit)

    ── Imputation flags ──────────────────────────────────────────────────────
    power_imputed / hr_imputed / speed_imputed:
        True when value was NaN before imputation AND is now non-NaN.

    ── Timeline quality flags (activity-level, same for all rows) ────────────
    timeline_is_dense_1hz        — overall timeline density
    power_channel_is_low_coverage — heuristic coverage signal for power
    speed_channel_is_low_coverage — heuristic coverage signal for speed
    See module docstring for full semantics.

    ── Derived coach columns ─────────────────────────────────────────────────
    power_zone: None when FTP absent OR power_channel_is_low_coverage is True
                (per-second zone assignment unreliable on low-coverage channels)
    """
    source_file: str
    athlete: Optional[str]
    export_schema_version: str
    timestamp: Optional[str]
    heart_rate: Optional[float]
    cadence: Optional[float]
    power: Optional[float]
    speed: Optional[float]
    distance: Optional[float]
    altitude: Optional[float]
    temperature: Optional[float]
    latitude_deg: Optional[float]
    longitude_deg: Optional[float]
    power_imputed: bool
    hr_imputed: bool
    speed_imputed: bool
    timeline_is_dense_1hz: bool
    power_channel_is_low_coverage: bool
    speed_channel_is_low_coverage: bool
    elapsed_time_s: Optional[float]
    distance_km: Optional[float]
    speed_kmh: Optional[float]
    is_moving: int
    power_zone: Optional[int]
    hr_zone: Optional[int]
    lap_index: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class FitActivity:
    """
    All session-level metrics for a single activity.

    ── Power metrics quality ─────────────────────────────────────────────────
    power_metrics_quality : Optional[str]
        Machine-readable categorical field.  See module docstring for values
        and their exact implications.  None when no power channel present.

    power_quality_note : Optional[str]
        Human-readable explanation, always consistent with power_metrics_quality.
        None when power_metrics_quality == "full" or None.

    Both fields are the authoritative source of truth for what was and was
    not computed for this activity's power metrics.
    """
    source_file: str
    export_schema_version: str
    athlete: Optional[str] = None
    file_size_bytes: Optional[int] = None
    manufacturer: Optional[str] = None
    product: Optional[str] = None
    serial_number: Optional[str] = None
    time_created: Optional[str] = None
    sport: Optional[str] = None
    sub_sport: Optional[str] = None
    activity_type: Optional[str] = None
    event: Optional[str] = None
    start_time: Optional[str] = None
    local_start_time: Optional[str] = None
    total_elapsed_time_s: Optional[float] = None
    total_timer_time_s: Optional[float] = None
    moving_time_s: Optional[float] = None
    total_distance_m: Optional[float] = None
    total_ascent_m: Optional[float] = None
    total_descent_m: Optional[float] = None
    elevation_gain_from_records_m: Optional[float] = None
    avg_grade_pct: Optional[float] = None
    max_grade_pct: Optional[float] = None
    avg_speed_mps: Optional[float] = None
    max_speed_mps: Optional[float] = None
    avg_heart_rate_bpm: Optional[float] = None
    max_heart_rate_bpm: Optional[float] = None
    avg_power_w: Optional[float] = None
    max_power_w: Optional[float] = None
    normalized_power_w: Optional[float] = None
    normalized_power_moving_w: Optional[float] = None
    avg_cadence_rpm: Optional[float] = None
    max_cadence_rpm: Optional[float] = None
    avg_cadence_no_zeros_rpm: Optional[float] = None
    avg_temperature_c: Optional[float] = None
    max_temperature_c: Optional[float] = None
    calories_kcal: Optional[float] = None
    training_stress_score: Optional[float] = None
    intensity_factor: Optional[float] = None
    variability_index: Optional[float] = None
    work_kj: Optional[float] = None
    pw_hr_decoupling_pct: Optional[float] = None
    trimp: Optional[float] = None
    left_right_balance_pct: Optional[float] = None
    balance_left_pct: Optional[float] = None
    balance_right_pct: Optional[float] = None
    pedal_smoothness_pct: Optional[float] = None
    torque_effectiveness_pct: Optional[float] = None
    trainer_mode: Optional[str] = None
    indoor: Optional[bool] = None
    indoor_source: Optional[str] = None
    has_structured_workout: bool = False
    workout_step_count: int = 0
    ftp_used_w: Optional[float] = None
    ftp_source: Optional[str] = None
    estimated_ftp_w: Optional[float] = None
    lthr_used_bpm: Optional[float] = None
    resting_hr_used_bpm: Optional[float] = None
    max_hr_used_bpm: Optional[float] = None
    hr_zone_source: Optional[str] = None
    has_power: bool = False
    has_hr: bool = False
    has_cadence: bool = False
    has_gps: bool = False
    has_speed: bool = False
    is_complete_for_power_analysis: bool = False
    is_complete_for_hr_analysis: bool = False
    timeline_is_dense_1hz: bool = False
    power_channel_is_low_coverage: bool = False
    speed_channel_is_low_coverage: bool = False
    timeline_median_gap_s: Optional[float] = None
    timeline_max_gap_s: Optional[float] = None
    power_metrics_quality: Optional[str] = None   # "full" | "sparse_warning" | "withheld" | None
    power_quality_note: Optional[str] = None
    record_count: int = 0
    lap_count: int = 0
    z1_recovery_sec: Optional[float] = None
    z2_endurance_sec: Optional[float] = None
    z3_tempo_sec: Optional[float] = None
    z4_threshold_sec: Optional[float] = None
    z5_vo2_sec: Optional[float] = None
    z6_anaerobic_sec: Optional[float] = None
    z7_neuromuscular_sec: Optional[float] = None
    hr_z1_sec: Optional[float] = None
    hr_z2_sec: Optional[float] = None
    hr_z3_sec: Optional[float] = None
    hr_z4_sec: Optional[float] = None
    hr_z5_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["total_distance_km"] = meters_to_km(self.total_distance_m)
        d["avg_speed_kmh"] = ms_to_kmh(self.avg_speed_mps)
        d["max_speed_kmh"] = ms_to_kmh(self.max_speed_mps)
        d["duration_hms"] = seconds_to_hms(self.total_elapsed_time_s)
        d["timer_hms"] = seconds_to_hms(self.total_timer_time_s)
        d["moving_time_hms"] = seconds_to_hms(self.moving_time_s)
        return d


# ---------------------------------------------------------------------------
# FIT field helpers
# ---------------------------------------------------------------------------


FIELD_ALIASES: Dict[str, str] = {
    "position_lat": "position_lat_semicircles",
    "position_long": "position_long_semicircles",
}


def semicircles_to_degrees(value: Optional[float]) -> Optional[float]:
    """Convert FIT semicircle coordinates to decimal degrees."""
    return None if value is None else float(value) * (180.0 / 2**31)


def extract_fields(msg: fitdecode.records.FitDataMessage) -> Dict[str, Any]:
    """Extract all fields from a FIT data message into a plain dict."""
    out: Dict[str, Any] = {}
    for f in msg.fields:
        if not f.name:
            continue
        out[f.name] = f.value
        alias = FIELD_ALIASES.get(f.name)
        if alias:
            out[alias] = f.value
    return out


def _str_field(fields: Dict[str, Any], key: str) -> Optional[str]:
    v = fields.get(key)
    return str(v) if v is not None else None


# ---------------------------------------------------------------------------
# FIT message handlers
# ---------------------------------------------------------------------------


def _handle_file_id(activity: FitActivity, fields: Dict[str, Any]) -> None:
    activity.time_created = to_iso(fields.get("time_created"))
    activity.manufacturer = _str_field(fields, "manufacturer")
    activity.product = _str_field(fields, "product")
    activity.serial_number = _str_field(fields, "serial_number")


def _handle_device_info(activity: FitActivity, fields: Dict[str, Any]) -> None:
    if activity.manufacturer is None:
        activity.manufacturer = _str_field(fields, "manufacturer")
    if activity.product is None:
        activity.product = _str_field(fields, "product")
    if activity.serial_number is None:
        activity.serial_number = _str_field(fields, "serial_number")


def _handle_sport(activity: FitActivity, fields: Dict[str, Any]) -> None:
    if fields.get("sport") is not None:
        activity.sport = str(fields["sport"])
    if fields.get("sub_sport") is not None:
        activity.sub_sport = str(fields["sub_sport"])


def _handle_session(activity: FitActivity, fields: Dict[str, Any]) -> None:
    if activity.sport is None:
        activity.sport = _str_field(fields, "sport")
    if activity.sub_sport is None:
        activity.sub_sport = _str_field(fields, "sub_sport")
    activity.start_time = to_iso(fields.get("start_time"))
    activity.local_start_time = to_iso(fields.get("local_timestamp"))
    activity.total_elapsed_time_s = safe_float(fields.get("total_elapsed_time"))
    activity.total_timer_time_s = safe_float(fields.get("total_timer_time"))
    activity.total_distance_m = safe_float(fields.get("total_distance"))
    activity.total_ascent_m = safe_float(fields.get("total_ascent"))
    activity.total_descent_m = safe_float(fields.get("total_descent"))
    activity.avg_speed_mps = safe_float(fields.get("avg_speed"))
    activity.max_speed_mps = safe_float(fields.get("max_speed"))
    activity.avg_heart_rate_bpm = safe_float(fields.get("avg_heart_rate"))
    activity.max_heart_rate_bpm = safe_float(fields.get("max_heart_rate"))
    activity.avg_power_w = safe_float(fields.get("avg_power"))
    activity.max_power_w = safe_float(fields.get("max_power"))
    activity.avg_cadence_rpm = safe_float(fields.get("avg_cadence"))
    activity.max_cadence_rpm = safe_float(fields.get("max_cadence"))
    activity.avg_temperature_c = safe_float(fields.get("avg_temperature"))
    activity.max_temperature_c = safe_float(fields.get("max_temperature"))
    activity.calories_kcal = safe_float(fields.get("total_calories"))
    activity.training_stress_score = safe_float(fields.get("training_stress_score"))
    activity.normalized_power_w = safe_float(fields.get("normalized_power"))
    raw_balance = safe_float(fields.get("left_right_balance"))
    activity.left_right_balance_pct = raw_balance
    if raw_balance is not None:
        raw_int = int(raw_balance)
        left_pct = float((raw_int & 0x7F) if raw_int >= 128 else raw_int)
        left_pct = max(0.0, min(100.0, left_pct))
        activity.balance_left_pct = left_pct
        activity.balance_right_pct = 100.0 - left_pct
    activity.pedal_smoothness_pct = safe_float(fields.get("avg_left_pedal_smoothness"))
    activity.torque_effectiveness_pct = safe_float(fields.get("avg_left_torque_effectiveness"))


def _handle_activity(activity: FitActivity, fields: Dict[str, Any]) -> None:
    if fields.get("type") is not None:
        activity.activity_type = str(fields["type"])
    if fields.get("event") is not None:
        activity.event = str(fields["event"])
    if activity.local_start_time is None:
        activity.local_start_time = to_iso(fields.get("local_timestamp"))
    if activity.total_timer_time_s is None:
        activity.total_timer_time_s = safe_float(fields.get("total_timer_time"))


def _handle_event(activity: FitActivity, fields: Dict[str, Any]) -> None:
    ev = fields.get("event")
    ev_type = fields.get("event_type")
    if str(ev).lower() == "timer" and str(ev_type).lower() == "start":
        if activity.start_time is None:
            activity.start_time = to_iso(fields.get("timestamp"))
    trainer_val = _str_field(fields, "data")
    if trainer_val is not None and activity.trainer_mode is None:
        activity.trainer_mode = trainer_val


def _build_lap_row(
    source_file: str,
    athlete: Optional[str],
    fields: Dict[str, Any],
) -> LapRow:
    elapsed = safe_float(fields.get("total_elapsed_time"))
    dist_m = safe_float(fields.get("total_distance"))
    spd_mps = safe_float(fields.get("avg_speed"))
    max_spd_mps = safe_float(fields.get("max_speed"))
    return LapRow(
        source_file=source_file,
        athlete=athlete,
        export_schema_version=EXPORT_SCHEMA_VERSION,
        start_time=to_iso(fields.get("start_time")),
        timestamp=to_iso(fields.get("timestamp")),
        event=_str_field(fields, "event"),
        event_type=_str_field(fields, "event_type"),
        lap_trigger=_str_field(fields, "lap_trigger"),
        total_elapsed_time_s=elapsed,
        total_timer_time_s=safe_float(fields.get("total_timer_time")),
        total_distance_m=dist_m,
        total_distance_km=meters_to_km(dist_m),
        avg_speed_mps=spd_mps,
        avg_speed_kmh=ms_to_kmh(spd_mps),
        max_speed_mps=max_spd_mps,
        max_speed_kmh=ms_to_kmh(max_spd_mps),
        avg_heart_rate_bpm=safe_float(fields.get("avg_heart_rate")),
        max_heart_rate_bpm=safe_float(fields.get("max_heart_rate")),
        avg_power_w=safe_float(fields.get("avg_power")),
        max_power_w=safe_float(fields.get("max_power")),
        normalized_power_w=safe_float(fields.get("normalized_power")),
        avg_cadence_rpm=safe_float(fields.get("avg_cadence")),
        max_cadence_rpm=safe_float(fields.get("max_cadence")),
        total_ascent_m=safe_float(fields.get("total_ascent")),
        total_descent_m=safe_float(fields.get("total_descent")),
        intensity=_str_field(fields, "intensity"),
        duration_hms=seconds_to_hms(elapsed),
    )


def _build_record_extracted_row(
    source_file: str,
    athlete: Optional[str],
    fields: Dict[str, Any],
) -> RecordExtractedRow:
    """Build one RecordExtractedRow verbatim — no channel merging here."""
    lat_sc = safe_float(fields.get("position_lat"))
    lon_sc = safe_float(fields.get("position_long"))
    return RecordExtractedRow(
        source_file=source_file,
        athlete=athlete,
        export_schema_version=EXPORT_SCHEMA_VERSION,
        timestamp=to_iso(fields.get("timestamp")),
        heart_rate=safe_float(fields.get("heart_rate")),
        cadence=safe_float(fields.get("cadence")),
        power=safe_float(fields.get("power")),
        speed=safe_float(fields.get("speed")),
        distance=safe_float(fields.get("distance")),
        altitude=safe_float(fields.get("altitude")),
        temperature=safe_float(fields.get("temperature")),
        latitude_deg=semicircles_to_degrees(lat_sc),
        longitude_deg=semicircles_to_degrees(lon_sc),
        grade=safe_float(fields.get("grade")),
        vertical_speed=safe_float(fields.get("vertical_speed")),
        left_right_balance=safe_float(fields.get("left_right_balance")),
        fractional_cadence=safe_float(fields.get("fractional_cadence")),
        enhanced_speed=safe_float(fields.get("enhanced_speed")),
        enhanced_altitude=safe_float(fields.get("enhanced_altitude")),
        gps_accuracy=safe_float(fields.get("gps_accuracy")),
    )


def _build_workout_step_row(
    source_file: str,
    athlete: Optional[str],
    fields: Dict[str, Any],
) -> WorkoutStepRow:
    return WorkoutStepRow(
        source_file=source_file,
        athlete=athlete,
        export_schema_version=EXPORT_SCHEMA_VERSION,
        step_index=safe_int(fields.get("message_index")),
        wkt_name=_str_field(fields, "wkt_step_name"),
        duration_type=_str_field(fields, "duration_type"),
        duration_value=safe_float(fields.get("duration_value")),
        target_type=_str_field(fields, "target_type"),
        target_low=safe_float(fields.get("target_value_low")),
        target_high=safe_float(fields.get("target_value_high")),
        intensity=_str_field(fields, "intensity"),
    )


# ---------------------------------------------------------------------------
# Step 1 — extract_fit_messages
# ---------------------------------------------------------------------------


def extract_fit_messages(
    file_path: Path,
    athlete: Optional[str],
    activity: FitActivity,
) -> Tuple[List[LapRow], List[WorkoutStepRow], List[RecordExtractedRow]]:
    """
    Read the FIT file and dispatch each message to its handler.

    Returns (laps, workout_steps, records_extracted).
    Side-effects: populates *activity* header fields.
    Raises: any fitdecode exception — caller wraps in try/except.
    """
    laps: List[LapRow] = []
    workout_steps: List[WorkoutStepRow] = []
    records_extracted: List[RecordExtractedRow] = []
    device_info_seen = False

    with fitdecode.FitReader(str(file_path)) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            name = frame.name
            fields = extract_fields(frame)

            if name == "file_id":
                _handle_file_id(activity, fields)
            elif name == "device_info" and not device_info_seen:
                device_info_seen = True
                _handle_device_info(activity, fields)
            elif name == "sport":
                _handle_sport(activity, fields)
            elif name == "session":
                _handle_session(activity, fields)
            elif name == "activity":
                _handle_activity(activity, fields)
            elif name == "event":
                _handle_event(activity, fields)
            elif name == "lap":
                laps.append(_build_lap_row(file_path.name, athlete, fields))
            elif name == "record":
                records_extracted.append(
                    _build_record_extracted_row(file_path.name, athlete, fields)
                )
            elif name == "workout_step":
                activity.has_structured_workout = True
                activity.workout_step_count += 1
                workout_steps.append(
                    _build_workout_step_row(file_path.name, athlete, fields)
                )

    return laps, workout_steps, records_extracted


# ---------------------------------------------------------------------------
# Step 2 — records_clean_input preparation
# ---------------------------------------------------------------------------


def _apply_enhanced_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce records_clean_input from a records_extracted DataFrame.

    For each (base, enhanced) pair:
    - Both present: fill NaN gaps in base from enhanced.
    - Only enhanced: create base column from enhanced.
    - Only base: unchanged.
    - Both absent: column remains absent.

    Applied to speed / altitude only.  Returns a separate copy.
    """
    df = df.copy()
    for base, enhanced in [("speed", "enhanced_speed"), ("altitude", "enhanced_altitude")]:
        if enhanced not in df.columns:
            continue
        if base in df.columns:
            df[base] = df[base].fillna(df[enhanced])
        else:
            df[base] = df[enhanced]
    return df


# ---------------------------------------------------------------------------
# Step 3 — build_1hz_timeline
# ---------------------------------------------------------------------------


def _prepare_for_resample(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Parse timestamps, drop invalid rows, sort.  Returns None when empty."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df if not df.empty else None


def _apply_effort_channel_imputation(
    df: pd.DataFrame,
    quality: TimelineQuality,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Impute effort channels in-place; return pre-imputation NaN masks.

    Returns (power_was_nan, hr_was_nan, speed_was_nan).
    """
    power_was_nan = (
        df["power"].isna() if "power" in df.columns
        else pd.Series(False, index=df.index)
    )
    hr_was_nan = (
        df["heart_rate"].isna() if "heart_rate" in df.columns
        else pd.Series(False, index=df.index)
    )
    speed_was_nan = (
        df["speed"].isna() if "speed" in df.columns
        else pd.Series(False, index=df.index)
    )

    for col in ["power", "cadence", "speed"]:
        if col in df.columns:
            df[col] = df[col].ffill(limit=_EFFORT_DROPOUT_FFILL_LIMIT_S)

    if quality.timeline_is_dense_1hz:
        for col in ["power", "cadence", "speed"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

    if "heart_rate" in df.columns:
        df["heart_rate"] = df["heart_rate"].ffill(limit=_HR_FFILL_LIMIT_S)

    for col in ["distance", "altitude"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="time")

    if "temperature" in df.columns:
        df["temperature"] = df["temperature"].ffill()

    return power_was_nan, hr_was_nan, speed_was_nan


def _attach_imputation_and_quality_flags(
    df: pd.DataFrame,
    power_was_nan: pd.Series,
    hr_was_nan: pd.Series,
    speed_was_nan: pd.Series,
    quality: TimelineQuality,
) -> None:
    """Attach imputation and quality flags to *df* in-place."""

    def _filled(was_nan: pd.Series, col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return was_nan.reindex(df.index, fill_value=False) & df[col].notna()

    hr_still_nan = (
        df["heart_rate"].isna() if "heart_rate" in df.columns
        else pd.Series(True, index=df.index)
    )
    df["power_imputed"] = _filled(power_was_nan, "power")
    df["hr_imputed"] = hr_was_nan.reindex(df.index, fill_value=False) & ~hr_still_nan
    df["speed_imputed"] = _filled(speed_was_nan, "speed")
    df["timeline_is_dense_1hz"] = quality.timeline_is_dense_1hz
    df["power_channel_is_low_coverage"] = quality.power_channel_is_low_coverage
    df["speed_channel_is_low_coverage"] = quality.speed_channel_is_low_coverage


def build_1hz_timeline(records_clean_input: pd.DataFrame) -> pd.DataFrame:
    """
    Resample records_clean_input to a true 1 Hz timeline.

    ── Input contract ────────────────────────────────────────────────────────
    - Must have a ``timestamp`` column.
    - Enhanced channels must already be merged (via _apply_enhanced_channels).

    ── Duplicate timestamp policy ────────────────────────────────────────────
    Multiple records at the same second → keep LAST (by sort order).

    ── Controlled no-timeline outcomes (empty DataFrame returned) ────────────
    These are controlled outcomes, NOT exceptions:
      1. records_clean_input is empty.
      2. No ``timestamp`` column.
      3. All timestamps invalid / unparseable.
      4. Empty after deduplication or resampling.
    Callers must check emptiness and set inventory.status = "ok_no_timeline".
    """
    if records_clean_input.empty or "timestamp" not in records_clean_input.columns:
        return pd.DataFrame()

    prepared = _prepare_for_resample(records_clean_input)
    if prepared is None:
        return pd.DataFrame()

    quality = assess_timeline_quality(prepared)
    prepared = prepared.drop_duplicates(subset=["timestamp"], keep="last")
    df = prepared.set_index("timestamp").resample("1s").asfreq()

    power_was_nan, hr_was_nan, speed_was_nan = _apply_effort_channel_imputation(df, quality)
    _attach_imputation_and_quality_flags(df, power_was_nan, hr_was_nan, speed_was_nan, quality)

    return df.reset_index()


# ---------------------------------------------------------------------------
# Step 4 — detect_indoor
# ---------------------------------------------------------------------------


def detect_indoor(
    records_df: pd.DataFrame,
    sub_sport: Optional[str],
    has_structured_workout: bool,
    trainer_mode: Optional[str],
) -> Tuple[Optional[bool], str]:
    """
    Classify activity as indoor / outdoor / unknown.

    Returns (is_indoor, detection_source).
    Absence of GPS columns is NOT an indoor signal.
    """
    if sub_sport is not None:
        sub = sub_sport.lower()
        if any(k in sub for k in ("indoor", "trainer", "virtual", "spin")):
            return True, "sub_sport"
        if any(k in sub for k in ("road", "mountain", "gravel", "cyclocross")):
            return False, "sub_sport"

    if trainer_mode is not None:
        return True, "trainer_mode_event"

    gps_cols = [c for c in ["latitude_deg", "longitude_deg"] if c in records_df.columns]
    gps_data_present = bool(gps_cols) and not records_df[gps_cols].dropna().empty

    if has_structured_workout and bool(gps_cols) and not gps_data_present:
        return True, "structured_workout_no_gps"

    if gps_data_present:
        return False, "gps_data_present"

    return None, "unknown"


# ---------------------------------------------------------------------------
# Step 5 — compute_session_metrics and helpers
# ---------------------------------------------------------------------------


def _build_moving_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    """Boolean mask: speed > 0.5 m/s OR power > 10 W.  None when no channels."""
    has_spd = "speed" in df.columns
    has_pwr = "power" in df.columns
    if has_spd and has_pwr:
        return (df["speed"].fillna(0) > 0.5) | (df["power"].fillna(0) > 10)
    if has_spd:
        return df["speed"].fillna(0) > 0.5
    if has_pwr:
        return df["power"].fillna(0) > 10
    return None


def _resolve_ftp(
    activity: FitActivity,
    power_1hz: List[Optional[float]],
    user_ftp: Optional[float],
) -> Optional[float]:
    """Resolve FTP; priority: user CLI → estimated from best 20'."""
    if user_ftp is not None:
        activity.ftp_used_w = user_ftp
        activity.ftp_source = "user_provided"
        return user_ftp

    best_20min = rolling_mean_best(power_1hz, 1200)
    if best_20min is not None:
        estimated = round(best_20min * 0.95, 1)
        activity.estimated_ftp_w = estimated
        activity.ftp_used_w = estimated
        activity.ftp_source = "estimated"
        return estimated

    return None


def _fill_session_fallbacks_from_records(
    activity: FitActivity,
    df_1hz: pd.DataFrame,
) -> None:
    """Fill session fields from 1 Hz records when FIT session message absent."""
    if activity.start_time is None and "timestamp" in df_1hz.columns:
        activity.start_time = to_iso(df_1hz["timestamp"].dropna().min())

    if (
        activity.total_distance_m is None
        and "distance" in df_1hz.columns
        and df_1hz["distance"].notna().any()
    ):
        activity.total_distance_m = safe_float(df_1hz["distance"].dropna().max())

    if (
        activity.total_elapsed_time_s is None
        and "timestamp" in df_1hz.columns
        and df_1hz["timestamp"].notna().any()
    ):
        ts = pd.to_datetime(df_1hz["timestamp"], errors="coerce", utc=True).dropna().sort_values()
        if len(ts) >= 2:
            activity.total_elapsed_time_s = float(
                (ts.iloc[-1] - ts.iloc[0]).total_seconds()
            )

    for col, attr_avg, attr_max in [
        ("speed", "avg_speed_mps", "max_speed_mps"),
        ("heart_rate", "avg_heart_rate_bpm", "max_heart_rate_bpm"),
        ("power", "avg_power_w", "max_power_w"),
        ("cadence", "avg_cadence_rpm", "max_cadence_rpm"),
        ("temperature", "avg_temperature_c", "max_temperature_c"),
    ]:
        if col not in df_1hz.columns:
            continue
        if getattr(activity, attr_avg) is None:
            setattr(activity, attr_avg, mean_ignore_none(df_1hz[col]))
        if getattr(activity, attr_max) is None and df_1hz[col].notna().any():
            setattr(activity, attr_max, safe_float(df_1hz[col].dropna().max()))


def compute_basic_activity_stats(
    activity: FitActivity,
    df_1hz: pd.DataFrame,
    quality: TimelineQuality,
) -> None:
    """
    Set has_* flags, moving_time_s, timeline quality, and session fallbacks.

    moving_time_s is None when speed_channel_is_low_coverage — unreliable.
    """
    _fill_session_fallbacks_from_records(activity, df_1hz)

    activity.has_power = "power" in df_1hz.columns and df_1hz["power"].gt(0).any()
    activity.has_hr = "heart_rate" in df_1hz.columns and df_1hz["heart_rate"].notna().any()
    activity.has_cadence = "cadence" in df_1hz.columns and df_1hz["cadence"].gt(0).any()
    activity.has_speed = "speed" in df_1hz.columns and df_1hz["speed"].gt(0).any()

    gps_cols = [c for c in ["latitude_deg", "longitude_deg"] if c in df_1hz.columns]
    activity.has_gps = bool(gps_cols) and not df_1hz[gps_cols].dropna().empty

    if quality.speed_channel_is_low_coverage:
        activity.moving_time_s = None
    else:
        moving_mask = _build_moving_mask(df_1hz)
        activity.moving_time_s = float(moving_mask.sum()) if moving_mask is not None else None

    activity.timeline_is_dense_1hz = quality.timeline_is_dense_1hz
    activity.power_channel_is_low_coverage = quality.power_channel_is_low_coverage
    activity.speed_channel_is_low_coverage = quality.speed_channel_is_low_coverage
    activity.timeline_median_gap_s = quality.median_gap_s
    activity.timeline_max_gap_s = quality.max_gap_s


def _compute_load_metrics(
    activity: FitActivity,
    df_1hz: pd.DataFrame,
    effective_ftp: Optional[float],
) -> None:
    """Compute VI, IF, TSS, work_kj, cadence-without-zeros.  Mutates *activity*."""
    if activity.variability_index is None:
        activity.variability_index = variability_index(
            activity.normalized_power_w, activity.avg_power_w
        )
    if activity.intensity_factor is None:
        activity.intensity_factor = intensity_factor(
            activity.normalized_power_w, effective_ftp
        )
    if activity.training_stress_score is None:
        duration = (
            activity.total_timer_time_s
            if activity.total_timer_time_s is not None
            else activity.total_elapsed_time_s
        )
        activity.training_stress_score = training_stress_score(
            duration, activity.normalized_power_w, effective_ftp
        )
    if activity.work_kj is None:
        if activity.total_timer_time_s is not None and activity.avg_power_w is not None:
            activity.work_kj = float(
                activity.avg_power_w * activity.total_timer_time_s / 1000.0
            )
        elif "power" in df_1hz.columns and not df_1hz["power"].dropna().empty:
            activity.work_kj = float(df_1hz["power"].dropna().sum() / 1000.0)

    if "cadence" in df_1hz.columns:
        cad_nz = df_1hz.loc[df_1hz["cadence"] > 0, "cadence"]
        activity.avg_cadence_no_zeros_rpm = (
            float(cad_nz.mean()) if not cad_nz.empty else None
        )


def _compute_power_zones(
    activity: FitActivity,
    df_1hz: pd.DataFrame,
    effective_ftp: Optional[float],
) -> None:
    """Compute Coggan 7-zone seconds and store on *activity*."""
    zp = time_in_power_zones(df_1hz, effective_ftp)
    activity.z1_recovery_sec = zp["z1_recovery_sec"]
    activity.z2_endurance_sec = zp["z2_endurance_sec"]
    activity.z3_tempo_sec = zp["z3_tempo_sec"]
    activity.z4_threshold_sec = zp["z4_threshold_sec"]
    activity.z5_vo2_sec = zp["z5_vo2_sec"]
    activity.z6_anaerobic_sec = zp["z6_anaerobic_sec"]
    activity.z7_neuromuscular_sec = zp["z7_neuromuscular_sec"]


def compute_power_metrics(
    activity: FitActivity,
    df_1hz: pd.DataFrame,
    user_ftp: Optional[float],
    quality: TimelineQuality,
) -> Optional[float]:
    """
    Compute power metrics, gated by power_metrics_quality.

    The quality category (from _determine_power_metrics_quality) is the single
    decision point.  All downstream behaviour derives from it:

    "full":
        Compute NP/NP_moving/IF/TSS/work/VI/zones with standard algorithm.
        power_quality_note = None.

    "sparse_warning":
        Compute with sparse-mode NP (gaps skipped, not zeroed).
        power_quality_note carries explicit overestimation warning.

    "withheld":
        Leave NP/NP_moving/IF/TSS/work/VI/zones at default None.
        power_quality_note explains what was withheld.
        effective_ftp is still resolved (FTP ≠ timeline quality).

    None (no power):
        Same as "withheld" for computation purposes; power_quality_note = None.

    Returns effective_ftp (float | None).  Mutates *activity*.
    """
    power_1hz: List[Optional[float]] = df_1hz.get(
        "power", pd.Series(dtype=float)
    ).tolist()

    effective_ftp = _resolve_ftp(activity, power_1hz, user_ftp)
    activity.is_complete_for_power_analysis = (
        activity.has_power and activity.ftp_used_w is not None
    )

    pmq = _determine_power_metrics_quality(quality)
    activity.power_metrics_quality = pmq

    if pmq == _PMQ_WITHHELD:
        activity.power_quality_note = (
            f"Power metrics withheld: low-coverage power channel with only "
            f"{quality.power_valid_sample_count} valid samples "
            f"(minimum required: {_SPARSE_POWER_MIN_VALID_SAMPLES}). "
            "Withheld: NP, NP_moving, IF, TSS, work_kj, VI, z1–z7, "
            "per-row power_zone, best_efforts.  "
            "avg_power and max_power from FIT session message are preserved."
        )
        return effective_ftp

    if pmq is None:
        # No power channel — nothing to compute.
        return effective_ftp

    # pmq is "full" or "sparse_warning" — compute metrics.
    is_dense = quality.timeline_is_dense_1hz

    if activity.normalized_power_w is None:
        activity.normalized_power_w = normalized_power(
            power_1hz, timeline_is_dense=is_dense
        )

    moving_mask = _build_moving_mask(df_1hz)
    if moving_mask is not None and moving_mask.any() and "power" in df_1hz.columns:
        activity.normalized_power_moving_w = normalized_power(
            df_1hz.loc[moving_mask, "power"].tolist(),
            timeline_is_dense=is_dense,
        )

    if pmq == _PMQ_SPARSE_WARNING:
        activity.power_quality_note = (
            f"Sparse timeline with {quality.power_valid_sample_count} valid "
            "power samples (sufficient to compute, but results may overestimate "
            "training load because coasting in unrecorded gaps is not modelled "
            "as zero watts).  NP, TSS, IF, work_kj computed with gaps-skipped "
            "algorithm."
        )

    _compute_load_metrics(activity, df_1hz, effective_ftp)
    _compute_power_zones(activity, df_1hz, effective_ftp)

    return effective_ftp


def compute_hr_metrics(
    activity: FitActivity,
    df_1hz: pd.DataFrame,
    resting_hr: Optional[float],
    max_hr: Optional[float],
    lthr: Optional[float],
) -> None:
    """Compute TRIMP, Pw:Hr decoupling, and HR zones.  Mutates *activity*."""
    activity.is_complete_for_hr_analysis = activity.has_hr and (
        lthr is not None or max_hr is not None
    )
    hr_1hz: List[Optional[float]] = df_1hz.get(
        "heart_rate", pd.Series(dtype=float)
    ).tolist()
    activity.trimp = trimp_score(hr_1hz, resting_hr, max_hr)
    activity.pw_hr_decoupling_pct = calculate_pw_hr_decoupling(df_1hz)

    zh = time_in_hr_zones(df_1hz, lthr, max_hr)
    activity.hr_z1_sec = zh["hr_z1_sec"]
    activity.hr_z2_sec = zh["hr_z2_sec"]
    activity.hr_z3_sec = zh["hr_z3_sec"]
    activity.hr_z4_sec = zh["hr_z4_sec"]
    activity.hr_z5_sec = zh["hr_z5_sec"]


def compute_grade_metrics(activity: FitActivity, df_1hz: pd.DataFrame) -> None:
    """Compute grade / elevation gain.  Only runs when indoor is False."""
    avg_grade, max_grade, elev_gain = derive_grade_from_records(df_1hz, activity.indoor)
    activity.avg_grade_pct = avg_grade
    activity.max_grade_pct = max_grade
    activity.elevation_gain_from_records_m = elev_gain


def compute_session_metrics(
    activity: FitActivity,
    df_1hz: pd.DataFrame,
    user_ftp: Optional[float],
    resting_hr: Optional[float],
    max_hr: Optional[float],
    lthr: Optional[float],
    quality: TimelineQuality,
) -> Optional[float]:
    """Orchestrate session-level metrics.  Returns effective_ftp."""
    compute_basic_activity_stats(activity, df_1hz, quality)
    effective_ftp = compute_power_metrics(activity, df_1hz, user_ftp, quality)
    compute_hr_metrics(activity, df_1hz, resting_hr, max_hr, lthr)
    return effective_ftp


# ---------------------------------------------------------------------------
# Step 6 — enrich_records_1hz
# ---------------------------------------------------------------------------


def _assign_lap_index(
    ts_series: pd.Series,
    lap_start_times: List[Optional[str]],
) -> pd.Series:
    """
    Assign a 1-based lap index to each record timestamp.

    Boundaries ≤ first_record_ts are dropped.  Duplicates removed.
    Returns nullable Int64 Series aligned to ts_series.index.
    """
    result = pd.Series(1, index=ts_series.index, dtype="Int64")
    if not lap_start_times:
        return result

    ts_parsed = pd.to_datetime(ts_series, utc=True, errors="coerce")
    first_record_ts = ts_parsed.dropna().min()
    if pd.isna(first_record_ts):
        return result

    raw = pd.to_datetime(
        [t for t in lap_start_times if t is not None], utc=True, errors="coerce"
    ).dropna()
    qualifying = raw[raw > first_record_ts].drop_duplicates().sort_values()
    if qualifying.empty:
        return result

    for i, boundary in enumerate(qualifying, start=2):
        result[ts_parsed >= boundary] = i

    return result


def _compute_zone_series(
    df: pd.DataFrame,
    effective_ftp: Optional[float],
    lthr: Optional[float],
    max_hr: Optional[float],
    power_channel_is_low_coverage: bool,
) -> Tuple[Any, Any]:
    """
    Compute per-row power_zone and hr_zone arrays.

    power_zone is all-None when power_channel_is_low_coverage — per-second
    zone assignment is unreliable when the channel has < 50 % coverage.
    """
    if (
        effective_ftp is not None
        and effective_ftp > 0
        and "power" in df.columns
        and not power_channel_is_low_coverage
    ):
        pwr_bounds = [0.0, 0.55, 0.75, 0.90, 1.05, 1.20, 1.50, float("inf")]
        r = df["power"].fillna(-1) / effective_ftp
        conds = [(r >= pwr_bounds[i]) & (r < pwr_bounds[i + 1]) for i in range(7)]
        pz_raw = np.select(conds, list(range(1, 8)), default=-1)
        pz_series = pd.array(
            [None if v == -1 else int(v) for v in pz_raw], dtype="Int64"
        )
    else:
        pz_series = pd.array([None] * len(df), dtype="Int64")

    ref = lthr if lthr is not None else max_hr
    if ref is not None and ref > 0 and "heart_rate" in df.columns:
        hr_bounds = [0.0, 0.81, 0.89, 0.94, 1.00, float("inf")]
        r_hr = df["heart_rate"].fillna(-1) / ref
        conds_hr = [(r_hr >= hr_bounds[i]) & (r_hr < hr_bounds[i + 1]) for i in range(5)]
        hz_raw = np.select(conds_hr, list(range(1, 6)), default=-1)
        hz_series = pd.array(
            [None if pd.isna(v) or v == -1 else int(v) for v in hz_raw], dtype="Int64"
        )
    else:
        hz_series = pd.array([None] * len(df), dtype="Int64")

    return pz_series, hz_series


def _iter_1hz_rows(
    df: pd.DataFrame,
    source_file: str,
    athlete: Optional[str],
    elapsed: pd.Series,
    dist_km: pd.Series,
    spd_kmh: pd.Series,
    is_moving: pd.Series,
    pz_series: Any,
    hz_series: Any,
    lap_idx: pd.Series,
    quality: TimelineQuality,
) -> Generator[Record1HzRow, None, None]:
    """Yield typed Record1HzRow objects via itertuples — no iloc loops."""
    tmp = df.copy()
    tmp["_elapsed"] = elapsed.values
    tmp["_dist_km"] = dist_km.values
    tmp["_spd_kmh"] = spd_kmh.values
    tmp["_is_moving"] = is_moving.values
    tmp["_pz"] = np.array(
        [None if pd.isna(v) else int(v) for v in pz_series], dtype=object
    )
    tmp["_hz"] = np.array(
        [None if pd.isna(v) else int(v) for v in hz_series], dtype=object
    )
    tmp["_lap"] = np.array(
        [None if pd.isna(v) else int(v) for v in lap_idx.values], dtype=object
    )

    def _g(row: Any, col: str) -> Optional[float]:
        return safe_float(getattr(row, col, None))

    def _b(row: Any, col: str) -> bool:
        v = getattr(row, col, False)
        return bool(v) if v is not None else False

    for row in tmp.itertuples(index=False, name="R"):
        yield Record1HzRow(
            source_file=source_file,
            athlete=athlete,
            export_schema_version=EXPORT_SCHEMA_VERSION,
            timestamp=to_iso(getattr(row, "timestamp", None)),
            heart_rate=_g(row, "heart_rate"),
            cadence=_g(row, "cadence"),
            power=_g(row, "power"),
            speed=_g(row, "speed"),
            distance=_g(row, "distance"),
            altitude=_g(row, "altitude"),
            temperature=_g(row, "temperature"),
            latitude_deg=_g(row, "latitude_deg"),
            longitude_deg=_g(row, "longitude_deg"),
            power_imputed=_b(row, "power_imputed"),
            hr_imputed=_b(row, "hr_imputed"),
            speed_imputed=_b(row, "speed_imputed"),
            timeline_is_dense_1hz=quality.timeline_is_dense_1hz,
            power_channel_is_low_coverage=quality.power_channel_is_low_coverage,
            speed_channel_is_low_coverage=quality.speed_channel_is_low_coverage,
            elapsed_time_s=safe_float(getattr(row, "_elapsed", None)),
            distance_km=safe_float(getattr(row, "_dist_km", None)),
            speed_kmh=safe_float(getattr(row, "_spd_kmh", None)),
            is_moving=int(getattr(row, "_is_moving", 0)),
            power_zone=getattr(row, "_pz", None),
            hr_zone=getattr(row, "_hz", None),
            lap_index=getattr(row, "_lap", None),
        )


def enrich_records_1hz(
    df: pd.DataFrame,
    source_file: str,
    athlete: Optional[str],
    effective_ftp: Optional[float],
    lthr: Optional[float],
    max_hr: Optional[float],
    lap_start_times: List[Optional[str]],
    quality: TimelineQuality,
) -> List[Record1HzRow]:
    """Produce typed Record1HzRow objects from the 1 Hz DataFrame."""
    if df.empty:
        return []

    df = df.copy()

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        t0 = ts.dropna().min()
        elapsed: pd.Series = (ts - t0).dt.total_seconds()
        lap_idx = _assign_lap_index(ts, lap_start_times)
    else:
        elapsed = pd.Series([None] * len(df), dtype=object)
        lap_idx = pd.Series([None] * len(df), dtype=object)

    dist_km = (
        df["distance"] / 1000.0 if "distance" in df.columns
        else pd.Series([None] * len(df), dtype=object)
    )
    spd_kmh = (
        df["speed"] * 3.6 if "speed" in df.columns
        else pd.Series([None] * len(df), dtype=object)
    )
    spd_moving = (
        df["speed"].fillna(0) > 0.5 if "speed" in df.columns
        else pd.Series(False, index=df.index)
    )
    pwr_moving = (
        df["power"].fillna(0) > 10 if "power" in df.columns
        else pd.Series(False, index=df.index)
    )
    is_moving = (spd_moving | pwr_moving).astype(int)

    pz_series, hz_series = _compute_zone_series(
        df, effective_ftp, lthr, max_hr,
        quality.power_channel_is_low_coverage,
    )

    return list(_iter_1hz_rows(
        df, source_file, athlete,
        elapsed, dist_km, spd_kmh, is_moving,
        pz_series, hz_series, lap_idx,
        quality,
    ))


# ---------------------------------------------------------------------------
# Streaming CSV export infrastructure
# ---------------------------------------------------------------------------


def _validate_existing_csv_header(path: Path, expected_columns: List[str]) -> None:
    """
    Verify that *path* begins with a header row that exactly matches
    *expected_columns* (same names, same order).

    Raises CsvSchemaMismatchError in these cases:
      - File is empty (0 bytes or contains only whitespace).
      - First row cannot be read as CSV.
      - Header columns differ from expected (wrong names or wrong order).

    Does NOT raise when the file does not exist — that case is handled by the
    caller before construction.

    This check prevents silently appending rows from schema version A into a
    file written under schema version B.
    """
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise CsvSchemaMismatchError(
            f"Cannot stat existing CSV file {path}: {exc}"
        ) from exc

    if file_size == 0:
        raise CsvSchemaMismatchError(
            f"Existing CSV file is empty (0 bytes): {path}. "
            "Cannot safely append without a valid header row.  "
            "Delete the file or provide a correctly initialised CSV."
        )

    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            try:
                first_row = next(reader)
            except StopIteration:
                raise CsvSchemaMismatchError(
                    f"Existing CSV file has no readable rows: {path}"
                )
    except OSError as exc:
        raise CsvSchemaMismatchError(
            f"Cannot read existing CSV file {path}: {exc}"
        ) from exc

    if first_row != expected_columns:
        raise CsvSchemaMismatchError(
            f"CSV schema mismatch in {path}.\n"
            f"  Expected : {expected_columns}\n"
            f"  Found    : {first_row}\n"
            "Appending to a file with a different schema would corrupt the "
            "dataset.  To start fresh, delete the existing file."
        )


class CsvStreamWriter:
    """
    Incremental, safe-append CSV writer for a single output file.

    ── Append-mode contract ─────────────────────────────────────────────────
    New file (does not exist at construction time):
        _header_written = False.  First append_rows() / append_dicts() call
        creates the file in "w" mode, writes the header, then data rows.
        All subsequent calls use "a" mode.

    Existing file (exists at construction time):
        _validate_existing_csv_header() is called immediately.  If the
        existing header does not exactly match the expected columns (same
        names, same order), CsvSchemaMismatchError is raised and the writer
        is not usable.  If validation passes, _header_written = True and
        every subsequent write uses "a" mode, never overwriting content.

    ── Error semantics ───────────────────────────────────────────────────────
    CsvSchemaMismatchError is raised in the constructor.  Callers (BatchCsvWriter
    __init__) must propagate this as a fatal startup error — it is never safe
    to continue with a mismatched schema.

    ── Column schema ─────────────────────────────────────────────────────────
    - Columns absent from a row dict → written as empty string.
    - Extra keys in row dicts not in columns → silently dropped.
    """

    def __init__(self, path: Path, columns: List[str]) -> None:
        self._path = path
        self._columns = columns
        if path.exists():
            # Validate before allowing any writes.  Raises on mismatch.
            _validate_existing_csv_header(path, columns)
            self._header_written: bool = True
        else:
            self._header_written = False

    def _open_and_write(self, rows: Iterator[Dict[str, Any]]) -> None:
        """
        Core write routine shared by append_rows and append_dicts.

        Opens in "w" (new file) or "a" (existing) mode, writes header on
        first write to a new file, then iterates rows.
        """
        mode = "w" if not self._header_written else "a"
        with self._path.open(mode, newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=self._columns,
                extrasaction="ignore",
                lineterminator="\n",
            )
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in self._columns})

    def append_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Append a list of row dicts to the CSV."""
        self._open_and_write(iter(rows))

    def append_dicts(self, rows: Iterator[Dict[str, Any]]) -> None:
        """
        Streaming variant — accepts any iterator of dicts.

        Avoids materialising the full collection in memory; suitable for
        large record collections (records_1hz, records_extracted).
        """
        self._open_and_write(rows)

    def ensure_header(self) -> None:
        """
        Create the file with a header row if it does not yet exist.

        No-op when the file already exists (validation was done in __init__).
        After creation _header_written is True, so subsequent writes use "a".
        """
        if not self._path.exists():
            self.append_rows([])


def _iter_dicts(rows: List[Any]) -> Iterator[Dict[str, Any]]:
    """Yield to_dict() for each item in *rows* without building a full list."""
    for r in rows:
        yield r.to_dict()


class BatchCsvWriter:
    """
    Container for all seven CsvStreamWriter instances for one batch run.

    Construction validates every pre-existing CSV header.  If any file has a
    mismatched schema, CsvSchemaMismatchError propagates immediately and the
    batch does not start.  This is intentional: it is safer to abort with a
    clear error than to produce a silently corrupted dataset.

    Initialise once after confirming that at least one FIT file exists.
    Call write_parsed_file() once per file.  Data is flushed immediately.
    """

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        self.activities = CsvStreamWriter(
            output_dir / "activities.csv", ACTIVITIES_COLUMNS
        )
        self.laps = CsvStreamWriter(
            output_dir / "laps.csv", LAPS_COLUMNS
        )
        self.workout_steps = CsvStreamWriter(
            output_dir / "workout_steps.csv", WORKOUT_STEPS_COLUMNS
        )
        self.records_extracted = CsvStreamWriter(
            output_dir / "records_extracted.csv", RECORDS_EXTRACTED_COLUMNS
        )
        self.records_1hz = CsvStreamWriter(
            output_dir / "records_1hz.csv", RECORDS_1HZ_COLUMNS
        )
        self.best_efforts = CsvStreamWriter(
            output_dir / "best_efforts.csv", BEST_EFFORTS_COLUMNS
        )
        self.file_inventory = CsvStreamWriter(
            output_dir / "file_inventory.csv", FILE_INVENTORY_COLUMNS
        )

        for writer in self._all_writers():
            writer.ensure_header()

    def _all_writers(self) -> Iterator[CsvStreamWriter]:
        yield self.activities
        yield self.laps
        yield self.workout_steps
        yield self.records_extracted
        yield self.records_1hz
        yield self.best_efforts
        yield self.file_inventory

    def write_parsed_file(self, result: "ParsedFitFile") -> None:
        """Append all rows from *result* to the corresponding CSV files."""
        self.activities.append_rows([result.activity.to_dict()])
        self.laps.append_dicts(_iter_dicts(result.laps))
        self.workout_steps.append_dicts(_iter_dicts(result.workout_steps))
        self.records_extracted.append_dicts(_iter_dicts(result.records_extracted))
        self.records_1hz.append_dicts(_iter_dicts(result.records_1hz))
        self.best_efforts.append_dicts(_iter_dicts(result.best_efforts))
        self.file_inventory.append_rows([result.inventory.to_dict()])


# ---------------------------------------------------------------------------
# ParsedFitFile — ephemeral result of one file
# ---------------------------------------------------------------------------


@dataclass
class ParsedFitFile:
    """
    Complete parsed output for a single FIT file.

    Ephemeral — created by parse_fit_file(), written by BatchCsvWriter,
    then discarded.  Never accumulated across the batch.
    """
    activity: FitActivity
    laps: List[LapRow] = field(default_factory=list)
    workout_steps: List[WorkoutStepRow] = field(default_factory=list)
    records_extracted: List[RecordExtractedRow] = field(default_factory=list)
    records_1hz: List[Record1HzRow] = field(default_factory=list)
    best_efforts: List[BestEffortRow] = field(default_factory=list)
    inventory: FileInventoryRow = field(
        default_factory=lambda: FileInventoryRow(
            source_file="",
            export_schema_version=EXPORT_SCHEMA_VERSION,
            status="ok",
            error_class=None,
            error_stage=None,
            status_message=None,
        )
    )


# ---------------------------------------------------------------------------
# Orchestrator — parse_fit_file
# ---------------------------------------------------------------------------


def _make_error_result(
    activity: FitActivity,
    inventory: FileInventoryRow,
    exc: Exception,
    stage: str,
    is_operational: bool,
) -> ParsedFitFile:
    """
    Build a controlled error ParsedFitFile.

    is_operational=True  → known FIT format error; log at WARNING.
    is_operational=False → unexpected bug; log at ERROR with traceback.
    """
    if is_operational:
        logger.warning(
            "FIT error at stage '%s' in %s: %s", stage, activity.source_file, exc
        )
    else:
        logger.exception(
            "Unexpected error at stage '%s' in %s", stage, activity.source_file
        )
    inventory.status = "error"
    inventory.error_class = type(exc).__name__
    inventory.error_stage = stage
    inventory.status_message = str(exc)
    return ParsedFitFile(
        activity=activity,
        laps=[],
        workout_steps=[],
        records_extracted=[],
        records_1hz=[],
        best_efforts=[],
        inventory=inventory,
    )


def _no_timeline_reason(records_clean_input: pd.DataFrame) -> str:
    """Specific reason why no 1 Hz timeline was built."""
    if records_clean_input.empty:
        return "No record messages found in the FIT file."
    if "timestamp" not in records_clean_input.columns:
        return "Record messages have no timestamp field."
    ts = pd.to_datetime(records_clean_input["timestamp"], utc=True, errors="coerce")
    if ts.dropna().empty:
        return "All record timestamps are invalid or unparseable."
    return "Timeline is empty after deduplication and resampling."


def _make_no_timeline_result(
    activity: FitActivity,
    inventory: FileInventoryRow,
    laps: List[LapRow],
    workout_steps: List[WorkoutStepRow],
    records_extracted: List[RecordExtractedRow],
    reason: str,
) -> ParsedFitFile:
    """
    Build a ParsedFitFile for the controlled no-timeline case.

    status = "ok_no_timeline": file read without error, but no 1 Hz timeline.
    Laps, workout_steps, records_extracted preserved.
    records_1hz and best_efforts are empty.
    """
    logger.info("No usable 1 Hz timeline for %s: %s", activity.source_file, reason)
    inventory.status = "ok_no_timeline"
    inventory.status_message = reason
    return ParsedFitFile(
        activity=activity,
        laps=laps,
        workout_steps=workout_steps,
        records_extracted=records_extracted,
        records_1hz=[],
        best_efforts=[],
        inventory=inventory,
    )


def parse_fit_file(
    file_path: Path,
    athlete: Optional[str] = None,
    ftp: Optional[float] = None,
    resting_hr: Optional[float] = None,
    max_hr: Optional[float] = None,
    lthr: Optional[float] = None,
) -> ParsedFitFile:
    """
    Orchestrate the full pipeline for a single FIT file.

    Every stage is individually guarded.  Failures produce a controlled
    ParsedFitFile with status="error"; the batch continues.

    Stages:
      1. extract_fit_messages
      2. _apply_enhanced_channels → records_clean_input
      3. assess_timeline_quality
      4. build_1hz_timeline  (empty → ok_no_timeline)
      5. detect_indoor
      6. compute_session_metrics
      7. compute_grade_metrics
      8. enrich_records_1hz
      9. build_best_efforts (gated by power_metrics_quality)
    """
    activity = FitActivity(
        source_file=str(file_path.name),
        export_schema_version=EXPORT_SCHEMA_VERSION,
        athlete=athlete,
        file_size_bytes=file_path.stat().st_size,
        lthr_used_bpm=lthr,
        resting_hr_used_bpm=resting_hr,
        max_hr_used_bpm=max_hr,
        hr_zone_source=(
            "lthr" if lthr is not None
            else ("max_hr" if max_hr is not None else None)
        ),
    )
    inventory = FileInventoryRow(
        source_file=file_path.name,
        export_schema_version=EXPORT_SCHEMA_VERSION,
        status="ok",
        error_class=None,
        error_stage=None,
        status_message=None,
    )

    # Stage 1 ─────────────────────────────────────────────────────────────────
    try:
        laps, workout_steps, records_extracted = extract_fit_messages(
            file_path, athlete, activity
        )
    except fitdecode.FitError as exc:
        return _make_error_result(activity, inventory, exc,
                                  "extract_fit_messages", is_operational=True)
    except Exception as exc:
        return _make_error_result(activity, inventory, exc,
                                  "extract_fit_messages", is_operational=False)

    activity.record_count = len(records_extracted)
    activity.lap_count = len(laps)

    # Stage 2 ─────────────────────────────────────────────────────────────────
    try:
        records_extracted_df = pd.DataFrame([r.to_dict() for r in records_extracted])
        records_clean_input = (
            _apply_enhanced_channels(records_extracted_df)
            if not records_extracted_df.empty
            else records_extracted_df
        )
    except Exception as exc:
        return _make_error_result(activity, inventory, exc,
                                  "apply_enhanced_channels", is_operational=False)

    # Stage 3 ─────────────────────────────────────────────────────────────────
    try:
        quality = assess_timeline_quality(records_clean_input)
    except Exception as exc:
        return _make_error_result(activity, inventory, exc,
                                  "assess_timeline_quality", is_operational=False)

    # Stage 4 ─────────────────────────────────────────────────────────────────
    try:
        df_1hz = build_1hz_timeline(records_clean_input)
    except Exception as exc:
        return _make_error_result(activity, inventory, exc,
                                  "build_1hz_timeline", is_operational=False)

    if df_1hz.empty:
        reason = _no_timeline_reason(records_clean_input)
        return _make_no_timeline_result(
            activity, inventory, laps, workout_steps, records_extracted, reason
        )

    # Stage 5 ─────────────────────────────────────────────────────────────────
    try:
        activity.indoor, activity.indoor_source = detect_indoor(
            df_1hz, activity.sub_sport,
            activity.has_structured_workout, activity.trainer_mode,
        )
    except Exception as exc:
        return _make_error_result(activity, inventory, exc,
                                  "detect_indoor", is_operational=False)

    # Stage 6 ─────────────────────────────────────────────────────────────────
    try:
        effective_ftp = compute_session_metrics(
            activity, df_1hz, ftp, resting_hr, max_hr, lthr, quality
        )
    except Exception as exc:
        return _make_error_result(activity, inventory, exc,
                                  "compute_session_metrics", is_operational=False)

    # Stage 7 ─────────────────────────────────────────────────────────────────
    try:
        compute_grade_metrics(activity, df_1hz)
    except Exception as exc:
        return _make_error_result(activity, inventory, exc,
                                  "compute_grade_metrics", is_operational=False)

    # Stage 8 ─────────────────────────────────────────────────────────────────
    lap_start_times = [lap.start_time for lap in laps]
    try:
        records_1hz = enrich_records_1hz(
            df_1hz,
            source_file=file_path.name,
            athlete=athlete,
            effective_ftp=effective_ftp,
            lthr=lthr,
            max_hr=max_hr,
            lap_start_times=lap_start_times,
            quality=quality,
        )
    except Exception as exc:
        return _make_error_result(activity, inventory, exc,
                                  "enrich_records_1hz", is_operational=False)

    # Stage 9 ─────────────────────────────────────────────────────────────────
    try:
        power_1hz: List[Optional[float]] = df_1hz.get(
            "power", pd.Series(dtype=float)
        ).tolist()
        # Best efforts follow the same quality gate as all other power metrics.
        compute_best_efforts = (activity.power_metrics_quality in (_PMQ_FULL, _PMQ_SPARSE_WARNING))
        best_efforts: List[BestEffortRow] = []
        for sec in [5, 15, 30, 60, 180, 300, 600, 1200]:
            value = rolling_mean_best(power_1hz, sec) if compute_best_efforts else None
            best_efforts.append(BestEffortRow(
                source_file=file_path.name,
                athlete=athlete,
                export_schema_version=EXPORT_SCHEMA_VERSION,
                metric=f"best_power_{sec}s",
                seconds=sec,
                value_w=value,
            ))
        if activity.estimated_ftp_w is not None:
            best_efforts.append(BestEffortRow(
                source_file=file_path.name,
                athlete=athlete,
                export_schema_version=EXPORT_SCHEMA_VERSION,
                metric="estimated_ftp_from_20min",
                seconds=1200,
                value_w=activity.estimated_ftp_w,
            ))
    except Exception as exc:
        return _make_error_result(activity, inventory, exc,
                                  "build_best_efforts", is_operational=False)

    return ParsedFitFile(
        activity=activity,
        laps=laps,
        workout_steps=workout_steps,
        records_extracted=records_extracted,
        records_1hz=records_1hz,
        best_efforts=best_efforts,
        inventory=inventory,
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def collect_fit_files(input_path: Path, recursive: bool = True) -> List[Path]:
    """Return all .fit files under *input_path*, or *input_path* itself."""
    if input_path.is_file() and input_path.suffix.lower() == ".fit":
        return [input_path]
    pattern = "**/*.fit" if recursive else "*.fit"
    return sorted(p for p in input_path.glob(pattern) if p.is_file())


# ---------------------------------------------------------------------------
# Stable CSV schema definitions
# ---------------------------------------------------------------------------

ACTIVITIES_COLUMNS: List[str] = [
    "export_schema_version",
    "source_file", "athlete", "file_size_bytes",
    "manufacturer", "product", "serial_number", "time_created",
    "sport", "sub_sport", "activity_type", "event",
    "start_time", "local_start_time",
    "total_elapsed_time_s", "total_timer_time_s", "moving_time_s",
    "duration_hms", "timer_hms", "moving_time_hms",
    "total_distance_m", "total_distance_km",
    "total_ascent_m", "total_descent_m",
    "elevation_gain_from_records_m", "avg_grade_pct", "max_grade_pct",
    "avg_speed_mps", "avg_speed_kmh", "max_speed_mps", "max_speed_kmh",
    "avg_heart_rate_bpm", "max_heart_rate_bpm",
    "avg_power_w", "max_power_w",
    "normalized_power_w", "normalized_power_moving_w",
    "avg_cadence_rpm", "max_cadence_rpm", "avg_cadence_no_zeros_rpm",
    "avg_temperature_c", "max_temperature_c", "calories_kcal",
    "training_stress_score", "intensity_factor", "variability_index",
    "work_kj", "pw_hr_decoupling_pct", "trimp",
    "left_right_balance_pct", "balance_left_pct", "balance_right_pct",
    "pedal_smoothness_pct", "torque_effectiveness_pct",
    "trainer_mode", "indoor", "indoor_source",
    "has_structured_workout", "workout_step_count",
    "ftp_used_w", "ftp_source", "estimated_ftp_w",
    "lthr_used_bpm", "resting_hr_used_bpm", "max_hr_used_bpm", "hr_zone_source",
    "has_power", "has_hr", "has_cadence", "has_gps", "has_speed",
    "is_complete_for_power_analysis", "is_complete_for_hr_analysis",
    "timeline_is_dense_1hz",
    "power_channel_is_low_coverage", "speed_channel_is_low_coverage",
    "timeline_median_gap_s", "timeline_max_gap_s",
    "power_metrics_quality", "power_quality_note",
    "record_count", "lap_count",
    "z1_recovery_sec", "z2_endurance_sec", "z3_tempo_sec", "z4_threshold_sec",
    "z5_vo2_sec", "z6_anaerobic_sec", "z7_neuromuscular_sec",
    "hr_z1_sec", "hr_z2_sec", "hr_z3_sec", "hr_z4_sec", "hr_z5_sec",
]

LAPS_COLUMNS: List[str] = [
    "export_schema_version",
    "source_file", "athlete",
    "start_time", "timestamp",
    "event", "event_type", "lap_trigger",
    "total_elapsed_time_s", "total_timer_time_s",
    "total_distance_m", "total_distance_km",
    "avg_speed_mps", "avg_speed_kmh",
    "max_speed_mps", "max_speed_kmh",
    "avg_heart_rate_bpm", "max_heart_rate_bpm",
    "avg_power_w", "max_power_w", "normalized_power_w",
    "avg_cadence_rpm", "max_cadence_rpm",
    "total_ascent_m", "total_descent_m",
    "intensity", "duration_hms",
]

WORKOUT_STEPS_COLUMNS: List[str] = [
    "export_schema_version",
    "source_file", "athlete",
    "step_index", "wkt_name",
    "duration_type", "duration_value",
    "target_type", "target_low", "target_high",
    "intensity",
]

RECORDS_EXTRACTED_COLUMNS: List[str] = [
    "export_schema_version",
    "source_file", "athlete", "timestamp",
    "heart_rate", "cadence", "power", "speed", "distance", "altitude",
    "temperature", "latitude_deg", "longitude_deg",
    "grade", "vertical_speed", "left_right_balance", "fractional_cadence",
    "enhanced_speed", "enhanced_altitude", "gps_accuracy",
]

RECORDS_1HZ_COLUMNS: List[str] = [
    "export_schema_version",
    "source_file", "athlete", "timestamp",
    "elapsed_time_s", "lap_index",
    "heart_rate", "cadence", "power", "speed", "distance", "altitude",
    "temperature", "latitude_deg", "longitude_deg",
    "power_imputed", "hr_imputed", "speed_imputed",
    "timeline_is_dense_1hz",
    "power_channel_is_low_coverage", "speed_channel_is_low_coverage",
    "distance_km", "speed_kmh", "is_moving",
    "power_zone", "hr_zone",
]

BEST_EFFORTS_COLUMNS: List[str] = [
    "export_schema_version",
    "source_file", "athlete",
    "metric", "seconds", "value_w",
]

FILE_INVENTORY_COLUMNS: List[str] = [
    "export_schema_version",
    "source_file",
    "status",
    "error_class",
    "error_stage",
    "status_message",
]


# ---------------------------------------------------------------------------
# Batch summary
# ---------------------------------------------------------------------------


class BatchSummary:
    """Accumulates lightweight counters for the final per-run report."""

    def __init__(self) -> None:
        self.total: int = 0
        self.ok: int = 0
        self.ok_no_timeline: int = 0
        self.errors: int = 0
        self.raw_records: int = 0
        self.hz_records: int = 0
        self.laps: int = 0
        self.workout_steps: int = 0

    def update(self, result: ParsedFitFile) -> None:
        self.total += 1
        status = result.inventory.status
        if status == "ok":
            self.ok += 1
        elif status == "ok_no_timeline":
            self.ok_no_timeline += 1
        else:
            self.errors += 1
        self.raw_records += len(result.records_extracted)
        self.hz_records += len(result.records_1hz)
        self.laps += len(result.laps)
        self.workout_steps += len(result.workout_steps)

    def print(self) -> None:
        print(
            f"Parsed: {self.total} files | "
            f"OK: {self.ok} | "
            f"No timeline: {self.ok_no_timeline} | "
            f"Errors: {self.errors}"
        )
        print(f"Raw records      : {self.raw_records:,}")
        print(f"1 Hz records     : {self.hz_records:,}")
        print(f"Laps             : {self.laps:,}")
        print(f"Workout steps    : {self.workout_steps:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def validate_args(args: argparse.Namespace) -> List[str]:
    """
    Domain-level CLI argument validation.

    Returns a list of error strings; empty means valid.
    """
    errors: List[str] = []

    if args.ftp is not None and args.ftp <= 0:
        errors.append(f"--ftp must be > 0 W, got {args.ftp}")
    if args.resting_hr is not None and args.resting_hr <= 0:
        errors.append(f"--resting-hr must be > 0 bpm, got {args.resting_hr}")
    if args.max_hr is not None and args.max_hr <= 0:
        errors.append(f"--max-hr must be > 0 bpm, got {args.max_hr}")
    if args.lthr is not None and args.lthr <= 0:
        errors.append(f"--lthr must be > 0 bpm, got {args.lthr}")

    if args.resting_hr is not None and args.max_hr is not None:
        if args.resting_hr >= args.max_hr:
            errors.append(
                f"--resting-hr ({args.resting_hr}) must be less than "
                f"--max-hr ({args.max_hr})"
            )
    if args.lthr is not None and args.max_hr is not None:
        if args.lthr >= args.max_hr:
            errors.append(
                f"--lthr ({args.lthr}) must be less than "
                f"--max-hr ({args.max_hr})"
            )
    if args.resting_hr is not None and args.lthr is not None:
        if args.resting_hr >= args.lthr:
            errors.append(
                f"--resting-hr ({args.resting_hr}) must be less than "
                f"--lthr ({args.lthr})"
            )

    return errors


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert FIT files (Garmin Edge / Wahoo KICKR / MyWhoosh) "
            "into coach-friendly CSV exports."
        )
    )
    p.add_argument("--input", required=True, help="Input .fit file or directory")
    p.add_argument("--output", required=True, help="Output directory for CSV files")
    p.add_argument("--athlete", default=None, help="Athlete name added to all rows")
    p.add_argument(
        "--ftp", type=float, default=None,
        help="FTP in watts (omit to estimate from best 20')",
    )
    p.add_argument(
        "--resting-hr", type=float, default=None,
        help="Resting HR in bpm (enables TRIMP)",
    )
    p.add_argument("--max-hr", type=float, default=None, help="Maximum HR in bpm")
    p.add_argument(
        "--lthr", type=float, default=None,
        help="Lactate threshold HR in bpm",
    )
    p.add_argument(
        "--no-recursive", action="store_true",
        help="Do not search subfolders for .fit files",
    )
    return p


def main() -> int:
    """
    Entry point.

    Empty-batch contract:
      When no .fit files are found, exit code 3 is returned WITHOUT creating
      or modifying any output CSV files.  BatchCsvWriter is only instantiated
      after confirming that at least one file exists.

    Schema-mismatch contract:
      If any pre-existing output CSV has a mismatched header,
      CsvSchemaMismatchError is raised during BatchCsvWriter construction.
      The program prints the error and returns exit code 4.  No FIT files
      are processed and no data is written.
    """
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    validation_errors = validate_args(args)
    if validation_errors:
        for msg in validation_errors:
            print(f"Error: {msg}", file=sys.stderr)
        return 1

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        return 2

    fit_files = collect_fit_files(input_path, recursive=not args.no_recursive)
    if not fit_files:
        print("No .fit files found.", file=sys.stderr)
        return 3

    # BatchCsvWriter instantiated only after confirming files exist.
    # CsvSchemaMismatchError propagates here if any existing CSV is incompatible.
    try:
        csv_writer = BatchCsvWriter(output_dir)
    except CsvSchemaMismatchError as exc:
        print(f"CSV schema mismatch — cannot continue:\n{exc}", file=sys.stderr)
        return 4

    summary = BatchSummary()

    for i, fp in enumerate(fit_files, 1):
        print(f"[{i}/{len(fit_files)}] {fp.name}", end=" ... ", flush=True)
        result = parse_fit_file(
            file_path=fp,
            athlete=args.athlete,
            ftp=args.ftp,
            resting_hr=args.resting_hr,
            max_hr=args.max_hr,
            lthr=args.lthr,
        )
        csv_writer.write_parsed_file(result)
        summary.update(result)
        print(result.inventory.status)

    print()
    summary.print()
    print(f"\nCSV export written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
