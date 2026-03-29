# fit_coach_exporter

`fit_coach_exporter` is a Python CLI tool for processing `.fit` files from cycling workouts and exporting data to stable CSV files, ready for further analysis by a coach, in Excel, pandas, or BI tools.

Supports both:
- **indoor** rides — e.g. Wahoo KICKR, ERG workouts, MyWhoosh,
- **outdoor** rides — e.g. Garmin Edge and other devices that record FIT files.

The project is focused on **repeatable data export**, **deterministic CSV schema**, and **careful metric semantics**, so that the output is suitable for real-world work with training data.

---

## Why this tool exists

A standard FIT file is convenient for sports devices and apps, but not particularly easy to analyse manually, report on, or process further.

This tool:
- reads a single `.fit` file or an entire directory of files,
- extracts session data, laps, and time-series records,
- builds a **1 Hz** time axis for analysis,
- calculates the most important cycling metrics,
- exports the result to several CSV files with a stable schema.

This gives a coach or analyst data in a format that can be easily filtered, aggregated, and visualised.

---

## Key features

- export to **7 stable CSV files**,
- **streaming writes** — good behaviour with larger batches,
- explicit schema version: `export_schema_version`,
- deterministic column order,
- validation of existing CSV headers before appending data,
- distinction between:
  - full analytic success,
  - successful file read without the ability to build a 1 Hz timeline,
  - processing error,
- careful handling of sparse / smart-recording data,
- quality flags for imputed data and data quality limitations,
- support for user-supplied FTP / HR values.

---

## What the tool calculates and exports

Depending on data quality and completeness, the tool can calculate, among others:

- total time, timer time, moving time,
- distance, elevation gain, speed,
- average and maximum heart rate,
- average and maximum power,
- **Normalized Power (NP)**,
- **Intensity Factor (IF)**,
- **Training Stress Score (TSS)**,
- **Variability Index (VI)**,
- **TRIMP**,
- **Pw:Hr decoupling**,
- time in power zones,
- time in HR zones,
- **best efforts** for common durations,
- activity classification as `indoor`, `outdoor`, or `unknown`.

In addition, the tool exports structural data present in the FIT file, when available:
- `laps`,
- `workout_steps`,
- raw device records (`records_extracted`),
- analytical 1 Hz timeline (`records_1hz`).

Not all metrics are calculated every time. When data quality is too poor, the tool prefers to:
- return `None`,
- add a quality flag,
- or describe the limitation in quality fields,
rather than produce plausibly-looking but unreliable numbers.

> **Important:** Power-dependent metrics (IF, TSS, NP, power zones) require FTP
> to be supplied explicitly via `--ftp`. The tool does not estimate FTP
> automatically — outdoor rides with stops, coasting, and variable pacing do
> not reliably contain a clean 20-minute maximal effort, so any automatic
> estimate would be untrustworthy. Always pass the FTP value configured on
> your bike computer or from a dedicated FTP test.

---

## How the pipeline works

Processing a single file proceeds roughly as follows:

1. **FIT read**
   - reading FIT messages,
   - extracting file info, session, laps, workout steps, and records.

2. **Building input data layers**
   - `records_extracted` — raw records from FIT,
   - `records_clean_input` — working layer with base channels and `enhanced_*` channels merged.

3. **Building the 1 Hz timeline and assessing its quality**
   - timestamp deduplication,
   - resampling to 1 Hz,
   - imputation according to dense/sparse policy,
   - determining quality flags and assessing channel coverage.

4. **Calculating session metrics**
   - basic statistics,
   - power metrics,
   - HR metrics,
   - zones,
   - best efforts,
   - grade only for activities confirmed as outdoor.

5. **Streaming export to CSV**
   - each FIT file is processed and immediately appended to the corresponding CSV files,
   - the tool does not accumulate the entire batch in memory.

---

## Output files

The tool creates the following files:

### `activities.csv`
One row per activity. Contains session summary, load metrics, FTP/HR sources, quality flags, and power metrics quality status.

### `laps.csv`
One row per lap / segment. Useful for interval and workout structure analysis.

### `workout_steps.csv`
Planned structured workout steps, if present in the FIT file.

### `records_extracted.csv`
Raw FIT records exactly as recorded by the device — before resampling, imputation, and 1 Hz timeline construction. Useful for auditing and verifying what the device actually recorded.

### `records_1hz.csv`
Normalised 1 Hz timeline with helper fields, e.g.:
- `elapsed_time_s`,
- `lap_index`,
- `distance_km`,
- `speed_kmh`,
- `is_moving`,
- `power_zone`,
- `hr_zone`,
- imputation flags,
- timeline quality and channel coverage flags.

### `best_efforts.csv`
Best mean powers for standard durations:
- 5 s,
- 15 s,
- 30 s,
- 1 min,
- 3 min,
- 5 min,
- 10 min,
- 20 min.

### `file_inventory.csv`
List of processed files and their processing status. This is the primary file for batch quality control.

---

## Statuses in `file_inventory.csv`

### `ok`
Full analytic success.
- the 1 Hz timeline was successfully built,
- all applicable metrics were calculated,
- some may still be subject to quality gating if data was limited.

### `ok_no_timeline`
The file was read without error, but a usable 1 Hz timeline could not be built.
Possible reasons:
- no record messages in the file,
- no `timestamp` column,
- all timestamps are invalid,
- empty result after deduplication / resampling.

In this case some data may still be preserved, if FIT extraction succeeded, e.g.:
- `records_extracted`,
- `laps`,
- `workout_steps`,
but there will be no `records_1hz`, `best_efforts`, or metrics that depend on the 1 Hz timeline.

### `error`
A failure occurred at a named pipeline stage. The batch continues, but the affected file gets an error entry with:
- `error_class`,
- `error_stage`,
- `status_message`.

---

## Power metrics quality

`activities.csv` contains the fields:
- `power_metrics_quality`,
- `power_quality_note`.

### `power_metrics_quality`
Machine-readable quality status for power-dependent metrics:

- `full` — power metrics were calculated and `best_efforts` are present. On a dense timeline the result is fully reliable and `power_quality_note` is empty. On a sparser timeline with sufficient power channel coverage metrics are also calculated, but `power_quality_note` may contain a warning about overestimation risk — check that field before interpreting results.
- `sparse_warning` — power metrics and `best_efforts` are calculated, but the power data has limited coverage and results should be interpreted with caution; `power_quality_note` provides details.
- `withheld` — power-dependent metrics were deliberately withheld due to insufficient data; this includes `NP`, `NP_moving`, `IF`, `TSS`, `work_kj`, `VI`, time in power zones, `power_zone` in `records_1hz`, and `best_efforts`; `power_quality_note` lists the withheld metrics.
- `None` — no power channel present in the FIT file; quality assessment does not apply.

### `power_quality_note`
Human-readable description, always consistent with `power_metrics_quality`:
- `full` on a dense timeline — empty (`None`),
- `full` on a sparser timeline with sufficient coverage — warns about overestimation risk for NP / TSS / IF / work,
- `sparse_warning` — warns about limited data coverage and indicates that results may overstate training load,
- `withheld` — lists the specific metrics that were deliberately withheld and explains why.

---

## Requirements

- Python 3.10+ (3.11 or newer recommended)
- packages:

```bash
pip install pandas numpy fitdecode
```

---

## Usage

### Single file

```bash
python fit_coach_exporter.py \
  --input ./ride.fit \
  --output ./coach_csv \
  --ftp 205
```

### Directory of FIT files

```bash
python fit_coach_exporter.py \
  --input ./fit_files \
  --output ./coach_csv \
  --ftp 205
```

### With full athlete data

```bash
python fit_coach_exporter.py \
  --input ./fit_files \
  --output ./coach_csv \
  --athlete "Przemek" \
  --ftp 205 \
  --resting-hr 50 \
  --max-hr 185 \
  --lthr 170
```

### Without subdirectory search

```bash
python fit_coach_exporter.py \
  --input ./fit_files \
  --output ./coach_csv \
  --ftp 205 \
  --no-recursive
```

---

## CLI parameters

- `--input` — a single `.fit` file or a directory of files,
- `--output` — output directory for CSV files,
- `--athlete` — athlete name appended to all rows,
- `--ftp` — FTP in watts. **Required for power-dependent metrics** (IF, TSS, NP, power zones). FTP is not estimated automatically — always supply the value from your device settings or a recent dedicated test.
- `--resting-hr` — resting heart rate in bpm; required for TRIMP,
- `--max-hr` — maximum heart rate in bpm,
- `--lthr` — lactate threshold heart rate in bpm,
- `--no-recursive` — do not search subdirectories.

Argument validation checks, among other things, the consistency of HR value relationships.

---

## Exit codes

- `0` — success,
- `1` — invalid input arguments,
- `2` — invalid input path,
- `3` — no `.fit` files found,
- `4` — problem with existing CSVs in the output directory, e.g. mismatched header, different column order, empty 0-byte file, or unreadable file.

---

## Important assumptions and limitations

### 1. This is not a full replacement for TrainingPeaks / WKO / Intervals.icu
This tool is primarily an **analytic exporter**, not a complete coaching platform.

### 2. FTP must be supplied manually
The tool does not estimate FTP automatically. Outdoor rides with stops, traffic lights, coasting, and variable pacing do not reliably produce a clean 20-minute maximal effort, so any automatic estimate from such data would be untrustworthy and could severely understate your actual FTP. Always pass `--ftp` with the value configured on your bike computer or from a dedicated FTP test.

### 3. Quality flags matter
If a file was recorded at a lower rate, with gaps, or using smart recording, some metrics may be:
- flagged with a warning,
- withheld,
- or subject to greater uncertainty.

### 4. `power_channel_is_low_coverage` and `speed_channel_is_low_coverage` are heuristics
These are auxiliary signals based on data coverage, not an objective measure of sensor quality.

### 5. `Pw:Hr decoupling` is simplified
The metric is based on a simplified 50/50 split of the activity and should be interpreted with caution, especially for workouts containing warm-up, cool-down, and intervals.

### 6. Grade is calculated conservatively
Gradient metrics are only calculated for activities confirmed as outdoor.

---

## CSV schema safety

When appending to existing CSV files, the tool checks that the header matches the expected schema.

If an existing file:
- is empty,
- has a different header,
- has a different column order,
- cannot be read correctly,

the program **will not append data** and will exit with an error. This prevents accidental mixing of different schema versions in the same dataset.

---

## Example use cases

- preparing training data for a coach,
- building a dashboard in Power BI / Tableau / Looker Studio,
- session analysis in pandas / Jupyter,
- exporting data from a folder of archived workouts,
- auditing data quality from different devices (Garmin, Wahoo, MyWhoosh).

---

## Project structure

A minimal repository might look like this:

```text
.
├── fit_coach_exporter.py
├── README.md
└── requirements.txt
```

Example `requirements.txt`:

```text
pandas
numpy
fitdecode
```

---

## Licence

MIT, Apache-2.0

---

## Roadmap / possible next steps

Potential future additions:
- unit and integration tests,
- export to Parquet,
- separate quality statuses for speed/grade/HR metrics,
- better reason codes in `file_inventory.csv`,
- support for additional metrics and coaching reports.
