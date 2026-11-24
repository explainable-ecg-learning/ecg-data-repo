# ECG Dataset Repo

PyTorch dataset classes, download scripts, data-quality audits, and per-record error labels for clinical 12-lead ECG benchmarks: **PTB-XL**, **MIMIC-IV-ECG**, **Chapman-Shaoxing**, **Georgia 12-Lead ECG**, and **CODE-15%**.

---

## Repository Structure

Each dataset is self-contained inside its own Python package subdirectory, with raw waveforms in `<dataset>/data/` (gitignored), quality/error-label artifacts in `<dataset>/quality/`, and executed exploration notebooks in `<dataset>/notebooks/`:

```text
ecg-data-repo/
├── cinc_style_dataset.py       # Shared base Dataset for PhysioNet/CinC .hea/.mat datasets
├── download_all.sh             # Runs every <dataset>/download.sh sequentially
├── verify_datasets.py          # Smoke-tests all present datasets and splits
├── requirements.txt            # Core PyTorch/WFDB/HDF5 dependencies
├── ptbxl/
│   ├── dataset.py              # PTBXLDataset (100/500 Hz, soft segmentations, error labels)
│   ├── download.sh             # PhysioNet PTB-XL v1.0.3 + Zenodo 7610236 segmentations
│   ├── quality/                # error_labels.csv (21,799 rows) + generate_error_labels.py
│   ├── notebooks/              # 00–08 executed exploration & cleaning notebooks (9 total)
│   └── data/                   # Raw data (gitignored)
├── mimic_iv_ecg/
│   ├── dataset.py              # MIMICIVECGDataset (subject splits, sentinel cleaning, error labels)
│   ├── download.sh             # PhysioNet MIMIC-IV-ECG v1.0
│   ├── quality/                # error_labels.csv.gz (800,035 rows) + generate_error_labels.py
│   ├── notebooks/              # 00–05 executed exploration & cleaning notebooks (6 total)
│   └── data/                   # Raw data (gitignored)
├── georgia_ecg/
│   ├── dataset.py              # GeorgiaECGDataset (CinC 2020 SNOMED-CT mapping, error labels)
│   ├── download.sh             # PhysioNet/CinC 2020 training/georgia
│   ├── quality/                # error_labels.csv (10,344 rows), generate_error_labels.py, dx_mapping_*.csv
│   ├── notebooks/              # 00–05 executed exploration & cleaning notebooks (6 total)
│   └── data/                   # Raw data (gitignored)
├── chapman_shaoxing/
│   ├── dataset.py              # ChapmanShaoxingDataset
│   ├── download.sh             # PhysioNet ecg-arrhythmia 1.0.0
│   └── data/                   # Raw data (gitignored)
└── code15/
    ├── dataset.py              # CODE15Dataset (patient-stratified splits over HDF5 parts)
    ├── download.sh             # Zenodo CODE-15% (18 HDF5 parts + exams.csv)
    └── data/                   # Raw data (gitignored)
```

---

## Data Quality, Error Labels & Loader Fixes

Across **PTB-XL**, **MIMIC-IV-ECG**, and **Georgia 12-Lead ECG**, `21` executed Jupyter notebooks audit metadata, diagnostic labels, signal integrity, spectra, wave delineation (`Zenodo 7610236` & `NeuroKit2`), and anomalies, producing versioned per-record error-label tables under `<dataset>/quality/`:

| Dataset | Audited Records | Error Flags | `0_clean` | `1_minor_warning` | `2_moderate_anomaly` | `3_severe_error` | Notebooks |
|---|---:|---:|---:|---:|---:|---:|---:|
| **PTB-XL** (`ptbxl/quality/error_labels.csv`) | `21,799` | `24` | `10,154` (`46.6%`) | `8,036` (`36.9%`) | `2,695` (`12.4%`) | `914` (`4.2%`) | `9` (`00`–`08`) |
| **MIMIC-IV-ECG** (`mimic_iv_ecg/quality/error_labels.csv.gz`) | `800,035` | `15` | `421,460` (`52.7%`) | `333,848` (`41.7%`) | `6,865` (`0.9%`) | `37,862` (`4.7%`) | `6` (`00`–`05`) |
| **Georgia ECG** (`georgia_ecg/quality/error_labels.csv`) | `10,344` | `18` | `8,770` (`84.8%`) | `941` (`9.1%`) | `584` (`5.6%`) | `49` (`0.5%`) | `6` (`00`–`05`) |

### Key Loader Improvements & Bug Fixes
- **Unified quality filtering API**: `PTBXLDataset`, `MIMICIVECGDataset`, and `GeorgiaECGDataset` all support `load_error_labels=True`, `max_error_severity=0..3`, and `exclude_error_flags=[...]` backed by `<dataset>/quality/error_labels.*`.
- **PTB-XL soft segmentations**: `PTBXLDataset(load_segmentation=True)` reads 24-class delineation masks directly from `ptbxl/data/ptbxl_segmentations_8bit.zip` (handling the archive's `ptbxl_segmentations_8it/` folder typo without unpacking `21,837` files and re-normalizing the `uint8` simplex).
- **Georgia `.hea` header parsing & SNOMED-CT mapping**: Fixed `CinCStyleDataset._parse_header` regex (`#\s*Dx:`, `#\s*Age:`, `#\s*Sex:`) for Georgia's spaced header comments and bundled official CinC 2020 `dx_mapping_scored.csv` / `dx_mapping_unscored.csv` inside `georgia_ecg/quality/`.
- **MIMIC-IV-ECG note-link deduplication & sentinel cleaning**: Deduplicated `7` multi-note `study_id`s in `waveform_note_links.csv` so `load_note_links=True` preserves 1 row per study, and added `clean_sentinels=True` to convert `29999` measurement sentinels to `NaN`.

---

## Quick Start & Verification

```bash
# Download all datasets (skips datasets already on disk)
./download_all.sh

# Smoke-test all downloaded datasets and splits
.venv/bin/python verify_datasets.py

# Regenerate per-record error-label tables
.venv/bin/python ptbxl/quality/generate_error_labels.py
.venv/bin/python mimic_iv_ecg/quality/generate_error_labels.py
.venv/bin/python georgia_ecg/quality/generate_error_labels.py

# Launch Jupyter Lab using the .venv kernel (ecg-venv)
.venv/bin/jupyter lab
```

---

## Datasets

### PTB-XL (`ptbxl/`)

~21,000 clinical 12-lead ECGs at 100 Hz or 500 Hz with diagnostic labels from SCP codes.

**Download**

```bash
./ptbxl/download.sh
```

Data (waveforms + [Zenodo 7610236](https://zenodo.org/records/7610236) 24-class soft segmentations) is saved to `ptbxl/data/`. If the data is already present the script is a no-op. Data quality and cleaning exploration notebooks live in `ptbxl/notebooks/`.

**Usage**

```python
from ptbxl import PTBXLDataset

# Convenience split constructors (folds 1-8 / 9 / 10; defaults to ptbxl/data)
train_ds = PTBXLDataset.train("ptbxl/data")
val_ds   = PTBXLDataset.val("ptbxl/data")
test_ds  = PTBXLDataset.test("ptbxl/data")

# Or manually select folds and options (including 24-class soft segmentations & error labels)
ds = PTBXLDataset(
    root="ptbxl/data",
    sampling_rate=100,          # 100 or 500 Hz
    folds=[1, 2, 3],            # None = all folds
    filter_noise=True,          # drop records with signal-quality annotations
    min_class_fraction=0.0029,  # prune rare subclasses
    load_segmentation=False,    # True loads (12, 24, 1000) float32 delineation probabilities
    load_error_labels=False,    # True attaches per-record anomaly/error flags from ptbxl/quality/error_labels.csv
    max_error_severity=None,    # 0..3 optional severity cutoff (e.g. 2 drops '3_severe_error')
)

sample = ds[0]
# sample["signal"]              # float32 ndarray (12, seq_len)
# sample["diagnostic_class"]    # e.g. "NORM", "MI"
# sample["diagnostic_subclass"] # e.g. "IMI", "LAFB"
# sample["extra_labels"]        # list of secondary subclasses
# sample["scp_codes"]           # raw dict {code: likelihood}
# sample["meta"]                # remaining CSV columns as dict
# sample["segmentation"]        # (12, 24, 1000) float32 when load_segmentation=True
# sample["error_labels"]        # dict of severity_tier, error_tags, flags when load_error_labels=True

print(ds.class_names)           # sorted list of retained subclasses
```

**Data Exploration & Quality Audit (`ptbxl/notebooks/`)**

Nine executed Jupyter notebooks document PTB-XL v1.0.3 data quality, cleaning, soft segmentations, and anomalies:

| Notebook | Focus |
|---|---|
| [`00_sample_waveforms.ipynb`](ptbxl/notebooks/00_sample_waveforms.ipynb) | Quick 12-lead waveform & multi-lead lineplot visualizations for samples `11`, `13`, `19`, `92`, and `42` |
| [`01_cohort_metadata_cleaning.ipynb`](ptbxl/notebooks/01_cohort_metadata_cleaning.ipynb) | Missingness taxonomy, sentinel `age == 300` (>89 yr), anthropometric/BMI typos, longitudinal consistency, and human vs autogenerated validation |
| [`02_scp_labels_and_filter_audit.ipynb`](ptbxl/notebooks/02_scp_labels_and_filter_audit.ipynb) | SCP-ECG taxonomy, confidence weights, 0-diagnostic-code `"NORM"` fallback, hierarchy mismatches, and split `class_names` consistency |
| [`03_signal_quality_and_artifacts.ipynb`](ptbxl/notebooks/03_signal_quality_and_artifacts.ipynb) | Per-lead artifact parsing (`baseline_drift`, `static_noise`, `burst_noise`, `electrodes_problems`), defect gallery, and train/test noise shift |
| [`04_waveform_integrity_and_spectra.ipynb`](ptbxl/notebooks/04_waveform_integrity_and_spectra.ipynb) | Unannotated flatlines & rail spikes (`>10 mV`), Einthoven/Goldberger lead physics, `100 Hz` vs `500 Hz` spectra, HR sanity, and 12-lead beat templates |
| [`05_splits_leakage_and_batch_effects.ipynb`](ptbxl/notebooks/05_splits_leakage_and_batch_effects.ipynb) | Zero patient leakage across `strat_fold` 1–10, fold 9–10 human-validation skew, `device`/`site` shortcut checks, and PCA outlier detection |
| [`06_soft_segmentations_delineation.ipynb`](ptbxl/notebooks/06_soft_segmentations_delineation.ipynb) | [Zenodo 7610236](https://zenodo.org/records/7610236) 24-class soft segmentations (`v1.0.1` vs `v1.0.3` reconciliation, entropy SQI, PR/QRS/QTc intervals, and `NeuroKit2` benchmark) |
| [`07_anomalies_filtered_samples_and_anthropometrics.ipynb`](ptbxl/notebooks/07_anomalies_filtered_samples_and_anthropometrics.ipynb) | Exact census of filtered-out vs retained ECGs, clinical/signal differences, and forensic inspection of `height`/`weight` outliers and their ECGs |
| [`08_error_labels_catalog.ipynb`](ptbxl/notebooks/08_error_labels_catalog.ipynb) | Full-cohort catalog of [`ptbxl/quality/error_labels.csv`](ptbxl/quality/error_labels.csv) (`24` binary `err_*` flags + `4` severity tiers) with waveform galleries and `PTBXLDataset` filtering examples |

**Per-Record Error Labels (`ptbxl/quality/error_labels.csv`)**

Generated across all `21,799` records (`10,154` `0_clean`, `8,036` `1_minor_warning`, `2,695` `2_moderate_anomaly`, `914` `3_severe_error`) by:

```bash
python ptbxl/quality/generate_error_labels.py
```

---

### MIMIC-IV-ECG (`mimic_iv_ecg/`)

~800,000 ten-second 12-lead diagnostic ECGs at 500 Hz from ~160,000 unique patients (BIDMC, 2008–2019).

**Download**

```bash
./mimic_iv_ecg/download.sh
```

Data is saved to `mimic_iv_ecg/data/`. Total size is ~90 GB.

**Usage**

```python
from mimic_iv_ecg import MIMICIVECGDataset

# Subject-stratified splits (splits subjects, not records, to avoid leakage)
train_ds = MIMICIVECGDataset.from_subject_split("mimic_iv_ecg/data", split="train")
val_ds   = MIMICIVECGDataset.from_subject_split("mimic_iv_ecg/data", split="val")
test_ds  = MIMICIVECGDataset.from_subject_split("mimic_iv_ecg/data", split="test")

# Or load the full dataset with optional metadata tables & quality filtering
ds = MIMICIVECGDataset(
    root="mimic_iv_ecg/data",
    subject_ids=None,           # list[int] to filter specific patients
    load_measurements=True,     # merge machine_measurements.csv
    load_note_links=False,      # merge deduplicated waveform_note_links.csv
    clean_sentinels=False,      # True replaces 29999 measurement sentinels with NaN
    load_error_labels=False,    # True attaches per-study flags from mimic_iv_ecg/quality/error_labels.csv.gz
    max_error_severity=None,    # 0..3 optional cutoff (e.g. 2 drops '3_severe_error')
)

sample = ds[0]
# sample["signal"]       # float32 ndarray (12, 5000)  — 12 leads × 10 s × 500 Hz
# sample["subject_id"]   # int
# sample["study_id"]     # int
# sample["meta"]         # merged metadata columns as dict
# sample["error_labels"] # dict of severity_tier, error_tags, flags when load_error_labels=True
```

**Data Exploration & Quality Audit (`mimic_iv_ecg/notebooks/`)**

Six executed Jupyter notebooks audit all `800,035` MIMIC-IV-ECG studies (`161,352` patients, `500 Hz`):

| Notebook | Focus |
|---|---|
| [`00_sample_waveforms.ipynb`](mimic_iv_ecg/notebooks/00_sample_waveforms.ipynb) | Quick 12-lead waveform & multi-lead lineplot visualizations for samples `11`, `13`, `19`, `92`, and `42` |
| [`01_cohort_tables_and_timestamps.ipynb`](mimic_iv_ecg/notebooks/01_cohort_tables_and_timestamps.ipynb) | Table join integrity (`7` multi-note duplicate `study_id`s), heavy repeat-patient tail, same-timestamp duplicates, diurnal cadence, and `cart_id` fleet |
| [`02_machine_measurements_and_sentinels.ipynb`](mimic_iv_ecg/notebooks/02_machine_measurements_and_sentinels.ipynb) | `29999` sentinel values (up to `230k` in `p_end`), `int16`/`uint16` overflows (`65535`/`32767`/`-32768`), and cleaned HR/PR/QRS/QTc intervals |
| [`03_report_statements_and_filter_confounders.ipynb`](mimic_iv_ecg/notebooks/03_report_statements_and_filter_confounders.ipynb) | Top 20 machine report statements, `23,230` lead-reversal warnings, and hardware `bandwidth`/`filtering` (`50 Hz` US notch, `<not specified>`) confounders |
| [`04_waveform_integrity_500hz_and_neurokit2.ipynb`](mimic_iv_ecg/notebooks/04_waveform_integrity_500hz_and_neurokit2.ipynb) | Stratified `500 Hz` waveform scan (`NaN`, flatlines, `>10 mV`, Einthoven), `50 Hz` vs `60 Hz` notch PSD proof, and `NeuroKit2` vs MIMIC cart interval agreement |
| [`05_anomalies_and_error_labels_catalog.ipynb`](mimic_iv_ecg/notebooks/05_anomalies_and_error_labels_catalog.ipynb) | Full-cohort catalog of [`mimic_iv_ecg/quality/error_labels.csv.gz`](mimic_iv_ecg/quality/error_labels.csv.gz) (`15` `err_*` flags + `4` severity tiers), waveform galleries, and `MIMICIVECGDataset` filtering |

**Per-Study Error Labels (`mimic_iv_ecg/quality/error_labels.csv.gz`)**

Generated across all `800,035` studies (`421,460` `0_clean`, `333,848` `1_minor_warning`, `6,865` `2_moderate_anomaly`, `37,862` `3_severe_error`) by:

```bash
python mimic_iv_ecg/quality/generate_error_labels.py
```

---

### Chapman-Shaoxing (`chapman_shaoxing/`)

45,152 patients · 500 Hz · 10 s · SNOMED-CT labels (PhysioNet `ecg-arrhythmia 1.0.0`).

**Download**

```bash
./chapman_shaoxing/download.sh
```

Data is saved to `chapman_shaoxing/data/`. Total size is ~5 GB.

**Usage**

```python
from chapman_shaoxing import ChapmanShaoxingDataset

ds = ChapmanShaoxingDataset("chapman_shaoxing/data")

sample = ds[0]
# sample["signal"]       # float32 ndarray (12, 5000)
# sample["snomed_codes"] # list of SNOMED-CT code strings
# sample["label_names"]  # list of human-readable condition names
# sample["meta"]         # dict with age, sex, record_path
```

---

### Georgia 12-Lead ECG (`georgia_ecg/`)

~10,000 patients · 500 Hz · 10 s · 27 SNOMED-CT classes (PhysioNet/CinC 2020 Challenge).

**Download**

```bash
./georgia_ecg/download.sh
```

Data is saved to `georgia_ecg/data/`.

**Usage**

```python
from georgia_ecg import GeorgiaECGDataset

ds = GeorgiaECGDataset(
    "georgia_ecg/data",
    load_error_labels=False,   # True attaches per-record error flags from georgia_ecg/quality/error_labels.csv
    max_error_severity=None,   # 0..3 optional cutoff (e.g. 2 drops '3_severe_error', incl. 10 missing files)
)

sample = ds[0]
# sample["signal"]       # float32 ndarray (12, seq_len)
# sample["snomed_codes"] # list of SNOMED-CT code strings
# sample["label_names"]  # list of CinC 2020 condition abbreviations (e.g. ["NSR"])
# sample["meta"]         # dict with age, sex, record_path
# sample["error_labels"] # dict of severity_tier, error_tags, flags when load_error_labels=True
```

**Data Exploration & Quality Audit (`georgia_ecg/notebooks/`)**

Six executed Jupyter notebooks audit all `10,344` Georgia 12-Lead ECG records (`500 Hz`):

| Notebook | Focus |
|---|---|
| [`00_sample_waveforms.ipynb`](georgia_ecg/notebooks/00_sample_waveforms.ipynb) | Quick 12-lead waveform & multi-lead lineplot visualizations for samples `11`, `13`, `19`, `92`, and `42` |
| [`01_cohort_headers_and_demographics.ipynb`](georgia_ecg/notebooks/01_cohort_headers_and_demographics.ipynb) | Missing/corrupt `.hea`/`.mat` files (`10`), non-10s durations (`52`), `# Age:`/`# Dx:` header parsing fix, missing/pediatric ages, and `g1`–`g11` balance |
| [`02_snomed_labels_and_cinc2020_audit.ipynb`](georgia_ecg/notebooks/02_snomed_labels_and_cinc2020_audit.ipynb) | Scored (`27`) vs unscored (`84`) CinC 2020 SNOMED-CT coverage, official count audit, multi-label co-occurrence, and age-decade prevalence |
| [`03_waveform_integrity_and_500hz_spectra.ipynb`](georgia_ecg/notebooks/03_waveform_integrity_and_500hz_spectra.ipynb) | `500 Hz` signal defects (`NaN`, flatlines, `>10 mV` spikes), Einthoven/Goldberger lead physics, `60 Hz` US mains hum PSD, and 12-lead beat templates |
| [`04_neurokit2_delineation_and_intervals.ipynb`](georgia_ecg/notebooks/04_neurokit2_delineation_and_intervals.ipynb) | `NeuroKit2` `500 Hz` DWT delineation, rhythm/HR sanity (`SB`/`NSR`/`STach`/`AF`), PR/QRS/QTc clinical validation, and visual overlays |
| [`05_anomalies_and_error_labels_catalog.ipynb`](georgia_ecg/notebooks/05_anomalies_and_error_labels_catalog.ipynb) | Full-cohort catalog of [`georgia_ecg/quality/error_labels.csv`](georgia_ecg/quality/error_labels.csv) (`18` `err_*` flags + `4` severity tiers), waveform galleries, and `GeorgiaECGDataset` filtering |

**Per-Record Error Labels (`georgia_ecg/quality/error_labels.csv`)**

Generated across all `10,344` records (`8,770` `0_clean`, `941` `1_minor_warning`, `584` `2_moderate_anomaly`, `49` `3_severe_error`) by:

```bash
python georgia_ecg/quality/generate_error_labels.py
```

> **Note:** Chapman-Shaoxing and Georgia ECG share the same file format and
> base class (`CinCStyleDataset`), making it easy to combine them for
> multi-source training.

---

### CODE-15% (`code15/`)

345,779 exams · 233,770 patients · 400 Hz · 4,096 samples · 6 binary labels
(`1dAVb`, `RBBB`, `LBBB`, `SB`, `AF`, `ST`). Stored as HDF5 part files (Zenodo).

**Download**

```bash
./code15/download.sh
```

Data is saved to `code15/data/`. Total size is ~46 GB.

**Usage**

```python
from code15 import CODE15Dataset

# Patient-stratified splits (avoids data leakage)
train_ds = CODE15Dataset.from_patient_split("code15/data", split="train")
val_ds   = CODE15Dataset.from_patient_split("code15/data", split="val")
test_ds  = CODE15Dataset.from_patient_split("code15/data", split="test")

# Or load the full dataset
ds = CODE15Dataset("code15/data")

sample = ds[0]
# sample["signal"]  # float32 ndarray (12, 4096)
# sample["labels"]  # dict {"1dAVb": 0, "RBBB": 1, ...}
# sample["meta"]    # age, is_male, patient_id, ...
```

---

## Requirements

Core dataset loaders (`requirements.txt`):

```
torch
numpy
pandas
wfdb
h5py
```

Additional packages used by the exploration notebooks (`ptbxl/notebooks/`, `mimic_iv_ecg/notebooks/`, `georgia_ecg/notebooks/`) and `quality/generate_error_labels.py`:

```
matplotlib
seaborn
scipy
scikit-learn
neurokit2
jupyter
```
