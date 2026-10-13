# Evaluating Lightweight Source-Only Domain Generalization for Bearing Fault Diagnosis under Unseen Operating Conditions

Reproducibility materials for paper **AT172**, accepted by the **2026 6th International Conference on Control Theory and Applications (ICoCTA 2026)**.

## Overview

This repository contains the NumPy implementation, frozen experiment configuration, model checkpoints, prediction outputs, training histories, data manifests, automated tests, statistical analysis, and figure/table reconstruction code used for the paper. It evaluates a 228-parameter 1D-CNN with ERM, source-only CORAL, Mixup, and linear MMD under a held-out-load protocol.

## Scope and evidence levels

This archive supports two distinct activities:

1. **Artifact verification** recomputes metrics and paper tables from the retained prediction arrays, histories, checkpoints, manifests, and configuration. This path verifies the published numerical claims without retraining.
2. **Fresh rerun** downloads or reads the official CWRU MAT files and retrains all four methods and five seeds in a new environment. A fresh rerun is a new execution, not proof of the unknown historical software environment.

The original CWRU recordings are not redistributed. Historical Python and package versions were not logged and are therefore **UNKNOWN**. `requirements.txt` and `environment_report.json` describe the environment used for the fresh rerun performed during this audit.

## Protocol

- Official 12-kHz drive-end recordings.
- Classes: normal, 0.007-inch inner race, ball, and centered outer-race faults.
- File IDs: 97-100, 105-108, 118-121, and 130-133.
- Source training: 0 hp and 1 hp.
- Validation and checkpoint selection: 2 hp.
- Unseen final test: 3 hp, evaluated only after training.
- Per recording: first 80 non-overlapping windows of 1,024 samples.
- Per-window z-score normalization uses only that window.
- Methods: ERM, CORAL, Mixup, linear MMD.
- Seeds: 20260916, 20260917, 20260918, 42, 123.

The 2-hp domain is a validation/model-selection domain. It is not the unseen 3-hp test domain. The experiment is source-only with respect to the final 3-hp target.

## Install

Create a clean CPython 3.12 environment, then run:

```powershell
python -m pip install -r requirements.txt
```

## Artifact verification

No CWRU download is needed to recompute test metrics from the retained predictions:

```powershell
python reproduce.py verify-artifacts
```

To additionally verify the local MAT files against the recorded byte counts and SHA-256 hashes:

```powershell
python reproduce.py verify-artifacts --data-dir C:\path\to\cwru_mat_files
```

The command runs the integrity, domain-isolation, window-manifest, model/gradient, and prediction-metric tests. It also rebuilds Tables II-V under `audit/recomputed/`.

## Dataset source and download

The original vibration data are available from the [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter). This repository does **not** redistribute the CWRU MAT files. Users must obtain them from the official source and comply with any terms stated by Case Western Reserve University.

The required 12-kHz drive-end file identifiers are:

| Class | File IDs for 0, 1, 2, and 3 hp |
|---|---|
| Normal | 97, 98, 99, 100 |
| Inner-race fault, 0.007 in | 105, 106, 107, 108 |
| Ball fault, 0.007 in | 118, 119, 120, 121 |
| Centered outer-race fault, 0.007 in | 130, 131, 132, 133 |

The exact filenames, official download URLs, byte counts, and SHA-256 hashes are recorded in `data_manifests/dataset_manifest.csv`.

Official data pages:

The official entry points are:

- https://engineering.case.edu/bearingdatacenter/normal-baseline-data
- https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data

Automatic download with manifest verification:

```powershell
python reproduce.py download-data --data-dir C:\path\to\cwru_mat_files
```

The package accepts a file only when its byte count and SHA-256 match `data_manifests/dataset_manifest.csv`.

## Fresh rerun

Use a new, empty output directory:

```powershell
python reproduce.py fresh-rerun `
  --data-dir C:\path\to\cwru_mat_files `
  --output-dir C:\path\to\new_results
```

Or run download, artifact verification, and retraining together:

```powershell
.\run_all.ps1 -DataDir C:\path\to\cwru_mat_files -OutputDir C:\path\to\new_results
```

Linux/macOS:

```sh
sh run_all.sh /path/to/cwru_mat_files /path/to/new_results
```

The fresh-rerun command refuses to overwrite a non-empty output directory.

## Rebuild metrics and tables

```powershell
python analysis/recompute_metrics.py --artifact-root . --output-dir audit/recomputed
```

Outputs include the complete prediction-to-metric audit, class recalls, and Tables II-V. The exact paired sign-flip implementation is in `analysis/statistical_tests.py`; it enumerates all 2^5 sign assignments and retains zero differences in the five-pair randomization space.

Regenerate Figs. 1-2 from the retained metrics and histories:

```powershell
python analysis/make_figures.py --artifact-root . --output-dir audit/recomputed_figures
```

## Expected results

The retained unseen 3-hp results use five predeclared seeds and sample standard deviation:

| Method | Macro-F1, mean ± SD (%) |
|---|---:|
| ERM | 85.24 ± 20.41 |
| CORAL | 85.24 ± 20.41 |
| Mixup | 78.58 ± 20.45 |
| Linear MMD | 85.37 ± 20.34 |

`python reproduce.py verify-artifacts` recomputes the per-run metrics and Tables II-V from the retained artifacts. It is an artifact-verification path, not a replacement for retraining. A full fresh rerun requires the official MAT files and an empty output directory.

## Important random-order note

All methods use identical seed-specific initial parameters. ERM, CORAL, and linear MMD also use identical source permutations. In the retained implementation, Mixup uses the same seeded generator for source permutations and its additional Beta/interpolation draws; these extra draws change later source permutations. The package preserves this behavior so the retained experiment can be reproduced exactly.

## Directory guide

- `configs/`: frozen experimental configuration.
- `src/`: path-explicit, non-destructive training implementation.
- `data_manifests/`: file and window manifests; no raw CWRU data.
- `checkpoints/`, `predictions/`, `metrics/`, `histories/`: retained per-run evidence.
- `analysis/`: metric, statistical, and table reconstruction.
- `figures/`, `tables/`: retained and regenerated presentation artifacts.
- `tests/`: automated integrity, isolation, parameter, gradient, and metric tests.
- `audit/`: inventory, numerical verification, environment record, and unresolved limitations.

## Provenance and limits

- The retained checkpoints and predictions are directly verifiable against the supplied hashes.
- The exact historical interpreter, NumPy/SciPy versions, BLAS build, CPU thread settings, and dependency lock file are **UNKNOWN**.
- The audit fresh rerun was performed with CPython 3.12, NumPy 2.5.3, SciPy 1.18.1, and Pillow 12.3.0. The captured software and hardware disclosure is in `audit/environment_report.json`.
- The audit fresh rerun is recorded separately from artifact verification and must not be interpreted as recovery of the unknown historical environment.
- The evaluation covers one laboratory dataset, one seeded-fault diameter, four closed-set classes, one sensor channel, one held-out load, and five random seeds. It does not establish industrial field reliability or a universal ranking of domain-generalization methods.

## License and redistribution boundaries

The existing `LICENSE` applies only to original AT172 code and documentation. It does not license the CWRU dataset, third-party packages, conference templates, or cited publications. Raw CWRU recordings are deliberately excluded. Model checkpoints and prediction artifacts are retained for verification; users remain responsible for reviewing the official dataset terms before redistributing derived materials.
