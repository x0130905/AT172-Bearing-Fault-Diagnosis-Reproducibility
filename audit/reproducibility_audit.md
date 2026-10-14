# AT172 Reproducibility Audit

Audit date: 2026-10-13  
Paper: Evaluating Lightweight Source-Only Domain Generalization for Bearing Fault Diagnosis under Unseen Operating Conditions

## Executive conclusion

The retained evidence supports every reported Table II-V value and the headline conclusions. All 20 checkpoints replay to the retained 3-hp predictions; all prediction arrays recompute to the stored metrics; and a complete fresh rerun under the separately recorded audit environment reproduces all 20 class-prediction arrays and all metrics exactly. Floating probabilities, histories, and checkpoint values differ only at machine precision (maximum checkpoint absolute difference below 9e-15), so the rerun is functionally identical at 1e-12 but not byte-for-byte identical.

No training/test leakage was found. The 3-hp domain is evaluated after training and is not accessed by the training function. The 2-hp domain is used for early stopping and checkpoint selection and is therefore a validation domain, not an untouched target. The paper describes the final 3-hp load as unseen and target-isolated.

One protocol statement was too strong: Mixup does not retain the same later source permutation sequence as ERM, CORAL, and MMD because its Beta and interpolation draws consume the same seeded random generator. The result artifacts were not changed. The manuscript and README now state this behavior accurately.

## Status definitions

- **VERIFIED**: directly checked against readable data, code, or retained artifacts.
- **RECONSTRUCTABLE**: can be recomputed from supplied verified artifacts.
- **MISSING**: expected evidence is absent.
- **INCONSISTENT**: evidence conflicts with a claim or is not a valid dataset file.
- **UNVERIFIABLE**: the retained record cannot establish the historical fact.

## Data and domain isolation

| Item | Status | Evidence |
|---|---|---|
| Official CWRU source and protocol | VERIFIED | CWRU official normal-baseline and 12-kHz drive-end pages; official MAT URLs recorded in the manifest. |
| File IDs and class/load mapping | VERIFIED | 97-100 normal; 105-108 inner race; 118-121 ball; 130-133 centered outer race; loads 0-3 hp. |
| Sixteen MAT file byte counts and SHA-256 hashes | VERIFIED | Local files match `data_manifests/dataset_manifest.csv`. |
| Window length, stride, and count | VERIFIED | 1,024-sample length and stride; 80 windows per recording; 1,280 total windows. |
| Split counts | VERIFIED | 640 training, 320 validation, 320 unseen-test windows. |
| Window overlap | VERIFIED | Consecutive intervals within every file satisfy prior end <= next start. |
| Cross-split file/window leakage | VERIFIED | Every file belongs to one load and one split; no file or window crosses splits. |
| Normalization | VERIFIED | Mean and standard deviation are calculated independently along each window. |
| 3-hp use during training/selection | VERIFIED ABSENT | Dynamic guard test rejects any key-3 access from `train()`; code uses loads 0/1 for updates and load 2 for selection. |
| 2-hp role | VERIFIED | Used for validation Macro-F1, early stopping, and checkpoint selection. It is not described as untouched in the corrected package. |
| No historical post-hoc hyperparameter or seed search | UNVERIFIABLE | Retained code/configs use fixed settings and five reported seeds, but absent historical activity cannot be proven from files alone. |

The directory also contained incomplete download remnants `99.http.mat`, `99.part`, and `99.range.mat`. They are not referenced by the manifest or experiment and are excluded from the public package. The first and third appear in the indexed file types and are marked INCONSISTENT as dataset candidates, not as evidence of a scientific inconsistency.

Official source pages checked on 2026-10-13:

- https://engineering.case.edu/bearingdatacenter/normal-baseline-data
- https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
- https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures

## Model and training audit

| Claim | Status | Finding |
|---|---|---|
| 228 trainable parameters | VERIFIED | Shapes are (9,1,16), (16,), (16,4), and (4,), totaling 144+16+64+4=228. |
| Shared inference architecture | VERIFIED | All four methods instantiate the same CNN class; methods change only the training objective/data interpolation. |
| Same seed-specific initialization | VERIFIED | Parameter arrays are identical for repeated construction with the same seed. |
| Same source order for all methods | INCONSISTENT in prior wording | ERM/CORAL/MMD share permutations; Mixup consumes additional RNG draws and changes later permutations. Manuscript corrected. |
| Adam learning rate 0.002 | VERIFIED | Fixed configuration and update equations. |
| 32 samples per source domain | VERIFIED | Two separately permuted domain batches of 32 are concatenated. |
| Maximum 30 epochs, patience 8 | VERIFIED | Fixed configuration; histories terminate consistently with early stopping. |
| Validation metric Macro-F1 | VERIFIED | Load-2 Macro-F1 selects the best checkpoint. |
| CORAL | VERIFIED | Unbiased source covariance; squared Frobenius difference divided by 4d^2; lambda 1.0. |
| Mixup | VERIFIED | Beta(0.2,0.2), input and one-hot target interpolation, soft-label cross-entropy. |
| Linear MMD | VERIFIED | Squared source mean difference divided by feature dimension; lambda 1.0. |
| Gradient implementation | VERIFIED for representative numerical checks | Central finite differences pass for ERM, CORAL, Mixup soft labels, and MMD. |
| Saved checkpoints match predictions | VERIFIED | All 20 checkpoints replay to saved labels/predictions; probabilities agree within 2e-15. |

## Numerical result verification

The full prediction-level comparison is in `audit/recomputed/paper_results_verification.csv`.

| Method | Recomputed Macro-F1 mean (%) | Sample SD (%) | Paper value (%) | Status |
|---|---:|---:|---:|---|
| ERM | 85.242536 | 20.405157 | 85.24 +/- 20.41 | VERIFIED after rounding |
| CORAL | 85.242536 | 20.405157 | 85.24 +/- 20.41 | VERIFIED after rounding |
| Mixup | 78.581668 | 20.445661 | 78.58 +/- 20.45 | VERIFIED after rounding |
| Linear MMD | 85.365611 | 20.343217 | 85.37 +/- 20.34 | VERIFIED after rounding |

Additional findings:

- CORAL and ERM predictions are exactly identical for all five seeds.
- MMD changes predictions relative to ERM only for seeds 20260916 and 42.
- MMD minus ERM mean paired Macro-F1 difference is +0.123075 percentage points; exact two-sided sign-flip p=0.50.
- Mixup minus ERM mean paired Macro-F1 difference is -6.660868 percentage points; p=0.8125.
- Mean outer-race recall is 61.75% for ERM/CORAL, 43.50% for Mixup, and 62.00% for MMD.
- The maximum source-to-test relative Macro-F1 drop is 16.253641% for ERM/CORAL/MMD, seed 20260917.
- ERM seed 20260917 has 80 outer-race test windows: 67 predicted as ball and 13 predicted correctly.
- Table II uses 99 logged final-mini-batch observations per objective across five seeds. These are diagnostic logged batches, not epoch averages.
- The exact sign-flip test enumerates all 2^5 sign assignments and retains zero paired differences; their sign has no numerical effect but remains in the five-pair randomization space.

## Fresh rerun

Fresh rerun environment:

- Windows 11, AMD64, Intel processor.
- CPython 3.12.3.
- NumPy 2.5.3, SciPy 1.18.1, Pillow 12.3.0.
- CPU NumPy; no GPU.
- OpenBLAS reported by NumPy; relevant thread environment variables were unset.
- Full environment details: `audit/environment_report.json`.

All 20 runs completed in 77.89 seconds in an independent output directory. All test labels, class predictions, and metrics match the retained artifacts exactly. Floating probabilities differ by at most approximately 1e-15; checkpoint arrays differ by at most 8.72e-15; logged loss/history values differ by at most 4.45e-16. These are machine-precision differences and do not change any reported result.

The fresh rerun establishes reproducibility in the recorded audit environment. Separate forensic recovery verifies Windows AMD64, CPython 3.12.x, a CPU-only NumPy path, and SciPy 1.18.1 for the original experiment; the exact Python patch, NumPy/BLAS build, Windows release/build, CPU model, RAM, and thread settings remain unknown.

## Reproducibility package completeness

The archive contains configuration, portable training code, file/window manifests, checkpoints, prediction arrays, metrics, histories, statistical scripts, tests, reconstructed tables, final figures, an environment report, inventory, and SHA-256 list. It excludes the CWRU MAT files and gives official download locations and integrity hashes.

The package is publicly available at https://github.com/x0130905/AT172-Bearing-Fault-Diagnosis-Reproducibility. Conference upload and venue metadata remain author-side administrative actions.

## Overall assessment

- Artifact verification: **VERIFIED**.
- Data isolation: **VERIFIED**.
- Fresh rerun: **VERIFIED in the recorded audit environment**.
- Historical environment provenance: **PARTIALLY VERIFIED**, with unresolved fields explicitly listed.
- Public accessibility: **VERIFIED** at the repository URL above.
- Submission readiness: **Technically ready**; conference upload, registration, payment, and copyright remain author-side actions.

