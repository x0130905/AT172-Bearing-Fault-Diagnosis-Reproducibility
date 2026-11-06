# Sampling-rate and interpretation documentation correction

Date: 2026-11-06

This documentation update supersedes the blanket description of all selected recordings as 12 kHz. It changes no experiment code, configuration, original manifest, model, prediction, checkpoint, history, numerical result or statistical method.

## Corrected protocol description

- Normal files 97-100 have a native sampling rate of 48 kHz; selected fault files 105-108, 118-121 and 130-133 have a native rate of 12 kHz, as identified in Smith and Randall (2015), Tables A1/A2.
- Official CWRU pages establish the file identities and load mapping. The existing source loader establishes direct native-sample windowing followed by per-window normalization, with no resampling.
- Windows contain 1,024 samples: approximately 21.33 ms for normal recordings and 85.33 ms for fault recordings.
- Historical `sampling_frequency_hz=12000` values are retained declarations, not evidence of resampling. Class, sampling rate and physical window duration are associated, and the magnitude of this potential confound has not been independently quantified. The retained results do not prove a sampling-rate shortcut.
- The existing shared random generator couples Mixup interpolation with later source-order changes. The comparison cannot isolate their effects.

The updated README and `data_manifests/SAMPLING_RATE_NOTE.md` separate literature rates, official file identity and code-path evidence. `data_manifests/sampling_rate_metadata.csv` is additive metadata; the original manifests are not edited.

## Checks for this documentation update

The release work independently checked the original file/window manifests and raw-file hashes without retraining. SHA256SUMS.txt is regenerated with the existing repository script and verified against the complete repository. Historical audit statements remain dated records, not claims that this documentation update performed their earlier training or replay checks.

## Table IV presentation units

The manuscript header now reads `Paired Δ vs ERM (pp)`; percentage-valued median/minimum/maximum columns carry `(%)`, and p-values are unitless. No table values change. The existing `analysis/recompute_metrics.py` continues to write its historical CSV field `paired_delta_vs_erm_pct`. That field stores differences between percentage-valued Macro-F1 scores and is interpreted as percentage points. This documentation-only update does not rename historical output fields or modify the script.

For the authoritative state of this update, use the default-branch commit and pull-request history containing this file. There is no separate patch-application step for a checkout that already contains these documents. Earlier locally delivered patch instructions describe their delivery time and are superseded by the merged repository version.
