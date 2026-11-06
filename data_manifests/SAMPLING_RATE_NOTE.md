# Native sampling-rate provenance

This additive note corrects a historical blanket 12-kHz description. Original configurations, file and window manifests, source code, raw recordings, and experimental artifacts are unchanged.

## Evidence

- Smith, W. A., and Randall, R. B. (2015), Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study, DOI https://doi.org/10.1016/j.ymssp.2015.04.021 . Table A1 lists normal files 97-100 at 48 kHz; Table A2 lists the selected fault files 105-108, 118-121, and 130-133 at 12 kHz. The author-uploaded full text is available at https://www.researchgate.net/publication/276248554_Rolling_Element_Bearing_Diagnostics_Using_the_Case_Western_Reserve_University_Data_A_Benchmark_Study .
- Official file identity and load mapping: https://engineering.case.edu/bearingdatacenter/normal-baseline-data and https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data .
- `data_manifests/dataset_manifest.csv` records the 16 exact file hashes; `domain_split_manifest.csv` records the 1,280 windows and native sample boundaries.
- `src/experiment.py`, `de_signal` and `load_dataset`, reads native DE_time vectors, slices windows, and applies per-window normalization, with no resampling. `sampling_frequency_hz` occurs only in its configuration declaration and is not consumed by the loader.

## Metadata interpretation

`sampling_rate_metadata.csv` adds documentary metadata keyed by file ID and original recorded SHA-256. Sampling rates come from the cited reference tables; they are not values read from a MAT-file rate field. Window duration is `window_samples / native_sampling_frequency_hz * 1000` milliseconds (21.333333 or 85.333333). The historical blanket setting 12000 is preserved and must not be used as evidence that normal recordings were resampled.

Class and acquisition rate/physical window duration are associated. This is a potential confound in interpreting normal-state recognition, not proof that the model exploited it. The size of this potential confounding effect has not been independently quantified; existing results cannot isolate the effect of native rate or window duration. The Mixup comparison additionally couples interpolation with later source-order changes because one generator supplies both types of draws.

No training, new evaluation, or modified preprocessing is introduced by this documentation correction. Original frozen manifests are deliberately not edited. Local raw-file SHA-256 checks and source/manifest inspection are distinct from a fresh scientific reproduction.
