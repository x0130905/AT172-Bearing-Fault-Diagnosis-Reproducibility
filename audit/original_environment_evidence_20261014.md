# Original environment evidence recovery

This record separates the original E2 five-seed experiment environment from the controlled 2026-10-14 verification environment.

## Verified historical facts

- Platform family: Microsoft Windows on AMD64/64-bit.
- Interpreter: CPython 3.12.x. The exact patch and build are not retained.
- Numerical path: CPU-only NumPy implementation; PyTorch, scikit-learn, CUDA, and GPU acceleration were not used.
- SciPy: 1.18.1, loaded from the project-local dependency directory.
- Final E2 configuration: seeds `20260916`, `20260917`, `20260918`, `42`, and `123`; 30 maximum epochs; batch size 32 per source domain; Adam learning rate 0.002 with beta1 0.9, beta2 0.999, and epsilon 1e-8; early-stopping patience 8.
- E2 source: the separate `raw_results_v4` five-seed records used by the paper. E1 shared ERM/CORAL arrays for the three common seeds match E2 exactly.

Evidence consists of retained Python 3.12 bytecode metadata, the project-local `scipy-1.18.1-cp312-cp312-win_amd64` installation record, source imports and code paths, configuration snapshots, run timestamps, result arrays, and hashes. The complete author-held forensic archive is not needed to execute this public reproduction package.

## Inferred or unresolved historical fields

- Exact Python patch/build: UNKNOWN.
- Exact NumPy version and BLAS/LAPACK build: UNKNOWN.
- Exact Windows release/build: UNKNOWN.
- CPU model, RAM, physical GPU inventory, and thread settings: UNKNOWN.
- Original experiment Git commit: UNKNOWN; the historical working tree was not committed.

The controlled audit environment in `environment_verification_20261014.json` is a compatible independently recorded environment. It is not presented as the original environment.
