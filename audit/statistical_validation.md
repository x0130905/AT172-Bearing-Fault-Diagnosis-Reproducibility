# Statistical Validation

All test metrics were recalculated directly from the retained `y` and `prediction` arrays. No prediction array was reconstructed from aggregate results.

## Definitions

- Accuracy is the fraction of correct predictions.
- Per-class precision uses TP/(TP+FP), with precision set to zero if a class receives no predictions.
- Per-class recall uses TP/(TP+FN).
- Per-class F1 is the harmonic mean; it is zero when precision and recall are both zero.
- Macro metrics are unweighted means over four classes.
- Across-seed variability is the sample standard deviation (`ddof=1`).
- Paired differences use the same five seeds in the stated order.
- The exact two-sided sign-flip p-value enumerates all 32 sign assignments and compares the absolute mean difference with the observed absolute mean. Zero differences are retained.

## Verification result

Every recomputed test metric matches its retained `metrics.json` value within 1e-12. Tables II-V regenerate from histories and predictions. The headline means, sample standard deviations, distribution summaries, paired deltas, p-values, class recalls, and the seed-20260917 67/80 outer-race-to-ball count are verified.

The complete row-level audit is `recomputed/paper_results_verification.csv`, and the exact paired test is implemented in `../analysis/statistical_tests.py`.

