"""Statistical utilities used by the AT172 audit."""
from __future__ import annotations

import itertools
import numpy as np


def exact_two_sided_sign_flip_pvalue(differences) -> float:
    """Enumerate all 2^n sign assignments, retaining zero differences.

    Zero differences are not dropped. Their sign has no numerical effect but is
    retained in the predeclared five-pair randomization space used in the paper.
    """
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistics.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def paired_summary(candidate, baseline):
    differences = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    return {
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "minimum_difference": float(differences.min()),
        "maximum_difference": float(differences.max()),
        "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip_pvalue(differences),
        "differences": differences.tolist(),
    }

