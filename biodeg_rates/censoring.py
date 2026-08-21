"""Censoring convention for the Mann-Kendall / Theil-Sen module (reference Appendix A).

Committed choice: simple recensoring, valid under light censoring. It is internally
consistent across the test statistic, its variance, and the slope, which is the property
the rank-based confidence interval needs.

The engineering rule from the reference: one function takes the record and emits a single
object that both the S statistic and the Theil-Sen slope set consume, so the tie and
censoring conventions are defined exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Above this censored fraction the simple convention biases toward an attenuated trend;
# the estimator raises this as a warning and the confidence drops.
CENSORED_FRACTION_WARN = 0.20


@dataclass
class RecensoredSeries:
    """A well series recoded under the simple recensoring rule.

    censored[i] is True for every observation strictly below RL_star (any non-detect, or a
    detect reported below the highest non-detect limit). Those points collapse into one
    mutually tied class. Determinate detects (>= RL_star) keep their numeric value.
    """

    t: np.ndarray                 # time (years), ascending
    value: np.ndarray            # numeric concentration (RL substituted for non-detects)
    censored: np.ndarray         # bool: member of the single censored class
    rl_star: float               # highest reporting limit among the non-detects
    n: int
    n_censored: int

    @property
    def censored_fraction(self) -> float:
        return self.n_censored / self.n if self.n else 0.0

    @property
    def heavily_censored(self) -> bool:
        return self.censored_fraction > CENSORED_FRACTION_WARN


def recensor(t, value, detect, rl) -> RecensoredSeries:
    """Apply the simple recensoring rule (Appendix A.1) to one well series.

    Inputs are aligned arrays. ``value`` holds the reported concentration with the reporting
    limit already substituted for non-detects. ``detect`` is the quantified-detection flag.
    ``rl`` is the per-event reporting limit.
    """
    t = np.asarray(t, float)
    value = np.asarray(value, float)
    detect = np.asarray(detect, bool)
    rl = np.asarray(rl, float)

    nd = ~detect
    rl_star = float(np.max(rl[nd])) if np.any(nd) else 0.0
    # Member of the censored class: any non-detect, or any detect reported below RL_star.
    censored = nd | (detect & (value < rl_star))
    order = np.argsort(t, kind="mergesort")
    return RecensoredSeries(
        t=t[order], value=value[order], censored=censored[order],
        rl_star=rl_star, n=int(len(t)), n_censored=int(np.sum(censored)),
    )


def pair_sign(i: int, j: int, rec: RecensoredSeries) -> int:
    """Determinate sign of the (i, j) comparison under recensoring (Appendix A.2).

    Both determinate -> sign of the value difference. Censored vs determinate -> the
    determinate sign, because a censored value is strictly below RL_star and a retained
    detect is at or above it. Both censored -> a tie (0).
    """
    ci, cj = rec.censored[i], rec.censored[j]
    if not ci and not cj:
        d = rec.value[j] - rec.value[i]
        return int(np.sign(d))
    if ci and not cj:
        return +1
    if cj and not ci:
        return -1
    return 0


def tie_groups(rec: RecensoredSeries) -> list[int]:
    """Sizes of tied groups for the variance tie correction (Appendix A.3): the single
    censored group plus any groups of exactly equal determinate values."""
    groups = []
    if rec.n_censored > 0:
        groups.append(rec.n_censored)
    det_vals = rec.value[~rec.censored]
    if det_vals.size:
        _, counts = np.unique(np.round(det_vals, 12), return_counts=True)
        groups.extend(int(c) for c in counts if c > 1)
    return groups
