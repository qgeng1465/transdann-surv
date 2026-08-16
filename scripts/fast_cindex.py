#!/usr/bin/env python3
"""fast_cindex.py; Vectorised, bit-exact port of lifelines' concordance index.

lifelines' implementation is O(n log n) via a sorted BTree; we replicate its
exact summary statistics using a maintained sorted numpy pool + batch
``np.searchsorted``.  Validated bit-identical (diff == 0.0) against
``lifelines.utils.concordance_index`` on heavy-tie real data.
"""

import numpy as np


def fast_lifelines_cindex(times, pred, events):
    """C-index with lifelines' exact tie/censoring handling.

    Parameters mirror ``lifelines.utils.concordance_index``:
      times  : observed survival/censoring times
      pred   : predicted_scores (what lifelines receives; we pass ``-risks``)
      events : 1 = event, 0 = censored
    """
    times = np.asarray(times, dtype=float)
    pred = np.asarray(pred, dtype=float)
    events = np.asarray(events, dtype=float)
    died_mask = events.astype(bool)
    if not died_mask.any():
        return np.nan

    d_t = times[died_mask]; d_p = pred[died_mask]
    c_t = times[~died_mask]; c_p = pred[~died_mask]
    o = np.argsort(d_t, kind="mergesort"); d_t = d_t[o]; d_p = d_p[o]
    o = np.argsort(c_t, kind="mergesort"); c_t = c_t[o]; c_p = c_p[o]

    pool = np.empty(0, dtype=float)   # sorted preds of died samples "in play"
    num_pairs = 0
    num_correct = 0
    num_tied = 0
    di = 0
    ci = 0
    n_d = len(d_t)
    n_c = len(c_t)

    def handle_batch(preds):
        nonlocal num_pairs, num_correct, num_tied
        B = len(preds)
        if B == 0:
            return
        num_pairs += len(pool) * B
        if len(pool):
            left = np.searchsorted(pool, preds, side="left")
            right = np.searchsorted(pool, preds, side="right")
            num_correct += int(left.sum())
            num_tied += int((right - left).sum())

    while di < n_d or ci < n_c:
        if ci >= n_c or (di < n_d and d_t[di] <= c_t[ci]):
            # died batch at d_t[di]; query then insert
            t0 = d_t[di]
            di_end = di
            while di_end < n_d and d_t[di_end] == t0:
                di_end += 1
            handle_batch(d_p[di:di_end])
            pool = np.sort(np.concatenate([pool, d_p[di:di_end]]))
            di = di_end
        else:
            # censored batch at c_t[ci]; query only (never inserted)
            t0 = c_t[ci]
            ci_end = ci
            while ci_end < n_c and c_t[ci_end] == t0:
                ci_end += 1
            handle_batch(c_p[ci:ci_end])
            ci = ci_end

    if num_pairs == 0:
        return np.nan
    return (num_correct + num_tied / 2.0) / num_pairs
