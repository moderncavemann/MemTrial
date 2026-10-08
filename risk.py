"""Trailing covariance of daily returns before a decision date (no look-ahead). Unlisted assets: flat price."""
import datetime
import numpy as np
import scoring as SC
RS = SC.RS
_COV = {}
def cov_for(d, assets, L=60):
    key = (d, tuple(assets))
    if key in _COV: return _COV[key]
    i = RS.SIDX[datetime.date.fromisoformat(d)]; sess = RS.SESS[i - L: i + 1]
    M = np.full((len(assets), len(sess)), np.nan)
    for k, a in enumerate(assets):
        p = RS.PR[a]
        for j, s in enumerate(sess):
            if s in p: M[k, j] = p[s]
        row = M[k]
        for j in range(1, len(row)):
            if np.isnan(row[j]): row[j] = row[j - 1]
        for j in range(len(row) - 2, -1, -1):
            if np.isnan(row[j]): row[j] = row[j + 1]
        if np.isnan(row).all(): row[:] = 1.0
    R = M[:, 1:] / M[:, :-1] - 1
    _COV[key] = np.cov(R); return _COV[key]
