"""Exact utility of portfolio mixtures on a realised 20-session path, vectorised over mixtures.
Q: mixture weights over the rows of W (expert weight vectors); G: each expert's relative value path; prev: holdings before
the trade. The 15 bp fee is charged once on the traded amount (changes below 1e-6 ignored, as in PortBench);
utility = net 20-session return - (gamma / 2) * 20 * variance of the daily returns."""
import numpy as np


def U_batch(Q, W, G, prev, gamma):
    Wm = Q @ W; ad = np.abs(Wm - prev); to = np.where(ad >= 1e-6, ad, 0).sum(1)
    nav = 1 - 0.0015 * to
    path = np.concatenate([np.ones((len(Q), 1)), nav[:, None] * (Q @ G)], 1)
    r = path[:, 1:] / path[:, :-1] - 1; J = path[:, -1] - 1
    return J - 0.5 * gamma * r.var(1) * 20
