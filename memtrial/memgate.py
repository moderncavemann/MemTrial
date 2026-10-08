"""MemGate core (2026-10-04): learning when to trust memory.

Per decision date t the agent is run with designed subsets of the k = 4 retrieved experiences. After the outcome matures:
  (1) counterfactual contribution  c[t,h] = mean utility of subsets containing h - mean of subsets without h (Banzhaf);
  (2) experience value learning     hierarchical Bayesian model (Fay-Herriot) on date-centred contributions,
                                    m_h = v_h + e_h, e_h ~ N(0, s2/n_h);  v_h = z_h' beta + u_h, beta ~ N(0, sb2 I), u_h ~ N(0, tau2)
                                    z_h = content features of the experience (PCA of its retrieval embedding); hyperparameters
                                    (sb2, tau2) by marginal likelihood; posterior means predict the value of seen AND unseen
                                    experiences;
  (3) prequential trust gate        before each date the learner predicts the value of that date's experiences; once the date
                                    matures, the prediction is scored against the realised contributions (Pearson r across the
                                    retrieved experiences). The gate opens when the mean forward score is significantly > 0
                                    (one-sided t-test, level alpha, at least MIN_SCORES scored dates): the learned values have
                                    predicted the future, not only fitted the past;
  (4) anchored allocation           open  -> the subset with the highest predicted value;
                                    closed -> KL-anchored FTRL between the safe reference and the average of the designed
                                    drafts (prior alpha0 on the reference, lambda = lam0 * sigma / sqrt(n)).
Ablation switches: learner {'content', 'identity'}, gate {'prequential', 'ftest', 'none'}, closed {'ftrl', 'ens', 'ref', 'uniform'}."""
import math, collections
import numpy as np

GRID = np.linspace(0.0, 1.0, 201)
MIN_SCORES = 10


def betainc(a, b, x):
    from math import lgamma, exp, log
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    def cf(a, b, x):
        qab, qap, qam = a + b, a + 1.0, a - 1.0; c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300); h = d
        for m in range(1, 300):
            m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300); c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300); c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
            de = d * c; h *= de
            if abs(de - 1.0) < 3e-14: break
        return h
    lbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x)
    return exp(lbt) * cf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1.0 - exp(lbt) * cf(b, a, 1 - x) / b


def t_sf(t, df):
    """P(T_df >= t)."""
    if not np.isfinite(t): return 0.0 if t > 0 else 1.0
    p2 = betainc(df / 2.0, 0.5, df / (df + t * t))            # two-sided
    return p2 / 2.0 if t > 0 else 1.0 - p2 / 2.0


def f_sf(F, d1, d2):
    if not np.isfinite(F): return 0.0
    if F <= 0: return 1.0
    return betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * F))


def banzhaf(U, j, masks):
    on = [U[m] for m in masks if (m >> j) & 1]; off = [U[m] for m in masks if not (m >> j) & 1]
    return float(np.mean(on) - np.mean(off))


def centred(records):
    """records (date, h, c) -> per-date centred contributions [(date, h, y)] (one value per (date, h))."""
    agg = collections.defaultdict(list)
    for s, h, c in records: agg[(s, h)].append(float(c))
    by = collections.defaultdict(list)
    for (s, h), cs in agg.items(): by[s].append((h, float(np.mean(cs))))
    out = []
    for s, lst in by.items():
        if len(lst) < 2: continue
        mu = np.mean([c for _, c in lst]); out += [(s, h, c - mu) for h, c in lst]
    return out


class ValueLearner:
    """hierarchical Bayesian value model with content features (Fay-Herriot); 'identity' mode drops the features."""
    LOG_GRID = np.exp(np.linspace(np.log(1e-6), np.log(1e2), 25))

    def __init__(self, Z=None, mode="content"):
        self.Z, self.mode = (Z or {}), mode
        self.reset()

    def reset(self):
        self.mean, self.n, self.s2, self.sb2, self.tau2, self.beta, self.H, self.scale = {}, {}, 0.0, 0.0, 0.0, None, [], 1.0

    def fit(self, records):
        self.reset()
        cen = centred(records)
        if not cen: return self
        g = collections.defaultdict(list)
        for s, h, y in cen: g[h].append(y)
        H = sorted(g); S = len({s for s, _, _ in cen}); N = len(cen)
        m = np.array([np.mean(g[h]) for h in H]); n = np.array([len(g[h]) for h in H], float)
        ssw = sum(((np.array(g[h]) - np.mean(g[h])) ** 2).sum() for h in H); dfw = N - len(H) - (S - 1)
        s2 = ssw / dfw if dfw > 0 else float(np.var([y for _, _, y in cen])) or 1e-12
        sc = math.sqrt(s2) or 1.0                                   # work in units of the residual SD
        m_, s2_ = m / sc, 1.0
        use_z = self.mode in ("content", "content_lr") and self.Z and all(h in self.Z for h in H)
        Zm = np.array([self.Z[h] for h in H], float) if use_z else np.zeros((len(H), 0))
        sb_grid = np.concatenate([[0.0], self.LOG_GRID[::2]]) if Zm.shape[1] else np.array([0.0])
        tau_grid = np.concatenate([[0.0], self.LOG_GRID[::2]])
        SB, TA = np.meshgrid(sb_grid, tau_grid, indexing="ij"); SB, TA = SB.ravel(), TA.ravel()
        ZZ = Zm @ Zm.T if Zm.shape[1] else np.zeros((len(H), len(H)))
        V = SB[:, None, None] * ZZ[None] + np.einsum("g,ij->gij", TA, np.eye(len(H))) + np.diag(s2_ / n)[None]
        Lc = np.linalg.cholesky(V)
        a = np.linalg.solve(Lc, np.broadcast_to(m_, (len(SB), len(H)))[..., None])[..., 0]
        ll = -0.5 * (a * a).sum(1) - np.log(np.diagonal(Lc, axis1=1, axis2=2)).sum(1)
        i = int(np.argmax(ll)); best = (ll[i], float(SB[i]), float(TA[i]))
        if Zm.shape[1] and self.mode == "content_lr":   # keep the content term only if supported by the data:
            j0 = SB == 0.0                      # variance-component LR test at 5% (chi-bar-square, critical value 2.71)
            k0 = int(np.flatnonzero(j0)[np.argmax(ll[j0])])
            if 2.0 * (ll[i] - ll[k0]) < 2.71: best = (ll[k0], 0.0, float(TA[k0]))
        _, sb2, tau2 = best
        K = sb2 * (Zm @ Zm.T) if Zm.shape[1] else np.zeros((len(H), len(H)))
        Vinv_m = np.linalg.solve(K + np.diag(tau2 + s2_ / n), m_)
        self.beta = (sb2 * Zm.T @ Vinv_m) if Zm.shape[1] else None
        post = (K + tau2 * np.eye(len(H))) @ Vinv_m
        self.mean = {h: float(v * sc) for h, v in zip(H, post)}
        self.n = dict(zip(H, n)); self.s2, self.sb2, self.tau2, self.H, self.scale = s2, sb2, tau2, H, sc
        return self

    def predict(self, h):
        if h in self.mean: return self.mean[h]
        if self.beta is not None and h in self.Z: return float(np.asarray(self.Z[h]) @ self.beta * self.scale)
        return 0.0


def ftest(records):
    """one-way random-effects F test on date-centred contributions (the earlier gate); returns p."""
    cen = centred(records); g = collections.defaultdict(list)
    for s, h, y in cen: g[h].append(y)
    H = len(g); N = len(cen); S = len({s for s, _, _ in cen})
    if H < 2 or sum(len(v) >= 2 for v in g.values()) < 2: return 1.0
    allc = np.array([y for _, _, y in cen]); mu = allc.mean()
    ssb = sum(len(v) * (np.mean(v) - mu) ** 2 for v in g.values()); ssw = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in g.values())
    dfb, dfw = H - 1, max(1, N - H - (S - 1)); msw = ssw / dfw
    if msw <= 0: return 0.0
    return f_sf((ssb / dfb) / msw, dfb, dfw)


class Gate:
    def __init__(self, kind="prequential", alpha=0.05):
        self.kind, self.alpha, self.scores, self.pending = kind, alpha, [], {}

    def p_value(self, records):
        if self.kind == "none": return 0.0
        if self.kind == "ftest": return ftest(records)
        sc = [x for x in self.scores if np.isfinite(x)]
        if len(sc) < MIN_SCORES: return 1.0
        sd = np.std(sc, ddof=1)
        if sd == 0: return 0.0 if np.mean(sc) > 0 else 1.0
        return t_sf(np.mean(sc) / (sd / math.sqrt(len(sc))), len(sc) - 1)

    def predicted(self, t, preds): self.pending[t] = np.asarray(preds, float)

    def matured(self, t, realised):
        p = self.pending.pop(t, None)
        if p is None: return
        y = np.asarray(realised, float)
        if p.std() < 1e-12 or y.std() < 1e-12: return
        self.scores.append(float(np.corrcoef(p, y)[0, 1]))


def ftrl_q(sumG, n, sig, a0, lam0):
    if n == 0: return a0
    lam = lam0 * (sig / n + 1e-6) / math.sqrt(n)
    qq = np.clip(GRID, 1e-9, 1 - 1e-9); kl = qq * np.log(qq / a0) + (1 - qq) * np.log((1 - qq) / (1 - a0))
    return float(GRID[int(np.argmax(sumG / n - lam * kl))])


class MemGate:
    """one deployment world; call decide(...) then matured(...) for each date in time order."""
    def __init__(self, Z=None, learner="content", gate="prequential", closed="ftrl", alpha=0.05, a0=0.9, lam0=4.0, masks=None):
        self.L = ValueLearner(Z, learner if learner in ("content", "content_lr", "identity") else "content")
        self.G = Gate(gate, alpha); self.closed, self.a0, self.lam0 = closed, a0, lam0
        self.masks = masks or list(range(16)); self.records = []
        self.sumG = np.zeros(len(GRID)); self.n = 0; self.sig = 0.0; self.log = []

    def decide(self, t, ids, U, u_ref, u_ens, grid):
        """ids: retrieved experiences (bit j -> ids[j]); U: mask -> utility of that designed draft (today's outcome is NOT
        used here except to score the chosen action); grid: utility of q*ref + (1-q)*ens on GRID. Returns realised utility."""
        self.L.fit(self.records)
        v = [self.L.predict(h) for h in ids]
        self.G.predicted(t, v)
        p = self.G.p_value(self.records)
        best = max(self.masks, key=lambda m: (sum(v[j] for j in range(len(ids)) if (m >> j) & 1), -bin(m).count("1")))
        gain = sum(v[j] for j in range(len(ids)) if (best >> j) & 1)
        opened = p < self.G.alpha and gain > 0
        if opened: out = U[best]
        elif self.closed == "ens": out = u_ens
        elif self.closed == "ref": out = u_ref
        else:
            q = ftrl_q(self.sumG, self.n, self.sig, 0.5 if self.closed == "uniform" else self.a0, self.lam0)
            out = float(np.interp(q, GRID, grid))
        self.log.append({"t": t, "p": p, "open": opened, "mask": best if opened else None})
        return out

    def matured(self, t, ids, U, u_ref, u_ens, grid):
        c = [banzhaf(U, j, self.masks) for j in range(len(ids))]
        self.records += [(t, h, cj) for h, cj in zip(ids, c)]
        mu = np.mean(c); self.G.matured(t, [x - mu for x in c])
        self.sumG += grid; self.n += 1; self.sig += abs(u_ref - u_ens) / 2.0


class MemGateBank:
    """all MemGate variants of one deployment world, sharing learner fits, gate statistics and FTRL sums.
    variants: name -> dict(learner, gate, closed, alpha). decide() returns {name: realised utility}."""
    def __init__(self, Z, variants, masks=None, a0=0.9, lam0=4.0):
        self.V = variants; self.masks = masks or list(range(16)); self.a0, self.lam0 = a0, lam0
        modes = sorted({m for v in variants.values() for m in (("identity", "content") if v.get("learner", "content") == "auto" else (v.get("learner", "content"),))})
        self.L = {m: ValueLearner(Z, m) for m in modes}; self.G = {m: Gate("prequential") for m in modes}
        self.records = []; self.sumG = np.zeros(len(GRID)); self.n = 0; self.sig = 0.0; self.log = collections.defaultdict(list)

    def decide(self, t, ids, U, u_ref, u_ens, grid):
        st = {}
        for m, L in self.L.items():
            L.fit(self.records); v = [L.predict(h) for h in ids]; self.G[m].predicted(t, v)
            best = max(self.masks, key=lambda k: (sum(v[j] for j in range(len(ids)) if (k >> j) & 1), -bin(k).count("1")))
            st[m] = (best, sum(v[j] for j in range(len(ids)) if (best >> j) & 1), self.G[m].p_value(self.records))
        pf = ftest(self.records)
        q = {a: ftrl_q(self.sumG, self.n, self.sig, a, self.lam0) for a in (self.a0, 0.5)}
        out = {}
        for name, cfg in self.V.items():
            lm = cfg.get("learner", "content")
            if lm == "auto":                    # the learner whose earlier predictions scored better out of sample
                sc = {m: np.mean(self.G[m].scores) if len(self.G[m].scores) >= MIN_SCORES else -np.inf for m in ("identity", "content")}
                lm = "content" if sc["content"] > sc["identity"] else "identity"
            best, gain, ppre = st[lm]; kind = cfg.get("gate", "prequential")
            p = {"prequential": ppre, "ftest": pf, "none": 0.0}[kind]
            opened = p < cfg.get("alpha", 0.05) and gain > 0
            cl = cfg.get("closed", "ftrl")
            if opened: val = U[best]
            elif cl == "ens": val = u_ens
            elif cl == "ref": val = u_ref
            else: val = float(np.interp(q[0.5 if cl == "uniform" else self.a0], GRID, grid))
            out[name] = val; self.log[name].append((t, opened, p))
        return out

    def matured(self, t, ids, U, u_ref, u_ens, grid):
        c = [banzhaf(U, j, self.masks) for j in range(len(ids))]
        self.records += [(t, h, cj) for h, cj in zip(ids, c)]
        mu = np.mean(c)
        for G in self.G.values(): G.matured(t, [x - mu for x in c])
        self.sumG += grid; self.n += 1; self.sig += abs(u_ref - u_ens) / 2.0
