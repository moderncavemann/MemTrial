"""Baselines with MemTrial's anchor (Appendix D.5, Table D.5; offline, no API calls).

Each LLM-based baseline keeps its own decisions, but instead of deploying its portfolio x_t it deploys
q_t * reference + (1 - q_t) * x_t, with q_t chosen exactly as MemTrial's anchored action (Eq. (11): memgate.ftrl_q,
prior alpha0 = 0.9 on the reference, lambda0 = 4) from the baseline's own matured dates. The reference is 1/N on the real
benchmarks and the average of eight memory-free drafts in PlantedMem, as for MemTrial. Holdings, fees, the timing of the
outcomes and the scoring are those of each benchmark's evaluation (pb_all.py, ib_turnover.py, cb_eval.py, suite.py), whose
code is reused unchanged; the baselines' own decisions are captured from that code. Self-consistency averages each seed's
own three memory-free draws (as sc_eval.py). Also: Hedge with prior weight 0.9 on the reference instead of a uniform prior.
Checks printed by each stage: the anchored Draft averaging equals MemTrial whose trust test never passes, and on PortBench
every unanchored baseline reproduces pb_all.py.

usage (after the steps of reproduce.sh that write pb_all_*.json, plantedmem/results/suite5_*.json and nt_pm_*.json):
  python3 anchored_baselines.py pb <cfg> <investor>   -> ANCHORED_pb_<cfg>_<investor>.json
  python3 anchored_baselines.py ib                    -> ANCHORED_ib.json
  python3 anchored_baselines.py cb                    -> ANCHORED_cb.json
  python3 anchored_baselines.py pm <s0> <s1> [procs]  -> ANCHORED_pm_<s0>_<s1>.json (add --bank to rerun MemTrial as well)
  python3 anchored_baselines.py report                -> ANCHORED.json (Table D.5; MemTrial against the best anchored baseline)"""
import sys, json, math, random, inspect, collections, pickle, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
for _p in ("memtrial", "portbench", "investorbench", "classalloc_and_robustness", "plantedmem", "tables"):
    sys.path.insert(0, str(LAB / _p))
import memgate as MG
from memgate import GRID, ftrl_q, MemGateBank
A0, LAM0 = 0.9, 4.0
NT = {"MemGate": dict(learner="auto"), "NT|never trust": dict(learner="auto", alpha=0.0)}


class Anchor:
    """MemTrial's anchored action for any portfolio: the same FTRL statistics as memgate.MemGateBank."""
    def __init__(self): self.sumG = np.zeros(len(GRID)); self.n = 0; self.sig = 0.0
    def q(self): return ftrl_q(self.sumG, self.n, self.sig, A0, LAM0)
    def matured(self, grid, u_ref, u_x): self.sumG += grid; self.n += 1; self.sig += abs(u_ref - u_x) / 2.0


def exec_patched(fn, module, edits, extra=None):
    """the source of fn with text edits applied, executed in a copy of its module's namespace."""
    src = inspect.getsource(fn)
    for a, b in edits:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    ns = dict(module.__dict__); ns.update(extra or {}); exec(src, ns); return ns[fn.__name__], ns


# ---------------------------------------------------------------- PortBench (each decision starts from the equal-weight book)
def stage_pb(cfg, p):
    import pb_all as PA, monthly_lib as ML, monthly_policies as MP
    rows, ids, dates, split = ML.load_monthly(); inp = json.load(open(LAB / "portbench/data/inputs.json"))
    D = ML.Data(rows, ids, dates, split, cache=ML.load_cache()); Z = PA.content_features(D, inp)
    D2 = ML.Data(rows + ML.load_rows("sc"), ids, dates, split, cache=D.cache)      # extra memory-free draws (Self-consistency)
    key = (cfg, p); MASKS = MP.MASKS
    valid = [d for d in D.dates if d in D.proj[key] and all(any(m in D.proj[key][d][r] for r in D.proj[key][d]) for m in MASKS)]
    fams = [f for f in MP.FAMILIES if all(any(f in D.proj[key][d][r] for r in D.proj[key][d]) for d in valid)]
    assert all(D2.uni[d] == D.uni[d] for d in valid)
    prior9 = np.array([0.9] + [0.1 / len(fams)] * len(fams))
    res = collections.defaultdict(lambda: collections.defaultdict(list)); qlog = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in (0, 1, 2):
        hist = []; up_obs = []; votes = collections.defaultdict(float); rnd = random.Random(f"{cfg}|{p}|{r}")
        bank = MemGateBank(Z, NT, masks=[int(m) for m in MP.HALF]); A = collections.defaultdict(Anchor)
        for d in valid:
            W = D.proj[key][d]
            def X(arm):
                if arm in W.get(r, {}): return W[r][arm]
                for rr in sorted(W):
                    if arm in W[rr]: return W[rr][arm]
            Xs = {m: X(m) for m in MASKS}; Xs.update({f: X(f) for f in fams if f not in Xs})
            ref, mv = D.ref[(d, p, "1/N")], D.ref[(d, p, "MinVar")]
            ens8 = np.mean([Xs[m] for m in MP.HALF], 0); ens16 = np.mean([Xs[m] for m in MASKS], 0)
            names = ["1/N", "MinVar"] + MASKS + [f for f in fams if f not in MASKS] + ["ENS8", "ENS16"]
            u = dict(zip(names, D.U(np.array([ref, mv] + [Xs[m] for m in MASKS] + [Xs[f] for f in fams if f not in MASKS] + [ens8, ens16]), d, p)))
            ids_ = D.ids[d]
            x = {"Zero-shot (no memory)": (Xs["0"], u["0"]), "Similarity retrieval (top-2)": (Xs["3"], u["3"]),
                 "Similarity retrieval (top-4)": (Xs["15"], u["15"]), "FinMem": (Xs["FinMem"], u["FinMem"]), "MemRL": (Xs["MemRL"], u["MemRL"]),
                 "Draft averaging": (ens8, u["ENS8"])}
            if X("Reflexion") is not None: x["Reflexion"] = (X("Reflexion"), None)
            W2 = D2.proj[key][d]; vs = [W2[rr]["0"] for rr in (r, 3 + 2 * r, 4 + 2 * r) if "0" in W2.get(rr, {})]
            x["Self-consistency"] = (np.mean(vs, 0), None)
            if hist:
                cf = collections.defaultdict(list)
                for h in hist:
                    for j, hid in enumerate(h["ids"]): cf[hid].append(h["cf"][j])
                sc = [np.mean(cf[h]) if h in cf else 0.0 for h in ids_]; o = sorted(range(4), key=lambda j: (-sc[j], j))
                mk = str((1 << o[0]) | (1 << o[1]))
            else:
                mk = "3"
            x["Counterfactual selection"] = (Xs[mk], u[mk])
            if len(up_obs) >= 3:
                allid = sorted({h for o_ in up_obs for h in o_[0]} | set(ids_)); ix = {h: i for i, h in enumerate(allid)}
                Xr = np.array([[1.0, o_[2]] + [1.0 if h in o_[0] else 0.0 for h in allid] for o_ in up_obs]); yr = np.array([o_[1] for o_ in up_obs])
                pen = np.eye(Xr.shape[1]); pen[0, 0] = pen[1, 1] = 0.0
                coef = np.linalg.solve(Xr.T @ Xr + 1e-4 * pen * len(yr), Xr.T @ yr); upl = [coef[2 + ix[h]] for h in ids_]
            else:
                upl = [0.0] * 4
            o = rnd.sample(range(4), 2) if rnd.random() < 0.1 else sorted(range(4), key=lambda j: (-upl[j], j))[:2]
            mk = str((1 << o[0]) | (1 << o[1])); x["Uplift credit"] = (Xs[mk], u[mk])
            up_obs.append(({ids_[o[0]], ids_[o[1]]}, u[mk], u["1/N"]))
            o = sorted(range(4), key=lambda j: (-votes[ids_[j]], j))[:2]
            mk = str((1 << o[0]) | (1 << o[1])); x["ExpeL"] = (Xs[mk], u[mk])
            for j in o: votes[ids_[j]] += 1.0 if u[mk] > u["1/N"] else -1.0
            Wf = np.array([ref] + [Xs[f] for f in fams])
            qh = MP.hedge([h["fam_u"] for h in hist], np.full(len(fams) + 1, 1 / (len(fams) + 1))); x["Hedge"] = (qh @ Wf, None)
            q9 = MP.hedge([h["fam_u"] for h in hist], prior9)
            res["Hedge (prior 0.9 on 1/N)"][d].append(float(D.U(q9 @ Wf, d, p)[0]))
            for b, (xb, ub) in x.items():
                ub = float(D.U(xb, d, p)[0]) if ub is None else float(ub)
                grid = MP.ugrid(D.pack(np.array([ref, xb]), d, p)); qq = A[b].q()
                res[b][d].append(ub); res["A|" + b][d].append(float(np.interp(qq, GRID, grid))); qlog[b][d].append(qq)
                A[b].matured(grid, u["1/N"], ub)
            Uh = {m: u[str(m)] for m in bank.masks}; grid = MP.ugrid(D.pack(np.array([ref, ens8]), d, p))
            for n_, v_ in bank.decide(d, ids_, Uh, u["1/N"], u["ENS8"], grid).items(): res[n_][d].append(float(v_))
            bank.matured(d, ids_, Uh, u["1/N"], u["ENS8"], grid)
            res["1/N"][d].append(float(u["1/N"]))
            T = {m: u[m] for m in MASKS}
            hist.append({"ids": ids_, "cf": [MP.banzhaf(T, j) for j in range(4)], "fam_u": [u["1/N"]] + [u[f] for f in fams]})
    out = {"res": {k: dict(v) for k, v in res.items()}, "q": {k: dict(v) for k, v in qlog.items()}, "split": split}
    (HERE / f"ANCHORED_pb_{cfg}_{p}.json").write_text(json.dumps(out))
    # checks against the published per-seed values of pb_all.py
    P = json.load(open(LAB / "portbench" / f"pb_all_{cfg}_{p}.json"))["res"]
    MAP = {"Zero-shot (no memory)": "Zero-shot (no memory)", "Similarity retrieval (top-2)": "Similarity retrieval (top-2)",
           "Similarity retrieval (top-4)": "Similarity retrieval (top-4)", "FinMem": "FinMem", "MemRL": "MemRL", "Reflexion": "Reflexion",
           "Counterfactual selection": "Counterfactual selection (no gate)", "Uplift credit": "Uplift credit (UpliftMem-style)",
           "ExpeL": "ExpeL (adapted)", "Draft averaging": "Draft averaging (8 drafts)", "Hedge": "Hedge (memory families)", "MemGate": "MemGate", "1/N": "1/N"}
    dev = {k: max(abs(a - b) for d in res[k] for a, b in zip(res[k][d], P[v][d])) for k, v in MAP.items()}
    da = max(abs(a - b) for d in res["A|Draft averaging"] for a, b in zip(res["A|Draft averaging"][d], res["NT|never trust"][d]))
    print(cfg, p, "max |unanchored - published|:", {k: f"{v:.1e}" for k, v in dev.items()}, "| anchored Draft averaging - never trust:", f"{da:.1e}", flush=True)


# ---------------------------------------------------------------- InvestorBench (holdings carried day to day, fee on trades)
def _hedge_prior(src_lines, prior_name="PRIOR9"):
    return [(src_lines[0], src_lines[0].replace("zz = Uh.mean(0) / lam", f"zz = np.log({prior_name}) + Uh.mean(0) / lam")),
            (src_lines[1], src_lines[1].replace("qh = np.full(5, 0.2)", f"qh = {prior_name}.copy()"))]


def _sc_targets(folder, vec, W, seeds, dates, project, M, mfl):
    """Self-consistency as in sc_eval.py: each seed averages its own three memory-free draws (the logged draft and two extra draws)."""
    import sc_eval as SE
    X = SE.extras(LAB / "classalloc_and_robustness" / "sc_runs" / folder, vec)
    return {(si, di): project(np.mean([W[(t, s)]["m0"]] + [X[(t, s, e)] for e in (1, 2) if (t, s, e) in X], 0), M, mfl)
            for si, s in enumerate(seeds) for di, t in enumerate(dates)}


def _anchor_loop(names, T, dates, seeds, inv, project, util, drift, ref_of, R_of, nxt_of, delayed, K, sc_of):
    """deploy q * ref + (1 - q) * x_b for every baseline b; returns anchored utilities, q, and the unanchored Self-consistency."""
    nD, nS = len(dates), len(seeds)
    res = {b: np.full((nD, 3, nS), np.nan) for b in names}; qs = {b: np.full((nD, 3, nS), np.nan) for b in names}
    sc_plain = np.full((nD, 3, nS), np.nan)
    for pi, (p, (M, mfl, gam)) in enumerate(inv.items()):
        ew = project(np.full(K, 1.0 / K), M, mfl); SC = sc_of(M, mfl)
        for si, s in enumerate(seeds):
            h0 = ew.copy()
            for di, t in enumerate(dates):
                x = SC[(si, di)]; sc_plain[di, pi, si] = float(np.ravel(util(x, R_of(t), gam, h0)[0])[0]); h0 = drift(x, nxt_of(t))
            for b in names:
                A = Anchor(); h = ew.copy(); fb = delayed()
                for di, t in enumerate(dates):
                    for item in fb.ready(t): A.matured(*item)
                    xb = SC[(si, di)] if b == "Self-consistency" else T[(pi, si, di)][b]; R = R_of(t)
                    g_ = util(GRID[:, None] * ew[None, :] + (1 - GRID)[:, None] * xb[None, :], R, gam, h)
                    grid = np.asarray(g_[0] if isinstance(g_, tuple) else g_, float)
                    q = A.q(); Xd = q * ew + (1 - q) * xb; val = float(np.interp(q, GRID, grid))
                    c_ = util(Xd, R, gam, h); chk = float(np.ravel(c_[0] if isinstance(c_, tuple) else c_)[0]); assert abs(chk - val) < 1e-8, (chk, val)
                    r1 = util(ew, R, gam, h); r2 = util(xb, R, gam, h)
                    u_ref = float(np.ravel(r1[0] if isinstance(r1, tuple) else r1)[0]); u_x = float(np.ravel(r2[0] if isinstance(r2, tuple) else r2)[0])
                    res[b][di, pi, si] = val; qs[b][di, pi, si] = q
                    fb.add(t, (grid, u_ref, u_x)); h = drift(Xd, nxt_of(t))
    return res, qs, sc_plain


class _Now:
    """InvestorBench: the outcome of a day is known before the next day."""
    def __init__(self): self.q = []
    def add(self, t, item): self.q.append(item)
    def ready(self, t_now): out, self.q = self.q, []; return out


NAMES = ["Zero-shot (no memory)", "Self-consistency", "Similarity retrieval (top-2)", "Similarity retrieval (top-4)", "FinMem", "MemRL",
         "Reflexion", "ExpeL", "Uplift credit", "Counterfactual selection", "Draft averaging", "Hedge"]
HEDGE_LINES = ["Uh = np.array(hed); sg = float(np.mean(Uh.std(1))) + 1e-12; lam = 4.0 * sg / math.sqrt(len(Uh)); zz = Uh.mean(0) / lam",
               "                    qh = np.full(5, 0.2)"]
PRIOR9_5 = np.array([0.9, 0.025, 0.025, 0.025, 0.025])            # Hedge's experts: the reference first, then four memory configurations
tl = lambda a: np.where(np.isnan(np.asarray(a, float)), None, np.asarray(a, float)).tolist()


def stage_ib():
    import ib_turnover as IT, ib_all as IA
    C = IT.load(); W, seeds, dates, ret = C["W"], C["seeds"], C["dates"], C["ret"]
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="anchored_ib_"))          # scratch for the outputs of ib_turnover's stages
    cap = "                for k in NAMES:\n                    X = target[k]; g, c = parts(X, r, H[k])"
    TG = []
    base, _ = exec_patched(IT.stage_base, IT, [(cap, "                TG.append((pi, si, di, {k: np.array(v, float).copy() for k, v in target.items()}))\n" + cap)],
                           dict(TG=TG, OUT=tmp))
    base(); B0 = json.load(open(tmp / "base.json"))
    base9, _ = exec_patched(IT.stage_base, IT, _hedge_prior(HEDGE_LINES), dict(OUT=tmp, PRIOR9=PRIOR9_5))
    base9(); B9 = json.load(open(tmp / "base.json"))
    T = {(pi, si, di): tg for pi, si, di, tg in TG}
    res, qs, scp = _anchor_loop(NAMES, T, dates, seeds, IT.INV, IT.project, IT.util, IT.drift, None, lambda t: ret[t], lambda t: ret[t], _Now, 5,
                                lambda M, mfl: _sc_targets("ib_main", IA.vec, W, seeds, dates, IT.project, M, mfl))
    IT.VARIANTS["NT|never trust"] = dict(learner="auto", alpha=0.0); keep = IT.OUT; IT.OUT = tmp
    try:
        mres, mop = IT.run_variant("MemGate", C); nres, _ = IT.run_variant("NT|never trust", C)
    finally:
        IT.OUT = keep
    unanch = {b: B0["res"][b] for b in NAMES}; unanch["Self-consistency"] = tl(scp)
    out = {"dates": dates, "seeds": seeds, "anchored": {b: tl(v) for b, v in res.items()}, "q": {b: tl(v) for b, v in qs.items()},
           "unanchored": unanch, "Hedge (prior 0.9 on 1/N)": B9["res"]["Hedge"], "1/N": B0["res"]["1/N"], "MemGate": tl(mres), "NT|never trust": tl(nres),
           "check_anchored_draft_averaging_minus_never_trust": float(np.nanmax(np.abs(res["Draft averaging"] - nres)))}
    (HERE / "ANCHORED_ib.json").write_text(json.dumps(out))
    print("InvestorBench: anchored Draft averaging - never trust:", out["check_anchored_draft_averaging_minus_never_trust"], flush=True)


def stage_cb(run="run"):
    import cb_eval as CE
    ev = CE.HERE / "classalloc" / run / "eval"; C = pickle.load(open(ev / "cache.pkl", "rb"))
    W, seeds, dates, rel = C["W"], C["seeds"], C["dates"], C["rel"]; nxt = CE.to_next(dates)
    cap = "                for k in BASE:\n                    X = target[k]; u, c, _ = util(X, R, gam, H[k])"
    TG = []
    base, _ = exec_patched(CE.evaluate_base, CE, [(cap, "                TG.append((pi, si, di, {k: np.array(v, float).copy() for k, v in target.items()}))\n" + cap)], dict(TG=TG))
    res0, cost0, held0 = base(C)
    base9, _ = exec_patched(CE.evaluate_base, CE, _hedge_prior(HEDGE_LINES), dict(PRIOR9=PRIOR9_5))
    res9, _, _ = base9(C)
    T = {(pi, si, di): tg for pi, si, di, tg in TG}
    res, qs, scp = _anchor_loop(NAMES, T, dates, seeds, CE.INV, CE.project, CE.util, CE.drift, None, lambda t: rel[t], lambda t: nxt[t], CE.Feedback, CE.K,
                                lambda M, mfl: _sc_targets("cb_main", lambda w: np.array([w[c] for c in CE.CL]), W, seeds, dates, CE.project, M, mfl))
    CE.MGV["NT|never trust"] = dict(learner="auto", alpha=0.0)
    mg = CE.evaluate_method(C, "MemGate")[0]; nt = CE.evaluate_method(C, "NT|never trust")[0]
    unanch = {b: tl(res0[b]) for b in NAMES}; unanch["Self-consistency"] = tl(scp)
    out = {"dates": dates, "seeds": seeds, "anchored": {b: tl(v) for b, v in res.items()}, "q": {b: tl(v) for b, v in qs.items()},
           "unanchored": unanch, "Hedge (prior 0.9 on 1/N)": tl(res9["Hedge"]), "1/N": tl(res0["1/N"]), "MemGate": tl(mg), "NT|never trust": tl(nt),
           "check_anchored_draft_averaging_minus_never_trust": float(np.nanmax(np.abs(res["Draft averaging"] - nt)))}
    (HERE / "ANCHORED_cb.json").write_text(json.dumps(out))
    print("ClassAlloc: anchored Draft averaging - never trust:", out["check_anchored_draft_averaging_minus_never_trust"], flush=True)


# ---------------------------------------------------------------- PlantedMem (each decision starts from the equal-weight book)
PM_NAMES = {"no memory": "Zero-shot (no memory)", "no-memory 8-draw ensemble (reference)": "Self-consistency",
            "similarity top-2": "Similarity retrieval (top-2)", "all 4": "Similarity retrieval (top-4)", "FinMem (adapted)": "FinMem",
            "outcome credit top-2 (MemRL/FinMem-style)": "MemRL", "Reflexion (adapted)": "Reflexion", "ExpeL (adapted)": "ExpeL",
            "uplift credit top-2 (regression-adjusted, UpliftMem/UCOB-style)": "Uplift credit",
            "counterfactual credit argmax top-2": "Counterfactual selection", "uniform aggregation of the 8 members": "Draft averaging",
            "Hedge (adapted)": "Hedge"}
_PM_EDITS = [
    ('        out["no memory"] = U(b(), t)\n',
     '        _x = b(); XV["no memory"] = _x; out["no memory"] = U(_x, t)\n'),
    ('        out["similarity top-2"] = U(act(ret4[:2], preds, b()), t)\n',
     '        _x = act(ret4[:2], preds, b()); XV["similarity top-2"] = _x; out["similarity top-2"] = U(_x, t)\n'),
    ('        out["all 4"] = U(act(ret4, preds, b()), t)\n',
     '        _x = act(ret4, preds, b()); XV["all 4"] = _x; out["all 4"] = U(_x, t)\n'),
    ('        u = U(act(used, preds, b()), t); out["outcome credit top-2 (MemRL/FinMem-style)"] = u\n',
     '        _x = act(used, preds, b()); XV["outcome credit top-2 (MemRL/FinMem-style)"] = _x; u = U(_x, t); out["outcome credit top-2 (MemRL/FinMem-style)"] = u\n'),
    ('        out["counterfactual credit argmax top-2"] = U(act([ret4[j] for j in top], preds, b()), t)\n',
     '        _x = act([ret4[j] for j in top], preds, b()); XV["counterfactual credit argmax top-2"] = _x; out["counterfactual credit argmax top-2"] = U(_x, t)\n'),
    ('        out["uniform aggregation of the 8 members"] = U(ens_vec, t)\n',
     '        XV["uniform aggregation of the 8 members"] = ens_vec; out["uniform aggregation of the 8 members"] = U(ens_vec, t)\n'),
    ('        out["no-memory 8-draw ensemble (reference)"] = refavg\n',
     '        out["no-memory 8-draw ensemble (reference)"] = refavg; XV["no-memory 8-draw ensemble (reference)"] = ref_vec\n'),
    ('        u_up = U(act(up_used, preds, BASE[rng2.integers(len(BASE))]), t)\n',
     '        _x = act(up_used, preds, BASE[rng2.integers(len(BASE))]); XV["uplift credit top-2 (regression-adjusted, UpliftMem/UCOB-style)"] = _x; u_up = U(_x, t)\n'),
    ('        fm = sorted(ret4, key=lambda m: (-(score[m] + imp[m]), pos[m]))[:2]; out["FinMem (adapted)"] = u_fm = pair(*fm)\n',
     '        fm = sorted(ret4, key=lambda m: (-(score[m] + imp[m]), pos[m]))[:2]; out["FinMem (adapted)"] = u_fm = pair(*fm); XV["FinMem (adapted)"] = mem[(1 << pos[fm[0]]) | (1 << pos[fm[1]])]\n'),
    ('        xp = sorted(ret4, key=lambda m: (-votes[m], pos[m]))[:2]; out["ExpeL (adapted)"] = u_xp = pair(*xp)\n',
     '        xp = sorted(ret4, key=lambda m: (-votes[m], pos[m]))[:2]; out["ExpeL (adapted)"] = u_xp = pair(*xp); XV["ExpeL (adapted)"] = mem[(1 << pos[xp[0]]) | (1 << pos[xp[1]])]\n'),
    ('        out["Reflexion (adapted)"] = float(U_batch(((1 - beta) * base6 + beta * tilt if tilt is not None else base6)[None, :], rel, gamma)[0])\n',
     '        _x = (1 - beta) * base6 + beta * tilt if tilt is not None else base6; XV["Reflexion (adapted)"] = _x; out["Reflexion (adapted)"] = float(U_batch(_x[None, :], rel, gamma)[0])\n'),
    ('        out["Hedge (adapted)"] = float(U_batch((qh @ Wh)[None, :], rel, gamma)[0])\n',
     '        out["Hedge (adapted)"] = float(U_batch((qh @ Wh)[None, :], rel, gamma)[0]); XV["Hedge (adapted)"] = qh @ Wh\n'
     '        if hed:\n'
     '            zz9 = np.log(PRIOR9) + Uh.mean(0) / lam; q9 = np.exp(zz9 - zz9.max()); q9 /= q9.sum()\n'
     '        else:\n'
     '            q9 = PRIOR9.copy()\n'
     '        out["Hedge (prior 0.9 on the reference)"] = float(U_batch((q9 @ Wh)[None, :], rel, gamma)[0])\n'),
    ('        if t >= warm:\n            for k, v in out.items(): res[k].append(v)\n',
     '        for _k, _x in XV.items():\n'
     '            _g = U_batch(GRID[:, None] * ref_vec[None, :] + (1 - GRID)[:, None] * _x[None, :], rel, gamma); _q = ANC[_k].q()\n'
     '            out["A|" + _k] = float(np.interp(_q, GRID, _g)); ANC[_k].matured(_g, refavg, float(U(_x, t)))\n'
     '        XV.clear()\n'
     '        if t >= warm:\n            for k, v in out.items(): res[k].append(v)\n'),
    ('    imp = np.full(M, 0.5); votes = np.zeros(M); refl = []; hed = []; pastret = []\n',
     '    imp = np.full(M, 0.5); votes = np.zeros(M); refl = []; hed = []; pastret = []; XV = {}; ANC = collections.defaultdict(Anchor)\n'),
]


def _pm_episode(bank=False):
    """suite.episode with the baselines' portfolios recorded and anchored; MemTrial itself runs only with bank=True (its
    published values come from suite.py and ablation/nt_pm.py)."""
    import suite as V5
    V5.MG_VARIANTS = dict(NT) if bank else {}
    edits = _PM_EDITS + [('    return means, means["_open|MemGate"]\n', '    return means, means.get("_open|MemGate")\n')]
    extra = dict(Anchor=Anchor, PRIOR9=np.array([0.025, 0.9, 0.025, 0.025, 0.025]))
    if not bank: extra["MemGateBank"] = _NoBank
    fn, ns = exec_patched(V5.episode, V5, edits, extra)
    return fn, V5


class _NoBank:
    """stands in for MemGateBank when MemTrial itself is not rerun (it does not touch the random streams of the episode)."""
    def __init__(self, *a, **k): self.log = {}
    def decide(self, *a): return {}
    def matured(self, *a): pass


BANK = False


def _pm_job(a):
    g, n, s = a; fn, V5 = _pm_episode(BANK)
    means, _ = fn(seed=s, gamma=V5.GAMMA[g], **V5.REGIMES[n])
    keep = {k: v for k, v in means.items() if k.startswith("A|") or k in PM_NAMES or k in ("MemGate", "NT|never trust", "1/N", "_open|MemGate", "Hedge (prior 0.9 on the reference)")}
    return a, keep


def stage_pm(s0, s1, procs=2):
    import multiprocessing as mp, suite as V5
    jobs = [(g, n, s) for g in V5.GAMMA for n in V5.REGIMES for s in range(s0, s1)]; t0 = time.time()
    R = collections.defaultdict(lambda: collections.defaultdict(dict)); done = 0
    with mp.get_context("fork").Pool(procs) as pool:
        for (g, n, s), o in pool.imap_unordered(_pm_job, jobs, chunksize=4):
            R[g][n][str(s)] = o; done += 1
            if done % 100 == 0: print(f"{done}/{len(jobs)} episodes, {time.time() - t0:.0f}s", flush=True)
    (HERE / f"ANCHORED_pm_{s0}_{s1}.json").write_text(json.dumps(R)); print("done", f"{time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- report
def p_one_sided(d):
    d = np.asarray(d, float); n = len(d); sd = d.std(ddof=1)
    if sd == 0: return 0.0 if d.mean() > 0 else 1.0
    return float(MG.t_sf(d.mean() / (sd / math.sqrt(n)), n - 1))


def report():
    import glob
    G3 = ("conservative", "balanced", "aggressive"); names = list(PM_NAMES.values())
    T = {}; per_date = {}

    def put(col, k, per_seed, by_date=None):
        T.setdefault(col, {})[k] = dict(mean=float(np.mean(per_seed)), sd=float(np.std(per_seed, ddof=1)), per_seed=[float(v) for v in per_seed])
        if by_date is not None: per_date.setdefault(col, {})[k] = by_date
    for cfg, col in (("full-price", "PortBench-Full"), ("raw-price", "PortBench-Raw")):
        R = {p: json.load(open(HERE / f"ANCHORED_pb_{cfg}_{p}.json")) for p in G3}; split = R["balanced"]["split"]
        dates = sorted(d for d in R["balanced"]["res"]["1/N"] if d >= split)
        def arr(k):
            return 100 * np.array([[R[p]["res"][k][d] for p in G3] for d in dates], float)       # dates x investors x seeds, pp
        keys = ["1/N", "MemGate", "NT|never trust", "Hedge (prior 0.9 on 1/N)"] + names + ["A|" + b for b in names]
        for k in keys:
            if all(k in R[p]["res"] and all(d in R[p]["res"][k] and len(R[p]["res"][k][d]) == 3 for d in dates) for p in G3):
                a = arr(k); put(col, k, a.mean((0, 1)), a.mean((1, 2)).tolist())
    I = json.load(open(HERE / "ANCHORED_ib.json"))
    for k, a in [("1/N", I["1/N"]), ("MemGate", I["MemGate"]), ("NT|never trust", I["NT|never trust"]), ("Hedge (prior 0.9 on 1/N)", I["Hedge (prior 0.9 on 1/N)"])] + \
                [(b, I["unanchored"][b]) for b in names] + [("A|" + b, I["anchored"][b]) for b in names]:
        a = 1e4 * np.array(a, float); put("InvestorBench", k, np.nanmean(a, (0, 1)), np.nanmean(a, (1, 2)).tolist())
    Cb = json.load(open(HERE / "ANCHORED_cb.json"))
    for k, a in [("1/N", Cb["1/N"]), ("MemGate", Cb["MemGate"]), ("NT|never trust", Cb["NT|never trust"]), ("Hedge (prior 0.9 on 1/N)", Cb["Hedge (prior 0.9 on 1/N)"])] + \
                [(b, Cb["unanchored"][b]) for b in names] + [("A|" + b, Cb["anchored"][b]) for b in names]:
        a = 100 * np.array(a, float); put("ClassAlloc", k, np.nanmean(a, (0, 1)), np.nanmean(a, (1, 2)).tolist())
    PM = collections.defaultdict(dict)
    for f in glob.glob(str(HERE / "ANCHORED_pm_*_*.json")):
        for g, Rg in json.load(open(f)).items():
            for n, S in Rg.items():
                for s, v in S.items(): PM[(g, n)][s] = dict(v)
    for f in glob.glob(str(LAB / "plantedmem/results/suite5_*_*.json")):           # MemTrial: published by plantedmem/suite.py
        g = Path(f).stem.split("_")[1]
        for n, S in json.load(open(f)).items():
            for s, v in S.items():
                if s in PM[(g, n)]: PM[(g, n)][s]["MemGate"] = v[0]["MemGate"]
    for f in glob.glob(str(HERE / "nt_pm_*_*.json")):                              # never trusting: ablation/nt_pm.py
        for g, Rg in json.load(open(f)).items():
            for n, S in Rg.items():
                for s, v in S.items():
                    if s in PM[(g, n)]: PM[(g, n)][s]["NT|never trust"] = v[0]["NT|never trust"]
    GROUPS = {"PlantedMem no signal": ["no influence", "noise only"], "PlantedMem signal": ["beta 0.1", "beta 0.25", "beta 0.5"],
              "PlantedMem informative": ["many experiences, informative content"], "PlantedMem core": ["no influence", "noise only", "beta 0.1", "beta 0.25", "beta 0.5"]}
    if PM:
        seeds = sorted(PM[("balanced", "no influence")], key=int)
        inv = {v: k for k, v in PM_NAMES.items()}
        pm_keys = [("1/N", "1/N"), ("MemGate", "MemGate"), ("NT|never trust", "NT|never trust"), ("Hedge (prior 0.9 on 1/N)", "Hedge (prior 0.9 on the reference)")] + \
                  [(b, inv[b]) for b in names] + [("A|" + b, "A|" + inv[b]) for b in names]
        for col, regs in GROUPS.items():
            for k, kk in pm_keys:
                if all(kk in PM[(g, n)][s] for g in G3 for n in regs for s in seeds):
                    put(col, k, 100 * np.array([np.mean([PM[(g, n)][s][kk] for g in G3 for n in regs]) for s in seeds]))
        if all("NT|never trust" in PM[key][s] for key in PM for s in PM[key]):
            T["checks_pm"] = {"anchored_draft_averaging_minus_never_trust": float(max(abs(PM[key][s]["A|uniform aggregation of the 8 members"] - PM[key][s]["NT|never trust"]) for key in PM for s in PM[key]))}
    # MemTrial against the best anchored baseline of each column (paired over dates; over seeds in PlantedMem)
    cmp = {}
    for col, rows in T.items():
        if not col.startswith(("Port", "Inv", "Class", "Planted")): continue
        anch = {k[2:]: v for k, v in rows.items() if k.startswith("A|")}
        best = max(anch, key=lambda b: anch[b]["mean"])
        if col in per_date: d = np.array(per_date[col]["MemGate"]) - np.array(per_date[col]["A|" + best])
        else: d = np.array(rows["MemGate"]["per_seed"]) - np.array(rows["A|" + best]["per_seed"])
        cmp[col] = dict(best_anchored=best, memtrial_minus_best=float(rows["MemGate"]["mean"] - anch[best]["mean"]),
                        pct=float(100 * (rows["MemGate"]["mean"] - anch[best]["mean"]) / abs(anch[best]["mean"])), p_one_sided=p_one_sided(d),
                        p_two_sided=float(min(1.0, 2 * min(p_one_sided(d), p_one_sided(-d)))))
    (HERE / "ANCHORED.json").write_text(json.dumps({"table": T, "memtrial_vs_best_anchored": cmp}, indent=1))
    for col, rows in T.items():
        if not isinstance(rows, dict) or "MemGate" not in rows: continue
        print(f"\n{col}"); n1 = rows["1/N"]["mean"]
        for k in ["1/N", "MemGate", "Hedge (prior 0.9 on 1/N)"] + names:
            if k in rows:
                a = rows.get("A|" + k)
                print(f"  {k:32s} {rows[k]['mean']:8.3f} ± {rows[k]['sd']:.3f}" + (f"   anchored {a['mean']:8.3f} ± {a['sd']:.3f}" if a else ""))
        if col in cmp: print("  MemTrial vs best anchored:", cmp[col])


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "pb": stage_pb(sys.argv[2], sys.argv[3])
    elif st == "ib": stage_ib()
    elif st == "cb": stage_cb(sys.argv[2] if len(sys.argv) > 2 else "run")
    elif st == "pm":
        BANK = "--bank" in sys.argv; args = [a for a in sys.argv[2:] if a != "--bank"]
        stage_pm(int(args[0]), int(args[1]), int(args[2]) if len(args) > 2 else 2)
    elif st == "report": report()
