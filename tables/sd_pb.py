"""PortBench: sd over the three seeds (deployment worlds) for every PortBench number reported without one.
python3 sd_pb.py fast            -> SD_pb_fast.json   (main-table rows, anchor table, k-table trust rates; published files only)
python3 sd_pb.py replay CFG      -> SD_pb_replay_<CFG>.json (trust rate of every variant per seed: the SensBank replay of
                                     trust_rates.py, frozen memgate.py, per seed instead of pooled)
python3 sd_pb.py t1              -> SD_pb_t1.json     (Table 1 credits per seed; supplement decomposition per seed)
Per seed = mean over the 20 test months and the three investors (and, where a number pools them, both configurations)."""
import sys, os, json, collections
import numpy as np
from sd_common import *
STAGE = sys.argv[1]
MON = LAB / "portbench"
EXPK = {"FinMem": "FinMem", "MemRL": "MemRL", "Reflexion": "Reflexion", "ExpeL": "ExpeL (adapted)"}
CFGS = ("full-price", "raw-price")
if STAGE == "fast":
    from paper_tables import pb_load
    out = {"main": {}, "anchor": {}, "k_trust": {}}
    trust_all = collections.defaultdict(list)
    for cfg in CFGS:
        A, dates = pb_load(cfg); per = {k: 100 * v.mean((0, 1)) for k, v in A.items()}
        be = max(EXPK, key=lambda k: per[EXPK[k]].mean())
        tr = []
        for r in range(3):
            fl = []
            for p in G3:
                R = json.load(open(MON / f"pb_all_{cfg}_{p}.json")); fl += [g["open"] for g in R["gate"] if g["world"] == r and g["date"] >= R["split"]]
            tr.append(100 * np.mean(fl)); trust_all[r] += fl
        out["main"][cfg] = {"MemTrial": ms(per["MemGate"]), "best_experience_agent": be, "improv": ms(per["MemGate"] - per[EXPK[be]]),
                            "minus_1N": ms(per["MemGate"] - per["1/N"]), "trust_pct": ms(tr), "n_test": len(dates)}
        # anchor table
        S = {p: json.load(open(LAB / "memtrial/sens" / f"pb_{cfg}_{p}.json")) for p in G3}; split = S["balanced"]["split"]
        ds = sorted(d for d in S["balanced"]["res"]["1/N"] if d >= split)
        out["anchor"][cfg] = {}
        for n in [k for k in S["balanced"]["res"] if k.startswith("S|a0=")] + ["MemGate", "S|default"]:
            if all(n in S[p]["res"] for p in G3):
                x = np.array([[S[p]["res"][n][d] for p in G3] for d in ds]); out["anchor"][cfg][n] = ms(100 * x.mean((0, 1)))
        # k-table trust rates and utilities
        K = json.load(open(MON / f"pb_k_{cfg}.json")); split = K["split"]
        for dsg in K["out"]["balanced"]["open"]:
            o = np.array([[K["out"][p]["open"][dsg][d] for p in G3] for d in sorted(K["out"]["balanced"]["open"][dsg]) if d >= split])
            u = np.array([[K["out"][p]["res"][dsg][d] for p in G3] for d in sorted(K["out"]["balanced"]["res"][dsg]) if d >= split])
            out["k_trust"].setdefault(dsg, {})[cfg] = {"trust_pct": ms(100 * o.mean((0, 1))), "util": ms(100 * u.mean((0, 1))), "open_per_seed": o.mean((0, 1)).tolist()}
    for dsg, v in out["k_trust"].items():
        both = np.array([v[c]["open_per_seed"] for c in CFGS]).mean(0); v["pooled_trust_pct"] = ms(100 * both)
    out["main"]["pooled_trust_pct"] = ms([100 * np.mean(trust_all[r]) for r in range(3)])
    save("pb_fast", out)
    print(json.dumps(out["main"], indent=0)); print({k: v["pooled_trust_pct"] for k, v in out["k_trust"].items()})
    print({c: out["anchor"][c].get("MemGate") for c in CFGS}, list(out["anchor"]["full-price"])[:4])
else:
    import monthly_lib as ML, monthly_policies as MP, pb_all as PA
    rows, ids, dates, split = ML.load_monthly(); inp = json.load(open(MON / "data/inputs.json"))
    D = ML.Data(rows, ids, dates, split, cache=ML.load_cache())
if STAGE == "replay":
    import sens as SE
    cfg = sys.argv[2]; HALF = [int(m) for m in MP.HALF]; V = {**PA.VARIANTS, **SE.SENS_V}
    Z = PA.content_features(D, inp)
    acc = {n: [[], [], []] for n in V}
    for p in G3:
        key = (cfg, p)
        valid = [d for d in D.dates if d in D.proj[key] and all(any(m in D.proj[key][d][r] for r in D.proj[key][d]) for m in MP.MASKS)]
        for r in (0, 1, 2):
            bank = SE.SensBank(Z, V, masks=HALF)
            for d in valid:
                W = D.proj[key][d]
                def X(arm):
                    if arm in W.get(r, {}): return W[r][arm]
                    for rr in sorted(W):
                        if arm in W[rr]: return W[rr][arm]
                Xs = {m: X(m) for m in MP.MASKS}; ref = D.ref[(d, p, "1/N")]; ens8 = np.mean([Xs[m] for m in MP.HALF], 0)
                u = dict(zip(list(MP.MASKS) + ["1/N", "ENS8"], D.U(np.array([Xs[m] for m in MP.MASKS] + [ref, ens8]), d, p)))
                Uh = {m: float(u[str(m)]) for m in HALF}; grid = MP.ugrid(D.pack(np.array([ref, ens8]), d, p))
                bank.decide(d, D.ids[d], Uh, float(u["1/N"]), float(u["ENS8"]), grid)
                if d >= split:
                    for n in V: acc[n][r].append(float(bank.log[n][-1][1]))
                bank.matured(d, D.ids[d], Uh, float(u["1/N"]), float(u["ENS8"]), grid)
    out = {n: {"open_per_seed": [float(np.mean(a)) for a in v], "n_per_seed": [len(a) for a in v]} for n, v in acc.items()}
    save(f"pb_replay_{cfg}", out); print({n: v["open_per_seed"] for n, v in list(out.items())[:6]})
if STAGE == "t1":
    t1 = {}; dec = {}
    for cfg in CFGS:
        for p in G3:
            key = (cfg, p); kind = f"U_{p}"; S = D.score[key]; M16 = [str(m) for m in range(16)]
            ds = [d for d in D.dates if d in S and all(all(m in S[d][r] for m in M16) for r in (0, 1, 2) if r in S[d]) and all(r in S[d] for r in (0, 1, 2))]
            mk = []; oc = [[], [], []]; cf = [[], [], []]
            for d in ds:
                u1n = float(D.U(D.ref[(d, p, "1/N")], d, p)[0])
                for j in range(4):
                    on = [m for m in M16 if (int(m) >> j) & 1]; off = [m for m in M16 if not (int(m) >> j) & 1]
                    for r in range(3):
                        a = np.mean([S[d][r][m][kind] for m in on]); b = np.mean([S[d][r][m][kind] for m in off])
                        oc[r].append(a); cf[r].append(a - b)
                    mk.append(u1n)
            R = json.load(open(MON / f"pb_all_{cfg}_{p}.json")); te = sorted(d for d in R["res"]["1/N"] if d >= R["split"])
            per = {k: 100 * np.array([R["res"][k][d] for d in te]).mean(0) for k in R["res"]}
            be = max(EXPK, key=lambda k: per[EXPK[k]].mean())
            t1[f"{cfg}|{p}"] = {"n_dates": len(ds), "outcome_vs_1N": ms([spearman(oc[r], mk) for r in range(3)]), "cf_vs_1N": ms([spearman(cf[r], mk) for r in range(3)]),
                                "best_agent": be, "best_minus_1N": ms(per[EXPK[be]] - per["1/N"]),
                                "cfsel_minus_1N": ms(per["Counterfactual selection (no gate)"] - per["1/N"])}
            # decomposition (why_1n.py formulas), per seed
            gam = ML.GAMMA[p]; acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [[], [], []]))
            for d in sorted(D.proj[key]):
                if d < split: continue
                W = D.proj[key][d]; rel = D.rel[d]; prev = D.prev[d]
                for r in sorted(W):
                    arms = {"1/N": D.ref[(d, p, "1/N")], "zero-shot": W[r].get("0"), "similarity top-4": W[r].get("15"),
                            "draft average": np.mean([W[r][m] for m in MP.HALF], 0) if all(m in W[r] for m in MP.HALF) else None}
                    for nm, x in arms.items():
                        if x is None: continue
                        x = np.asarray(x) / np.sum(x); path = np.concatenate([[1.0], x @ rel]); rr = path[1:] / path[:-1] - 1
                        l1 = float(np.abs(x - prev)[np.abs(x - prev) >= 1e-6].sum()); nav = 1 - 0.0015 * l1
                        gross = path[-1] - 1; cost = gross - (nav * path[-1] - 1); risk = 0.5 * gam * np.var(nav * rr) * 20
                        u = float(D.U(x[None, :], d, p)[0])
                        for k, v in (("utility", 100 * u), ("gross", 100 * gross), ("cost", 100 * cost), ("risk", 100 * risk), ("L1", l1)):
                            acc[nm][k][r].append(v)
            dec[f"{cfg}|{p}"] = {nm: {k: [float(np.mean(v[r])) for r in range(3)] for k, v in kv.items()} for nm, kv in acc.items()}
    decs = {}
    for cfg in CFGS:
        decs[cfg] = {}
        for nm in ["1/N", "zero-shot", "similarity top-4", "draft average"]:
            decs[cfg][nm] = {k: ms(np.mean([dec[f"{cfg}|{p}"][nm][k] for p in G3], 0)) for k in ["utility", "gross", "cost", "risk", "L1"]}
    save("pb_t1", {"table1": t1, "decomposition": decs})
    for k, v in t1.items(): print(k, v)
    print(json.dumps(decs)[:1500])
