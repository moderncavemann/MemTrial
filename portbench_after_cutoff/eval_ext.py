"""Offline evaluation of the PortBench forward extension (no LLM calls). Frozen before the extension was run.

Same code as the paper's PortBench numbers: ../portbench/monthly_lib.py (exact settlement, investor projection,
utilities, the pre-specified data-gap exclusion) and ../portbench/pb_all.py (run(): every method time-forward over
the dates, MemTrial = the frozen MemGate). The 59 frozen dates are followed by the new dates, so every method carries its
state from 2019-2024 into 2025-2026 without re-tuning. Content features of MemTrial: pb_all.content_features() on the 59
frozen dates (unchanged); an experience first retrieved on a new date is projected on the same 8 components.
Checks: the 2019-2024 results must equal pb_all_<agent>_<investor>.json exactly (same rows, scores and code).
Usage (from this folder):  python3 eval_ext.py scores [seconds]    (scores of rows not in the frozen cache -> score_cache_ext.pkl; repeat until 0 remain)
                           python3 eval_ext.py run <agent> <investor> [--old-only]   -> ext_res_<agent>_<investor>.json
                           python3 eval_ext.py report [--old-only]  -> EXT_RESULTS.md, EXT_RESULTS.json"""
import atexit, glob, json, math, pickle, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; LAB = HERE.parent; FROZEN = LAB / "portbench"
sys.path.insert(0, str(FROZEN)); sys.path.insert(0, str(LAB / "memtrial"))
import monthly_lib as ML
import pb_all as PA
from memgate import t_sf
atexit.register(lambda: ML.SC._dirty.__setitem__(0, False))     # never write the shared scoring cache (runs before its saver)
CFGS = ("full-price", "raw-price"); INVS = ("conservative", "balanced", "aggressive")
SPLIT_EXT = "2025-01-01"; LLM_CUTOFF = "2024-06-01"            # gpt-4.1-mini knowledge cutoff (June 2024)
EXT_CACHE = HERE / "score_cache_ext.pkl"
METHODS = [("1/N", "1/N"), ("Minimum variance", "Minimum variance"), ("Zero-shot (no memory)", "Zero-shot (no memory)"),
           ("Self-consistency", "Self-consistency (3 drafts)"), ("Similarity retrieval (top-2)", "Similarity retrieval (top-2)"),
           ("Similarity retrieval (top-4)", "Similarity retrieval (top-4)"), ("FinMem", "FinMem"), ("MemRL", "MemRL"),
           ("Reflexion", "Reflexion"), ("ExpeL", "ExpeL (adapted)"), ("Uplift credit", "Uplift credit (UpliftMem-style)"),
           ("Counterfactual selection", "Counterfactual selection (no gate)"), ("Draft averaging", "Draft averaging (8 drafts)"),
           ("Hedge", "Hedge (memory families)"), ("MemTrial (ours)", "MemGate")]


def load(old_only=False):
    """rows of the 59 frozen dates (main run and Reflexion) and, unless old_only, of the twelve new dates."""
    inp = json.load(open(FROZEN / "data" / "inputs.json"))
    rows = ML.load_rows("main", "reflexion"); dates = list(inp["dates"]); ids = dict(inp["ids"])
    new = [] if old_only else list(inp["ext_dates"])
    if new:
        dates += new; ids.update(inp["ext_ids"])
        rows += ML.load_rows("ext") + [r for r in ML.load_rows("ext_reflexion") if r["date"] in set(new)]
    return rows, ids, dates, inp, new


def cache():
    c = ML.load_cache(FROZEN / "score_cache.pkl")
    if EXT_CACHE.exists(): c.update(pickle.load(open(EXT_CACHE, "rb")))
    return c


def features(D, inp, new):
    """pb_all.content_features on the frozen dates; new experiences projected on the same components and scaling."""
    Z = PA.content_features(D, inp)
    if not new: return Z, 0
    emb = json.load(open(FROZEN / "data" / "experience_embeddings.json"))
    ids = sorted({h for d in inp["dates"] for h in inp["ids"][d]}); X = np.array([emb[h] for h in ids], float)
    mu = X.mean(0); Xc = X - mu; _, _, Vt = np.linalg.svd(Xc, full_matrices=False); Zc = Xc @ Vt[:8].T; sd = Zc.std(0)
    assert np.allclose(Zc / sd, np.array([Z[h] for h in ids]), atol=1e-12), "replica of pb_all.content_features differs"
    added = 0
    for d in new:
        for h in inp["ext_ids"][d]:
            if h in Z: continue
            Z[h] = list(((np.array(emb[h], float) - mu) @ Vt[:8].T) / sd); added += 1
    return Z, added


def _score_key(args):
    key, d, uni, wn = args
    return key, ML._score_one((d, uni, wn))


def cmd_scores(budget=150.0):
    """Scores of rows not in the frozen cache, in parallel; saved after every call, resumable (same values as ML.build_cache)."""
    import hashlib, multiprocessing as mp
    rows, ids, dates, _, _ = load(); c = cache(); t0 = time.time()
    frozen_keys = set(ML.load_cache(FROZEN / "score_cache.pkl"))
    uni = {}
    for r in rows:
        if r.get("status") == "VALID_NATIVE_EXECUTION": uni.setdefault(r["date"], set()).update(r["action"])
    uni = {d: sorted(u) for d, u in uni.items()}
    todo, seen = [], set()
    for r in rows:
        if r.get("status") != "VALID_NATIVE_EXECUTION": continue
        w = r["action"]; tot = sum(w.values()); wn = {a: v / tot for a, v in w.items()}
        key = hashlib.sha1(json.dumps([r["date"], sorted(wn.items())]).encode()).hexdigest()
        if key in c or key in seen: continue
        seen.add(key); todo.append((key, r["date"], uni[r["date"]], wn))
    done = 0
    with mp.Pool(4) as pool:
        for key, (vecs, sc) in pool.imap_unordered(_score_key, todo, chunksize=2):
            c[key] = ({p: np.array(v) for p, v in vecs.items()}, sc); done += 1
            if time.time() - t0 > budget: pool.terminate(); break
    pickle.dump({k: v for k, v in c.items() if k not in frozen_keys}, open(EXT_CACHE, "wb"))
    print(f"scored {done} rows this call; remaining {len(todo) - done}; {time.time() - t0:.0f}s -> {EXT_CACHE.name}")


def cmd_run(cfg, p, old_only):
    rows, ids, dates, inp, new = load(old_only)
    D = ML.Data(rows, ids, dates, "2023-01-01", cache=cache())
    Z, added = features(D, inp, new)
    res, gl = PA.run(D, cfg, p, Z)
    out = HERE / (f"ext_res_{cfg}_{p}" + ("_oldonly" if old_only else "") + ".json")
    json.dump({"res": res, "gate": gl, "excluded": [list(x) for x in D.excluded], "new_features": added,
               "n_failed": D.n_failed}, open(out, "w"))
    ref = json.load(open(FROZEN / f"pb_all_{cfg}_{p}.json"))["res"]
    missing = [(k, d) for k in ref for d in ref[k] if d not in res.get(k, {}) or len(res[k][d]) != len(ref[k][d])]
    worst = max([abs(res[k][d][w] - ref[k][d][w]) for k in ref for d in ref[k] if (k, d) not in missing
                 for w in range(len(ref[k][d]))] or [float("nan")])
    print(f"{cfg} {p}: frozen-date reproduction max |diff| = {worst:.3g}, missing {len(missing)}; "
          f"dates {len(res['1/N'])}; excluded cells {D.excluded}; new experiences {added} -> {out.name}")


def ptest(a, b):
    d = np.asarray(a) - np.asarray(b); n = len(d)
    if n < 3 or d.std(ddof=1) == 0: return 1.0
    return t_sf(d.mean() / (d.std(ddof=1) / math.sqrt(n)), n - 1)


def cmd_report(old_only=False):
    sfx = "_oldonly" if old_only else ""
    R = {(c, p): json.load(open(HERE / f"ext_res_{c}_{p}{sfx}.json")) for c in CFGS for p in INVS}
    out = {"windows": {}}; L = ["# PortBench forward extension: results", ""]
    for c in CFGS:
        ref = json.load(open(FROZEN / f"pb_all_{c}_balanced.json"))["res"]["1/N"]
        for p in INVS:
            x = R[(c, p)]; f = json.load(open(FROZEN / f"pb_all_{c}_{p}.json"))["res"]
            worst = max(abs(x["res"][k][d][w] - f[k][d][w]) for k in f for d in f[k] for w in range(3))
            L.append(f"- reproduction {c} {p}: max |difference| on the 2019-2024 dates = {worst:.2g}")
        new_all = sorted(d for d in R[(c, "balanced")]["res"]["1/N"] if d >= SPLIT_EXT)
        L.append(f"- {c}: usable new dates {new_all}; excluded cells {R[(c, 'balanced')]['excluded']}")
    L.append("")
    for wname, lo in (("2025-2026 (new dates)", SPLIT_EXT), ("after the LLM knowledge cutoff (2024-06 onward)", LLM_CUTOFF),
                      ("test 2023-2026 (paper test + new dates)", "2023-01-01")):
        L += [f"## {wname}: mean +- sd over 3 seeds, averaged over the 3 investors (pp per month)", "",
              "| method | " + " | ".join(f"{c} (n dates)" for c in CFGS) + " |", "|---|" + "---|" * len(CFGS)]
        tab = {}
        if not all(any(d >= lo for d in R[(c, "balanced")]["res"]["1/N"]) for c in CFGS):
            L[-4:] = [f"## {wname}: no usable dates", ""]; continue
        for c in CFGS:
            ds = sorted(d for d in R[(c, "balanced")]["res"]["1/N"] if d >= lo)
            for lab, k in METHODS:
                A = np.array([[R[(c, p)]["res"][k][d] for p in INVS] for d in ds])          # dates x investors x seeds
                per_seed = A.mean((0, 1)); per_date = A.mean((1, 2))
                tab[(lab, c)] = dict(mean=float(100 * per_seed.mean()), sd=float(100 * per_seed.std(ddof=1)), n=len(ds),
                                     per_date=list(map(float, per_date)), dates=ds)
        for lab, _ in METHODS:
            L.append(f"| {lab} | " + " | ".join(f"{tab[(lab, c)]['mean']:.3f} ± {tab[(lab, c)]['sd']:.3f} ({tab[(lab, c)]['n']})" for c in CFGS) + " |")
        L.append("")
        for c in CFGS:
            mt = tab[("MemTrial (ours)", c)]["per_date"]
            cmp_ = []
            for lab, _ in METHODS:
                if lab == "MemTrial (ours)": continue
                b = tab[(lab, c)]["per_date"]; cmp_.append(f"{lab} {np.mean(mt) * 100 - np.mean(b) * 100:+.3f} (p={ptest(mt, b):.3f})")
            L.append(f"- {c}, MemTrial minus each method (one-sided paired p over dates): " + "; ".join(cmp_))
            g = [x["open"] for p in INVS for x in R[(c, p)]["gate"] if x["date"] >= lo]
            L.append(f"- {c}, MemTrial trust rate: {100 * np.mean(g):.1f}% of {len(g)} decisions")
        L.append("")
        out["windows"][wname] = {f"{lab}|{c}": v for (lab, c), v in tab.items()}
    (HERE / f"EXT_RESULTS{sfx}.md").write_text("\n".join(L)); json.dump(out, open(HERE / f"EXT_RESULTS{sfx}.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "scores": cmd_scores(float(a[1]) if len(a) > 1 else 150.0)
    elif a[0] == "run": cmd_run(a[1], a[2], "--old-only" in a)
    elif a[0] == "report": cmd_report("--old-only" in a)
