"""PortBench data layer of the offline evaluation (no LLM calls): logged drafts -> exact investor-projected weight
vectors, utilities and relative price paths for fast exact mixture utilities (mixture.U_batch).
data/executions.jsonl.gz: one row per logged PortBench execution (configuration full-price / raw-price, decision date,
arm = 'view' (a subset mask '0'..'15' of the four retrieved experiences, a memory provider, or 'Reflexion'), replicate,
status and the executed target weights); data/inputs.json: decision dates, dev/test split and the ids of the four
retrieved experiences per date. Investor projections of every draft are cached in score_cache.pkl
(python3 monthly_lib.py builds it in parallel; otherwise Data computes missing entries on the fly)."""
import sys, json, gzip, datetime, collections, hashlib, pickle
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ML = HERE.parent; DATA = HERE / "data"
sys.path.insert(0, str(ML))
import scoring as SC, risk as LR, mixture as CM
RS = SC.RS
INV = SC.INVESTORS; GAMMA = {p: RS.PROFILES[p]["gamma"] for p in INV}
PROVIDERS = ["FinMem", "MemRL", "M2_original"]


def load_rows(*sources):
    """logged executions of the given sources: main (every arm of the main run), reflexion (Reflexion baseline),
    sc (two extra memory-free drafts per seed for Self-consistency), ext and ext_reflexion (dates of 2025-2026)."""
    with gzip.open(DATA / "executions.jsonl.gz", "rt") as f:
        return [r for r in map(json.loads, f) if r["src"] in sources]


def load_monthly():
    inp = json.load(open(DATA / "inputs.json"))
    rows = load_rows("main", "reflexion")
    return rows, inp["ids"], inp["dates"], inp["split"]


def rel_matrix(d, uni):
    dd = datetime.date.fromisoformat(d); pos = RS.SIDX[dd]; fut = RS.SESS[pos + 1: pos + 21]
    M = np.zeros((len(uni), 20))
    for i, a in enumerate(uni):
        p0 = RS.last_price(a, dd); last = p0
        for j, t in enumerate(fut):
            if t in RS.PR[a]: last = RS.PR[a][t]
            M[i, j] = last / p0
    return M


class Data:
    """arms[(config, p)][d][r][arm] = projected vector; draft[(config)][d][r][arm] = normalised draft vector."""
    def __init__(self, rows, ids, dates, split, cache=None):
        self.ids, self.dates, self.split = ids, sorted(dates), split
        # pre-specified exclusion: a (config, date) cell with any data-gap failure ("missing realized observation") is
        # dropped whole, so surviving executions are never selected by which assets they happened to avoid
        gap = {(r.get("config"), r.get("date")) for r in rows
               if r.get("status") != "VALID_NATIVE_EXECUTION" and "missing realized observation" in str(r.get("message", ""))}
        self.excluded = sorted(gap)
        self.rows = [r for r in rows if r.get("status") == "VALID_NATIVE_EXECUTION" and (r["config"], r["date"]) not in gap]
        self.n_failed = sum(1 for r in rows if r.get("status") != "VALID_NATIVE_EXECUTION")
        self.configs = sorted({r["config"] for r in self.rows})
        self.uni = {}
        for r in self.rows: self.uni.setdefault(r["date"], set()).update(r["action"])
        self.uni = {d: sorted(u) for d, u in self.uni.items()}
        self.rel = {d: rel_matrix(d, self.uni[d]) for d in self.uni}
        self.prev = {d: np.full(len(self.uni[d]), 1 / len(self.uni[d])) for d in self.uni}
        cache = cache or {}
        self.draft = collections.defaultdict(lambda: collections.defaultdict(dict))
        self.proj = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
        self.score = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
        for r in self.rows:
            d, c, w = r["date"], r["config"], r["action"]
            tot = sum(w.values()); wn = {a: v / tot for a, v in w.items()}
            key = hashlib.sha1(json.dumps([d, sorted(wn.items())]).encode()).hexdigest()
            if key not in cache:
                vecs = {p: self._vec(RS.project(wn, self.uni[d], RS.PROFILES[p]), d) for p in INV}
                cache[key] = (vecs, SC.score(wn, d, self.uni[d]))
            vecs, sc = cache[key]
            vecs = {p: np.asarray(v) for p, v in vecs.items()}
            self.draft[c][d].setdefault(r["replicate"], {})[r["view"]] = self._vec(wn, d)
            for p in INV:
                self.proj[(c, p)][d][r["replicate"]][r["view"]] = vecs[p]
                self.score[(c, p)][d][r["replicate"]][r["view"]] = sc
        self.cache = cache
        self.ref = {}
        for d in self.uni:
            for p in INV:
                ew = {a: 1 / len(self.uni[d]) for a in self.uni[d]}
                self.ref[(d, p, "1/N")] = self._vec(RS.project(ew, self.uni[d], RS.PROFILES[p]), d)
                self.ref[(d, p, "MinVar")] = self._vec(self._minvar(d, p), d)

    def _vec(self, w, d): return np.array([w.get(a, 0.0) for a in self.uni[d]])

    def _minvar(self, d, p, iters=300):
        uni = self.uni[d]; S = LR.cov_for(d, uni) * 20; x = np.ones(len(uni)) / len(uni); L = 2 * np.linalg.eigvalsh(S).max()
        for _ in range(iters):
            x = x - (2 * S @ x) / L
            w = RS.project(dict(zip(uni, x)), uni, RS.PROFILES[p]); x = np.array([w.get(a, 0.0) for a in uni])
        return dict(zip(uni, x))

    def U(self, X, d, p):
        """exact investor utility of feasible weight rows X (n x assets) on date d."""
        X = np.atleast_2d(X); X = X / X.sum(1, keepdims=True); G = X @ self.rel[d]
        return CM.U_batch(np.eye(len(X)), X, G, self.prev[d], GAMMA[p])

    def pack(self, X, d, p):
        X = np.asarray(X); return (X, X @ self.rel[d], self.prev[d], GAMMA[p], list(range(len(X))))


def save_cache(cache, path=HERE / "score_cache.pkl"):
    pickle.dump(cache, open(path, "wb"))


def load_cache(path=HERE / "score_cache.pkl"):
    return pickle.load(open(path, "rb")) if Path(path).exists() else {}


def _score_one(args):
    d, uni, wn = args
    vecs = {p: [RS.project(wn, uni, RS.PROFILES[p]).get(a, 0.0) for a in uni] for p in INV}
    return vecs, SC.score(wn, d, uni)


def build_cache(rows, cache, workers=4):
    """score every valid row once (parallel); keys match Data.__init__."""
    import multiprocessing as mp
    uni = {}
    for r in rows:
        if r.get("status") == "VALID_NATIVE_EXECUTION": uni.setdefault(r["date"], set()).update(r["action"])
    uni = {d: sorted(u) for d, u in uni.items()}
    todo, keys = [], []
    for r in rows:
        if r.get("status") != "VALID_NATIVE_EXECUTION": continue
        w = r["action"]; tot = sum(w.values()); wn = {a: v / tot for a, v in w.items()}
        key = hashlib.sha1(json.dumps([r["date"], sorted(wn.items())]).encode()).hexdigest()
        if key in cache or key in keys: continue
        keys.append(key); todo.append((r["date"], uni[r["date"]], wn))
    if todo:
        with mp.Pool(workers) as pool:
            out = pool.map(_score_one, todo, chunksize=32)
        for key, (d, u, _), (vecs, sc) in zip(keys, todo, out):
            cache[key] = ({p: np.array(v) for p, v in vecs.items()}, sc)
    return len(todo)


def build_cache_resumable(rows, workers=4, chunk=200):
    """build_cache over all rows (same universes, keys and values), saved to score_cache.pkl after every chunk."""
    import multiprocessing as mp, time
    cache = load_cache(); uni = {}
    for r in rows:
        if r.get("status") == "VALID_NATIVE_EXECUTION": uni.setdefault(r["date"], set()).update(r["action"])
    uni = {d: sorted(u) for d, u in uni.items()}
    todo, keys = [], set()
    for r in rows:
        if r.get("status") != "VALID_NATIVE_EXECUTION": continue
        w = r["action"]; tot = sum(w.values()); wn = {a: v / tot for a, v in w.items()}
        key = hashlib.sha1(json.dumps([r["date"], sorted(wn.items())]).encode()).hexdigest()
        if key in cache or key in keys: continue
        keys.add(key); todo.append((key, (r["date"], uni[r["date"]], wn)))
    t0 = time.time(); print(f"{len(todo)} drafts to score ({len(cache)} cached)", flush=True)
    with mp.Pool(workers) as pool:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            for (key, _), (vecs, sc) in zip(part, pool.map(_score_one, [a for _, a in part])):
                cache[key] = ({p: np.array(v) for p, v in vecs.items()}, sc)
            save_cache(cache); print(f"  {i + len(part)}/{len(todo)} scored, {time.time() - t0:.0f}s", flush=True)
    return len(todo)


if __name__ == "__main__":
    # python3 monthly_lib.py [workers]: investor projections of every logged draft of the main run, Reflexion and the extra
    # Self-consistency draws -> score_cache.pkl (about 30 minutes on 4 cores; resumable: rerun after an interruption).
    # The extra draws use only assets of the main run's universe, so the universes (and values) are those of Data.
    import time
    build_cache_resumable(load_rows("main", "reflexion", "sc"), workers=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
