"""Data for the hyperparameter figure (Figure D.4; analysis only; reads published result files, writes SENS_FIG.json here).
Per setting: mean over seeds of the per-seed utility (mean over test dates and investors), as in Appendices D.7-D.9."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent; SD = LAB / "tables"
G3 = ("conservative", "balanced", "aggressive")
F = json.load(open(SD / "SD_pb_fast.json")); I = json.load(open(SD / "SD_ib.json")); P = json.load(open(SD / "SD_pm.json"))
K = json.load(open(SD / "SD_ibk_k2full_k3half_k3full_k4half_k4full.json"))
out = {"alpha": {}, "anchor": {}, "k": {}}
ALPHAS = ["0.01", "0.02", "0.05", "0.1", "0.2"]
for cfg, lab in (("full-price", "PB-Full"), ("raw-price", "PB-Raw")):
    S = {p: json.load(open(LAB / "memtrial/sens" / f"pb_{cfg}_{p}.json")) for p in G3}; split = S["balanced"]["split"]
    ds = sorted(d for d in S["balanced"]["res"]["1/N"] if d >= split)
    def per_seed(n):
        x = np.array([[S[p]["res"][n][d] for p in G3] for d in ds]); return 100 * x.mean((0, 1))
    out["alpha"][lab] = {a: float(per_seed("S|default" if a == "0.05" else f"S|alpha={a}").mean()) for a in ALPHAS}
    out["alpha"][lab]["MemTrial"] = float(per_seed("MemTrial").mean())
out["alpha"]["IB"] = {a: I["variants"]["MemTrial" if a == "0.05" else f"alpha={a}"]["util"][0] for a in ALPHAS}
for part, lab in (("core", "PM-core"), ("informative", "PM-inf")):
    out["alpha"][lab] = {a: P["util_sens"]["S|default" if a == "0.05" else f"S|alpha={a}"][part][0] for a in ALPHAS}
A0 = ["0.5", "0.75", "0.9", "0.95"]; L0 = ["1.0", "4.0", "16.0"]
for cfg, lab in (("full-price", "PB-Full"), ("raw-price", "PB-Raw")):
    out["anchor"][lab] = {f"{a},{l}": F["anchor"][cfg]["MemTrial" if (a, l) == ("0.9", "4.0") else f"S|a0={a},lam0={l}"][0] for a in A0 for l in L0}
out["anchor"]["IB"] = {f"{a},{l}": I["anchor"][f"{a},{l}"][0] for a in A0 for l in L0}
out["anchor"]["PM-core"] = {f"{a},{l}": P["anchor_core"]["default (MemTrial)" if (a, l) == ("0.9", "4.0") else f"S|a0={a},lam0={l}"][0] for a in A0 for l in L0}
DES = ["k=2, full (4 drafts)", "k=3, half (4 drafts)", "k=3, full (8 drafts)", "k=4, half (8 drafts)", "k=4, full (16 drafts)", "k=5, half (16 drafts)", "k=6, half (32 drafts)"]
for cfg, lab in (("full-price", "PB-Full"), ("raw-price", "PB-Raw")):
    out["k"][lab] = {d: F["k_trust"][d][cfg]["util"][0] for d in DES if d in F["k_trust"]}
out["k"]["IB"] = {d: K[d]["util"][0] for d in DES if d in K}
out["k"]["PM-core"] = {d: P["k_by_regime"][d]["core"][0] for d in DES}
out["k"]["PM-inf"] = {d: P["k_by_regime"][d]["many experiences, informative content"][0] for d in DES}
(HERE / "SENS_FIG.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out))
