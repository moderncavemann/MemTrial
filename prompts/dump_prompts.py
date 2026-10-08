"""Dump the prompts used on InvestorBench, ClassAlloc and PortBench, with real examples, to PROMPTS.json (offline, read-only).
python3 dump_prompts.py   (no API calls; reads saved inputs, executions and calls)"""
import json, glob, ast, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
out = {}
# ------------------------------------------------------------------ InvestorBench
sys.path.insert(0, str(LAB / "investorbench")); import run_ib as RB
IBR = LAB / "investorbench" / "run"
B = RB.Bench(); exps = {e["id"]: e for e in json.load(open(IBR / "experiences.json"))}
recs = [json.load(open(f)) for f in glob.glob(str(IBR / "executions" / "*.json"))]
DAY = "2021-02-25"
r15 = [r for r in recs if r.get("arm") == "subset" and r["date"] == DAY and r["seed"] == 0 and r["mask"] == 15][0]
ctx = B.context(DAY)
body = RB.decision_body(ctx, RB.block_experiences([exps[h]["text"] for h in r15["used"]]))
warm = sorted([r for r in recs if r.get("phase") == "warmup"], key=lambda r: r["date"])
out["ib"] = {"system": RB.SYSTEM, "ask": RB.ASK, "fix": RB.FIX, "day": DAY, "context": ctx, "used": r15["used"],
             "decision_user": body["messages"][1]["content"], "decision_reply": {"weights": r15["weights"], "reason": r15.get("reason")},
             "insights_block_example": RB.block_insights(json.load(open(IBR / "insights.json"))[:3]),
             "reflections_block_template": RB.block_reflections(["<note 1>", "<note 2>", "<note 3>"]),
             "warm_keys": sorted({k for r in warm[:3] for k in r}), "warm_example": warm[0] if warm else None}
e0 = exps["E000"]
if warm:
    w0 = [r for r in warm if r["date"] == e0["date"]]
    if w0:
        w = w0[0]["weights"]; rr = B.next_returns(e0["date"])
        out["ib"]["lesson_user"] = RB.feedback_body(B.context(e0["date"]), w, rr, "lesson")["messages"][1]["content"]
        out["ib"]["reflection_user"] = RB.feedback_body(B.context(e0["date"]), w, rr, "reflection")["messages"][1]["content"]
        out["ib"]["lesson_written"] = e0["lesson"]
out["ib"]["insight_user"] = RB.insight_body([exps[h] for h in sorted(exps)[:2]])["messages"][1]["content"]
# ------------------------------------------------------------------ ClassAlloc
sys.path.insert(0, str(LAB / "classalloc_and_robustness")); import run_cb as RC
CB = RC.Bench(); CR = LAB / "classalloc_and_robustness" / "classalloc" / "run"
cexps = json.load(open(CR / "experiences.json"))
out["cb"] = {"system": RC.SYSTEM, "ask": RC.ASK, "context": CB.context("2022-04-01"), "exp_keys": sorted(cexps[0].keys()), "exp0": cexps[0]}
bod = RC.Bodies({"model": "x"}, 0.7) if False else None
# ------------------------------------------------------------------ PortBench (saved native requests)
PR = LAB / "portbench" / "run" / "calls"
pb = {}
for f in glob.glob(str(PR / "*.json")):
    d = json.load(open(f)); lid = d["logical_id"]
    if lid.startswith("monthly/raw-price/") and ":2022-04-01:" in lid:
        c = lid.split(":")[-2]
        r = ast.literal_eval(d["request"]) if isinstance(d["request"], str) else d["request"]
        u = r["messages"][-1]["content"]
        if c not in pb or ("FROZEN HISTORICAL EXPERIENCE" in u and u.count('\\"id\\"') > pb[c]["n"]):
            pb[c] = {"lid": lid, "system": r["messages"][0]["content"], "user": u, "n": u.count('\\"id\\"'),
                     "params": {k: v for k, v in r.items() if k != "messages"},
                     "reply": (ast.literal_eval(d["response"]) if isinstance(d["response"], str) else d["response"])["choices"][0]["message"]["content"]}
out["pb"] = pb
(HERE / "PROMPTS.json").write_text(json.dumps(out, indent=1))
print("ib used", out["ib"]["used"], "warm keys", out["ib"]["warm_keys"], "lesson_user" in out["ib"])
print("cb exp keys", out["cb"]["exp_keys"])
print({c: (v["lid"], v["n"], len(v["user"])) for c, v in pb.items()})
