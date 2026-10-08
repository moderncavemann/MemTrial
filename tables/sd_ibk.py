"""InvestorBench, number of retrieved experiences k: per-seed utility and trust rate of each design
(python3 sd_ibk.py NAME [NAME ...] -> SD_ibk_<i>.json). Re-runs the unchanged ib_k.run (frozen memgate.py) on the logged drafts."""
import sys, json, time
import numpy as np
from sd_common import *
import ib_turnover as T, ib_k as IK
C = T.load(); out = {}
for name in sys.argv[1:]:
    t0 = time.time(); k, masks = IK.DESIGNS[name]; res, op = IK.run(k, masks, C)
    out[name] = {"util": ms(1e4 * res.mean((0, 1))), "trust_pct": ms(100 * op.mean((0, 1)))}
    print(name, out[name], f"{time.time() - t0:.0f}s", flush=True)
tag = "_".join(n.split(",")[0].replace("=", "") + n.split(",")[1].split("(")[0].strip()[:4] for n in sys.argv[1:])
(HERE / f"SD_ibk_{tag}.json").write_text(json.dumps(out, indent=1))
