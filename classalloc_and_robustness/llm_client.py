"""Shared LLM plumbing for the supplementary runs (standard library only; python3 >= 3.8).

Same rules as investorbench/run_ib.py:
  * every API response is saved under <run>/calls/ before it is used and replayed on later runs, so an interrupted run
    resumes and nothing is paid twice;
  * hard cap: a call is sent only if spent + reserved + the call's worst case stays below the cap; failed calls count;
  * keys come from the environment (OPENAI_API_KEY / OPENROUTER_API_KEY) or from the single key of that kind in
    .env, and are never printed;
  * --mock runs use deterministic fake responses (no network, no key).
"""
from __future__ import annotations
import hashlib, json, math, os, random, re, ssl, sys, threading, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # package root (optional .env with API keys)
URLS = {"openai": ("https://api.openai.com/v1/chat/completions", "https://api.openai.com/v1/embeddings"),
        "openrouter": ("https://openrouter.ai/api/v1/chat/completions", None)}
EMB_MODEL, P_EMB = "text-embedding-3-small", 0.02e-6

# Prices are USD per token (list prices; cached-input discounts are ignored, so the ledger over-counts).
# OpenRouter prices vary by provider: the ledger uses the cost OpenRouter reports (usage.cost) and these
# conservative numbers only to reserve budget before a call.
MODELS = {
    "gpt-4.1-mini": dict(provider="openai", model="gpt-4.1-mini-2025-04-14", reasoning=False, p_in=0.40e-6, p_out=1.60e-6),
    "gpt-4.1-nano": dict(provider="openai", model="gpt-4.1-nano-2025-04-14", reasoning=False, p_in=0.10e-6, p_out=0.40e-6),
    "gpt-5-mini":   dict(provider="openai", model="gpt-5-mini-2025-08-07", reasoning=True, effort="minimal", p_in=0.25e-6, p_out=2.00e-6),
    "llama-3.3-70b": dict(provider="openrouter", model="meta-llama/llama-3.3-70b-instruct", reasoning=False, p_in=0.60e-6, p_out=0.80e-6),
    # added 5 Oct 2026 (comparable mid-size models of other providers, all through OpenRouter; thinking switched off)
    "gemini-2.5-flash": dict(provider="openrouter", model="google/gemini-2.5-flash", reasoning=False, p_in=0.30e-6, p_out=2.50e-6,
                             extra={"reasoning": {"max_tokens": 0, "exclude": True}}),
    "claude-haiku-4.5": dict(provider="openrouter", model="anthropic/claude-haiku-4.5", reasoning=False, p_in=1.00e-6, p_out=5.00e-6),
    "qwen3-235b": dict(provider="openrouter", model="qwen/qwen3-235b-a22b-2507", reasoning=False, p_in=0.25e-6, p_out=1.00e-6),
    "deepseek-v3.1": dict(provider="openrouter", model="deepseek/deepseek-chat-v3.1", reasoning=False, p_in=0.30e-6, p_out=1.20e-6,
                          extra={"reasoning": {"enabled": False}}),
}


def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def sha(b): return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()
def canon(v): return json.dumps(v, sort_keys=True, separators=(",", ":"))
def clean(s): return re.sub(r"\s+", " ", str(s)).strip()
def content(resp): return ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def load_key(provider):
    var = "OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY"
    if os.environ.get(var): return os.environ[var]
    env = ROOT / ".env"
    if env.exists():
        pat = r"sk-(?!or-)[A-Za-z0-9_-]{20,}" if provider == "openai" else r"sk-or-[A-Za-z0-9_-]{20,}"
        keys = set(re.findall(pat, env.read_text()))
        if len(keys) == 1: return keys.pop()
    sys.exit(f"No {provider} key: export {var}=... (or keep exactly one such key in .env)")


def ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def adapt(body, spec, temperature):
    """Turn a gpt-4.1-style chat body into the body for `spec` (model id, temperature, token limit, reasoning effort)."""
    b = json.loads(json.dumps(body)); b["model"] = spec["model"]; mt = b.pop("max_tokens", None)
    if spec["reasoning"]:                  # reasoning models: no temperature; the limit also covers reasoning tokens
        b.pop("temperature", None); b["max_completion_tokens"] = max(2000, 4 * (mt or 500))
        if spec.get("effort"): b["reasoning_effort"] = spec["effort"]
    else:
        b["temperature"] = temperature
        if mt: b["max_tokens"] = mt
    if spec["provider"] == "openrouter": b["usage"] = {"include": True}
    if spec.get("extra"): b.update(json.loads(json.dumps(spec["extra"])))   # provider-specific switches (new models only)
    return b


class BudgetStop(RuntimeError): pass


class Ledger:
    def __init__(self, cap, out):
        self.cap, self.lock, self.spent, self.held = cap, threading.Lock(), 0.0, 0.0
        for p in (out / "calls").glob("*.json"): self.spent += json.loads(p.read_text())["charged_usd"]
        fp = out / "call_failures.jsonl"
        if fp.exists(): self.spent += sum(json.loads(x)["charged_usd"] for x in fp.read_text().splitlines() if x.strip())

    def reserve(self, amt):
        with self.lock:
            if self.spent + self.held + amt > self.cap + 1e-12: return False
            self.held += amt; return True

    def settle(self, held, charged):
        with self.lock: self.held -= held; self.spent += charged


def mock_response(url, body, logical_id):
    rnd = random.Random(sha(logical_id))
    if url.endswith("/embeddings"):
        vecs = []
        for x in body["input"]:
            r2 = random.Random(sha(x)); v = [r2.gauss(0, 1) for _ in range(32)]; n = math.sqrt(sum(a * a for a in v))
            vecs.append([a / n for a in v])
        return {"data": [{"embedding": v} for v in vecs], "usage": {"prompt_tokens": 10 * len(vecs)}}
    user = body["messages"][-1]["content"]
    if "Respond with a JSON object only" in user or user.startswith("Your reply was not"):
        allu = " ".join(m["content"] for m in body["messages"] if m["role"] == "user")
        ex = re.search(r"\{[^{}]*\"reason\"[^{}]*\}", allu); keys = [k for k in json.loads(ex.group(0)) if k != "reason"]
        a = [rnd.random() ** 2 for _ in keys]; s = sum(a)
        obj = {k: round(v / s, 3) for k, v in zip(keys, a)}; obj["reason"] = "mock"; txt = json.dumps(obj)
        if rnd.random() < 0.03: txt = "not json"
    elif "numbered list" in user:
        txt = "\n".join(f"{i}. Mock insight {i}: diversify when volatility rises." for i in range(1, 6))
    else:
        txt = f"Mock note {rnd.randint(0, 999)}: in mixed markets, hold more cash and trim the most volatile holding."
    return {"choices": [{"message": {"content": txt}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 80}}


def make_requester(ledger, out, spec, stop, mock):
    """returns req(logical_id, body, kind='chat'|'emb') -> response (saved, replayed, capped)."""
    chat_url, emb_url = URLS[spec["provider"]]
    key = None if mock else load_key(spec["provider"])
    emb_key = None if mock else (key if spec["provider"] == "openai" else None)
    ctx = ssl_context()

    def req(logical_id, body, kind="chat"):
        url = chat_url if kind == "chat" else emb_url
        if url is None:                     # embeddings always come from OpenAI
            url = URLS["openai"][1]
        k = emb_key if kind == "emb" else key
        if kind == "emb" and k is None and not mock: k = load_key("openai")
        wire = canon(body).encode(); rsha = sha(wire)
        path = out / "calls" / (sha(logical_id)[:40] + ".json")
        if path.exists():
            rec = json.loads(path.read_text())
            if rec["request_sha256"] != rsha: raise RuntimeError(f"saved call {logical_id} has a different request")
            return rec["response"]
        if stop.is_set(): raise BudgetStop("stopped")
        if kind == "emb": reserve = len(wire) / 3.0 * P_EMB
        else:
            mt = body.get("max_tokens") or body.get("max_completion_tokens") or 1000
            reserve = len(wire) / 3.0 * spec["p_in"] + mt * spec["p_out"]
        if not ledger.reserve(reserve): stop.set(); raise BudgetStop("hard cap reached; not sending")
        charged, last = 0.0, None
        try:
            for attempt in range(1, 7):
                if mock: resp = mock_response(url, body, logical_id)
                else:
                    req_ = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
                        "Authorization": f"Bearer {k}", "Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(req_, timeout=300, context=ctx) as r: resp = json.loads(r.read())
                    except urllib.error.HTTPError as e:
                        last = f"HTTP {e.code}: {e.read()[:400].decode(errors='replace')}"
                        if e.code in (401, 402, 403): stop.set(); raise RuntimeError(last)
                        if e.code == 429 or e.code >= 500:
                            try: wait = float(e.headers.get("retry-after") or 0)
                            except ValueError: wait = 0.0
                            time.sleep(min(max(wait, 2.0 ** attempt), 60)); continue
                        raise RuntimeError(last)
                    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                        last = f"{type(e).__name__}: {e}"; time.sleep(min(2.0 ** attempt, 30)); continue
                    if "error" in resp and not resp.get("choices") and kind == "chat":
                        last = f"API error: {str(resp['error'])[:300]}"; time.sleep(min(2.0 ** attempt, 30)); continue
                u = resp.get("usage") or {}
                if kind == "emb": charged = u.get("prompt_tokens", 0) * P_EMB if u else reserve
                elif isinstance(u.get("cost"), (int, float)): charged = float(u["cost"])          # OpenRouter
                else: charged = (u.get("prompt_tokens", 0) * spec["p_in"] + u.get("completion_tokens", 0) * spec["p_out"]) if u else reserve
                rec = {"logical_id": logical_id, "request_sha256": rsha, "charged_usd": charged, "saved_utc": now(),
                       "request": body, "response": resp}
                tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); os.replace(tmp, path)
                return resp
            raise RuntimeError(f"gave up after retries: {last}")
        except Exception as e:
            if not path.exists():
                with ledger.lock, (out / "call_failures.jsonl").open("a") as f:
                    f.write(json.dumps({"logical_id": logical_id, "error": str(e)[:300], "charged_usd": charged, "utc": now()}) + "\n")
            raise
        finally:
            ledger.settle(reserve, charged)
    return req


def list_models(provider, pattern=r"gpt-5|gpt-4\.1|o4|o3"):
    """print the chat models this key can use (OpenAI) or check a model id (OpenRouter); no cost."""
    if provider == "openai":
        r = urllib.request.Request("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {load_key('openai')}"})
        with urllib.request.urlopen(r, timeout=60, context=ssl_context()) as f: data = json.loads(f.read())
        ids = sorted(m["id"] for m in data.get("data", []) if re.search(pattern, m["id"]))
        print("OpenAI models available to this key:", ", ".join(ids))
    else:
        with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60, context=ssl_context()) as f: data = json.loads(f.read())
        ids = sorted(m["id"] for m in data.get("data", []))
        print(len(ids), "OpenRouter models; llama-3.3-70b present:", "meta-llama/llama-3.3-70b-instruct" in ids)
