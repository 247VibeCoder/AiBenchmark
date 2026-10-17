#!/usr/bin/env python3
"""
llmbench.py - benchmark every model behind a LiteLLM proxy (llama-swap backend).

Suites follow the lightweight eval mix that most 2026 local-model cards report
(GSM8K / HumanEval / IFEval / BFCL) plus speed, long context and embeddings:

  speed   first-request latency (includes llama-swap model load), TTFT,
          decode tokens/s, prefill tokens/s
  math    GSM8K-style word problems, exact numeric answer
  code    HumanEval-style completion, graded by EXECUTING unit tests
  ifeval  IFEval-style verifiable instruction constraints
  tools   BFCL-style function calling (simple, selection, enum args, parallel,
          irrelevance, answering from a tool result)
  needle  long-context retrieval at 10 / 50 / 90 % depth
  embed   embedding throughput + multilingual (en/de/it) retrieval: top-1, MRR
  parallel  (opt-in, --parallel 2,4) N simultaneous users: aggregate and per-user
          tokens/s, queueing (TTFT), and which LiteLLM deployment served each
          request - shows load balancing across GPUs or llama-server slots

The built-in task sets run fully offline. With --hf N, math and code pull N real
GSM8K / HumanEval items from the Hugging Face datasets-server API instead.

WARNING: the code suite executes model-generated Python in a subprocess
(temp dir, timeout). Run this somewhere you don't mind that happening.

Notes
  * Models run one after another, so llama-swap swaps cleanly and the
    "load" column shows the real cold-start time.
  * Keep --concurrency 1 for clean speed numbers. llama-server splits -c across
    its parallel slots, so concurrency > 1 can also break the needle suite.
  * If every tools case fails on a llama.cpp model, the server is probably not
    applying the jinja chat template: add --jinja to its llama-swap cmd.
  * Costs come from LiteLLM's x-litellm-response-cost header, i.e. whatever
    prices config.yaml defines (your Opus-reference pricing).

Configuration
  Credentials come from a .env file (cwd first, then next to this script;
  override with --env-file). Variables already set in the shell win.
    LITELLM_BASE_URL=https://api.ctollama.home.arpa
    LITELLM_API_KEY=sk-...            # or LITELLM_MASTER_KEY
    REQUESTS_CA_BUNDLE=/path/ca.crt   # optional, for an internal CA
    LITELLM_INSECURE=1                # optional, skip TLS verification

Usage
  pip install requests
  cp .env.example .env && $EDITOR .env
  python3 llmbench.py                                   # everything, via https://api.ctollama.home.arpa
  python3 llmbench.py --exclude MiniMax-M3              # local models only
  python3 llmbench.py --models compress,qwen3-embedding --suites speed,embed
  python3 llmbench.py --hf 50 --suites math,code        # real GSM8K/HumanEval
  python3 llmbench.py --no-think                        # disable Qwen thinking
  python3 llmbench.py --suites speed --parallel 2,4     # 1 vs 2 vs 4 simultaneous users
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median

try:
    import requests
    from requests.adapters import HTTPAdapter
except ImportError:
    sys.exit("This script needs 'requests':  pip install requests")

def load_env_file(path):
    """Minimal .env reader: KEY=VALUE, '#' comments, optional 'export', quotes.
    Never overrides variables that are already set in the environment."""
    if not path or not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            else:
                val = val.split(" #", 1)[0].rstrip()
            os.environ.setdefault(key, val)
    return True


def find_env_file():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file")
    known, _ = pre.parse_known_args()
    if known.env_file:
        if not os.path.isfile(known.env_file):
            sys.exit(f"--env-file not found: {known.env_file}")
        return known.env_file
    for cand in (os.path.join(os.getcwd(), ".env"),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
        if os.path.isfile(cand):
            return cand
    return None


ALIASES_TO_SKIP = {"homelab-coder", "homelab-doc"}
ALL_SUITES = ["speed", "math", "code", "ifeval", "tools", "needle", "embed", "parallel"]
DEFAULT_SUITES = [s for s in ALL_SUITES if s != "parallel"]  # parallel is opt-in (--parallel)


# =============================================================================
# HTTP client
# =============================================================================
@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)
    ttft: float | None = None
    total: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    error: str | None = None
    deployment: str | None = None  # x-litellm-model-id: which LiteLLM deployment answered
    api_base: str | None = None    # x-litellm-model-api-base
    started: float = 0.0           # perf_counter at send

    @property
    def decode_tps(self):
        if self.ttft is None or self.completion_tokens < 8:
            return None
        dt = self.total - self.ttft
        return self.completion_tokens / dt if dt > 0 else None

    @property
    def prefill_tps(self):
        if not self.ttft or not self.prompt_tokens:
            return None
        return self.prompt_tokens / self.ttft


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def visible(text: str) -> str:
    """Answer text without inline reasoning blocks."""
    text = THINK_RE.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


class _ForcedVerifyAdapter(HTTPAdapter):
    """Force `verify` on every send, bypassing urllib3's pre-built SSLContext.

    Python 3.14 + urllib3 can cache an SSLContext that ignores later mutations
    of ``Session.verify``; re-passing verify per request is the only reliable
    way to flip cert validation off (or to swap a CA bundle) at runtime.
    """
    def __init__(self, verify):
        super().__init__(pool_maxsize=64)  # room for --parallel users
        self._verify = verify

    def send(self, request, **kwargs):
        kwargs["verify"] = self._verify
        return super().send(request, **kwargs)


class Client:
    def __init__(self, base_url, api_key, timeout, extra_body=None, verify=True):
        self.base = base_url.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.s = requests.Session()
        # Force verify on every send so a later .verify= mutation actually
        # takes effect (see _ForcedVerifyAdapter docstring).
        self.s.mount("https://", _ForcedVerifyAdapter(verify))
        self.s.mount("http://", _ForcedVerifyAdapter(verify))
        self.s.verify = verify
        if api_key:
            self.s.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout
        self.extra = extra_body or {}
        self._lock = threading.Lock()
        self.cost = {}
        self.info = {}  # model_name -> LiteLLM /model/info entry

    def _add_cost(self, model, c):
        with self._lock:
            self.cost[model] = self.cost.get(model, 0.0) + c

    def list_models(self):
        """[(name, 'chat'|'embedding')] from LiteLLM, de-duplicated."""
        out = {}
        try:
            r = self.s.get(f"{self.base}/model/info", timeout=30)
            r.raise_for_status()
            for m in r.json().get("data", []):
                name = m.get("model_name")
                mode = (m.get("model_info") or {}).get("mode")
                if name:
                    self.info.setdefault(name, m)
                    out[name] = "embedding" if mode == "embedding" else "chat"
        except Exception:
            pass
        if not out:
            r = self.s.get(f"{self.base}/v1/models", timeout=30)
            r.raise_for_status()
            for m in r.json().get("data", []):
                out[m["id"]] = "embedding" if "embed" in m["id"].lower() else "chat"
        return list(out.items())

    def chat(self, model, messages, max_tokens=1024, tools=None, stream=True, temperature=0.0):
        body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, **self.extra}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        res = ChatResult()
        t0 = res.started = time.perf_counter()
        try:
            r = self.s.post(f"{self.base}/v1/chat/completions", json=body,
                            stream=stream, timeout=self.timeout)
            if r.status_code >= 400:
                res.error = f"HTTP {r.status_code}: {r.text[:300]}"
                res.total = time.perf_counter() - t0
                return res
            res.cost = _f(r.headers.get("x-litellm-response-cost"))
            res.deployment = r.headers.get("x-litellm-model-id")
            res.api_base = r.headers.get("x-litellm-model-api-base")

            if not stream:
                d = r.json()
                res.total = time.perf_counter() - t0
                msg = d["choices"][0]["message"]
                res.content = msg.get("content") or ""
                res.reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                res.tool_calls = [{"name": tc["function"]["name"],
                                   "arguments": tc["function"].get("arguments") or ""}
                                  for tc in (msg.get("tool_calls") or [])]
                u = d.get("usage") or {}
                res.prompt_tokens = u.get("prompt_tokens") or 0
                res.completion_tokens = u.get("completion_tokens") or 0
            else:
                parts, rparts, tcs = [], [], {}
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    u = chunk.get("usage")
                    if u:
                        res.prompt_tokens = u.get("prompt_tokens") or res.prompt_tokens
                        res.completion_tokens = u.get("completion_tokens") or res.completion_tokens
                    for ch in chunk.get("choices") or []:
                        delta = ch.get("delta") or {}
                        c = delta.get("content")
                        rc = delta.get("reasoning_content") or delta.get("reasoning")
                        tc = delta.get("tool_calls")
                        if (c or rc or tc) and res.ttft is None:
                            res.ttft = time.perf_counter() - t0
                        if c:
                            parts.append(c)
                        if rc:
                            rparts.append(rc)
                        for t in tc or []:
                            slot = tcs.setdefault(t.get("index", 0), {"name": "", "arguments": ""})
                            fn = t.get("function") or {}
                            if fn.get("name"):
                                slot["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["arguments"] += fn["arguments"]
                res.total = time.perf_counter() - t0
                res.content = "".join(parts)
                res.reasoning = "".join(rparts)
                res.tool_calls = [tcs[k] for k in sorted(tcs)]
                if not res.completion_tokens:  # server sent no usage: rough estimate
                    res.completion_tokens = max(1, len(res.content + res.reasoning) // 4)
        except Exception as e:
            res.error = f"{type(e).__name__}: {e}"
            res.total = time.perf_counter() - t0
        self._add_cost(model, res.cost)
        return res

    def embed(self, model, inputs):
        t0 = time.perf_counter()
        r = self.s.post(f"{self.base}/v1/embeddings", json={"model": model, "input": inputs},
                        timeout=self.timeout)
        dt = time.perf_counter() - t0
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        self._add_cost(model, _f(r.headers.get("x-litellm-response-cost")))
        d = r.json()
        vecs = [x["embedding"] for x in sorted(d["data"], key=lambda x: x.get("index", 0))]
        return vecs, (d.get("usage") or {}).get("prompt_tokens") or 0, dt


# =============================================================================
# Model properties (parameters, context window, quant, ...)
# =============================================================================
# model_info keys that may carry a parameter count / quant (custom entries in config.yaml)
PARAM_KEYS = ("params_b", "parameters", "parameter_count", "params", "n_params", "num_parameters")
ACTIVE_KEYS = ("active_params_b", "active_parameters", "active_params")
QUANT_KEYS = ("quant", "quantization")
SKIP_INFO_KEYS = {"id", "db_model", "key", "litellm_provider_model"}


def _params_b(v):
    """'35B', '35b', 35, 35e9, '3.5e10' -> billions."""
    if v is None or v == "":
        return None
    if isinstance(v, str):
        m = re.fullmatch(r"\s*([\d.]+(?:e\d+)?)\s*([kmbt]?)\s*", v, re.I)
        if not m:
            return None
        x = float(m.group(1)) * {"": 1, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[m.group(2).lower()]
        if not m.group(2) and x < 1e5:
            x *= 1e9  # bare small number: already in billions
    else:
        x = float(v) * (1e9 if float(v) < 1e5 else 1)
    return round(x / 1e9, 2)


def model_props(cl, name):
    """Model properties as reported by LiteLLM /model/info (model_info + upstream model id)."""
    entry = cl.info.get(name)
    if not entry:
        return {"sources": []}
    mi = entry.get("model_info") or {}
    lp = entry.get("litellm_params") or {}
    props = {k: v for k, v in mi.items()
             if k not in SKIP_INFO_KEYS and v not in (None, "", [], {}) and not k.endswith("_api_key")}
    if lp.get("model"):
        props["upstream_model"] = lp["model"]
    for k, out in (("input_cost_per_token", "input_cost_per_mtok"),
                   ("output_cost_per_token", "output_cost_per_mtok")):
        v = lp.get(k) if lp.get(k) is not None else mi.get(k)
        if v:
            props[out] = round(float(v) * 1e6, 4)
    for keys, out in ((PARAM_KEYS, "params_b"), (ACTIVE_KEYS, "active_params_b")):
        for k in keys:
            v = _params_b(mi.get(k))
            if v:
                props[out] = v
                break
    for k in QUANT_KEYS:
        if mi.get(k):
            props["quant"] = str(mi[k])
            break
    props["context_window"] = mi.get("max_input_tokens") or mi.get("max_tokens")
    props["sources"] = ["litellm"]
    return props


def pmap(fn, items, n):
    if n <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(fn, items))


def summarize(details):
    passed = sum(1 for d in details if d["pass"])
    toks = [d["tokens"] for d in details if d.get("tokens")]
    return {"score": passed / len(details) if details else None, "passed": passed,
            "total": len(details), "errors": sum(1 for d in details if d.get("error")),
            "avg_completion_tokens": round(sum(toks) / len(toks)) if toks else None,
            "details": details}


def clip(s, n=1500):
    s = s or ""
    return s if len(s) <= n else s[:n] + " …"


# =============================================================================
# Filler text (haystacks, prefill, embedding throughput)
# =============================================================================
_ADJ = ["quiet", "northern", "old", "busy", "narrow", "green", "wooden", "distant", "famous", "small"]
_NOUN = ["harbour", "library", "bakery", "bridge", "orchard", "workshop", "station", "garden", "museum", "market"]
_PLACE = ["Bergamo", "Graz", "Lyon", "Utrecht", "Porto", "Gdansk", "Brno", "Ghent", "Trieste", "Basel"]
_VERB = ["was repainted", "hosted a festival", "closed for repairs", "won a regional award",
         "doubled its visitors", "changed owners", "installed solar panels", "published a report"]
_TIME = ["in early spring", "after a long winter", "during the 1990s", "last autumn",
         "on a rainy Tuesday", "before the elections", "in mid-summer", "at the turn of the century"]


def filler(n_tokens: int, rng: random.Random) -> str:
    out, approx = [], 0
    while approx < n_tokens:
        s = (f"The {rng.choice(_ADJ)} {rng.choice(_NOUN)} in {rng.choice(_PLACE)} "
             f"{rng.choice(_VERB)} {rng.choice(_TIME)}, and {rng.randint(2, 999)} residents noticed.")
        out.append(s)
        approx += len(s) // 4 + 1
    return " ".join(out)


# =============================================================================
# Suite: speed
# =============================================================================
def suite_speed(cl, model, args):
    out = {}
    r = cl.chat(model, [{"role": "user", "content": "Reply with the single word: ready"}],
                max_tokens=512)
    out["first_request_s"] = round(r.total, 2)
    if r.error:
        out["error"] = r.error
        return out

    gens = []
    for i in range(args.runs):
        r = cl.chat(model, [{"role": "user", "content":
                             f"(run {i}) Write a 300-word story about a lighthouse keeper "
                             f"who finds a message in a bottle."}],
                    max_tokens=args.gen_tokens)
        if not r.error:
            gens.append(r)
    if gens:
        out["ttft_ms"] = round(median(g.ttft for g in gens if g.ttft) * 1000)
        tps = [g.decode_tps for g in gens if g.decode_tps]
        out["decode_tps"] = round(median(tps), 1) if tps else None
        out["gen_tokens"] = round(median(g.completion_tokens for g in gens))

    # unique nonce so llama-server's prompt cache cannot skip the prefill
    text = f"[{uuid.uuid4()}] " + filler(args.prefill_tokens, random.Random(time.time()))
    r = cl.chat(model, [{"role": "user", "content": text + "\n\nSummarize the text above in one sentence."}],
                max_tokens=64)
    if not r.error:
        out["prefill_tps"] = round(r.prefill_tps, 1) if r.prefill_tps else None
        out["prefill_prompt_tokens"] = r.prompt_tokens
    else:
        out["prefill_error"] = r.error
    return out


# =============================================================================
# Suite: parallel (N simultaneous users)
# =============================================================================
PARALLEL_TOPICS = ["a lighthouse keeper", "a mountain guide", "a submarine cook", "a desert botanist",
                   "a night-train conductor", "an arctic radio operator", "a beekeeper", "a city cartographer"]


def parallel_round(cl, model, n, args):
    """Fire n streamed generations at the same instant; return per-request results + wall time."""
    gate = threading.Barrier(n)

    def one(i):
        gate.wait()
        topic = PARALLEL_TOPICS[i % len(PARALLEL_TOPICS)]
        return cl.chat(model, [{"role": "user", "content":
                                f"[{uuid.uuid4()}] Write a 300-word story about {topic} "
                                f"who finds a message in a bottle."}],
                       max_tokens=args.gen_tokens)

    with ThreadPoolExecutor(n) as ex:
        res = list(ex.map(one, range(n)))
    ok = [r for r in res if not r.error]
    if not ok:
        return res, 0.0
    t0 = min(r.started for r in ok)
    wall = max(r.started + r.total for r in ok) - t0
    return res, wall


def suite_parallel(cl, model, args):
    """Compare 1 user with N simultaneous users. Scaling ~N means real parallelism (two GPUs
    behind LiteLLM, or llama-server slots); ~1 means requests are queued one after another."""
    levels = sorted({1, *args.parallel_levels})
    # warm-up so the first level doesn't include a llama-swap model load
    w = cl.chat(model, [{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=16)
    if w.error:
        return {"error": w.error}
    out, base, all_deps = {"levels": []}, None, {}
    for n in levels:
        res, wall = parallel_round(cl, model, n, args)
        ok = [r for r in res if not r.error]
        lv = {"n": n, "ok": len(ok), "errors": len(res) - len(ok)}
        if ok:
            toks = sum(r.completion_tokens for r in ok)
            tps = [r.decode_tps for r in ok if r.decode_tps]
            ttft = [r.ttft for r in ok if r.ttft]
            lv.update({
                "wall_s": round(wall, 2),
                "total_tokens": toks,
                "aggregate_tps": round(toks / wall, 1) if wall else None,
                "per_user_tps": round(median(tps), 1) if tps else None,
                # what the slowest user experiences: tokens / (queue wait + generation)
                "slowest_user_tps": round(min(r.completion_tokens / r.total for r in ok if r.total), 1),
                "ttft_ms": round(median(ttft) * 1000) if ttft else None,
                "ttft_ms_max": round(max(ttft) * 1000) if ttft else None,
            })
            deps = {}
            for r in ok:
                key = r.deployment or r.api_base or "?"
                deps[key] = deps.get(key, 0) + 1
                if r.deployment or r.api_base:
                    all_deps[key] = r.api_base or ""
            lv["deployments"] = deps
        if res and len(ok) < len(res):
            lv["error"] = next(r.error for r in res if r.error)
        if n == 1 and lv.get("aggregate_tps"):
            base = lv["aggregate_tps"]
        if base and lv.get("aggregate_tps"):
            lv["scaling"] = round(lv["aggregate_tps"] / base, 2)        # x times the single-user throughput
            lv["efficiency"] = round(lv["scaling"] / n, 2)               # 1.0 = perfect parallelism
        out["levels"].append(lv)
    out["deployments_seen"] = len(all_deps)
    out["deployment_api_bases"] = all_deps
    multi = [lv for lv in out["levels"] if lv["n"] > 1 and lv.get("scaling")]
    if multi:
        best = max(multi, key=lambda lv: lv["aggregate_tps"])
        out["best_n"], out["best_aggregate_tps"], out["best_scaling"] = best["n"], best["aggregate_tps"], best["scaling"]
    return out


def parallel_line(r):
    if r.get("error"):
        return f"ERROR {r['error']}"
    parts = [f"n={lv['n']}: {num(lv.get('aggregate_tps'))} t/s total, {num(lv.get('per_user_tps'))}/user"
             + (f", x{lv['scaling']}" if lv.get("scaling") and lv["n"] > 1 else "")
             + (f", {lv['errors']} err" if lv["errors"] else "") for lv in r["levels"]]
    return " | ".join(parts) + f"  [{r.get('deployments_seen', 0)} deployment(s)]"


# =============================================================================
# Suite: math (GSM8K-style)
# =============================================================================
MATH_TASKS = [
    ("A bakery sells 48 rolls in the morning and twice as many in the afternoon. "
     "Each roll costs 0.75 euros. How many euros does the bakery take in from rolls that day?", 108),
    ("A train travels 120 km at 80 km/h and then another 90 km at 60 km/h. "
     "How many minutes does the whole trip take?", 180),
    ("A marina charges 42 euros per night. Booking 7 nights at once gives 15% off those 7 nights. "
     "A boat stays 10 nights: the first 7 at the discounted weekly rate and the remaining 3 at the "
     "normal nightly price. What is the total cost in euros?", 375.9),
    ("A 600-litre tank already holds 100 litres. Water flows in at 25 litres per minute but leaks "
     "out at 5 litres per minute. How many minutes until the tank is full?", 25),
    ("Lisa has three times as many marbles as Tom. Together they have 64 marbles. "
     "Lisa then gives Tom 10 marbles. How many marbles does Lisa have now?", 38),
    ("A rectangle has a perimeter of 46 cm and its length is 5 cm more than its width. "
     "What is its area in square centimetres?", 126),
    ("A jacket costs 80 euros. Its price is raised by 20% and later the new price is reduced by 25%. "
     "What is the final price in euros?", 72),
    ("What is the sum of all integers from 1 to 200 inclusive that are divisible by 3 or by 5?", 9368),
    ("A worker earns 18 euros per hour for the first 40 hours of a week and 1.5 times that rate for "
     "every hour beyond 40. How much does the worker earn in a week of 47 hours?", 909),
    ("If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines need to make "
     "100 widgets?", 5),
    ("What is the smaller angle in degrees between the hour hand and the minute hand of an analog "
     "clock at 3:40?", 130),
    ("A recipe for 4 people needs 300 g of flour. How many grams of flour are needed for 10 people?", 750),
]
MATH_SUFFIX = ("\n\nSolve step by step, then finish with a final line of the form "
               "'ANSWER: <number>' (digits only, no units).")


def extract_number(text):
    m = re.findall(r"ANSWER:?\s*\**\s*[$€]?\s*(-?[\d,]*\.?\d+)", text, re.I)
    if m:
        cand = m[-1]
    else:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        if not nums:
            return None
        cand = nums[-1]
    try:
        return float(cand.replace(",", ""))
    except ValueError:
        return None


def suite_math(cl, model, args, tasks):
    def run(t):
        q, ans = t
        r = cl.chat(model, [{"role": "user", "content": q + MATH_SUFFIX}], max_tokens=args.max_tokens)
        txt = visible(r.content)
        pred = extract_number(txt)
        ok = pred is not None and abs(pred - ans) <= 1e-4 * max(1.0, abs(ans))
        return {"q": q[:80], "expected": ans, "got": pred, "pass": ok,
                "tokens": r.completion_tokens, "error": r.error, "answer": clip(txt, 400)}
    return summarize(pmap(run, tasks, args.concurrency))


# =============================================================================
# Suite: code (HumanEval-style, executed)
# =============================================================================
def _ct(entry, prompt, test):
    return {"task_id": f"local/{entry}", "entry_point": entry,
            "prompt": prompt.strip("\n") + "\n", "test": test.strip("\n") + "\n"}


CODE_TASKS = [
    _ct("is_palindrome", '''
def is_palindrome(s: str) -> bool:
    """Return True if s is a palindrome when ignoring case and every non-alphanumeric character.
    >>> is_palindrome("A man, a plan, a canal: Panama")
    True
    """
''', '''
def check(c):
    assert c("A man, a plan, a canal: Panama") is True
    assert c("race a car") is False
    assert c("") is True
    assert c("No 'x' in Nixon") is True
    assert c("ab") is False
'''),
    _ct("run_length_encode", '''
def run_length_encode(s: str) -> str:
    """Run-length encode s as each character followed by its run length.
    >>> run_length_encode("aaabcc")
    'a3b1c2'
    An empty string returns an empty string.
    """
''', '''
def check(c):
    assert c("aaabcc") == "a3b1c2"
    assert c("") == ""
    assert c("a") == "a1"
    assert c("aaaaaaaaaaaa") == "a12"
    assert c("abba") == "a1b2a1"
'''),
    _ct("merge_intervals", '''
def merge_intervals(intervals: list) -> list:
    """Merge overlapping or touching closed intervals [start, end]; return them sorted by start.
    >>> merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])
    [[1, 6], [8, 10], [15, 18]]
    """
''', '''
def check(c):
    assert c([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert c([[1, 4], [4, 5]]) == [[1, 5]]
    assert c([]) == []
    assert c([[5, 7], [1, 2]]) == [[1, 2], [5, 7]]
    assert c([[1, 10], [2, 3]]) == [[1, 10]]
'''),
    _ct("roman_to_int", '''
def roman_to_int(s: str) -> int:
    """Convert a Roman numeral (I, V, X, L, C, D, M, with subtractive notation) to an integer.
    >>> roman_to_int("MCMXCIV")
    1994
    """
''', '''
def check(c):
    assert c("III") == 3
    assert c("IV") == 4
    assert c("IX") == 9
    assert c("LVIII") == 58
    assert c("MCMXCIV") == 1994
    assert c("MMXXVI") == 2026
'''),
    _ct("flatten", '''
def flatten(nested: list) -> list:
    """Flatten arbitrarily nested lists into one flat list, preserving order. Strings are atoms.
    >>> flatten([1, [2, [3, [4]], 5]])
    [1, 2, 3, 4, 5]
    """
''', '''
def check(c):
    assert c([1, [2, [3, [4]], 5]]) == [1, 2, 3, 4, 5]
    assert c([]) == []
    assert c([[[]]]) == []
    assert c(["ab", ["cd"]]) == ["ab", "cd"]
'''),
    _ct("top_k_frequent", '''
def top_k_frequent(words: list, k: int) -> list:
    """Return the k most frequent words, ordered by frequency (highest first);
    words with equal frequency are ordered alphabetically.
    """
''', '''
def check(c):
    assert c(["i", "love", "leetcode", "i", "love", "coding"], 2) == ["i", "love"]
    assert c(["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], 4) == ["the", "is", "sunny", "day"]
    assert c(["b", "a"], 2) == ["a", "b"]
'''),
    _ct("valid_brackets", '''
def valid_brackets(s: str) -> bool:
    """Return True if every (), [] and {} in s is correctly matched and nested.
    Characters other than brackets are ignored.
    """
''', '''
def check(c):
    assert c("()[]{}") is True
    assert c("([)]") is False
    assert c("{[]}") is True
    assert c("") is True
    assert c("((") is False
    assert c("a(b)c") is True
    assert c(")(") is False
'''),
    _ct("spiral_order", '''
def spiral_order(matrix: list) -> list:
    """Return the elements of a 2-D matrix in clockwise spiral order, starting top-left.
    >>> spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    [1, 2, 3, 6, 9, 8, 7, 4, 5]
    """
''', '''
def check(c):
    assert c([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
    assert c([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) == [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
    assert c([]) == []
    assert c([[1], [2], [3]]) == [1, 2, 3]
'''),
    _ct("parse_duration", '''
def parse_duration(s: str) -> int:
    """Parse a duration like "1d2h30m15s" into total seconds. Units d, h, m, s may each
    appear at most once, in that order, and any of them may be missing.
    >>> parse_duration("1h30m15s")
    5415
    """
''', '''
def check(c):
    assert c("1h30m15s") == 5415
    assert c("45s") == 45
    assert c("2h") == 7200
    assert c("1d2h") == 93600
    assert c("10m5s") == 605
'''),
    _ct("lis_length", '''
def lis_length(nums: list) -> int:
    """Return the length of the longest strictly increasing subsequence of nums.
    >>> lis_length([10, 9, 2, 5, 3, 7, 101, 18])
    4
    """
''', '''
def check(c):
    assert c([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert c([0, 1, 0, 3, 2, 3]) == 4
    assert c([7, 7, 7]) == 1
    assert c([]) == 0
'''),
]

CODE_HEADER = "from typing import *\nimport math, re, string, collections, itertools, functools, heapq\n\n"


def extract_code(text, entry):
    blocks = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", text, re.S)
    for b in sorted(blocks, key=len, reverse=True):
        if f"def {entry}" in b:
            return b
    return blocks[0] if blocks else text


def run_code_test(task, completion):
    code = extract_code(completion, task["entry_point"])
    if f"def {task['entry_point']}" not in code:
        code = task["prompt"] + code
    program = CODE_HEADER + code + "\n\n" + task["test"] + f"\n\ncheck({task['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candidate.py")
        with open(path, "w") as fh:
            fh.write(program)
        try:
            r = subprocess.run([sys.executable, path], cwd=d, capture_output=True, text=True,
                               timeout=20, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return r.returncode == 0, r.stderr[-400:]


def suite_code(cl, model, args, tasks):
    def run(t):
        prompt = ("Complete the following Python function. Reply with the complete implementation "
                  "(including the def line and any imports) in a single ```python code block.\n\n"
                  f"```python\n{t['prompt']}```")
        r = cl.chat(model, [{"role": "user", "content": prompt}], max_tokens=args.max_tokens)
        txt = visible(r.content)
        ok, err = (False, r.error) if r.error else run_code_test(t, txt)
        return {"task": t["task_id"], "pass": ok, "tokens": r.completion_tokens,
                "error": r.error, "stderr": clip(err or "", 300)}
    return summarize(pmap(run, tasks, args.concurrency))


# =============================================================================
# Suite: instruction following (IFEval-style)
# =============================================================================
def _words(t):
    return re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", t)


def _nonempty_lines(t):
    return [l.strip() for l in t.splitlines() if l.strip()]


def _strip_fences(t):
    t = t.strip()
    m = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", t, re.S)
    return m.group(1) if m else t


def _chk_json(t):
    try:
        d = json.loads(_strip_fences(t))
    except Exception:
        return False
    return (isinstance(d, dict) and set(d) == {"city", "population"} and isinstance(d["city"], str)
            and isinstance(d["population"], int) and not isinstance(d["population"], bool))


IF_TASKS = [
    ("Write exactly 3 bullet points about solar energy. Each line must start with '- '. "
     "Output nothing else.",
     lambda t: len(_nonempty_lines(t)) == 3 and all(l.startswith("- ") for l in _nonempty_lines(t))),
    ("Describe the city of Munich in two sentences. Your entire response must be in lowercase "
     "letters; do not use any capital letters.",
     lambda t: t == t.lower() and len(_words(t)) >= 8),
    ("Write one sentence of at least 6 words about the sea without using the letter 'e' anywhere "
     "(neither uppercase nor lowercase).",
     lambda t: "e" not in t.lower() and len(_words(t)) >= 6),
    ("Write a single paragraph about bicycles with between 50 and 80 words. Do not use any commas.",
     lambda t: 50 <= len(_words(t)) <= 80 and "," not in t),
    ("Antworte mit genau einem Wort auf Deutsch: Wie heißt die Hauptstadt von Italien?",
     lambda t: re.sub(r"\W", "", t).lower() == "rom"),
    ("Give me two tips for brewing better coffee. End your response with the exact phrase: "
     "Is there anything else I can help with?",
     lambda t: t.rstrip().endswith("Is there anything else I can help with?")),
    ("Write a short paragraph about sailing that uses the word 'anchor' at least twice and the "
     "word 'tide' at least twice.",
     lambda t: len(re.findall(r"\banchor", t, re.I)) >= 2 and len(re.findall(r"\btide", t, re.I)) >= 2),
    ("Wrap your entire response in double quotation marks. Question: what is the boiling point of "
     "water at sea level in degrees Celsius?",
     lambda t: len(t) > 2 and t[0] in "\"“" and t[-1] in "\"”"),
    ("Explain what a VPN is. Your response must have exactly 2 sections. Mark the beginning of each "
     "section with 'SECTION X', where X is the section number (SECTION 1, SECTION 2).",
     lambda t: len(re.findall(r"SECTION \d", t)) == 2 and "SECTION 1" in t and "SECTION 2" in t),
    ("Return the city of Rome as a JSON object with exactly two keys: \"city\" (string) and "
     "\"population\" (integer, approximate). Output only the JSON, nothing else.",
     _chk_json),
    ("Answer with exactly 5 words: what colour is the sky on a clear day?",
     lambda t: len(_words(t)) == 5),
]


def suite_ifeval(cl, model, args):
    def run(t):
        prompt, checker = t
        r = cl.chat(model, [{"role": "user", "content": prompt}], max_tokens=args.max_tokens)
        txt = visible(r.content)
        try:
            ok = bool(checker(txt)) and not r.error
        except Exception:
            ok = False
        return {"prompt": prompt[:80], "pass": ok, "tokens": r.completion_tokens,
                "error": r.error, "answer": clip(txt, 400)}
    return summarize(pmap(run, IF_TASKS, args.concurrency))


# =============================================================================
# Suite: tool calling (BFCL-style)
# =============================================================================
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Munich'"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "convert_currency", "description": "Convert an amount of money between currencies.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string", "description": "ISO 4217 code, e.g. EUR"},
            "to_currency": {"type": "string", "description": "ISO 4217 code, e.g. USD"}},
            "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {
        "name": "create_issue", "description": "Create an issue in the issue tracker.",
        "parameters": {"type": "object", "properties": {
            "project": {"type": "string", "description": "Project key, e.g. HOME"},
            "title": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "normal", "high", "critical"]}},
            "required": ["project", "title"]}}},
]


def _args(c):
    try:
        a = json.loads(c.get("arguments") or "{}")
        return a if isinstance(a, dict) else None
    except Exception:
        return None


def _only(calls, name):
    return len(calls) == 1 and calls[0]["name"] == name and _args(calls[0]) is not None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


TOOL_CASES = [
    {"name": "simple",
     "messages": [{"role": "user", "content": "What's the weather like in Munich right now? Use celsius."}],
     "check": lambda calls, txt: _only(calls, "get_weather")
        and "munich" in str(_args(calls[0]).get("city", "")).lower()
        and _args(calls[0]).get("unit") == "celsius"},
    {"name": "selection",
     "messages": [{"role": "user", "content": "How much is 250 euros in US dollars?"}],
     "check": lambda calls, txt: _only(calls, "convert_currency")
        and _num(_args(calls[0]).get("amount")) == 250
        and str(_args(calls[0]).get("from_currency", "")).upper() == "EUR"
        and str(_args(calls[0]).get("to_currency", "")).upper() == "USD"},
    {"name": "enum_args",
     "messages": [{"role": "user", "content":
                   "Open a critical ticket in project HOME titled 'NAS fan is failing'."}],
     "check": lambda calls, txt: _only(calls, "create_issue")
        and str(_args(calls[0]).get("project", "")).upper() == "HOME"
        and _args(calls[0]).get("priority") == "critical"
        and "nas fan" in str(_args(calls[0]).get("title", "")).lower()},
    {"name": "parallel",
     "messages": [{"role": "user", "content": "What's the weather in Rome and in Vienna? Celsius please."}],
     "check": lambda calls, txt: len(calls) == 2
        and all(c["name"] == "get_weather" and _args(c) for c in calls)
        and any("rome" in str(_args(c).get("city", "")).lower() or "roma" in str(_args(c).get("city", "")).lower() for c in calls)
        and any(x in str(_args(c).get("city", "")).lower() for c in calls for x in ("vienna", "wien"))},
    {"name": "irrelevance",
     "messages": [{"role": "user", "content": "Tell me a short joke about cats."}],
     "check": lambda calls, txt: not calls and len(txt) > 10},
    {"name": "use_tool_result",
     "messages": [
         {"role": "user", "content": "What's the weather in Munich?"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "call_1", "type": "function", "function": {
                 "name": "get_weather", "arguments": "{\"city\": \"Munich\", \"unit\": \"celsius\"}"}}]},
         {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(
             {"city": "Munich", "temperature": 17, "unit": "celsius", "condition": "light rain"})}],
     "check": lambda calls, txt: not calls and "17" in txt and "rain" in txt.lower()},
]


def suite_tools(cl, model, args):
    def run(case):
        r = cl.chat(model, case["messages"], max_tokens=args.max_tokens, tools=TOOLS, stream=False)
        txt = visible(r.content)
        try:
            ok = not r.error and bool(case["check"](r.tool_calls, txt))
        except Exception:
            ok = False
        return {"case": case["name"], "pass": ok, "tokens": r.completion_tokens, "error": r.error,
                "tool_calls": r.tool_calls, "answer": clip(txt, 300)}
    return summarize(pmap(run, TOOL_CASES, args.concurrency))


# =============================================================================
# Suite: needle in a haystack
# =============================================================================
def suite_needle(cl, model, args):
    def run(depth):
        rng = random.Random(f"{depth}-{uuid.uuid4()}")  # distinct haystacks defeat prompt caching
        code = str(rng.randint(1000, 9999))
        sents = filler(args.needle_tokens, rng).split(". ")
        pos = int(len(sents) * depth)
        sents.insert(pos, f"The secret access code for the marina gate is {code}")
        hay = ". ".join(sents)
        q = (hay + "\n\nQuestion: What is the secret access code for the marina gate? "
             "Answer with the code only.")
        r = cl.chat(model, [{"role": "user", "content": q}], max_tokens=args.max_tokens)
        txt = visible(r.content)
        return {"depth": depth, "pass": code in txt and not r.error, "tokens": r.completion_tokens,
                "prompt_tokens": r.prompt_tokens,
                "prefill_tps": round(r.prefill_tps, 1) if r.prefill_tps else None,
                "error": r.error, "answer": clip(txt, 120)}
    return summarize([run(d) for d in (0.1, 0.5, 0.9)])  # sequential: big prompts


# =============================================================================
# Suite: embeddings
# =============================================================================
EMBED_PAIRS = [  # (query, relevant document); several are cross-lingual
    ("How do I reset my router to factory settings?",
     "To restore the router's factory defaults, hold the reset button on the back for about ten "
     "seconds until the lights flash."),
    ("Wie lange muss man Nudeln kochen?",
     "Pasta is usually boiled in salted water for eight to twelve minutes, depending on its shape "
     "and thickness."),
    ("Quali documenti servono per ormeggiare in un porto turistico?",
     "Für einen Liegeplatz im Yachthafen benötigt man in der Regel die Bootspapiere, eine "
     "Haftpflichtversicherung und einen Ausweis."),
    ("symptoms of dehydration",
     "Common signs that the body lacks water include thirst, dark urine, dizziness and fatigue."),
    ("Welche Programmiersprache eignet sich für Datenanalyse?",
     "Python ist dank Bibliotheken wie pandas und NumPy eine der beliebtesten Sprachen für "
     "Datenanalyse."),
    ("best time to go hiking in the Dolomites",
     "Le Dolomiti sono ideali per l'escursionismo tra fine giugno e settembre, quando i sentieri "
     "sono liberi dalla neve."),
    ("how does public key encryption work",
     "Asymmetric cryptography uses a key pair: data encrypted with the public key can only be "
     "decrypted with the matching private key."),
    ("Come si prepara il risotto?",
     "Il risotto si prepara tostando il riso, sfumando con vino bianco e aggiungendo brodo caldo "
     "poco alla volta."),
    ("Warum ist der Himmel blau?",
     "The sky looks blue because air molecules scatter the short blue wavelengths of sunlight "
     "more strongly than red ones (Rayleigh scattering)."),
    ("train delays because of a strike",
     "Wegen eines Streiks der Lokführer fallen heute zahlreiche Regional- und Fernzüge aus oder "
     "verspäten sich."),
]
EMBED_DISTRACTORS = [  # hard negatives sharing vocabulary with the queries
    "The router's Wi-Fi password is printed on a sticker on the bottom of the device.",
    "Rice should be rinsed before cooking to remove excess starch.",
    "Die Liegeplatzgebühren im Yachthafen steigen jedes Jahr im Frühling.",
    "Sunsets look red because the light passes through much more atmosphere near the horizon.",
    "JavaScript läuft hauptsächlich im Browser und wird für interaktive Webseiten genutzt.",
    "Symmetric encryption uses the same secret key for both encryption and decryption.",
    "Il treno regionale per Venezia parte ogni ora dal binario tre.",
    "Drinking too much water in a short time can dilute sodium levels in the blood.",
]


def embed_prefixes(model):
    m = model.lower()
    if "nomic" in m:
        return "search_query: ", "search_document: "
    if "qwen3-embedding" in m:
        return ("Instruct: Given a web search query, retrieve relevant passages that answer "
                "the query\nQuery: "), ""
    return "", ""


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def suite_embed(cl, model, args):
    out = {}
    t0 = time.perf_counter()
    vecs, _, _ = cl.embed(model, ["warm-up"])
    out["first_request_s"] = round(time.perf_counter() - t0, 2)
    out["dims"] = len(vecs[0])

    # retrieval quality
    qp, dp = embed_prefixes(model)
    docs = [d for _, d in EMBED_PAIRS] + EMBED_DISTRACTORS
    dvec = [_norm(v) for v in cl.embed(model, [dp + d for d in docs])[0]]
    qvec = [_norm(v) for v in cl.embed(model, [qp + q for q, _ in EMBED_PAIRS])[0]]
    top1, rr, details = 0, 0.0, []
    for i, q in enumerate(qvec):
        sims = [sum(a * b for a, b in zip(q, d)) for d in dvec]
        rank = sorted(range(len(docs)), key=lambda j: -sims[j]).index(i) + 1
        top1 += rank == 1
        rr += 1.0 / rank
        details.append({"query": EMBED_PAIRS[i][0], "rank": rank, "pass": rank == 1})
    out["top1"] = top1 / len(EMBED_PAIRS)
    out["mrr"] = round(rr / len(EMBED_PAIRS), 3)
    out["details"] = details

    # throughput
    rng = random.Random(1)
    passages = [dp + filler(120, rng) for _ in range(args.embed_docs)]
    tok, t = 0, 0.0
    for i in range(0, len(passages), args.embed_batch):
        _, n, dt = cl.embed(model, passages[i:i + args.embed_batch])
        tok += n
        t += dt
    out["docs_per_s"] = round(len(passages) / t, 1) if t else None
    out["tokens_per_s"] = round(tok / t, 1) if t and tok else None
    return out


# =============================================================================
# Hugging Face datasets (optional, --hf N)
# =============================================================================
HF_ROWS = "https://datasets-server.huggingface.co/rows"


def hf_rows(dataset, config, split, n, seed=0):
    rows, offset = [], 0
    need = max(n, 100)
    while len(rows) < need:
        r = requests.get(HF_ROWS, params={"dataset": dataset, "config": config, "split": split,
                                          "offset": offset, "length": 100}, timeout=60)
        r.raise_for_status()
        batch = [x["row"] for x in r.json().get("rows", [])]
        if not batch:
            break
        rows += batch
        offset += 100
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_hf_tasks(n):
    print(f"Fetching {n} GSM8K + {n} HumanEval items from Hugging Face …")
    gsm = [(r["question"], float(r["answer"].split("####")[-1].strip().replace(",", "")))
           for r in hf_rows("openai/gsm8k", "main", "test", n)]
    he = [{"task_id": r["task_id"], "entry_point": r["entry_point"], "prompt": r["prompt"],
           "test": r["test"]} for r in hf_rows("openai/openai_humaneval", "openai_humaneval", "test", n)]
    return gsm, he


# =============================================================================
# Reporting
# =============================================================================
def pct(x):
    return "-" if x is None else f"{x * 100:.0f}%"


def num(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def fmt_ctx(x):
    if not x:
        return "-"
    return f"{x / 1024:.0f}k" if x >= 1024 else str(x)


def fmt_params(p):
    if not p or not p.get("params_b"):
        return "-"
    s = f"{p['params_b']:g}B"
    if p.get("active_params_b"):
        s += f" (A{p['active_params_b']:g}B)"
    return s


def props_line(p):
    if not p.get("sources"):
        return "no LiteLLM model info"
    parts = [f"params {fmt_params(p)}", f"ctx {fmt_ctx(p.get('context_window'))}"]
    if p.get("max_output_tokens"):
        parts.append(f"max out {fmt_ctx(p['max_output_tokens'])}")
    if p.get("quant"):
        parts.append(p["quant"])
    return "  ".join(parts)


def render_tables(results):
    lines = []
    chat = results["chat"]
    if chat:
        hdr = ["Model", "Params", "Quant", "Ctx", "load s", "TTFT ms", "gen t/s", "pp t/s", "Math", "Code", "IF",
               "Tools", "Needle", "Avg", "cost $", "time s"]
        rows = []
        for m, r in chat.items():
            sp = r.get("speed", {})
            scores = {k: (r.get(k) or {}).get("score") for k in ("math", "code", "ifeval", "tools", "needle")}
            have = [v for v in scores.values() if v is not None]
            avg = sum(have) / len(have) if have else None
            p = r.get("props") or {}
            rows.append([m, fmt_params(p), p.get("quant") or "-", fmt_ctx(p.get("context_window")),
                         num(sp.get("first_request_s")), num(sp.get("ttft_ms"), 0),
                         num(sp.get("decode_tps")), num(sp.get("prefill_tps"), 0),
                         pct(scores["math"]), pct(scores["code"]), pct(scores["ifeval"]),
                         pct(scores["tools"]), pct(scores["needle"]), pct(avg),
                         num(r.get("cost"), 4), num(r.get("wall_s"), 0)])
        lines += _table(hdr, rows)
    par = {m: r["parallel"] for m, r in chat.items() if (r.get("parallel") or {}).get("levels")}
    if par:
        lines.append("")
        hdr = ["Parallel", "users", "total t/s", "t/s per user", "slowest incl. wait", "TTFT ms", "TTFT max",
               "scaling", "efficiency", "deployments"]
        rows = []
        for m, p in par.items():
            for lv in p["levels"]:
                rows.append([m if lv is p["levels"][0] else "", lv["n"], num(lv.get("aggregate_tps")),
                             num(lv.get("per_user_tps")), num(lv.get("slowest_user_tps")),
                             num(lv.get("ttft_ms"), 0), num(lv.get("ttft_ms_max"), 0),
                             f"x{lv['scaling']}" if lv.get("scaling") else "-", pct(lv.get("efficiency")),
                             len(lv.get("deployments") or {}) or "-"])
        lines += _table(hdr, rows)
    emb = results["embedding"]
    if emb:
        lines.append("")
        hdr = ["Embedding model", "Params", "Ctx", "load s", "dims", "top-1", "MRR", "docs/s", "tok/s", "time s"]
        rows = [[m, fmt_params(r.get("props")), fmt_ctx((r.get("props") or {}).get("context_window")),
                 num(r.get("first_request_s")), str(r.get("dims", "-")), pct(r.get("top1")),
                 num(r.get("mrr"), 3), num(r.get("docs_per_s")), num(r.get("tokens_per_s"), 0),
                 num(r.get("wall_s"), 0)] for m, r in emb.items()]
        lines += _table(hdr, rows)
    return lines


def _table(hdr, rows):
    w = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)]
    fmt = lambda r: "| " + " | ".join(str(x).ljust(w[i]) for i, x in enumerate(r)) + " |"
    return [fmt(hdr), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [fmt(r) for r in rows]


def failures_md(results):
    out = []
    for m, r in results["chat"].items():
        fails = []
        for s in ("math", "code", "ifeval", "tools", "needle"):
            for d in (r.get(s) or {}).get("details", []):
                if not d["pass"]:
                    label = d.get("q") or d.get("task") or d.get("prompt") or d.get("case") or d.get("depth")
                    why = d.get("error") or d.get("stderr") or d.get("answer") or d.get("tool_calls")
                    fails.append(f"- **{s}** · {label} → `{clip(str(why), 160)}`".replace("\n", " "))
        if fails:
            out += [f"\n### {m}", *fails]
    return out


def save(results, out_dir, stamp):
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, f"llmbench-{stamp}.json")
    with open(jp, "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=str)
    return jp


# =============================================================================
# Main
# =============================================================================
def main():
    env_file = find_env_file()
    load_env_file(env_file)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-file", help="path to .env (default: ./.env, then next to the script)")
    ap.add_argument("--base-url", default=os.environ.get("LITELLM_BASE_URL", "https://api.ctollama.home.arpa"))
    ap.add_argument("--api-key", default=os.environ.get("LITELLM_MASTER_KEY") or os.environ.get("LITELLM_API_KEY"))
    ap.add_argument("--models", help="comma-separated model names (default: all from LiteLLM)")
    ap.add_argument("--exclude", default="", help="comma-separated model names to skip")
    ap.add_argument("--suites", default=",".join(DEFAULT_SUITES),
                    help=f"subset of {','.join(ALL_SUITES)} (default: all but parallel)")
    ap.add_argument("--parallel", default="", metavar="N[,M...]",
                    help="run the parallel suite with these numbers of simultaneous users, e.g. 2 or 2,4 "
                         "(1 user is always measured as the baseline)")
    ap.add_argument("--hf", type=int, default=0, metavar="N",
                    help="use N real GSM8K/HumanEval items from Hugging Face instead of built-ins")
    ap.add_argument("--max-tokens", type=int, default=4096, help="cap for quality answers (thinking needs room)")
    ap.add_argument("--gen-tokens", type=int, default=768, help="tokens generated per speed run")
    ap.add_argument("--runs", type=int, default=3, help="speed runs per model (median reported)")
    ap.add_argument("--prefill-tokens", type=int, default=4000)
    ap.add_argument("--needle-tokens", type=int, default=16000, help="approx. haystack size")
    ap.add_argument("--embed-docs", type=int, default=64)
    ap.add_argument("--embed-batch", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--no-think", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=false (Qwen-style templates)")
    ap.add_argument("--insecure", action="store_true",
                    default=os.environ.get("LITELLM_INSECURE", "").lower() in ("1", "true", "yes"),
                    help="skip TLS verification (self-signed cert); better: set REQUESTS_CA_BUNDLE or --ca-bundle")
    ap.add_argument("--ca-bundle", default=os.environ.get("REQUESTS_CA_BUNDLE"),
                    help="path to CA bundle for the LiteLLM endpoint (overrides REQUESTS_CA_BUNDLE)")
    ap.add_argument("--no-props", action="store_true", help="don't read model properties from LiteLLM /model/info")
    ap.add_argument("--out", default="bench-results")
    args = ap.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    try:
        args.parallel_levels = sorted({int(x) for x in args.parallel.split(",") if x.strip()})
    except ValueError:
        ap.error("--parallel expects numbers, e.g. 2 or 2,4")
    if any(n < 1 for n in args.parallel_levels):
        ap.error("--parallel values must be >= 1")
    if args.parallel_levels and "parallel" not in suites:
        suites.append("parallel")
    if "parallel" in suites and not args.parallel_levels:
        args.parallel_levels = [2]
    bad = set(suites) - set(ALL_SUITES)
    if bad:
        ap.error(f"unknown suite(s): {', '.join(bad)}")

    if not args.api_key:
        sys.exit("No API key: set LITELLM_API_KEY (or LITELLM_MASTER_KEY) in .env or pass --api-key")
    print(f"Config: {env_file or 'no .env file'} -> {args.base_url}")

    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_think else {}
    if args.insecure:
        verify = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    elif args.ca_bundle:
        verify = args.ca_bundle
    else:
        verify = True
    cl = Client(args.base_url, args.api_key, args.timeout, extra, verify=verify)

    try:
        models = cl.list_models()
    except requests.exceptions.SSLError as e:
        sys.exit(f"TLS error talking to {cl.base}: {e}\n"
                 "-> set REQUESTS_CA_BUNDLE to your CA certificate, or LITELLM_INSECURE=1 in .env")
    except requests.exceptions.RequestException as e:
        sys.exit(f"Cannot reach LiteLLM at {cl.base}: {e}")
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        known = dict(models)
        models = [(m, known.get(m, "embedding" if "embed" in m.lower() else "chat")) for m in wanted]
    excl = {m.strip() for m in args.exclude.split(",") if m.strip()} | ALIASES_TO_SKIP
    models = [(m, mode) for m, mode in models if m not in excl]
    if not models:
        sys.exit("No models to benchmark.")

    math_tasks, code_tasks = MATH_TASKS, CODE_TASKS
    if args.hf and ({"math", "code"} & set(suites)):
        math_tasks, code_tasks = load_hf_tasks(args.hf)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = {"meta": {"started": stamp, "base_url": cl.base, "suites": suites,
                        "hf_items": args.hf, "no_think": args.no_think,
                        "math_items": len(math_tasks), "code_items": len(code_tasks),
                        "parallel_levels": args.parallel_levels if "parallel" in suites else []},
               "chat": {}, "embedding": {}}

    print(f"Benchmarking {len(models)} model(s) via {cl.base}: {', '.join(m for m, _ in models)}\n")
    for name, mode in models:
        t0 = time.perf_counter()
        if mode == "embedding":
            if "embed" not in suites:
                continue
            print(f"▶ {name} (embedding)")
            try:
                res = suite_embed(cl, name, args)
                print(f"    top-1 {pct(res['top1'])}  MRR {res['mrr']}  {res['docs_per_s']} docs/s")
            except Exception as e:
                res = {"error": f"{type(e).__name__}: {e}"}
                print(f"    ERROR {res['error']}")
            if not args.no_props:
                res["props"] = model_props(cl, name)
                print(f"    props   {props_line(res['props'])}")
            res["wall_s"] = round(time.perf_counter() - t0, 1)
            res["cost"] = round(cl.cost.get(name, 0.0), 6)
            results["embedding"][name] = res
        else:
            chat_suites = [s for s in suites if s != "embed"]
            if not chat_suites:
                continue
            print(f"▶ {name}")
            res = {}
            for s in chat_suites:
                ts = time.perf_counter()
                try:
                    if s == "speed":
                        res[s] = suite_speed(cl, name, args)
                        sp = res[s]
                        msg = (f"load {num(sp.get('first_request_s'))}s  TTFT {num(sp.get('ttft_ms'), 0)}ms  "
                               f"gen {num(sp.get('decode_tps'))} t/s  pp {num(sp.get('prefill_tps'), 0)} t/s")
                        if sp.get("error"):
                            msg = f"ERROR {sp['error']}"
                    else:
                        fn = {"math": lambda: suite_math(cl, name, args, math_tasks),
                              "code": lambda: suite_code(cl, name, args, code_tasks),
                              "ifeval": lambda: suite_ifeval(cl, name, args),
                              "tools": lambda: suite_tools(cl, name, args),
                              "needle": lambda: suite_needle(cl, name, args),
                              "parallel": lambda: suite_parallel(cl, name, args)}[s]
                        res[s] = fn()
                        r = res[s]
                        if s == "parallel":
                            msg = parallel_line(r)
                        else:
                            msg = f"{r['passed']}/{r['total']} ({pct(r['score'])})"
                            if r["errors"]:
                                msg += f"  [{r['errors']} request errors]"
                            if r["avg_completion_tokens"]:
                                msg += f"  ~{r['avg_completion_tokens']} tok/answer"
                except Exception as e:
                    res[s] = {"error": f"{type(e).__name__}: {e}"}
                    msg = f"ERROR {res[s]['error']}"
                print(f"    {s:<7} {msg}  ({time.perf_counter() - ts:.0f}s)")
                if s == "speed" and res[s].get("error"):
                    print("    skipping remaining suites for this model")
                    break
            if not args.no_props:
                res["props"] = model_props(cl, name)
                print(f"    props   {props_line(res['props'])}")
            res["wall_s"] = round(time.perf_counter() - t0, 1)
            res["cost"] = round(cl.cost.get(name, 0.0), 6)
            results["chat"][name] = res
        save(results, args.out, stamp)  # incremental, survives a crash

    results["meta"]["finished"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    jp = save(results, args.out, stamp)
    table = render_tables(results)
    print("\n" + "\n".join(table))

    md = [f"# LLM benchmark {stamp}", "",
          f"Endpoint `{cl.base}` · suites: {', '.join(suites)} · "
          f"math items {len(math_tasks)} · code items {len(code_tasks)}"
          + (" · thinking disabled" if args.no_think else ""), "",
          "load = first request incl. llama-swap model load · TTFT/gen from streamed runs (median) · "
          "pp = prefill tokens/s on a cache-busted prompt · Params / Quant / Ctx from LiteLLM model_info "
          "(Ctx = max_input_tokens)"
          + (" · Parallel: N users at once, scaling = total t/s vs. 1 user, efficiency = scaling / N, "
             "deployments = distinct LiteLLM deployments that answered" if "parallel" in suites else ""), "", *table, "", "## Failures", *failures_md(results)]
    mp = jp[:-5] + ".md"
    with open(mp, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"\nSaved {jp}\n      {mp}")


if __name__ == "__main__":
    main()
