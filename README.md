# llmbench

Benchmarks every model behind the LiteLLM proxy on CTollama (llama-swap / llama.cpp backend,
plus any cloud models in the LiteLLM config): speed, answer quality, tool calling, long context
and embeddings. One run produces a comparison table plus JSON and Markdown reports.

## Quick start

```bash
chmod +x start.sh
./start.sh            # first run: creates .env from env.example and stops
nano .env             # set LITELLM_API_KEY
./start.sh            # creates .venv, installs requests, runs all models
```

Requirements: Python 3.10+ with `venv` (Debian/Ubuntu: `sudo apt install python3-venv`)
and network access to the LiteLLM endpoint.

## Files

| File | Purpose |
|---|---|
| `llmbench.py` | The benchmark (only dependency: `requests`) |
| `start.sh` | Creates `.venv` on first run, checks `.env`, passes all arguments to the script |
| `env.example` | Template for `.env` |
| `.env` | Your credentials - keep it out of git |
| `bench-results/` | Output: `llmbench-<timestamp>.json` and `.md` |

## Configuration (`.env`)

```ini
LITELLM_BASE_URL=https://api.ctollama.home.arpa
LITELLM_API_KEY=sk-...                 # virtual key or master key (LITELLM_MASTER_KEY also works)
# REQUESTS_CA_BUNDLE=/path/to/ca.crt   # if the internal certificate isn't trusted
# LITELLM_INSECURE=1                   # fallback: skip TLS verification
```

The script reads `.env` from the current folder, then from its own folder; `--env-file PATH`
picks another one. Variables already exported in the shell take precedence.

## What gets tested

| Suite | What it measures | Modelled on |
|---|---|---|
| `speed` | Load time (first request, includes the llama-swap swap), time to first token, generation tok/s, prompt-processing tok/s (cache-busted) | llama-bench-style throughput |
| `math` | 12 word problems with exact numeric answers | GSM8K |
| `code` | 10 functions; generated code is **executed** against unit tests | HumanEval |
| `ifeval` | 11 automatically checkable instructions (lowercase only, exact JSON, word limits, ...) | IFEval |
| `tools` | Simple call, tool selection, enum args, parallel calls, no-tool case, answering from a tool result | BFCL |
| `needle` | Code hidden in a ~16k-token text at 10 / 50 / 90 % depth | Needle-in-a-haystack |
| `embed` | Embedding throughput; en/de/it retrieval with hard negatives (top-1, MRR) | MTEB-style retrieval |
| `parallel` | *Opt-in* (`--parallel 2,4`): N simultaneous users vs. 1 user - total and per-user tok/s, queueing, which LiteLLM deployment served each request | multi-user load test |

Chat models run every suite except `embed`; embedding models run only `embed`. The aliases
`homelab-coder` and `homelab-doc` are skipped because they point at a model already in the list.

The built-in task sets are small and run offline - good for comparing your models against each
other, not for matching published leaderboard numbers. `--hf N` swaps the math and code suites
for N real GSM8K and HumanEval items fetched from Hugging Face.

## Common runs

```bash
./start.sh --exclude MiniMax-M3                      # local models only (no API spend)
./start.sh --models qwen3.6-35b-a3b-UD-Q4_K_M,compress
./start.sh --suites speed                            # speed only, fast
./start.sh --suites embed                            # embedding models only
./start.sh --hf 50 --suites math,code                # real GSM8K / HumanEval subsets
./start.sh --no-think                                # disable Qwen-style thinking
./start.sh --suites speed --parallel 2,4             # 1 vs 2 vs 4 simultaneous users
./start.sh --help                                    # all options
```

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--models` / `--exclude` | all | Comma-separated model names |
| `--suites` | all | `speed,math,code,ifeval,tools,needle,embed` |
| `--hf N` | off | Use N real GSM8K / HumanEval items |
| `--max-tokens` | 4096 | Answer cap for quality suites (thinking models need room) |
| `--runs` | 3 | Speed runs per model, median reported |
| `--needle-tokens` | 16000 | Haystack size; must fit each model's `-c` |
| `--concurrency` | 1 | Parallel requests in quality suites |
| `--no-think` | off | Sends `chat_template_kwargs.enable_thinking=false` |
| `--parallel N[,M]` | off | Add the parallel suite with these user counts (1 user is always the baseline) |
| `--no-props` | off | Don't collect model properties |
| `--out` | `bench-results` | Output folder |

## Reading the results

The console table and the Markdown report show, per chat model:

- **Params / Quant / Ctx** - model properties (see below)
- **load s** - first request, including the model load by llama-swap
- **TTFT ms**, **gen t/s**, **pp t/s** - latency, generation and prompt-processing speed
- **Math / Code / IF / Tools / Needle** - pass rate per suite, **Avg** across them
- **cost $** - from LiteLLM's `x-litellm-response-cost`, i.e. the prices in its config
  (Opus-reference pricing for local models, so a comparison figure, not real spend)

The Markdown report lists every failed test with the model's answer. The JSON contains all
details, including average tokens per answer (shows how much a thinking model reasons).
Results are saved after each model, so an interrupted run keeps what finished.

### Model properties

Each model gets a `props` block in the JSON, taken only from LiteLLM's `/model/info`: every
`model_info` field (max input/output tokens, capability flags, prices, mode, ...) plus the
upstream model id. `context_window` is `max_input_tokens` (else `max_tokens`).

LiteLLM has no parameter count of its own; add it to the model's `model_info` in LiteLLM's
`config.yaml` and the script picks it up (also `active_parameters` and `quantization`):

```yaml
- model_name: qwen3.6-35b-a3b-UD-Q4_K_M
  litellm_params: { model: openai/qwen3.6-35b-a3b-UD-Q4_K_M, api_base: http://llama-swap:8080/v1 }
  model_info:
    max_input_tokens: 65536
    parameters: 35B
    active_parameters: 3B
    quantization: UD-Q4_K_M
```

### Parallel users

`--parallel 2` (or `2,4`, ...) sends N identical generation requests at the same instant, after
a 1-user baseline, and reports per level:

- **total t/s** - all tokens / wall time; **scaling** - that divided by the 1-user value
- **t/s per user** - each request's own generation speed (median)
- **slowest incl. wait** - tokens / (queue wait + generation) of the worst-off user
- **TTFT / TTFT max** - how long users wait for the first token (queueing shows here)
- **deployments** - how many distinct LiteLLM deployments answered (`x-litellm-model-id`)

How to read it: if LiteLLM balances the model across two GPUs, 2 users give about **x2** and
2 deployments. With one card and one llama-server slot the requests queue: **x1**, TTFT max ≈ a
whole answer, and the slowest user gets half the speed. With one card but `--parallel`/`-np` > 1
in llama-server, expect something in between (batching: total t/s up, per-user t/s down).
Note that llama-server splits `-c` across its slots, so each user gets a smaller context.

### Viewing results in the browser

Open `html/analyse.html` directly from disk (no server needed) and drop one or more
`bench-results/*.json` files onto it: sortable tables, model properties, charts, and every
test answer.

## Notes and troubleshooting

- **The code suite runs model-generated Python** in a subprocess with a timeout. Run the
  benchmark on a machine or container where that's acceptable.
- **All tools tests fail on a llama.cpp model**: add `--jinja` to that model's `cmd` in
  `llama-swap.yaml` so the chat template's tool-call format is used.
- **Needle errors on one model**: the haystack is larger than its context (`-c`). Lower
  `--needle-tokens`.
- **TLS error**: set `REQUESTS_CA_BUNDLE` to your CA certificate, or `LITELLM_INSECURE=1`.
- **Timeouts on first request**: llama-swap's `healthCheckTimeout` and LiteLLM's 900 s
  timeout apply; very large models can take a while to load.
- Keep `--concurrency 1` for clean speed numbers - llama-server shares `-c` across its
  parallel slots, so concurrent requests also shrink the context per request.
- `MiniMax-M3` is a cloud model: running it costs real money. Exclude it with
  `--exclude MiniMax-M3`.
