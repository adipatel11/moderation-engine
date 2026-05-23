# moderation-engine

> Low-latency toxicity classification for real-time chat, optimized for CPU.

![status](https://img.shields.io/badge/status-shipped-success)
![python](https://img.shields.io/badge/python-3.11-blue)
![runtime](https://img.shields.io/badge/runtime-ONNX%20%2B%20INT8-informational)
![hardware](https://img.shields.io/badge/target-AWS%20c6i.large%20(2%20vCPU)-orange)
![license](https://img.shields.io/badge/license-MIT-green)

A FastAPI service wrapping `unitary/toxic-bert` (BERT-base, 6 labels), deployed
to a $25/month AWS box. Built end-to-end as a portfolio MLE project: naive
PyTorch baseline → ONNX → INT8 → length-bucketed dynamic batching → ORT
threading tune → INT4 long-shot (failed deliberately) → identity-disaggregated
bias eval.

## Headline numbers

Same hardware (`c6i.large`, 2 vCPU, $25/mo), same locked benchmark protocol
(Phase 1; see [`docs/benchmarks.md`](docs/benchmarks.md)):

| Metric | Naive baseline | Final (INT8 + ORT tune) | Δ |
|---|---:|---:|---:|
| p99 latency @ 1 user | 860 ms | ~400 ms | **−54%** |
| p99 latency @ 10 users | 4,800 ms | 2,000 ms | **−58%** |
| Throughput @ 10 users | 7.0 req/s | **23.6 req/s** | **+237%** |
| Container size | 873 MB | 173 MB | **−80%** |
| Macro-F1 (Jigsaw) | 0.6101 | **0.6146** | +0.0045 |

Macro-F1 went *up* under INT8 — counter to the usual quantization-trades-accuracy
narrative. The full story (why precision went up on 5/6 labels while recall went
down) is in [`docs/benchmarks.md`](docs/benchmarks.md#int8-accuracy-phase-2-opt-2).

### The chart (Phase 2 Opt 3: batching window sweep)

![Dynamic batching window sweep — c6i.large, 10-user closed loop](docs/charts/batching_window_sweep.png)

Bypass mode (`window=0`) wins on throughput, which is the axis that saturates
this 2-vCPU host. p99 actually *nudges down* at `window=2` (1,800 ms vs
2,200 ms), but throughput drops 32% to buy that. The optimization that
mattered wasn't the batcher — it was the one-line ORT threading knob from
Opt 4 (`intra_op_num_threads=1`).

## Quick start

```bash
docker compose up                 # builds the image, boots on :8000
curl http://localhost:8000/health # {"status":"ok","model_loaded":true}
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"I love this!"}'
# {"labels":{"toxic":0.0007,...},"model_version":"unitary/toxic-bert@onnx-int8"}
```

First boot takes ~30s (model loads into memory once at startup). The image is
multi-stage; the INT8-quantized ONNX export is baked into the image at build
time so the runtime has no Hugging Face fetch and works offline
(`TRANSFORMERS_OFFLINE=1`).

## Architecture

```mermaid
flowchart LR
  C[client] -->|POST /classify| API[FastAPI]
  API --> B{Batcher<br/>window_ms?}
  B -->|0 default| ORT[ONNX Runtime<br/>INT8, intra_op=1]
  B -->|>0| Q[length-bucketed queues<br/>64 / 256 / 512 tok] --> ORT
  ORT --> API --> C
  subgraph host[c6i.large · 2 vCPU · 4 GB · $25/mo]
    API
    B
    Q
    ORT
  end
```

The batcher is in-tree but disabled by default — the EC2 sweep showed bypass
wins on a 2-vCPU host (the ORT intra-op threads already saturate both cores
at batch=1). Flipping `BATCHING_WINDOW_MS=5` re-enables the bucketed path for
multi-vCPU instances.

## Optimization journey

The headline artifact of the project. Every per-opt row was measured under the
locked Phase 1 protocol (closed-loop locust sweep at 1/5/10/25/50/100 users
× 60s, on `c6i.large`, against the same 1000-row Jigsaw sample).

| Stage | p99 @ 1u/10u (ms) | Throughput @ 10u (req/s) | Macro-F1 | Container |
|---|---:|---:|---:|---:|
| Baseline (PyTorch CPU) | 860 / 4,800 | 7.0 | 0.6101 | 873 MB |
| + ONNX Runtime | 930 / 7,300 | 8-9 | 0.6101 | 521 MB |
| + INT8 dynamic quant | 430 / 3,100 | 22-23 | **0.6146** | 173 MB |
| + Dynamic batching (bucketed) | bypass wins | 20-21 (w=0) | 0.6146 | 173 MB |
| + ORT threading tune (`intra_op=1`) | ~400 / **2,000** | **23.6** | 0.6146 | 173 MB |
| Long-shot — INT4 NF4 ❌ | bs=1 = 47.7 ms (11× slower) | 25.3 samples/s eval | 0.6127 | 146 MB |

### Opt 1 — ONNX Runtime (lossless)

Export PyTorch → ONNX via `optimum`; serve with `onnxruntime`. Element-wise
parity vs PyTorch: max abs prob diff **3.76e-06** on 63,978 Jigsaw rows × 6
labels. p50 −25%, throughput +30%. **What surprised**: p99 *widened* at
mid concurrency (+90% at 25 users) — ORT defaults
`intra_op_num_threads=num_cores`, so one inference grabs both vCPUs. Great
for p50 at 1 user; bad for p99 with concurrent requests. **Foreshadows Opt 4.**

### Opt 2 — INT8 dynamic quantization (the big win)

`onnxruntime.quantization.quantize_dynamic(QuantType.QInt8)` — weights only,
no calibration data. p50 **−50 to −66%** across the curve; throughput **+120
to +191%**; container 521 MB → 173 MB. Macro-F1 went **up** by 0.0045 (5/6
labels improved). **Why F1 went up**: precision up + recall down on every
label — INT8 mildly compresses logits toward 0, and the FP32 baseline was
precision-limited at threshold 0.5. The trade landed on the right side of
the bias/variance curve for this baseline. Full breakdown:
[`docs/benchmarks.md`](docs/benchmarks.md#int8-accuracy-phase-2-opt-2).

**Hardware tangent worth knowing**: the eval almost didn't happen on Colab T4
— ORT has no CUDA kernel for `MatMulInteger`/`DynamicQuantizeLinear`, so the
whole graph fell back per-op to CPU with bus copies (13 h estimate). Colab
CPU was worse — free-tier Xeons predate AVX-VNNI, no INT8 SIMD. Ran locally
on M3 Pro (ARM SDOT) instead. **A quantization format is only as fast as the
SIMD that supports it** — comes back hard in the INT4 long-shot below.

### Opt 3 — Length-bucketed dynamic batching (implemented, measured, bypass wins)

From-scratch `asyncio` batcher in [`moderation_engine/batcher.py`](moderation_engine/batcher.py) —
no `mosec` or `litserve` (built it to learn the failure modes). Round 1 — naive
single-queue batching — collapsed throughput 4× because Jigsaw's heavy-tailed
seq_len (p99 = 512 tokens) + `padding=True` made every batch pay the longest-
item compute. Round 2 — route into `[64, 256, 512]` buckets sized against the
58/37/5 traffic split — recovered most of the loss (5.4 → 14 req/s) but bypass
still won (20.7 req/s). **Why**: ORT defaults to 2 intra-op threads on a 2-vCPU
host, so a single inference already pegs both cores; batching adds queueing
without adding compute capacity. **The deliverable isn't "I shipped batching"** —
it's the curve in [`docs/charts/batching_window_sweep.png`](docs/charts/batching_window_sweep.png)
and the diagnosis of why naive batching failed (padding), what fixed it
(bucketing), and why even the fix can't beat bypass on 2 vCPUs (intra-op
saturation). Code stays in-tree behind `BATCHING_WINDOW_MS` for the
multi-vCPU configuration where it'd flip positive.

### Opt 4 — One-line ORT threading knob (the surprise winner)

`SessionOptions.intra_op_num_threads = 1`. One line. **+7.4% throughput** over
the INT8 baseline at 10 users. Two single-threaded inferences run truly in
parallel on the two vCPUs instead of fighting for shared intra-op threads —
exactly the Opt 1 diagnosis read in reverse. **The interaction**: `intra_op=1`
is a Pareto win in bypass mode but loses combined with the batcher (−25% vs
either bypass row). Per-bucket batches give up the intra-op parallelism that
bypass gets to keep. Optimizations don't always stack — you have to evaluate
the combination, not the deltas.

### Long-shot — INT4 NF4 quantization (the deliberate failure)

Hypothesis: if INT8 halved size and tripled speed, INT4 should at minimum
shrink the container further at unchanged latency. Wrapped
`MatMulBnb4Quantizer(quant_type=NF4)` over the FP32 export.

**Three predicted failure modes all turned out wrong**: quantization ran
cleanly (72/96 MatMul ops rewritten to `com.microsoft::MatMulBnb4`); ORT 1.26
*does* ship a CPU `MatMulBnb4` kernel; and **accuracy was actually *better*
than INT8** — only 608/383,868 = 0.158% of decisions flipped vs FP32, vs INT8's
0.505%. NF4's learned codebook fits transformer weight distributions better
than INT8's uniform grid.

**The actual failures**: 11.4× *slower* than INT8 on bs=1 (47.7 ms vs 4.2 ms),
3× slower on the full Jigsaw eval, and 33% *larger* on disk (146 MB vs 110 MB,
because Bnb4 only packs MatMul weights, not the 92 MB embedding table).
**Root cause**: no INT4 SIMD on x86 VNNI or ARM SDOT. INT8 has `vpdpbusd` and
`SDOT`. INT4 has nothing equivalent — the CPU kernel must unpack 4-bit codes
→ look up NF4 LUT → multiply by per-block FP16 scale → upcast → run FP32
matmul. INT8's `MatMulInteger` skips all of that.

**The takeaway**: NF4 is the most accurate quantizer I tried. It's still
strictly worse than INT8 on this CPU because of instruction-set asymmetry, not
code quality. **The platform-cost vector of a quantization format is part of
the format.** Production stays at INT8. Full writeup:
[`docs/benchmarks.md`](docs/benchmarks.md#long-shot-int4-bitsandbytes-style-quantization--the-thing-that-didnt-work).

## Bias evaluation

Production INT8 evaluated for unintended demographic bias on the Jigsaw
"Unintended Bias in Toxicity Classification" test set (42,870 identity-annotated
rows, the slice Borkan et al. 2019 and the Jigsaw competition leaderboard
evaluate on). Full methodology and tables:
[`docs/bias_evaluation.md`](docs/bias_evaluation.md).

| Subgroup | n | FPR | Δ vs overall (1.39%) |
|---|---:|---:|---:|
| LGB | 1,065 | **3.61%** | **+2.22 pp (2.6×)** |
| Mental illness | 511 | 3.21% | +1.82 pp (2.3×) |
| Female | 5,155 | 1.99% | +0.60 pp |
| Black | 1,519 | 1.77% | +0.38 pp |
| Male | 4,386 | 1.75% | +0.36 pp |
| Jewish | 835 | 1.72% | +0.33 pp |
| White | 2,452 | 1.64% | +0.25 pp |
| Muslim | 2,040 | 1.22% | −0.17 pp |
| Christian | 4,226 | 0.81% | −0.58 pp |

The pattern matches what Borkan et al. report across toxicity classifiers —
over-flagging of identity mentions, with the largest gaps on identities
frequently discussed in adversarial contexts online. This is **inherited bias
from the training data distribution**, not introduced by quantization (FP32
would show the same disparities). The eval **measures and documents** without
attempting to debias — naive debiasing approaches commonly trade one harm for
another. Jigsaw competition power-mean bias score: **0.8705** (overall AUC
0.9019).

## Design decisions

- **From-scratch batcher** instead of `mosec` / `litserve` — the project's
  goal was to learn the failure modes, not to ship a black box. The diagnosis
  in [`docs/benchmarks.md`](docs/benchmarks.md#dynamic-batching-phase-2-opt-3)
  is only possible because we own the code.
- **INT8 over INT4** in production despite INT4's better accuracy — the
  long-shot writeup explains why. ISA features set the floor; algorithms set
  the ceiling.
- **Bypass batching default** despite shipping the batcher — measured on the
  production target instead of trusting "batching always wins." Stays in-tree
  behind `BATCHING_WINDOW_MS` for multi-vCPU configs.
- **`intra_op_num_threads=1` default** — counterintuitive on a 2-vCPU host
  (the obvious setting is 2). The Opt 4 sweep showed it wins by +7.4% under
  closed-loop concurrency. Override with `ONNX_INTRA_OP_THREADS=0` on
  beefier instances.
- **Locked benchmark protocol in Phase 1** — same locust sample, same sweep
  shape (1/5/10/25/50/100 users × 60 s), same host. Every per-opt comparison
  is apples-to-apples. The protocol is documented in
  [`docs/benchmarks.md`](docs/benchmarks.md#protocol-locked-in-phase-1--do-not-change-once-set).

## Known limitations

- **Inherited demographic bias** (see Bias evaluation above) — LGB-mention FPR
  is 2.6× the overall rate; mental-illness mentions 2.3×. Documented; not
  fixed. Naive fixes commonly make things worse.
- **No adversarial-robustness audit yet** — misspellings, leetspeak,
  zero-width unicode, polite-sounding threats. Plan §3 calls for a half-day
  probing exercise — not yet done.
- **No Prometheus instrumentation** — out of scope for this project. The
  service emits structured JSON logs only (via `structlog`); aggregation
  would be a downstream concern if this were deployed for real.
- **Single-instance deployment** — no autoscaling, no load balancer. The
  $25/mo box ceiling is ~23.6 req/s at p99 = 2 s. Beyond that you scale
  horizontally; the container is stateless.
- **Bypass batching default is hardware-specific** — flip
  `BATCHING_WINDOW_MS=5` on a multi-vCPU instance to recoup the
  batching win. Not auto-detected.
- **Multi-arch images** (`linux/arm64` + `linux/amd64`) build but the
  production deploy is amd64 only; arm64 images are for local Mac dev.

## What I'd do next

- Adversarial probing → populate a "Known Failure Modes" section with
  concrete examples that fool the model.
- Per-subgroup threshold tuning as a bias-mitigation experiment (with proper
  hold-out evaluation, since it can easily make things worse).
- Try the Bnb4 long-shot again on Sapphire Rapids (AMX-INT4) or a GPU target —
  the format is right; the silicon was wrong.

## Reproducing

```bash
# Eval the accuracy floor (PyTorch baseline, ~33 min on M3 Pro)
uv run python scripts/eval_baseline.py --backend pytorch

# Quantize to INT8 (~30 s)
uv run python scripts/quantize_onnx.py

# Eval INT8 (~14 min on M3 Pro)
uv run python scripts/eval_baseline.py --backend onnx \
  --onnx-dir models/onnx-toxic-bert-int8

# Bias eval (~10 min on M3 Pro; requires Kaggle competition rules acceptance —
# see scripts/download_civil_comments.py)
uv run python scripts/download_civil_comments.py
uv run python scripts/eval_bias.py
uv run python scripts/render_bias_writeup.py

# Latency sweep against a running container (closed-loop, 1/5/10/25/50/100 users)
./scripts/run_locust_sweep.sh http://<host>:8000

# Re-render the batching window-sweep chart
uv run python scripts/render_batching_chart.py
```

Every committed predictions parquet (`docs/baseline_predictions.parquet`,
`docs/onnx_predictions.parquet`, `docs/int8_predictions.parquet`,
`docs/int4_nf4_predictions.parquet`, `docs/bias_predictions.parquet`)
contains per-row probabilities so future experiments can diff against any
prior backend without re-running the inference pass.

## Project structure

```
moderation_engine/      # the service (deploys in the container)
  api.py                # FastAPI app, /classify + /health
  batcher.py            # length-bucketed async batcher (in-tree, default off)
  config.py             # pydantic-settings: env vars + defaults
  model.py              # backend-agnostic ToxicityClassifier protocol
  backends/             # PyTorchToxicityClassifier, ONNXToxicityClassifier
scripts/                # eval, quantize, load-test, chart-render (dev only)
docs/
  benchmarks.md         # source of truth for every measured number
  bias_evaluation.md    # Phase 3 bias eval writeup
  charts/               # rendered figures
  locust/               # per-sweep CSVs (phase1-baseline, phase2-*, batching, threading)
  *_predictions.parquet # raw per-row probs from each backend
tests/                  # pytest, including the batcher parallelism contract
Dockerfile              # multi-stage: uv builder + slim runtime, INT8 baked in
docker-compose.yml      # local dev (port 8000, healthcheck)
```

## Tech stack

`Python 3.11` · `FastAPI` · `pydantic` · `ONNX Runtime` (INT8 dynamic quant) ·
`Transformers` / `optimum` · `Docker` (multi-stage, multi-arch) ·
`AWS EC2` / `ECR` · `locust` (closed-loop load testing) · `structlog` ·
`uv` · `ruff` · `pre-commit` · `pytest`

## License

MIT. See [`LICENSE`](LICENSE).

## Author

Adit Patel | AI student building production-grade ML infrastructure projects.
`aditdpatel05@gmail.com` · [GitHub](https://github.com/adipatel11)
