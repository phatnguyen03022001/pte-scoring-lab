# TASK-05 — Serverless GPU Feasibility & Final Technical Assessment

**Date:** 2026-11-11  
**Repository:** `phatnguyen03022001/pte-scoring-lab`  
**Investigated HEAD:** `b35c27ba51fa91100e27a5ccab2cbe6f8bce14ea`  
**Scope:** non-billable technical investigation, baseline regression verification, Cloud Run documentation and conditional cost assessment.  
**Final verdict:** **CLOUD EXECUTION BLOCKED — NOT VERIFIED**. This is **not** an engineering PASS or an operational-scoring authorization.

## A. Separate verdicts

| Dimension | Verdict | Evidence / blocker |
| --- | --- | --- |
| Container compatibility | NOT VERIFIED | No Linux container exists in the repo; local Docker daemon is unavailable. |
| CUDA compatibility | NOT VERIFIED | CUDA/cuBLAS/cuDNN wheels and inference devices were not exercised on Linux/NVIDIA. |
| Actual GPU inference | NOT VERIFIED | No GPU execution was authorized or performed. |
| End-to-end serverless scoring | NOT VERIFIED | No Cloud Run deployment or minimal HTTP interface. |
| CPU/GPU scoring parity | NOT VERIFIED | No matched Linux CPU/GPU inference outputs. |
| Cold-start, latency and throughput | NOT VERIFIED | Neither first request nor warm request was measured in cloud. |
| Reliability under concurrency | NOT VERIFIED | Existing Mac tests are not a Cloud Run load test. |
| Cloud price model | PARTIAL | Official published list rates checked; runtime, billing, registry/egress and regional SKU costs unknown. |
| Technical cloud feasibility | PARTIAL / BLOCKED | Source inspection suggests feasible acceleration for ASR after engineering work; full-engine compatibility unproven. |
| Pearson ≥85% agreement | NOT VERIFIED | Frozen TASK-04 scientific evidence unchanged. |
| Operational learner scoring | **NO-GO** | Scientific and GPU parity evidence insufficient. |

**Cloud authorization boundary:** An active `gcloud` account, configured project and billing-enabled project were confirmed read-only. None of these constitutes the Human Owner's explicit authorization for a billable experiment. No paid resource was created, modified or deleted, no quota request was filed, and no billed GPU test was attempted. Regional GPU quota and permissions to deploy were **not verified**. Existing cloud identities and project identifiers are intentionally omitted.

**Local build boundary:** `docker` and `buildx` are installed, but `docker version` could not connect to the configured OrbStack daemon. No host service was restarted and no Linux/AMD64 build was attempted. `nvidia-smi` is unavailable locally.

## B. Verified execution map

| Stage | Existing implementation / device | Proposed minimum GPU investigation | Evidence |
| --- | --- | --- | --- |
| Audio decode | PyAV/FFmpeg, CPU | Remain CPU | `score_integrated.py`, `score_pronunciation.py`. |
| ASR | faster-whisper/CTranslate2; explicit `device="cpu"`, `compute_type="int8"` | CUDA 12 with cuBLAS and cuDNN 9; evaluate fp16/int8 variants **without** assuming parity | `score_integrated.py:397-404`; `uv.lock`: faster-whisper 1.2.1, CTranslate2 4.8.5. |
| VAD, content, prosody/fluency | Python/native CPU | Remain CPU unless profiling proves otherwise | `score_integrated.py`. |
| MFCC, i-vectors, acoustic posteriors, alignment, GOP | Kaldi 5.5-era local source/build for **macOS arm64**, CPU | Rebuild pinned Kaldi source **for Linux/AMD64** and verify all subprocess utilities | `.local/kaldi` not suitable as an x86-64 runtime binary package. |
| GOPT | PyTorch, explicit `torch.load(..., map_location="cpu")` and CPU tensors | Remain CPU initially; test CUDA only if justified and numerically checked | `score_pronunciation.py:775-923`. |
| Aggregation | Python CPU | Remain CPU | `score_integrated.py`. |

**Principal findings / root causes of missing compatibility:**

1. The current scorer **will not use an NVIDIA GPU** just because it is deployed on a GPU instance: the ASR device is hardcoded to CPU and GOPT runs on CPU. GPU access and GPU inference are different propositions.
2. The current `uv.lock` covers the application Python dependencies, **not** PyTorch, `kaldiio`, the separately managed GOPT runtime, Kaldi native binaries or all model assets. A reproducible Linux/AMD64 installation remains to be specified and verified, using fixed versions and hashes. No driver or CUDA stack has been verified on a real L4.
3. `score_integrated.py` starts a separate pronunciation Python process, and its ASR function instantiates `WhisperModel` per invocation. GOPT also loads its checkpoint per scoring invocation. A warm Cloud Run instance does **not** imply warm model weights. Changing reuse/device semantics requires explicit paired-parity evidence before acceptance.
4. `score_pronunciation.py:315-345,1028` reads `GOPT_DATA` (`.local/gopt/data.zip`, ~167 MB) to reconstruct the training-derived phone mapping, even though the expected symbols are also represented in `GOPT_PHONE_ORDER`. Do **not** ship the full research archive to a cloud image to satisfy this lookup without checking rights. A future minimal mapping-materialization change must show exact mapping and scoring parity against the frozen baseline.
5. The local pretrained Kaldi M13 chain/extractor/language archives (~430 MB compressed total), GOPT checkpoint and GOPT source are stored under ignored `.local`. The public repo has neither a Linux build manifest nor a safely provisioned, license-reviewed cloud asset source. Asset presence on the Mac does not prove Cloud Run readiness.
6. A minimal authenticated HTTP inference interface, cancellation propagation, readiness probe, cleanup proof, and GPU telemetry do not exist yet. An API that merely returns JSON is not sufficient evidence of safe serverless scoring.

**Preserved model identities:** Kaldi M13 chain source archive SHA-256 `47cb7cc2fbe91c7a60fd3d46266c3d459bd766aaffa1dfad789ef57875f1be01`; GOPT LibriSpeech checkpoint SHA-256 `ab07451e51648f9d2455505a51055b20ac4ad7921d771ccc5170ff486a826259` (recorded local baseline, not re-uploaded). No model, dataset or cloud credential was committed.

## C. Performance evidence, strictly separated

| Measure | Mac CPU baseline (historical; one 4.4 s sample, pronunciation-only) | Linux CPU | Cloud Run L4 |
| --- | ---: | ---: | ---: |
| Total pronunciation pipeline | 2.850 s | NOT MEASURED | NOT MEASURED |
| Kaldi features + i-vectors | 0.806 s | NOT MEASURED | NOT MEASURED |
| Acoustic posteriors | 0.396 s | NOT MEASURED | NOT MEASURED |
| Forced alignment | 0.361 s | NOT MEASURED | NOT MEASURED |
| GOP extraction | 0.031 s | NOT MEASURED | NOT MEASURED |
| GOPT checkpoint initialization | 0.0074 s | NOT MEASURED | NOT MEASURED |
| GOPT first/warm forward | 0.00074 / 0.00036 s | NOT MEASURED | NOT MEASURED |
| Cold start / model readiness / warm p50/p95 | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Peak single-process RSS | ~348 MiB (historical) | NOT MEASURED | NOT MEASURED |
| Peak VRAM, GPU utilization, concurrency | N/A | N/A | NOT MEASURED |

Historical metrics are from the project's previously saved `.local/audio-scoring/evidence-validated/result.json` and README, **not** a fresh head-to-head benchmark. They are not sufficient to predict L4 latency: CPU/Apple Silicon and Linux CPUs differ, and the integrated ASR path adds more work. The **GOPT forward pass is tiny** on this sample, so moving only GOPT to GPU would be unjustified without a representative workload demonstrating improvement.

**Frozen scientific baseline from TASK-04:** Vietnamese-L1 pronunciation coverage 22.72%, precision 0.2726, recall 0.1720, F1 0.2109, conditional ROC-AUC 0.7600 and false-positive rate 59.08%; Pearson ≥85% agreement remains NOT VERIFIED. No inference thresholds, score scales, acoustic support gates or scientific evidence were changed.

## D. Official Cloud Run constraints and cost

Documentation reviewed 2026-11-11:

- Cloud Run Services NVIDIA L4: **24 GB VRAM**; minimum **4 vCPU and 16 GiB RAM**; `asia-southeast1` (Singapore) listed as supported, subject to actual quota/capacity. The service permits scale-to-zero. Driver libraries are provided by the platform; container CUDA user-space dependencies remain an application concern. [Official GPU service guidance](https://docs.cloud.google.com/run/docs/configuring/services/gpu).
- GPU services use **instance-based billing**, not per-request billing; GPUs bill for the instance lifecycle and idle time on a running minimum instance. The published pricing table shows an instance-based **one-minute minimum**. See [Cloud Run pricing](https://cloud.google.com/run/pricing).
- Default published instance-based rates used **only for illustrative base compute**: L4 without zonal redundancy `$0.0001867/GPU-s`, vCPU `$0.000018/vCPU-s`, memory `$0.000002/GiB-s`; at 4 vCPU + 16 GiB the sum is **$0.0002907 per billable instance-second**. With GPU zonal redundancy the analogous sum is **$0.0003949/s**. These are default list values; actual Singapore SKU, any applicable free-tier offsets, taxes, discounts, artifacts, egress, startup pull, and storage have not been verified for the project.
- Currency illustration: **1 USD = 25,611 VND**, State Bank of Vietnam reference rate reported for **2026-11-11** by [VietnamPlus/VNA](https://en.vietnamplus.vn/reference-exchange-rate-goes-up-on-november-11-post355187.vnp). This is not the user's bank settlement rate.
- Cloud Run docs cite approximately five seconds for GPU-equipped *platform instance* startup. This **does not include model loading and full application readiness**; no claim of a five-second ready scorer is made.

### Cost sensitivity — not measured

**Explicit modeling assumptions**, for illustration only: one 4 vCPU / 16 GiB / L4 non-zonal instance, default listed prices, **30 seconds total per burst for startup/model preparation** (hypothetical), **five billable seconds of warm inference per request** (hypothetical), sequential requests, immediate termination after traffic, and the one-minute instance minimum. No claims that any model actually meets these latencies.

| Scenario | Hypothetical billable time | Compute-only USD | VND using above reference | Actual task measurement |
| --- | ---: | ---: | ---: | --- |
| One short request (5 s work) | 60 s minimum | $0.017442 | ~447 ₫ | NOT MEASURED |
| One medium request (20 s assumed work) | 60 s minimum | $0.017442 | ~447 ₫ | NOT MEASURED |
| One longer request (60 s assumed work) | 90 s | $0.026163 | ~670 ₫ | NOT MEASURED |
| Ten short requests in one burst | 80 s | $0.023256 | ~596 ₫ | NOT MEASURED |
| One hundred short requests in one burst | 530 s | $0.154071 | ~3,946 ₫ | NOT MEASURED |
| One thousand short requests in one burst | 5,030 s | $1.462221 | ~37,449 ₫ | NOT MEASURED |
| One thousand requests with **one hour total source audio**, assuming 3.6 s audio/5 s compute each | 5,030 s | $1.462221 | ~37,449 ₫ | NOT MEASURED |
| Sensitivity: one thousand isolated scale-to-zero bursts each billed at least 60 s | 60,000 s | $17.442 | ~446,707 ₫ | NOT MEASURED |
| Sensitivity: instance exists for one continuous hour | 3,600 s | $1.046520 | ~26,802 ₫ | NOT MEASURED |

**Do not interpret the modeled five-second runtime or 30-second readiness as actual measured values.** One hour of learner **source audio** and one hour of GPU **billable instance time** are not equivalent. Per-request modeled unit economics vary radically with burst pattern, concurrency, idle time and instance lifetime. Additional billable components (Cloud Build, registry, data, network, logging, minimum instances) are **not included** and could increase totals. The scenario table does not qualify as a production cost forecast, measured USD/request or CPU/GPU price comparison.

For interactive requests a **Cloud Run Service** is the preferred evaluation target. A Job is not the primary interactive scoring boundary; do not add both. CPU-only is a valid candidate if Cloud Run GPU speeds up little of the end-to-end pipeline. No alternative cloud provider was investigated because Cloud Run has not been proven unsuitable.

## E. Scoring-parity / reliability status

There are **zero matched Linux CPU/GPU evaluations**, no parity tolerances agreed on actual device results, and no observed CUDA GPU utilization. Therefore **GPU SCORING PARITY: NOT VERIFIED**, rather than PASS. Existing ASR CPU int8 outputs cannot be assumed identical to CUDA float16 outputs. The comparison must freeze reference/audio, transcript, alignment, pronunciation native-scale scores, component failures/abstentions, null Pearson fields and schema; include invalid/silent/oversized/cancelled requests.

Cloud-side latency p50/p95, time to readiness, peak RAM/VRAM, error rate, bounded concurrency and removal of orphan Kaldi/FFmpeg processes are **NOT MEASURED**. No read of an existing local regression suite establishes these cloud properties.

## F. Non-billable verification executed

1. `mcp repo_observer` and fresh `repo_remote_observer`: local `main` clean and same SHA as public `origin/main` at start.
2. `python3` source and asset inspection: verified hardcoded CPU device paths, installed Mac binaries, two Python environments, model inputs and required GOPT training archive lookup.
3. `docker version --format '{{json .Server}}'`: **FAIL / daemon unavailable** (configured OrbStack socket missing); `docker buildx version` works. No daemon restart attempted.
4. `gcloud` read-only configuration, account and billing checks: project configured, account active, billing enabled. No quota/deployment/spending authorization inferred.
5. `uv lock --check --offline`: **PASS** (29 resolved packages; no lock update).
6. `.venv/bin/python -m unittest discover -s tests -q`: **PASS**, 127 tests, 1 skipped, ~10.0 s. This verifies local regression only.
7. Current Google Cloud GPU, pricing, faster-whisper and CTranslate2 documentation checked. No Cloud Run deployments, cloud builds, downloads of restricted corpora, data uploads, model modifications or billing actions performed.

**New cloud charges caused by TASK-05:** no charge-generating operations performed; account invoice not audited.  
**Resource cleanup:** no experimental cloud resources created, so nothing to remove.  
**Code or model changes:** none; local scientific results and restricted datasets unchanged.

## G. Answers and decision

1. Can the engine run on Serverless GPU? **Not yet proven**: it requires a reproducible Linux runtime with Kaldi, models, GOPT, API and permissioned deployment.
2. Does it currently use GPU? **No**: the inspected inference code specifies CPU. No GPU execution was performed.
3. Which stages remain CPU? Audio, Kaldi, Python descriptors and aggregation; GOPT currently CPU.
4. Is GPU faster than CPU end-to-end? **NOT VERIFIED**.
5. Cost per request and one hour source audio? **Not measured**; illustrative, assumption-dependent compute figures above only.
6. Is scale-to-zero beneficial? Potentially for sporadic demand, but long cold starts/idle decisions and one-minute minimum could dominate; **NOT VERIFIED**.
7. Stable bounded concurrency? **NOT VERIFIED**.
8. CPU/GPU scoring parity? **NOT VERIFIED**.
9. Ready for operational learner scoring? **NO-GO** regardless of future GPU engineering success.
10. Pearson ≥85% agreement? **NOT VERIFIED**; requires genuine Pearson-labeled held-out scoring evidence, validated metric definition and scientific evaluation. GPU acceleration cannot supply it.
11. Economically justified Cloud Run GPU path? **NOT VERIFIED**; compare measured Linux CPU vs GPU end-to-end, including full billable instance lifecycle, only after authorized tests.

### Stop and bounded continuation gate

**STOP after TASK-05.** Before any billable trial, Human Owner must explicitly authorize a **named cloud project**, a **bounded maximum spend**, deployment/benchmark scope and cleanup. Confirm quota and model/dataset rights first. Re-enable a local container engine or use an explicitly authorized build execution surface before claiming a reproducible Linux container. Do not silently move CPU baselines to GPU or upload restricted models/data. A future controlled trial must stop when its spending cap or scoring-parity gate is reached.

**Final conclusion: CLOUD EXECUTION BLOCKED — NOT VERIFIED.** Scientific scoring quality remains independently unverified for learner use.
