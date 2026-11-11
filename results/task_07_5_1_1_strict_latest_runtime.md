# TASK-07.5.1.1 — Strict Latest-Technology Enforcement & Runtime Modernization

**Executed:** 2026-11-12, macOS Apple Silicon (16 GiB). **Owner-policy six-month cutoff:** 2026-05-12 (inclusive). **Starting `main` HEAD:** `15d3449ffd1a665403ad2f0ba687146a5f6454bb`, identical to GitHub `origin/main`; worktree clean. **Final verdict: `BLOCKED`** — runtime modernization partly executed, but strict compliance impossible with current scientifically defensible dependencies and model evidence. **`PEARSON_AGREEMENT_NOT_VERIFIED`.**

## A. Complete active technology inventory

The project has **two actual runtime environments**, not just the `pyproject.toml` declaration: main ASR/Content/Fluency `.venv` (**25 installed distributions**) and isolated Kaldi/GOPT `.local/gopt-venv` (**12 distributions**), **33 distinct package names**. The table lists all 33, including transitive dependencies. `ASR` and `GOPT` denote environment membership; `ASR+GOPT` denotes both. **Selected release date** is the earliest official PyPI release-file upload for the installed version; age is completed calendar months at execution time. **YES** requires both *installed = latest stable* and *selected date >= cutoff*. A current release that is still too old is always **NO**. The table's `License` values are PyPI metadata, **not** independent redistribution- or model-rights clearance.

| Python package / environment | Installed | Latest stable | Installed release | Age | Compliant | Decision | License metadata |
|---|---|---|---|---|---|---|---|
| anyio (ASR) | 4.16.0 | 4.16.0 | 2026-11-01 | 0mo | YES | KEEP_CURRENT | MIT |
| av (ASR) | 18.1.0 | 19.0.2 | 2026-08-12 | 3mo | NO (not latest) | BLOCKED_COMPATIBILITY_OR_UNTESTED | BSD-3-Clause |
| click (ASR) | 8.5.1 | 8.5.1 | 2026-11-06 | 0mo | YES | KEEP_CURRENT | BSD-3-Clause |
| ctranslate2 (ASR) | 4.9.0 | 4.9.0 | 2026-11-10 | 0mo | YES | KEEP_CURRENT | MIT |
| faster-whisper (ASR) | 1.2.1 | 1.2.1 | 2025-10-31 | 12mo | NO (stale) | BLOCKED_NO_FRESH_STABLE | MIT |
| filelock (ASR+GOPT) | 4.2.10 | 4.2.10 | 2026-11-09 | 0mo | YES | KEEP_CURRENT | MIT |
| flatbuffers (ASR) | 25.12.19 | 25.12.19 | 2025-12-19 | 10mo | NO (stale) | BLOCKED_NO_FRESH_STABLE | Apache 2.0 |
| fsspec (ASR+GOPT) | 2026.9.0 | 2026.9.0 | 2026-09-18 | 1mo | YES | KEEP_CURRENT | BSD-3-Clause |
| h11 (ASR) | 0.16.0 | 0.16.0 | 2025-04-24 | 18mo | NO (stale) | BLOCKED_NO_FRESH_STABLE | MIT |
| hf-xet (ASR) | 1.8.0 | 1.8.0 | 2026-10-30 | 0mo | YES | KEEP_CURRENT | Apache-2.0 |
| httpcore2 (ASR) | 2.16.0 | 2.16.0 | 2026-10-21 | 0mo | YES | KEEP_CURRENT | BSD-3-Clause |
| httpx2 (ASR) | 2.16.0 | 2.16.0 | 2026-10-21 | 0mo | YES | KEEP_CURRENT | BSD-3-Clause |
| huggingface-hub (ASR) | 2.4.0 | 2.4.0 | 2026-10-29 | 0mo | YES | KEEP_CURRENT | Apache-2.0 |
| idna (ASR) | 3.20 | 3.20 | 2026-09-17 | 1mo | YES | KEEP_CURRENT | BSD-3-Clause |
| jinja2 (GOPT) | 3.1.6 | 3.1.6 | 2025-03-05 | 20mo | NO (stale) | BLOCKED_NO_FRESH_STABLE | UNKNOWN |
| jiwer (ASR) | 4.0.0 | 4.0.0 | 2025-06-19 | 16mo | NO (stale) | BLOCKED_NO_FRESH_STABLE | Apache-2.0 |
| kaldiio (GOPT) | 2.18.1 | 2.18.1 | 2025-03-06 | 20mo | NO (stale) | BLOCKED_NO_FRESH_STABLE | UNKNOWN |
| markupsafe (GOPT) | 3.0.4 | 3.0.4 | 2026-10-02 | 1mo | YES | KEEP_CURRENT | BSD-3-Clause |
| mpmath (GOPT) | 1.3.0 | 1.4.1 | 2023-03-07 | 44mo | NO (not latest) | BLOCKED_BY_SYMPY_CONSTRAINT | BSD-3-Clause |
| networkx (GOPT) | 3.7 | 3.7 | 2026-09-21 | 1mo | YES | KEEP_CURRENT | BSD-3-Clause |
| numpy (ASR+GOPT) | 2.5.4 | 2.5.4 | 2026-10-10 | 1mo | YES | KEEP_CURRENT | BSD-3-Clause AND 0BSD AND MI |
| onnxruntime (ASR) | 1.31.1 | 1.31.1 | 2026-10-26 | 0mo | YES | KEEP_CURRENT | MIT License |
| packaging (ASR) | 26.3 | 26.3 | 2026-08-04 | 3mo | YES | KEEP_CURRENT | Apache-2.0 OR BSD-2-Clause |
| protobuf (ASR) | 7.36.2 | 7.36.2 | 2026-09-17 | 1mo | YES | KEEP_CURRENT | 3-Clause BSD License |
| pyyaml (ASR) | 6.0.3 | 6.0.3 | 2025-09-25 | 13mo | NO (stale) | BLOCKED_NO_FRESH_STABLE | MIT |
| rapidfuzz (ASR) | 3.14.6 | 3.14.6 | 2026-08-30 | 2mo | YES | KEEP_CURRENT | MIT |
| setuptools (GOPT) | 84.2.1 | 84.2.1 | 2026-10-20 | 0mo | YES | KEEP_CURRENT | MIT |
| sympy (GOPT) | 1.14.0 | 1.14.0 | 2025-04-27 | 18mo | NO (stale) | BLOCKED_NO_FRESH_STABLE | BSD |
| tokenizers (ASR) | 0.23.3 | 0.23.3 | 2026-10-09 | 1mo | YES | KEEP_CURRENT | UNKNOWN |
| torch (GOPT) | 2.15.0 | 2.15.0 | 2026-10-28 | 0mo | YES | KEEP_CURRENT | Apache-2.0 AND Apache-2.0 WI |
| tqdm (ASR) | 4.70.1 | 4.70.1 | 2026-09-11 | 2mo | YES | KEEP_CURRENT | MPL-2.0 AND MIT |
| truststore (ASR) | 0.10.5 | 0.10.5 | 2026-10-12 | 1mo | YES | KEEP_CURRENT | MIT |
| typing-extensions (ASR+GOPT) | 4.16.0 | 4.16.0 | 2026-07-02 | 4mo | YES | KEEP_CURRENT | PSF-2.0 |

**Package status:** 10 noncompliant package names after applied compatible updates. Fresh PyPI releases do not prove compatibility by themselves; all table dates were checked against the [PyPI release API](https://docs.pypi.org/api/json/) in a dated, ignored local inventory (`.local/research/task-07.5.1.1/package_inventory.json`). Source labels are *declared/locked/installed*, not interchangeable: `pyproject.toml` declares faster-whisper/JiWER/PyAV; `uv.lock` pins 28 resolved distributions for supported Python; project `.venv` actually has 25 distributions; `requirements-gopt.txt` pins 12 additional isolated packages. Optional test code uses the same installed environments, not separate active production tooling.

### Interpreter, tooling, native binaries, models and first-party processing

| Component / class | Actually selected | Latest eligible or available | Release evidence and age | Policy | Disposition / exact provenance |
|---|---|---|---|---|---|
| CPython interpreter (both venvs) | **3.14.10** | 3.15.2 stable generally; **3.14.10 latest compatible with ONNX Runtime 1.31.1** | 3.14.10 released **2026-10-25**, ~0mo; 3.15.2 also 2026-10-25 | **YES, compatible selection** | **UPDATE** from Python 3.12.13 (2026-03-03, stale). Frozen `.python-version` and `requires-python>=3.14,<3.15`. A dry-run on 3.15.2 fails: ONNX Runtime 1.31.1 has no `cp315` wheel |
| uv project tool | **0.13.7**, installed only under `.local/uv-tool` | 0.13.7 | 2026-11-03, ~0mo | **YES for project-executed commands** | **UPDATE**, require `>=0.13.7,<0.14`. Host Homebrew uv 0.12.5 remains unchanged and is **not** a compliant command for this project |
| FFmpeg CLI and native codec libraries | **8.1.2** host; libavcodec 62.28.102, libavformat 62.12.102, libswresample 6.3.102 | **9.0.2** / libavcodec 63.1.102 | installed 2026-06-17 (~4mo), latest 2026-09-18 | **NO — not latest stable** | **BLOCKED**: host-global upgrades prohibited, project-scoped 9.0.2 build and existing Kaldi/GOPT/silence parity not established. PyAV bundles separate FFmpeg libraries; updating CLI cannot fix faster-whisper's PyAV API |
| ffprobe CLI | **8.1.2** | **9.0.2** | same FFmpeg release dates | **NO** | **BLOCKED** with FFmpeg; codec inspection remains based on stream content, not suffix |
| Kaldi source/executables / GOP & forced alignment | **Kaldi 5.5** extracted source; `.git` absent | No official GitHub stable release demonstrated | exact installed build commit/release date **UNVERIFIED**; original installation notes cite `e02e35f0254bb033fab73d1df99fc34123e31d56` but checkout fingerprint cannot prove it | **NO — UNVERIFIED** | Retain reproducible research executable hashes: `compute-gop` SHA-256 `493ef586e22c457f27bc7e9d47ed1be1741810b544b099941d616e519db368db`, `compute-mfcc-feats` `5c974511d6411d0842cc3ea30dc2cfcedb0f8eb3e6f5e01abf022fd2963c6475`, `nnet3-compute` `dc6dd7c71667b30f107904d38bbff3b2f57594198ba3e66a9e9d5dfd8b846f6a` |
| Kaldi LibriSpeech M13 `tdnn_1d_sp` acoustic checkpoint | **84,128,700 B** | No new matching, independently validated active replacement found | actual weight publication date **UNVERIFIED**; release provenance predates cutoff | **NO** | Frozen SHA-256 `6c4cae54e5d3ecfd5d61c34d5cc138733a769abd88b6969dd62dde56ca2ac955`; raw `output` supplies GOP, `output-xent` supplies alignment/safeguards |
| GOPT source / pronunciation-scoring trained head | `YuanGongND/gopt`, documented commit `bed909daf8eca035095871e51642525acc5b9b55`; local extracted source without Git history | No fresh, versioned validated replacement | source/model family ICASSP **2022**; exact checkpoint build date **UNVERIFIED** | **NO** | Frozen actual trained GOPT, BSD-3 source; checkpoint `best_audio_model.pth` SHA-256 `ab07451e51648f9d2455505a51055b20ac4ad7921d771ccc5170ff486a826259`; strict state-dict and `torch.load(weights_only=True)` remain |
| Whisper `base.en` production ASR weights | `Systran/faster-whisper-base.en@3d3d5dee26484f91867d81cb899cfcf72b96be6c` | No verified new compatible checkpoint with eligible weight date and valid scoring evidence | original Whisper weights published years before cutoff; model repo state updated 2023, **not 2026 new weights** | **NO** | SHA-256 `2a166925539a16005f14ff328359f9b9adb9dc4fb631bb3b227526862e93e2ef`. Frozen CPU INT8, four threads, beam 5, English, VAD |
| Silero VAD bundled ONNX model | `silero_vad_v6.onnx` from faster-whisper 1.2.1 | updated VAD model checkpoint with matched interfaces not verified | actual weight release date **UNVERIFIED**; bundled in 2025 faster-whisper distribution | **NO** | SHA-256 `4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2`; current VAD failure safeguards frozen |
| `small.en` experimental historical model | `Systran/faster-whisper-small.en@d1d751a5f8271d482d14ca55d9e2deeebbae577f` | only research candidate | original model years old; download date not weight-release date | **NOT ACTIVE** | SHA-256 `62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a`. Kept private for comparison, never silently promoted |
| First-party Content/Fluency & subprocess orchestration | tracked code at starting/ending repository revision | no separately released stable external product | first-party version `0.1.0` does **not** establish component release dates | **UNVERIFIED** | Deterministic normalization/alignment, ASR uncertainty/abstention, VAD, FFmpeg silence timing and NumPy RMS descriptors unchanged; no fake release date or scoring calibration |

License/release provenance: [Python release history](https://www.python.org/doc/versions/), [uv versions](https://pypi.org/project/uv/), [FFmpeg releases](https://ffmpeg.org/download.html), [Kaldi upstream releases](https://github.com/kaldi-asr/kaldi/releases), [Kaldi M13](https://kaldi-asr.org/models/m13), [GOPT README and release checkpoint](https://github.com/YuanGongND/gopt), [Whisper model repository](https://huggingface.co/Systran/faster-whisper-base.en), [faster-whisper package](https://pypi.org/project/faster-whisper/). License and commercial evaluation rights of audio corpora and derived checkpoints are **not** inferred from a Python wheel's license.

## B. Applied changes (implementation, not paper-only)

1. **Project-scoped runtime modernization:** both live `.venv` and `.local/gopt-venv` now run Python **3.14.10**. Python 3.15.2 was tested for dependency resolution only; `onnxruntime==1.31.1` lacks `cp315` wheels, so it was **not** selected. Original 3.12.13 environments are preserved at `.local/venv-py312-rollback` and `.local/gopt-venv-py312-rollback`. No model/data replacement.
2. **Latest project tooling:** installed `uv==0.13.7` under ignored `.local/uv-tool`; `pyproject.toml` enforces compatible uv version. The host-global Homebrew uv remains unchanged (0.12.5). Added `.python-version` to pin 3.14.10 and narrowed declared interpreter support.
3. **Compatible dependency updates:** main-lock transitive `filelock 4.2.9→4.2.10`; GOPT `filelock 4.2.9→4.2.10`. **Rejected** a candidate update `mpmath 1.3.0→1.4.1`: 35-phone output parity passed, but `uv pip check` found `sympy 1.14.0` requires `mpmath>=1.1,<1.4`. Restored GOPT to 1.3.0 and verified the environment resolves cleanly. `uv.lock` regenerated. Added minimal `requirements-gopt.txt` (12 explicit frozen versions) for reproducible isolated GOPT inference, independent of its outdated upstream example requirements.
4. **No unsafe substitutions:** faster-whisper, PyAV, Kaldi, GOPT, historical acoustic and ASR checkpoint, normalization, score logic, weights, abstentions and thresholds unchanged. No PyAV 19 compatibility hack or unreleased branch. No cloud or paid API, and no user-provided/restricted dataset committed.
5. **Environment containment:** Python installers and uv installed project-scoped for supported runtime. An incidental uv-managed global 3.14.10 download during dependency lock was immediately uninstalled; a second read confirmed the global Python inventory reverted to its original versions. All intended active 3.14.10 interpreters resolve inside the repository.

## C. Unresolved Owner Policy violations — mandatory release blockers

| Blocker and actual version | Why not compliant | Safe replacement / missing evidence |
|---|---|---|
| `faster-whisper 1.2.1`, **2025-10-31** | Newest stable itself >12 months old | Fresh alternatives `whisper.cpp 1.9.5` (2026-10-06), `sherpa-onnx 2.0.1` (2026-10-26) exist but require different decoding/model/VAD/word-timing semantics. No justified full-pipeline replacement backed by independent accented-English evidence. [Releases](https://github.com/ggml-org/whisper.cpp/releases) / [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases) |
| `PyAV 18.1.0`, **2026-08-12** | Newest is 19.0.2 (2026-10-21); released faster-whisper still passes removed `metadata_errors` | Must replace ASR decoder/runtime or receive stable upstream fix; upgrading to PyAV 19 immediately breaks decoding. [Upstream issue](https://github.com/SYSTRAN/faster-whisper/issues/1589), [PyAV releases](https://github.com/PyAV-Org/PyAV/releases) |
| `jiwer 4.0.0` (2025-06-19), `kaldiio 2.18.1` (2025-03-06) | Both already **newest stable** but pre-cutoff | Replace with maintained, demonstrably equivalent libraries or reduce genuine active dependency scope after numeric/audio parity; don't reinvent alignment/archives or game release labels |
| `flatbuffers 25.12.19` (2025-12-19), `h11 0.16.0` (2025-04-24), `pyyaml 6.0.3` (2025-09-25) | No latest release within six months | Transitive dependency graph replacement required (ONNX Runtime/HTTP/Whisper stack), not forced override |
| `jinja2 3.1.6` (2025-03-05), `sympy 1.14.0` (2025-04-27), `mpmath 1.3.0` (2023-03-07) | No fresh stable release. Newest mpmath 1.4.1 (2026-03-15) is **also stale** and conflicts with SymPy's declared `<1.4` constraint | PyTorch dependency graph relies on these; the 1.4.1 trial was rolled back despite score parity; must replace the incompatible dependency chain or wait for independently verified compatible stable releases |
| `FFmpeg/ffprobe 8.1.2` (2026-06-17) | Fresh enough but **not latest stable** (9.0.2) | Only a project-scoped, licensed 9.0.2 binary build plus codec/silence/Kaldi integration verification could replace host tool; host-global upgrades unauthorized |
| `Kaldi 5.5`, M13 acoustic checkpoint, GOPT source/checkpoint | Exact release provenance missing or substantially old. Kaldi has **no official tagged GitHub release**; GOPT's model family derives from 2022 research | A validated, modern *trained* phoneme scoring head with compatible phone identities and unbiased Vietnamese-L1 validation is required; forced alignment alone is not a pronunciation grader |
| Whisper `base.en` and bundled Silero VAD V6 weights | Old weights or unverified original training date; repackaging/reupload would not satisfy freshness | A genuinely newer released checkpoint with train/source provenance, rights, Apple Silicon CPU viability and independently labeled evaluation. `small.en` remains experimental and old |
| First-party Content/Fluency and deployment-native parity | No separately independently verified release/provenance for every internal subcomponent; Linux inference not run | Do not invent product release dates; verify actual distribution in a separately authorized deploy-readiness experiment |

Promising **research-only** pronunciation alternatives: [E2E-GOP on SpeechOcean762](https://huggingface.co/anonym-coling2027/E2E-GOP-SpeechOcean762) actually advertises a WavLM-based *trained phoneme regression head*, but custom model code/training-domain rights, identical target-phone scoring units and Vietnamese-L1 independent holdout remain unresolved. [OpenPronounce](https://pypi.org/project/openpronounce/) is newly packaged but independent performance and the underlying scorer's actual checkpoint/quality training are not established. A generic WavLM/HuBERT/wav2vec2 encoder, phoneme CTC system or forced aligner is not by itself a scorer. Newer ASR candidates (including NVIDIA Parakeet) have incompatible output/resource assumptions; a new runtime around **old** model weights does not make the weights fresh.

**Critical classification:** There is **no safe, currently demonstrated end-to-end fully compliant replacement**. The continuing active baseline is therefore declared **noncompliant**; not magically exempted by the experiment or the preserved historical archive.

## D. Verification and rollback readiness

- **Resolver:** `uv 0.13.7 lock --check` / `uv sync --locked --no-dev` PASS with Python 3.14.10; main 25 and GOPT 12 packages, `uv pip check` PASS separately. Project-staged 3.14.10 clean environments resolved with stable ARM64 wheels; **Python 3.15.2 failed exact resolution**: no ONNX Runtime `cp315` wheel.
- **Full regression:** **151 tests executed; 150 PASS, 1 SKIPPED, 0 FAIL** on promoted Python 3.14.10, including content alignment, VAD, Kaldi/GOPT safeguards, timeout/cleanup and failure contracts.
- **Real codec decoding:** 4/4 PASS: WAV PCM, genuine MP3 stream, M4A/AAC, and **AAC audio inside a file named .mp3**. Inspect actual streams, not extension.
- **Frozen ASR paired runtime comparison:** 54/54 successful clips; identical manifest/record IDs, speech transcripts, substitutions/deletions/insertions, token alignment, Content diagnostics, VAD intervals, segments and pronunciation text-gate statuses compared with Python 3.12.13. **0/54 output differences**. Verified human actual-speech WER unchanged **92/364 = 25.27%** (4 clips); prompt disagreement unchanged **109/435 = 25.06%** (48 clips). Existing prior research `small.en` stays **79/364 = 21.70%**, no new population claim. Main 54-clip elapsed sum **28.66s (Python 3.12)** versus **28.42s (Python 3.14)**, cold peak RSS **599,834,624 B** versus **606,453,760 B**; single sequential measured runs, not a speedup benchmark.
- **GOPT real-audio numerical parity:** one identical 35-phone WAV/reference pair run under Python 3.12.13 and 3.14.10: **35/35 exact predictions, max absolute drift 0.0, identical status, utterance predictions and warnings**. A bounded mpmath 1.4.1 trial yielded the same 35 phone scores (**0.0 drift**) but **FAILED dependency compatibility** because SymPy requires `<1.4`; GOPT was restored to 1.3.0. Score parity never overrides dependency metadata. These measurements assess numerical behavior, **not pronunciation accuracy**.
- **End-to-end Read Aloud:** after promoted environments, returned `PARTIAL_SUCCESS` with Content, Pronunciation, Oral Fluency, Prosody and Word Stress appropriately available/partial; `pearson_pte_score` and `overall_pte_score` remain `null`. No unchecked score reweighting.
- **Dependency advisories:** OSV `querybatch` checked all 33 current installed version pairs; **0 advisory matches** in returned 33/33 responses. This is not a guarantee of no vulnerability and does not certify native Kaldi/FFmpeg or model supply chain. Filelock was updated after verification; the incompatible mpmath upgrade was explicitly rejected and rolled back.
- **Linux/production deploy:** macOS ARM64 native wheels and local inference verified. Linux AMD64/ARM64 package wheel presence can be inferred from upstream distribution artifacts, **but Linux binary execution, GPU/Metal parity and cloud deployment were NOT VERIFIED**. No release-quality claims beyond local environment.
- **Historical integrity:** retained TASK-07.5 manifest `b52cafec4cfe23e49f602a36e5d2ae2c43ab89b9fd1e40f2a9ed9ba7d8ce497d` and separate TASK-07.5.1 predictions; no re-labeling or overwriting old WER/F1/FPR. Existing GOPT acoustic-feature branch (`output`/`output-xent`) unchanged.
- **Rollback:** known-good source HEAD `15d3449ffd1a665403ad2f0ba687146a5f6454bb`; retain local `.local/venv-py312-rollback` and `.local/gopt-venv-py312-rollback`. A normal `git revert` of this task's commit plus `uv sync` on the restored old Python requirement reverses source/lock changes; do not delete baseline weights or study artifacts. Exact new commit SHA and fresh origin/main synchronization must be checked **after** commit/push; report does not manufacture a self-referential SHA.

## E. Final verdict

**`BLOCKED`** — Partial compatible modernization **was implemented and verified**, but the hard freshness criterion fails for active faster-whisper, PyAV, multiple direct/transitive dependencies, FFmpeg, Kaldi, GOPT and old/unverified model weights. **`FRESHNESS_PASS: FAIL`; `COMPATIBILITY_PASS: PASS (local macOS scope)`; `REGRESSION_PASS: PASS`; `SCIENTIFIC_QUALITY_PASS: NOT VERIFIED`; `PRODUCTION_PROMOTION_PASS: FAIL`.** No speculative replacement was promoted, and no noncompliant component is misrepresented as compliant.

## F. Separate scientific verdicts

| Area | Status / boundary |
|---|---|
| ASR reliability | Narrow evidence only, 4 actual-speech human references, baseline WER 25.27%; model-quality improvement and population generalization **NOT VERIFIED** |
| Pronunciation | Historical 4-speaker Vietnamese-L1 L2-ARCTIC GOPT phone coverage **22.72%**, scored-correct false-positive rate **59.08%**; no legitimate independently validated modern replacement selected |
| Content / Fluency | Deterministic lexical diagnostics and timing/energy measurements remain uncalibrated; **semantic PTE Content / human fluency agreement NOT VERIFIED** |
| Independent ground truth | Human phone annotations exist in L2-ARCTIC but no independent sufficient Pearson score pairs or adjudicated 48-read actual-speech transcripts |
| Production promotion | **NOT APPROVED** for claims of PTE-scoring quality or modern, six-month compliant stack |
| Pearson | **`PEARSON_AGREEMENT_NOT_VERIFIED`** |

## G. Pearson status

**`PEARSON_AGREEMENT_NOT_VERIFIED`.** Version recency, 151 passing tests, WER changes and exact numerical parity do not constitute Pearson score agreement.

## H. Git publication

Starting main/origin SHA `15d3449ffd1a665403ad2f0ba687146a5f6454bb` (clean). New SHA/push/worktree must be verified by the publishing executor; commit SHAs cannot be correctly embedded into their own content. Rollback source reference is the starting SHA. No PR, no checkpoint binaries, no private datasets or unrelated files are committed.

## I. Exactly one next task (NOT EXECUTED)

**TASK-07.5.1.2 — Licensed, Independent Modern ASR and Pronunciation Replacement Qualification.** Acquire evidence on genuine post-cutoff model weights with trained pronunciation-scoring heads and sufficient independently rated Vietnamese-accented English, then run bounded same-unit comparisons before proposing any production scorer replacement. Do not execute this task without its own scope and Owner approval where legal rights/costs apply.
