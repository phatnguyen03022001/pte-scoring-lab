# TASK-07.5.1 — PTE Scoring Technology Modernization & Compatibility Validation

**Executed:** 2026-11-11 (local macOS Apple Silicon, Python 3.12.13). **Scope:** production dependency compatibility, historical parity, one experimental ASR model. **Starting `main`:** `f6aada30476f30683c52680b7883105640f58707` (fresh GitHub origin/main matched; clean). **Overall verdict:** `PARTIAL_MODERNIZATION`. **Pearson:** `PEARSON_AGREEMENT_NOT_VERIFIED`.

## A. Actual technology inventory / upstream decision

| Component / type | Installed before → after; model/source pin where relevant | Current upstream or alternative | Decision and evidence |
|---|---|---|---|
| Python interpreter / uv host tool | 3.12.13; uv 0.12.5; project `>=3.11,<3.15` | uv 0.13.x available; interpreter 3.12 supported | **KEEP_COMPATIBLE**; global toolchain upgrades prohibited by workspace policy, no relevant failure |
| ASR Python package / decoder | faster-whisper **1.2.1 → 1.2.1** | 1.2.1 latest stable; PyAV 19 fix exists in unreleased upstream source | **KEEP_COMPATIBLE** released API; no unpinned Git runtime |
| CTranslate2 / native runtime | **4.8.5 → 4.9.0**; Apple Silicon native wheel, CPU INT8 / four threads | 4.9.0, 2026-11-10; optional experimental MPS backend | **UPDATE**; tested CPU; **KEEP** MPS unselected/unverified |
| PyAV / native decoding | **17.1.0 → 18.1.0** | 19.0.2 newer but removed `av.open(metadata_errors=...)` still passed by released faster-whisper | **UPDATE** to latest compatible 18.1.0; **BLOCKED_BY_COMPATIBILITY** for 19.x; explicit `av>=18.1,<19` |
| NumPy / native numerical dependency | 2.5.4 unchanged in both envs | 2.5.4 current at review | **KEEP_COMPATIBLE** |
| JiWER / token measurement | 4.0.0 | 4.0.0 current | **KEEP_COMPATIBLE** |
| ONNX Runtime / faster-whisper VAD dependency | 1.31.1; Silero VAD bundled in faster-whisper | 1.31.1 installed compatible; ONNX runtime used for VAD, not Whisper decoding | **KEEP_COMPATIBLE** |
| Hugging Face Hub, Tokenizers / ASR transitive | 2.4.0, 0.23.3 | Upstream Python dependencies | **KEEP_COMPATIBLE**; not direct dependencies, no evidenced upgrade |
| `base.en` / production ASR checkpoint | `Systran/faster-whisper-base.en@3d3d5dee26484f91867d81cb899cfcf72b96be6c`; model SHA-256 `2a166925539a16005f14ff328359f9b9adb9dc4fb631bb3b227526862e93e2ef` | Whisper large-v3, turbo, Distil, whisper.cpp, Parakeet all require new resource / validation assumptions | **KEEP_COMPATIBLE** default, frozen decode/VAD settings |
| `small.en` / experimental checkpoint | Newly downloaded locally: `Systran/faster-whisper-small.en@d1d751a5f8271d482d14ca55d9e2deeebbae577f`; model SHA-256 `62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a` | 244M-parameter Whisper English model, MIT model card | **REPLACE_CANDIDATE** research only; **NOT** a production default |
| Kaldi / native executable + acoustic model | Kaldi 5.5; extracted source without `.git`; README installation pin `e02e35f0254bb033fab73d1df99fc34123e31d56`; LibriSpeech M13 chain `tdnn_1d_sp`, raw `output` and alignment `output-xent` | Kaldi has no semver-style releases; GOP recipe cautions about chain acoustic models | **KEEP_COMPATIBLE** frozen acoustic checkpoint; exact local Git commit **UNVERIFIED**, not parent repo SHA |
| GOPT / trained pronunciation head | `YuanGongND/gopt@bed909daf8eca035095871e51642525acc5b9b55`, `best_audio_model.pth` SHA-256 `ab07451e51648f9d2455505a51055b20ac4ad7921d771ccc5170ff486a826259`, BSD-3 | New generic wav2vec2/HuBERT/WavLM encoders are **not** demonstrated pretrained scoring heads; alternative trained heads lack paired validated Vietnamese labels | **KEEP_COMPATIBLE** historical model; safer `torch.load(weights_only=True)`, strict state dict |
| Isolated GOPT runtime / Python packages | PyTorch 2.15.0, NumPy 2.5.4, kaldiio 2.18.1 | Existing supported standalone environment | **KEEP_COMPATIBLE**; no parity evidence for replacing acoustic model |
| FFmpeg/ffprobe / native audio | 8.1.2 installed host | supports format probe, resample and silence detection | **KEEP_COMPATIBLE** |
| Content / fluency | Existing deterministic token normalization/alignment, conditional uncertainty, VAD, `silencedetect`, duration/speech-rate/RMS | No demonstrated advantage for embeddings, LLM graders, new rhythm/pitch libraries | **KEEP_COMPATIBLE**; no scoring weights or threshold changes |

**Official upstream references:** [faster-whisper PyPI](https://pypi.org/project/faster-whisper/), [PyAV 19 removal of metadata_errors](https://pyav.basswood.io/docs/stable/development/changelog.html), [faster-whisper PyAV regression](https://github.com/SYSTRAN/faster-whisper/issues/1589), [CTranslate2 v4.9.0 changelog](https://github.com/OpenNMT/CTranslate2/blob/master/CHANGELOG.md), [CTranslate2 PyPI wheels](https://pypi.org/project/ctranslate2/4.9.0/), [Whisper models](https://github.com/openai/whisper), [small.en model](https://huggingface.co/Systran/faster-whisper-small.en), [Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [GOPT](https://github.com/YuanGongND/gopt), [Kaldi](https://github.com/kaldi-asr/kaldi). Releases checked on task execution date. A comprehensive authenticated CVE/security database scan was **not** performed; successful dependency resolution is not a security certification.

## B. Applied updates and compatibility decisions

- `pyproject.toml`, `uv.lock`: compatible PyAV **18.1.0** and CTranslate2 **4.9.0** only; no unrelated package upgrades; Python version policy unchanged. `uv sync --locked --no-dev` and native imports pass.
- Regression tests exercise in-memory WAV decode using faster-whisper/PyAV and CPU INT8 availability in CTranslate2.
- Fixed **false Kaldi provenance**: `git -C .local/kaldi rev-parse HEAD` previously walked up to the scoring repository and reported its commit as Kaldi's. The scorer now reports `unknown` for an extracted Kaldi tree without its own `.git`; the separately documented original install pin remains a documented claim, not a verified local checkout revision. No phoneme/scoring calculations changed.
- No direct score, normalization, VAD, GOPT checkpoint, acoustic feature, pipeline API, or scoring threshold alteration; no cloud/API access, paid services, or model binary commits. The local experimental checkpoint can be deleted independently from `.local/models/asr-small-en-d1d751a5f827`.

## C. Reproducible paired ASR comparison

Frozen TASK-07.5 manifest fingerprint: `b52cafec4cfe23e49f602a36e5d2ae2c43ab89b9fd1e40f2a9ed9ba7d8ce497d`. All 54 original recording IDs, original audio and references, four Vietnamese-L1 actual-speech `words` tiers, 48 read-prompt references, two Common Voice sentence references, token normalization, and decode settings stayed fixed. CPU INT8, four threads, beam 5, en-only, `vad_filter=True`, `condition_on_previous_text=False`; one model loaded per process, no parallel model load. Per-row private evidence: `.local/research/task-07.5.1-tech/{base,small}.json`; reference baseline: `.local/research/task-07.5-asr/results.json`.

| Group / metric | Frozen base.en / modernized base.en | Experimental small.en |
|---|---:|---:|
| Human-transcribed **actual-speech WER** (4 clips; 364 words) | 92/364 = **25.27%** | 79/364 = **21.70%** |
| Actual-speech substitutions / deletions / insertions | 45 / 41 / 6 | 36 / 39 / 4 |
| Actual-speech character error ratio | 13.91% | 12.58% |
| Actual-speech exact clips | 0/4 | 0/4 |
| Reading-prompt **disagreement, NOT WER** (48 clips; 435 words) | 109/435 = **25.06%** | 98/435 = **22.53%** |
| Reading-prompt exact / text attribution gate accepted | 11/48 | 14/48 |
| Common Voice prompt disagreement, NOT verified ASR WER | 2/19 | 1/19 |
| Decode/VAD/ASR/control failures, empty speech transcripts | 0/54 | 0/54 |
| Measured 54-clip sum of elapsed seconds | 28.66s | 83.18s (**2.90×**) |
| Cold model load time | 0.113s | 0.417s |
| Peak macOS process RSS | 599,834,624 B (~572 MiB) | 1,099,841,536 B (~1,049 MiB) |

Actual-speech per-speaker word edits: HQTV **5→6** (worse), PNV **21→21**, THV **24→14**, TLV **42→38**. On read-prompt data, 39/48 transcriptions and five individual text-gate decisions changed; **37→34** read clips had text-gate rejection. These are not adjudicated speaker errors or demonstrated full-GOPT recoveries. The small experimental corpus, only **four** actual-speech samples/four speakers and 48 unverified intended prompts, cannot support population superiority. **No production model promotion.**

Reproducibility: `.venv/bin/python task_07_5_asr.py verify`; `.venv/bin/python task_07_5_1_asr.py --model base|small` in distinct processes, **only when private outputs are absent** (runner refuses overwrite), with pinned local model assets; never regenerate historic files under original IDs. No candidate was evaluated beyond this one.

## D. Pronunciation and fluency technical decision

GOPT is **an actual trained pronunciation assessment model**, not a generic speech representation encoder. It remains uncalibrated to PTE; its fixed Vietnamese-L1 historical benchmark had 4,575/20,136 (**22.72%**) phoneme coverage and **59.08%** false positives among scored-correct target phones. Generic wav2vec2, HuBERT and WavLM checkpoints, MFA forced alignment, or modern phoneme-CTC recognizers do not by themselves supply validated quality/error scores. No credible fully licensed, pretrained compatible scorer with independent Vietnamese-labeled paired evidence was verified for safe replacement in this scope. Current GOPT uses `weights_only=True` and strict checkpoint state loading. **KEEP**, no threshold adjustment. Content lexical equality remains distinct from semantic PTE Content. Fluency pause/energy descriptions remain descriptive, not empirically graded. No scientific scoring-quality gain is asserted.

## E. Regression and numerical preservation

| Verification | Result |
|---|---|
| Manifest freeze integrity | PASS |
| Dependency resolution/sync/installed compatibility | PASS: `uv lock`, `uv sync --locked --no-dev`, `uv pip check` |
| Native import / CPU INT8 | PASS |
| WAV, MP3, M4A, **AAC codec inside .mp3** decode | **4/4 PASS**; verified actual codec instead of extension |
| Frozen modernized base versus historic 54 outputs | **54/54 identical**: transcripts, content alignments, classifications, VAD counts, text gates, segment counts, failures |
| Experimental candidate full frozen paired run | 54/54 success; details above |
| Kaldi/GOPT real audio | `SCORING_COMPLETED`, 35 phones; **two current-run repeats show max absolute score/timing difference 0.0** |
| Integrated Read Aloud end-to-end local smoke | `PARTIAL_SUCCESS`; Content, Pronunciation, Oral Fluency, Prosody, Word Stress available/appropriately partial; Pearson/overall scores null |
| Unit/regression suite after updates | **151 executed, 150 passed, 1 skipped, 0 failures** |
| Security advisories and cross-platform native parity | Full advisory audit, Linux ARM64/AMD64 execution, GPU/MPS, and clean-room external-machine installs **NOT VERIFIED** |

**Non-comparable earlier GOPT fixture:** `.local/audio-scoring/evidence-validated/result.json` was produced using **`output-xent`** GOP input, before the previously accepted QUALITY-05 correction; its phone outputs differ by up to 0.923355 from the **current** `output` branch. It is **not** a valid same-version numerical regression baseline. Current `output`-branch run repeated twice with all 35 phone scores, 35 start/end timestamps and utterance prediction exactly equal. Historical fixture was preserved, not rewritten.

## F. Production status and release/rollback

**Production:** compatible native Python dependency updates, provenance correctness fix and regression tests. **Frozen:** `Whisper base.en` default, CPU INT8/decode/VAD parameters, Kaldi/GOPT checkpoints, attribution/abstention, all scoring metrics and weights. **Experimental only:** pinned `Whisper small.en` asset and `task_07_5_1_asr.py` runner; no automatic switch. Protected historical files and licensed corpora remain ignored under `.local`.

Starting commit: `f6aada30476f30683c52680b7883105640f58707`. Ending commit is the Git commit containing this report; obtain its exact SHA with `git rev-parse HEAD` after publication (a commit cannot embed its own SHA). Rollback: use `git revert <TASK-07.5.1 commit SHA>` on `main` with owner-controlled normal Git workflow, then `uv sync --locked --no-dev`; keep historic models and scientific evidence intact. This task does not independently validate a deployment or a cloud environment.

## G. Final acceptance gates

- **Technology audit:** PASS — major direct/transitive/native/checkpoint components mapped and upstream reviewed.
- **Dependency modernization:** PASS — two justified compatible updates, locked and locally verified.
- **Model evaluation:** PASS **as paired engineering experiment only**; quality superiority **NOT VERIFIED**.
- **Production compatibility:** PASS **on the tested macOS Apple Silicon local setup**, not a release-grade multi-platform certification.
- **Scientific scoring quality:** **NOT VERIFIED** — no independent sufficiently powered expert scoring study.
- **Pearson agreement:** **PEARSON_AGREEMENT_NOT_VERIFIED**.
- **Final verdict: `PARTIAL_MODERNIZATION`** — verified dependency modernization with unchanged historical baseline, experimental ASR retained, no evidence-based pronunciation replacement.

## H. Git verification

Before change: clean `main`, fresh `origin/main=f6aada30476f30683c52680b7883105640f58707`. After task: inspect `git log -1`, `git status --porcelain`, and remote `origin/main` for the published commit and `SYNCED_CLEAN` (final result verified in executor completion response). No branch, PR, unrelated files, checkpoint binary or restricted dataset is committed.

## I. Exactly one next task — NOT EXECUTED

**TASK-07.5.2 — Independent Read-Aloud Actual-Speech Transcript Adjudication:** obtain rights-cleared, independently annotated verbatim spoken words for the **same 48 frozen reading clips**; attribute baseline-versus-small recognition differences and pronunciation text-gate refusals to recognizer errors versus real learner deviations, while retaining both models as frozen comparators. Human permissions and any cost require Owner authorization; do not promote a model without later independent holdout evidence.
