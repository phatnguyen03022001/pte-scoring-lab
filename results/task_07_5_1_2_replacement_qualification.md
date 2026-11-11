# TASK-07.5.1.2 — Independent Modern ASR & Pronunciation Replacement Qualification

**Executed:** 2026-11-12 · **Repository:** `phatnguyen03022001/pte-scoring-lab`, `main` · **Starting/rollback HEAD:** `847ae175c68d4a76655985bef9344e216b32bf3d` (fresh `origin/main` matched, clean) · **Mode:** offline, project-local, research-only; no paid services, UI, cloud or production changes. **Overall scientific promotion verdict: `BLOCKED — INSUFFICIENT_INDEPENDENT_EVIDENCE`.** **`PEARSON_AGREEMENT_NOT_VERIFIED`.**

## A. Candidate selection, original provenance and eligibility

| Technology | Pinned artifact and independent original-release evidence | Usage, rights and Owner-policy disposition |
|---|---|---|
| Production ASR **Whisper base.en**, `faster-whisper==1.2.1` | Historical model `Systran/faster-whisper-base.en@3d3d5dee26484f91867d81cb899cfcf72b96be6c`; weight SHA-256 `2a166925539a16005f14ff328359f9b9adb9dc4fb631bb3b227526862e93e2ef`; wrapper released 2025-10-31 | **Active, freshness FAIL.** Historical benchmark unchanged |
| **Parakeet Ultra**, *new ASR research candidate* | `moondream/parakeet-ultra@73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b`; first Hugging Face model repository publication **2026-09-22**, checkpoint upload 2026-09-23, **1,255,353,386 B**, model SHA-256 `c9608f36d0ab956c14bfcc525479b0746b3b42a56f6949ec85c14eb7466717dc` | Model card calls it **post-trained** from older Parakeet v3, not just quantized; CC BY 4.0. **New checkpoint publication VERIFIED; exact training run/original-weight-component dates NOT independently established.** Research CPU with `moondream==2.6.1` + Photon. Training lineage and runtime license/dependency chain still need commercial review. Model [card](https://huggingface.co/moondream/parakeet-ultra), [commit](https://huggingface.co/moondream/parakeet-ultra/commit/73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b) |
| **Parakeet Redux**, rejected without download | Newly quantized ternary version of **2025 Parakeet v3**; 178 MB advertised | Repacking/quantization of old pretrained weights is **not** fresh-weight evidence. No duplicate model download. [Model card](https://huggingface.co/moondream/parakeet-redux) |
| Production **Kaldi M13 / GOP / GOPT** | M13 acoustic SHA `6c4cae54e5d3ecfd5d61c34d5cc138733a769abd88b6969dd62dde56ca2ac955`; GOPT scorer SHA `ab07451e51648f9d2455505a51055b20ac4ad7921d771ccc5170ff486a826259` | **Active, freshness FAIL/UNVERIFIED.** Historical threshold, 2022-era trained head, phoneme inventory and alignment safeguards untouched |
| **E2E-GOP-SpeechOcean762**, *new pronunciation research candidate* | `anonym-coling2027/E2E-GOP-SpeechOcean762@2062306816c2a14022a26bc63937999160bd42d8`; HF repository created **2026-10-14**, weights **1,302,710,896 B** SHA `b3df0e271d72acb2afd49f3004b16c0f69bb3585ad79d3826a0e651f1606baa3`; custom-code SHA `b055ffa159a7b8de1688f269e2958d9c7489ff8c30a359c32979349f8063c59e` | MIT-model metadata, **real** WavLM+CTC phoneme recognition *and* trained BiLSTM/regression pronunciation-scoring head, 39 stress-free ARPAbet phones, claimed training on SpeechOcean762. Backbone based on **older** WavLM, date of training original tensors **NOT VERIFIED**, so composite stack **not freshness PASS**. Custom code was inspected for obvious unsafe imports/external execution, then loaded only from immutable local checkout with Hub/Transformers offline; not a full code/security audit. [Model card](https://huggingface.co/anonym-coling2027/E2E-GOP-SpeechOcean762) |

No client calls uploaded human audio to any vendor. Two candidate binary downloads are ignored under `.local/models/`; experiments and raw predictions are ignored under `.local/research/task-07.5.1.2/`. Historical runs, 12-clip annotation hashes, transcripts and score rows were not committed. Research environments are separately installed under `.local/research/task-07.5.1.2/{asr-venv,pronunciation-venv}`; **no active production dependency, model selector or scoring threshold changed**. Note that several *research-environment* transitive releases predate 2026-05-12; research isolation is **not** active-runtime compliance.

## B. Paired ASR benchmark — actual speech versus prompt text

Reused original frozen **TASK-07.5 manifest**, fingerprint `b52cafec4cfe23e49f602a36e5d2ae2c43ab89b9fd1e40f2a9ed9ba7d8ce497d`, all 54 audio identities; no overwriting baseline predictions. All transcriptions were compared under the existing `score_integrated.normalize_tokens`, `task_07_5_asr.comparison` and JiWER semantics.

| Metric | base.en (historical) | small.en (historical) | **Parakeet Ultra (new)** |
|---|---:|---:|---:|
| **Human-verified actual-speech WER**, only 4 clips, 364 reference words, 4 Vietnamese-L1 speakers | 92/364 = **25.27%** (S45 D41 I6) | 79/364 = **21.70%** (S36 D39 I4) | **49/364 = 13.46%** (S37 D4 I8) |
| **Actual-speech CER**, independent per-utterance character edits summed / 1,804 normalized reference characters | 251/1804 = **13.91%** | 227/1804 = **12.58%** | **121/1804 = 6.71%** |
| **Read Aloud prompt disagreement**, 48 intended texts / 435 words (**NOT human spoken-word WER**) | 109/435 = **25.06%** | 98/435 = **22.53%** | **73/435 = 16.78%** (S57 D8 I8) |
| Exact Read Aloud prompt matches, only a *necessary* condition for attribution | 11/48 | 14/48 | **17/48** |
| Decode success / errors on fixed corpus | 54/54 / 0 | 54/54 / 0 | **54/54 / 0** |
| CPU inference elapsed, sum of 54 clips | 28.66s | 83.18s | **15.04s** |
| Cold load, one process | 0.113s | 0.417s | **3.849s** |
| Peak macOS process RSS | 599,834,624 B | 1,099,841,536 B | **4,842,045,440 B** |

Actual-speech word edits by speaker: `HQTV 5/6/4`, `PNV 21/21/5`, `THV 24/14/17`, `TLV 42/38/23` for base/small/Ultra respectively. The new run used **CPU, 3 threads** whereas the historical runner used its prior CPU settings; elapsed times reflect those configurations and separate invocations, **not** an equal-resource throughput claim. Ultra consumes roughly 8.1× the base model's observed RSS. No GPU, MPS or Linux inference was evaluated.

**Native integration guard:** Ultra supplies word/segment timestamps but **does not expose matching Whisper native `avg_logprob`, `no_speech_prob` and `compression_ratio` per-segment diagnostics**. Existing Content's decoder-support markers and `gate_pronunciation_attribution` cannot be assumed to mean the same thing. **17/48 text matches are NOT 17 verified downstream gate passes.** End-to-end GOPT score/abstention counterfactuals under Ultra were **NOT VERIFIED**. Production selection is unchanged.

### ASR failure and format checks

Single-model offline bounded runs: MP3, M4A/AAC, and AAC content in a file incorrectly named `.mp3` all produced the same transcript on the same source speech; WAV had passed the real ASR corpus. 2s silence and a 0.125s audio clip returned empty transcripts (not hallucinations). Invalid audio returned `ValueError`, not a crash. Format/negative smoke tests **PASS** on the candidate; they are **not** a complete error/timeout/stability benchmark.

## C. Paired pronunciation evidence — actual human phoneme labels

**Design:** exactly **12 deterministic L2-ARCTIC v5.0** read utterances: four Vietnamese-L1 speakers (HQTV, PNV, THV, TLV), three consecutive official IDs per speaker (`arctic_a0052`–`arctic_a0054`). These are **retrospective research recordings**, not unseen independent speaker holdout; the same corpus has already been used elsewhere in this repository. L2-ARCTIC license is **CC BY-NC 4.0**, so these data and restricted rows stay under `.local` and do not authorize commercial reuse.

Target **input** phoneme strings were generated only from each official *prompt transcript* and the existing fixed Kaldi LibriSpeech lexicon's first variant. **Evaluation** labels were separately loaded from the official human-annotated TextGrid **`phones` tier**, never from forced alignments. Stress and documented accent markers normalized; phone identity matched through the existing `quality_08_benchmark.edit_align`. No positional score shifts, synthetic human labels or threshold fitting. Candidate scores outside identity matches **abstained**.

| Metric | Frozen Kaldi/GOPT baseline on these 12 | **E2E-GOP candidate on these 12** |
|---|---:|---:|
| Recordings with successful inference | **2/12** (10 `ALIGNMENT_INVALID`) | **12/12** |
| Eligible human target phones | 320 | 320 |
| Phones scored on **exact identity match** | **56/320 = 17.50%** | **309/320 = 96.56%** |
| Error-labelled / correct amongst candidate-scored phones | Not recomputed as a new full experiment | 75 error / 234 correct |
| ROC-AUC for **correct phone as higher score**, separately on each scorer's scored subset | **0.837** (56 phones) | **0.726** (309 phones) |
| **Paired ROC-AUC on the identical 56 phones scored by both** | **0.837** | **0.698** |
| Local inference total (12 files), excluding model initialization | Historical cached per-run timings, not comparable to this process | **3.61s** |
| Peak process RSS | No new comparable GOPT measurement | **2,078,867,456 B** |

**Interpretation:** E2E-GOP dramatically increases *diagnostic phone coverage* in this small reference-conditioned test, but discriminates correct versus error phones **worse than GOPT on the 56 common units**. The broad 309-versus-56 ROC-AUC comparison uses different denominators and **cannot** establish which full-pipeline scorer is better. There is no validated E2E-GOP operating threshold, no preregistered calibration/false-positive tradeoff, no sentence-/Pearson-scoring mapping, no blindly held-out Vietnamese speakers and no demonstrated production-level attribution safeguards. Its `0..2` regression scores are **not** Pearson scores or probabilities. Model card self-reports trained scoring heads, but its fresh *repository date* does not prove all component weights satisfy six-month freshness.

## D. Compatibility, safety, scientific decisions

- **Local resolver:** isolated Ultra environment **35 installed packages compatible**, isolated E2E-GOP environment **42 installed packages compatible** (`uv pip check` PASS). Candidate runtimes used **Python 3.14.10**, `moondream==2.6.1`, `torch==2.15.0`, `transformers==5.22.2`, `safetensors==0.8.0`. Both candidate load paths pinned to local model directories; Hub and Transformers offline during model inference.
- **Production regression (unchanged):** `uv lock --check` PASS; **151 tests, 150 PASS, 1 SKIPPED, 0 FAIL**; TASK-07.5 frozen manifest integrity PASS; previous baseline predictions and model hashes untouched. No unexplained production drift was introduced because no production code, policy, GOPT or ASR runtime was modified.
- **Production release gate:** **FAIL**. Ultra's recognizer makes materially different transcriptions and native uncertainty features. E2E-GOP's phone coverage improvement is not sufficient evidence of correct attribution or calibrated error detection. Multiple production dependencies and historical models still violate the Owner's **2026-05-12 freshness cutoff**. **Neither candidate promoted**.
- **Security/rights:** Model cards publicly state CC BY 4.0 (Ultra) and MIT (E2E-GOP); review of dynamic model code was bounded/static, not a supply-chain audit. Exact training dataset/rights, inherited WavLM checkpoint provenance and commercial deployment terms are not fully verified. No trust in unpinned remote branches, paid API calls, audio uploads or legal agreements.
- **Scientific verdicts:** ASR has an **encouraging but four-recording, four-speaker** actual-speech accuracy signal. Pronunciation has a genuine trained alternative and higher observed small-sample coverage, **but insufficient controlled discriminatory quality**. Content, Fluency, Prosody and Pearson agreement remain **NOT VERIFIED**. **`PEARSON_AGREEMENT_NOT_VERIFIED`.**

## E. Reproducibility and publication

- **Private local evidence:** `.local/research/task-07.5.1.2/{parakeet-ultra.json,ultra-metrics.json,ultra-edge-cases.json,e2egop-inputs.json,e2egop-predictions.json,e2egop-metrics.json}` plus local-only benchmark scripts, exact model revisions and SHA-256 digests above. No historical reports or cached predictions were overwritten.
- **Model snapshots:** `.local/models/asr-parakeet-ultra-73175eb`, `.local/models/pronunciation-e2egop-2062306`; no binary committed. Storage is project-scoped. Current research scripts are local only; reproducing elsewhere requires authorized copies of the restricted corpus and separately installed research dependencies.
- **Source rollback:** `847ae175c68d4a76655985bef9344e216b32bf3d`. Production source untouched. This task publishes **only this aggregate Markdown report**; raw learner audio, row-level labels, model weights and exploratory scripts stay ignored. At publication, verify new local HEAD equals GitHub `origin/main` and worktree clean. Exact new commit SHA is intentionally not self-referentially embedded in its own contents.

## F. Exactly one next action — NOT EXECUTED

**TASK-07.5.2 — Independent Read-Aloud Actual-Speech Transcript Adjudication & Attribution Gate Verification.** Obtain independently transcribed *actual* speech for the 48 frozen read-aloud recordings and define an auditable mapping of Ultra native diagnostics onto existing fail-closed Content/Pronunciation attribution requirements; then re-evaluate on speaker-disjoint human labels before any change in production model selection. This remains separate from Pearson calibration.
