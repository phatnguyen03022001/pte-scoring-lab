# TASK-07.5.2 — Independent Read-Aloud Actual-Speech Transcript Adjudication & Attribution Gate Verification

**Date:** 2026-11-12. **Repository:** phatnguyen03022001/pte-scoring-lab, main.
**Starting HEAD / rollback:** b9828764dcc161327146c57323828929af4928c5; fresh origin/main matched, worktree initially clean.
**Sole final verdict: INDEPENDENT_TRANSCRIPTS_UNAVAILABLE.**
**Production:** PRODUCTION_MODEL_UNCHANGED. **Pearson:** PEARSON_AGREEMENT_NOT_VERIFIED.

## A. Ground Truth Audit

Frozen TASK-07.5 manifest: .local/research/task-07.5-asr/manifest.json, SHA-256 fingerprint **b52cafec4cfe23e49f602a36e5d2ae2c43ab89b9fd1e40f2a9ed9ba7d8ce497d**. Existing verify_manifest checked canonical fingerprint, all original audio SHA-256 values and reference-source SHA-256 values across 54 frozen recordings. The new task independently checked 12 deterministically selected annotated recording IDs from each of HQTV, PNV, THV and TLV (48 Read Aloud records total), matching the original selection from 150 source annotations per speaker. All 48 annotations and forced TextGrids are present. No original audio or model result was regenerated or replaced.

Local L2-ARCTIC v5.0 README identifies textgrid/ as Montreal Forced Aligner output, annotation/ as manually corrected **phone** annotation, and transcript/ as the scripted orthographic source. The README describes independent, cross-checked actual-speech transcription for spontaneous Suitcase narratives, but **does not establish independently transcribed actual-spoken words for scripted reading**. The source LICENSE is CC BY-NC 4.0. Audio, local row hashes, transcripts and recording-level evidence remain ignored under .local/ for bounded noncommercial research.

| Source | Count | Classification | Eligible actual-speech ASR WER |
|---|---:|---|---:|
| Read Aloud manually corrected annotation/ TextGrid word tiers | 48/48 | HUMAN_REVIEWED_BUT_PROMPT_CONDITIONED | **0/48** |
| Read Aloud forced textgrid/ word tiers | 48/48 | Prompt-conditioned forced alignment | **0/48** |
| Read Aloud transcript/ text | 48/48 | PROMPT_ONLY | **0/48** |
| Cached ASR-generated transcripts, each model | 48/48 | MODEL_GENERATED_REFERENCE | **0/48** |
| Spontaneous Suitcase manual words | 4/4 | Previously documented independently annotated actual speech | **4/4, but 0/48 Read Aloud** |

**All 48 manually annotated word sequences match their reading prompts after canonical normalization.** Manual and forced word interval-label sequences differ in **5/48** recordings (including differences in blank intervals), but normalized lexical sequences differ in **0/48**. This cannot establish whether a learner actually omitted, substituted or repeated words. Phone-level human review is not evidence of independent orthographic transcription.

**Independent human actual-spoken-word labels accepted for the frozen Read Aloud subset: 0/48.** No blinded review record or qualifying independent adjudication chain was found. CC BY-NC 4.0 does not authorize treating corpus audio as commercial training or freely disclosing it externally. No external audio upload, payment or annotation contract occurred.

## B. Actual Speech Versus Prompt

For Read Aloud, confirmed genuine speaker deletions, substitutions, insertions, repetitions and simultaneous ASR/speaker errors are **NOT MEASURABLE (0/48 independently eligible)**. Zero confirmed cases means absence of independently eligible evidence, **not absence of actual deviations**.

A research-only three_way_diagnostic supports prompt→independent actual-speech and independent actual-speech→ASR edit events, including repeated-token ambiguity, uncertain human word spans, event indices, mixed-error recording flags and separately identified text-gate conditions. It cannot run without an explicit independently verified actual-speech premise; uncertain words exclude unqualified WER. Synthetic test cases establish software behavior, not empirical human-label validity. **No Read Aloud actual-speech oracle labels or measurements were manufactured.**

## C. Attribution of the 37 Text-Gate Refusals

| Root-cause flag | Confirmed cases | Eligible human-annotated denominator | Evidential status |
|---|---:|---:|---|
| ASR_ERROR_CONFIRMED | 0 | 0/37 | Cannot distinguish ASR errors from spoken deviations |
| SPEAKER_DEVIATION_CONFIRMED | 0 | 0/37 | No independent actual words |
| BOTH_CONFIRMED | 0 | 0/37 | Mixed causality is possible but unverified |
| ALIGNMENT_AMBIGUITY, human-adjudicated | 0 | 0/37 | No valid human alignment adjudication |
| REFERENCE_PROVENANCE_UNVERIFIED | **37/37 flagged** | Not a causal human-label denominator | All refused recordings have intended reference only |
| INSUFFICIENT_EVIDENCE | **37/37 unresolved** | **0/37** | No meaningful ASR-causality rate can be estimated |

The **observed text predicate** is verified: cached Whisper base.en reference alignment has **11/48 exact matches** and **37/48 mismatches**, matching all cached research gate statuses (11 ASR_MATCH_UNVERIFIED, 37 UNSUPPORTED_ASR_MISMATCH_OR_UNCERTAINTY). This is **not** downstream GOPT or safe Pronunciation attribution evidence.

| Speaker (12 each) | base.en refusals | base.en exact | small.en exact | Ultra exact |
|---|---:|---:|---:|---:|
| HQTV | 11 | 1 | 2 | 3 |
| PNV | 5 | 7 | 9 | 10 |
| THV | 9 | 3 | 2 | 3 |
| TLV | 12 | 0 | 1 | 1 |
| **Total** | **37/48** | **11/48** | **14/48** | **17/48** |

Text counterfactual requirements are separate: a true actual-spoken transcript can disagree with the intended prompt if the learner deviated. Replacing an ASR hypothesis with independent actual speech therefore does **not** guarantee the prompt-text gate passes. An oracle transcript compared to itself passes identity by construction, but says nothing about decoder confidence, acoustic support, forced alignment, GOPT outputs or correct learner attribution. No eligible Read Aloud actual-speech counterfactual could be executed.

## D. Three-Model Same-Unit Comparison

All outputs reuse original frozen results with matching manifest fingerprint and all 54 expected IDs. WER and CER below refer to **four spontaneous Suitcase recordings only**; the separate Read Aloud figures measure disagreement with an *intended script*.

| Metric | Whisper base.en | Whisper small.en | Parakeet Ultra |
|---|---:|---:|---:|
| Actual-speech WER, Suitcase, 364 true words / 4 recordings | **92/364 = 25.27%**, S45 D41 I6 | **79/364 = 21.70%**, S36 D39 I4 | **49/364 = 13.46%**, S37 D4 I8 |
| Actual-speech CER, Suitcase, 1,804 normalized characters | 251/1804 = 13.91% | 227/1804 = 12.58% | 121/1804 = 6.71% |
| **Read Aloud prompt disagreement (NOT actual-speech WER)**, 435 prompt words | **109/435 = 25.06%**, S90 D8 I11 | **98/435 = 22.53%**, S76 D9 I13 | **73/435 = 16.78%**, S57 D8 I8 |
| Read Aloud prompt-character disagreement (NOT actual-speech CER), 2,323 chars | 305/2323 = 13.13% | 278/2323 = 11.97% | 216/2323 = 9.30% |
| Exact Read Aloud prompt text | 11/48 | 14/48 | 17/48 |
| Frozen corpus ASR execution (previous run) | 54/54 | 54/54 | 54/54 |

Actual-speech Suitcase word edits by speaker (reference words HQTV=49, PNV=118, THV=74, TLV=123): base.en **5/49, 21/118, 24/74, 42/123**; small.en **6/49, 21/118, 14/74, 38/123**; Ultra **4/49, 5/118, 17/74, 23/123**. These are four previously exposed recordings, not a representative holdout.

Paired with base.en on 48 Read Aloud recordings, small.en produces **18** units with fewer intended-prompt edits, **9** with more and **21** unchanged; **4/37** baseline mismatches become exact and **1/11** baseline exact becomes mismatched. Ultra produces **25** units with fewer prompt edits, **5** with more and **18** unchanged; **9/37** baseline mismatches become exact but **3/11** baseline exact become mismatched. Base/Ultra have **8** mutually exact and **28** mutually mismatched cases. These outcomes can neither confirm an actual ASR correction nor prove a newly introduced recognition error: a genuine learner deviation may simply be masked by a different recognizer. Cross-model agreement and consensus are not ground truth.

No speaker- or prompt-generalization claims, narrow word-independent confidence intervals, blind-test claims or model promotion are supported. All four Vietnamese-L1 speakers and shared ARCTIC prompts are retrospective/exposed.

## E. Parakeet Confidence and Attribution Compatibility

The experimental runner called Parakeet Ultra with word-timestamp output requested. Its saved 48 Read Aloud predictions contain **one segment each**, with only text, start, end and word_count. Individual native word entries were discarded by the earlier runner. The cached model result has **no avg_logprob, no_speech_prob, compression_ratio or temperature**. The production Whisper path has native decoder-segment diagnostics, but the historical benchmark results retained only transcripts, comparison metrics and derived gate statuses, not raw decoder fields.

| Gate requirement | Existing Whisper evidence | Parakeet cached evidence | Equivalent? | Status |
|---|---|---|---|---|
| Ordered normalized transcript/reference identity | Text and deterministic token alignment | Text available | Text-only yes | Not actual-speech proof |
| Repetition/duplicate-token ambiguity | Token algorithm | Same tokenizer/algorithm can run | Conditional | Cannot prove spoken occurrence |
| Speech-region/VAD validity | Integrated VAD and no-speech handling | No documented equivalent in cached output | Not verified | Validate independently or abstain |
| Per-token decoder support | Native segment diagnostics associated with ASR tokens | Missing | **No** | Fail-closed |
| Average log probability | Native Whisper avg_logprob | Missing | **No** | No valid arbitrary replacement |
| No-speech probability | Native Whisper no_speech_prob | Missing | **No** | Timestamps are not VAD confidence |
| Compression ratio | Native Whisper compression_ratio | Missing | **No** | No equivalent |
| Temperature fallback | Whisper decoder temperature | Missing | **No** | No equivalent |
| Word/segment timestamps | Whisper segment timestamps | Segment start/end only; native per-word values not preserved | Timing only | Not acoustic correctness |
| Forced phone alignment | Kaldi full-reference alignment | Not run under Ultra pipeline | Not evaluated | Retain safety gate |
| Acoustic support | Reference-conditioned output-xent support checks | Not run under Ultra pipeline | Not evaluated | No coverage claim |
| GOPT inference/phone attribution | Existing Kaldi/GOPT conditional pipeline | No end-to-end candidate test | Not evaluated | No pronunciation claim |
| Component failure/abstention | Existing scorer error branches | Research transcription-only run | Not verified | No integration |

**Code-level caution:** the current Content path leaves recognition_support absent when a recognized token has no mapped Whisper segment. The current Pronunciation text gate rejects only recognition_support == LOW, meaning an absent support marker is not itself rejected. A synthetic empty-segment *research shell* can therefore produce a text-only ASR_MATCH_UNVERIFIED result. This is **not an observed production scoring failure**, but forbids treating a metadata-empty Parakeet adapter as proven safe. The task-specific decoder_support_compatibility check returns MISSING_DECODER_SUPPORT_FAIL_CLOSED for the real cached Ultra output; even superficial name-matching decoder fields remain UNCALIBRATED_DIAGNOSTIC_SEMANTICS_FAIL_CLOSED. No production code or gate threshold was modified.

**Parakeet result: PARAKEET_ATTRIBUTION_NOT_VERIFIED.** No documented/calibrated native equivalent, independent acoustic check or validated word-level decoder support was established in this task. Timestamps and lower prompt disagreement do not grant safe attribution.

## F. Downstream Pronunciation Findings

The pipeline stages must remain causally distinct:

- **ASR transcription:** independently measurable only against 4 spontaneous Suitcase labels, not 48 Read Aloud actual words.
- **Genuine learner departures:** not independently adjudicated for any of the 48 read-speech units.
- **Text-reference mismatch:** 37/48 observed baseline refusals; causal origin unknown.
- **Decoder-confidence gate:** Whisper diagnostics available in code; candidate equivalence missing.
- **Acoustic-support / forced-alignment failures:** no valid replacement or new oracle run. Earlier GOPT results (including 649/1000 acoustic-support refusals) have separate denominators and cannot be merged with 37/48.
- **GOPT scores, correct phone-level attribution, legitimate abstentions:** no independent Read Aloud ground truth and no complete downstream counterfactual. Perfect ASR would not resolve existing GOPT coverage, discrimination and reference-conditioned acoustic concerns.

No new score thresholds, weights, content formulas or Pearson score mapping were implemented.

## G. Implemented Changes and Local Handoff

**New tracked files only:**

1. task_07_5_2_attribution.py — frozen hash/selection verification; 48 TextGrid provenance checks; same-unit prompt comparisons; conditional three-way edit diagnostics; structurally guarded annotation submission validation; explicit Parakeet confidence fail-closed evaluation.
2. tests/test_task_07_5_2_attribution.py — 21 synthetic tests for missing labels, source identity, human-review/blinding evidence, uncertain word spans, repetitions, omissions, insertions, substitutions, simultaneous sources, invalid confidence and paired regression cases.
3. results/task_07_5_2_read_aloud_adjudication.md — this **only aggregate** public report.

**Private ignored artifacts**, never included in Git:

- .local/research/task-07.5.2/diagnostics.json — hashes, provenance, row-level gate outcomes, aggregate model comparisons.
- .local/research/task-07.5.2/handoff.json — 48 stable IDs with audio SHA-256, local-only audio references, empty actual-speech transcripts, status UNANNOTATED, audio-first/model-blinded human-adjudication rules. No ASR hypotheses exposed to prospective reviewers.

The handoff validator checks recording hashes, traceable annotator identity, independent reviewer identity when present, audio-only initial pass, model blinding, prompt-aware adjudication, rights acknowledgment, external evidence hash, transcript nonemptiness and uncertainty indexes. A single human submission returns **SCHEMA_VALID_SINGLE_ANNOTATOR_UNVERIFIED**, while two-party metadata returns **SCHEMA_VALID_INDEPENDENCE_REQUIRES_EXTERNAL_VERIFICATION**. Neither proves actual human-ground-truth acceptance; only externally verified annotation evidence can do that.

## H. Verification

- Frozen manifest canonical fingerprint, all 54 audio/reference-source hashes, 48 Read Aloud selected IDs and 150-annotation inventory per speaker: **PASS**.
- Saved base/small/Ultra result fingerprints and 54/54 unique IDs per model: **PASS**.
- 48 TextGrid annotations, 48 forced counterparts and prompt lexical checks: **PASS**.
- Newly added tests: **21/21 PASS** using unittest.
- Full regression suite: **172 run; 171 PASS, 1 SKIPPED, 0 FAIL** (fresh observed run, 7.191 seconds).
- Offline lockfile check: local uv lock --check --offline **PASS** (28 packages resolved).
- Existing production scorer files, pyproject.toml and uv.lock unchanged. Both local research JSON outputs ignored.
- No new ASR/GOPT inference, GPU use, cloud, downloads, paid APIs or outbound learner audio transfer. No new peak RAM measurement; cached historical inference timings are not rebenchmarked.
- Synthetic test success establishes mechanics only, **not** human transcript quality, attribution validity, model superiority or Pearson agreement.

## I. Final Acceptance Gates

**Final verdict (exactly one): INDEPENDENT_TRANSCRIPTS_UNAVAILABLE.**

| Gate | Result |
|---|---|
| Engineering | **PASS** — frozen integrity, provenance audit, reproducible bounded analysis, 172 tests, no production modification |
| Scientific Read Aloud human actual-speech evidence | **UNAVAILABLE (0/48)** |
| ASR causal attribution for 37 refusals | **NOT VERIFIED (0/37 eligible)** |
| Correct Pronunciation attribution | **NOT VERIFIED** |
| Parakeet native diagnostic compatibility | **PARAKEET_ATTRIBUTION_NOT_VERIFIED** |
| Production readiness | **NOT VERIFIED / PRODUCTION_MODEL_UNCHANGED** |
| Pearson agreement | **PEARSON_AGREEMENT_NOT_VERIFIED** |

This task produced a valid partial scientific investigation, not a completed human annotation experiment. No percentage of the 37 refusals can be assigned to ASR causality from the present ground truth.

## J. Git Verification and Rollback

Starting HEAD and rollback reference: **b9828764dcc161327146c57323828929af4928c5**. Target is one validated commit on main, safe push to origin/main and SYNCED_CLEAN status. The final commit SHA and observed remote/worktree state are recorded in the execution response instead of self-referencing a future commit inside this report. Restricted corpus, private transcripts, model snapshots and per-recording results remain excluded from Git.

## K. Exactly One Next Task — NOT EXECUTED

**TASK-07.5.3 — Owner-Authorized Independent Human Read-Aloud Actual-Speech Adjudication.** Obtain genuine audio-first, model-blinded, independently reviewed actual-word transcripts for **the same frozen 48 recordings**, preserving substitutions, omissions, repetitions, false starts, uncertain words and hashes. Human Owner must authorize any costs, contractual terms, licensing decisions and external disclosure before annotation. Only thereafter perform human-grounded three-way attribution and independent decoder-support qualification; do not alter production safeguards until evidence supports it. No part of this next task was executed.
