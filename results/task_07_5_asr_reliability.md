# TASK-07.5 — Vietnamese-English ASR Reliability and Scoring Error Propagation

**Date:** 2026-11-11. **Scope:** scoring engine only; local, retrospective, noncommercial research.  
**Starting `main` HEAD:** `92eeecbc60e9548c00b58f928b1486c4822347d8`; matched fresh `origin/main`, clean.  
**Engineering decision:** **INSUFFICIENT_EVIDENCE**. **Pearson:** **PEARSON_AGREEMENT_NOT_VERIFIED**.

## A. Verdicts

| Gate | Verdict |
|---|---|
| Engineering | **PASS** — frozen auditable protocol, 54/54 controlled ASR cases, existing alignment/gate tracing, explicit reference provenance, regression tests. No production scoring modification. |
| ASR ground truth | **NARROW VERIFIED ACTUAL-SPEECH EVIDENCE** — four L2-ARCTIC spontaneous recordings have manually annotated word tiers; 364 human-reference words across four Vietnamese-L1 speakers. Non-blind, retrospective. |
| ASR reliability | **MEASURED, NOT GENERALIZABLE** — actual-speech WER 92/364 = **25.27%**. Not a population claim. |
| Downstream causal attribution | **SCORING_PROPAGATION_NOT_VERIFIED** end-to-end — text-level Content and Pronunciation attribution gate measured, not GOPT/acoustic score counterfactuals. |
| Scientific improvement | **NOT DEMONSTRATED / NO_CHANGE_REQUIRED**. No supported model, VAD, threshold or normalization production change. |
| Pearson validation | **PEARSON_AGREEMENT_NOT_VERIFIED** — no independent audio/numeric Pearson ground truth. |

## B. Frozen data and reference provenance

Frozen before ASR execution at `.local/research/task-07.5-asr/manifest.json`; manifest SHA-256 fingerprint:
`b52cafec4cfe23e49f602a36e5d2ae2c43ab89b9fd1e40f2a9ed9ba7d8ce497d`.
Audio and human-source SHA-256 values, local IDs, exact transcripts and per-unit predictions remain under ignored `.local/`. The 48 Read Aloud recordings were deterministically selected by lowest SHA-256 on `TASK-07.5|speaker|record_id`, 12 of 150 annotated records per speaker. Protocol and metrics are retrospective: all four Vietnamese speakers were exposed in previous TASK-07 work. The shared CMU ARCTIC prompt inventory means speaker separation does not imply prompt separation.

| Evaluated corpus | Audio / speakers | Confirmed Vietnamese L1 | Human actual spoken transcript | Intended-only | License / limitations |
|---|---:|---:|---:|---:|---|
| L2-ARCTIC v5.0 Suitcase, spontaneous narration | 4 / 4 | 4 | **4**, manual `words` TextGrid tier | 0 | CC BY-NC 4.0 research; no intended read prompt |
| L2-ARCTIC v5.0 annotated reading | 48 / 4 | 4 | **0** independently verified actual orthographic transcripts | **48** | CC BY-NC 4.0; previously exposed; prompts shared |
| Common Voice English v17, self-reported Vietnamese accent | 2 / **1** anonymous speaker | **0 L1 verified** | 0 | **2** | CC0; self-reported accent ≠ independently verified L1; tiny sample |

L2-ARCTIC README documents manual word/phone tiers and independent review for the spontaneous suitcase recording annotations. The **manual words tier** is the actual-speech reference. The release's second orthographic transcript is not silently substituted for this tier. Its differences from the manual words tier total **10 word edit units** across four recordings (HQTV 2, PNV 1, THV 4, TLV 3). Do not use source prompt transcripts, forced alignment outputs or a Whisper transcript as actual-speech truth.

Available but *not evaluated*: GMU Speech Accent Archive (40 Vietnamese L1 recordings, only 11 IPA phone-transcription files; no established human orthographic actual transcript; CC BY-NC-SA 4.0); SANDI Cambridge dev subset (Vietnamese-L1 linkage and metric-publication rights unverified for this task); SpeechOcean762 (Mandarin and GOPT-exposed); EpaDB (Spanish and threshold-exposed); Pearson sources (108 audio instances, no verified numerical audio-score pairs, inference rights unresolved). No corpus reacquisition, paid API, deployment or Pearson inference.

## C. Frozen ASR and runtime

Production paths reviewed: `score.py`, `score_integrated.py`, `score_pronunciation.py`, `evaluate_quality06_asr.py`, `l2arctic_benchmark.py`, reports TASK-06 through TASK-07.4.

- ASR: `faster-whisper 1.2.1`, `Systran/faster-whisper-base.en` local cache revision `3d3d5dee26484f91867d81cb899cfcf72b96be6c`; no changed weights or model downloads.
- CPU INT8; 4 threads; 1 worker; `local_files_only=True`; `language="en"`; beam 5; `vad_filter=True`; `condition_on_previous_text=False`; library-default temperature fallback.
- Decode to 16 kHz through existing faster-whisper library, use `score_integrated._vad_intervals`, `normalize_tokens`, `_build_content`, `classify_alignment` and `gate_pronunciation_attribution`. One reused model, one recording at a time.
- Host: Apple Silicon arm64 / M3 / 16 GiB memory, macOS. Local cold model load **0.281 seconds**, peak macOS process RSS **601,849,856 bytes** (~574 MiB). No cloud GPU claims.
- **54/54 success**; **0/54** decode/VAD/ASR/Content diagnostic failures, empty ASR transcripts or no-speech VAD cases. Silence-only hallucination rate was **not measured**, because no silence-only samples were selected. Stage-specific future failures are retained by the research runner; conservative gate refusals are not ASR execution failures.
- Median per-recording decode + VAD + warm ASR + alignment: Suitcase **1.8975s / 4**, L2 Read Aloud **0.4122s / 48**, Common Voice **0.4802s / 2**; total measured work **28.196s** excluding one-time initialization. These are per-recording observations on different durations, not comparable normalized throughput. Per-record CPU time recorded privately.

## D. Recognition / prompt-disagreement results

Levenshtein S/D/I counts use existing normalized word tokens; no invented meaning for omitted words. CER uses space-joined normalized characters. Counts below are **not pooled across different reference types**.

| Reference category | Clips | Reference words | Matched words | S / D / I | Total word edits | Measured ratio | Exact matches | CER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Manual actual speech: legitimate ASR WER** | 4 | **364** | 278 | 45 / 41 / 6 | **92** | **25.27% WER** | 0/4 | 13.91% |
| Reading prompt: **NOT ASR WER** | 48 | 435 | 337 | 90 / 8 / 11 | 109 | **25.06% reference disagreement** | 11/48 | 13.13% |
| Common Voice sentence: **NOT verified actual speech** | 2 | 19 | 18 | 1 / 0 / 1 | 2 | **10.53% reference disagreement** | 1/2 | 3.64% |

Human actual-speech per speaker (one utterance each): **HQTV 5/49 (10.20%), PNV 21/118 (17.80%), THV 24/74 (32.43%), TLV 42/123 (34.15%)**. Four speaker clusters are too few for a useful population-level 95% CI; no spurious precision or representative population claim is made. Speaker, fluency, utterance length and recording type remain confounded.

Reading prompt per-speaker mismatch: HQTV **32/100 words**, PNV **7/110**, THV **27/110**, TLV **43/115**; no attribution to ASR vs actual spoken deviations without human actual transcriptions. Two Common Voice examples are **one speaker**, not two independent speakers.

## E. Downstream propagation and uncertainty

Inspected path: **audio → decode → VAD → ASR → normalized words → reference alignment → Content token evidence → pronunciation attribution text gate → conditional acoustic/GOPT (not rerun).**

| Outcome | Count / denominator | Scientific meaning |
|---|---:|---|
| ASR lexical discrepancies against manual actual speech | **92 edits / 364 words** | 45 S, 41 D, 6 I attributable to ASR *relative to the human word tier*. Corpus reference error is possible; manual annotation is not infallible. |
| Actual-speech clips failing exact ASR-to-human transcript | **4/4** | Text-gate diagnostic only. These are spontaneous narratives, **not** Read Aloud tasks with a required text. |
| Verified genuine speaker departures from a **reading prompt** in the spontaneous set | **Not applicable (0 prompts / 4 clips)** | Cannot invent genuine prompt deviations from prompt-free speech. |
| Reading prompt mismatches | **109 edits / 435 prompt words; 37/48 clips** | Causal allocation between ASR error and genuine learner deviation **unverified** (0/48 human-verified actual word transcripts). |
| Reading exact-text Pronunciation attribution gate | **11/48 pass, 37/48 withhold** | Necessary gate condition, **not actual GOPT pass/fail or full pronunciation coverage**. |
| Common Voice exact-text gate | **1/2 pass, 1/2 withhold** | Single speaker, uncertain actual adherence. |
| ASR runtime or VAD failure | **0/54** | No observed execution error; does not prove future reliability. |
| Content alignment component returned ASR-unverified output | **54/54** | No semantic/Pearson Content ground truth; lexical equality is not DI/RL/SGD/RTS correctness. |
| Acoustically confirmed Pronunciation causal error from ASR | **0 validated full-pipeline comparisons** | **SCORING_PROPAGATION_NOT_VERIFIED**. |

Existing Content classifications on four manual actual references: **278 MATCH, 45 SUBSTITUTION, 41 OMISSION, 5 INSERTION, 1 UNCERTAIN** (last one is a downgraded potential insertion). `_build_content` explicitly uses `UNCERTAIN_ASR_ONLY` rather than accusing the speaker of omission or substitution.

`gate_pronunciation_attribution` requires a complete ordered ASR/reference match, no low decoder-support tokens, no ambiguous occurrences, and available reference-conditioned pronunciation diagnostics. The task exercised the **real gate function on an empty diagnostic shell** — no GOPT scores were simulated. The previous TASK-07.1 benchmark reported 649/1,000 recordings failing **acoustic alignment support**, not ASR; those must remain a separate denominator and failure class.

## F. Controlled counterfactuals

**A — Actual speech vs observed ASR:** Same four audio files, manual word-tier references, recognizer output, text normalization and IDs. Observed **92/364** word edits, **0/4** exact-text-gate passes. Replacing only the *diagnostic transcript* with the manual actual-spoken transcript yields zero textual edits and **4/4** text-gate passes **by construction**. This is a bounded attribution experiment, **not a feasible production ASR system, verified GOPT recovery or scored pronunciation improvement**.

**B — Transcript provenance sensitivity:** Same audio and ASR, but using the corpus-provided orthographic transcripts instead of the human word tiers yields **91/363 (25.07%)** reference disagreement instead of **92/364 (25.27%)**. Ten manual/source word edits across four clips explain why label provenance is part of the protocol. This is a comparison of reference sources, not an ASR intervention.

**C — Normalization/attribution safeguards:** Synthetic unit tests exercise punctuation/case, Unicode apostrophes, small numbers, duplicate tokens, missing actual transcript, no-VAD uncertainty, reference mismatch and exact-match gate. Synthetic cases validate implementation invariants, **not model recognition accuracy**. No evidence of a normalization regression justifies a production patch. No VAD, model, score scale or threshold changes were made. All comparison units, reference types, abstentions and evaluation split exposure are retained.

Private per-clip details and both counterfactual outcomes: `.local/research/task-07.5-asr/` (ignored, not committed).

## G. Root causes and minimal production decision

| Hypothesis | Classification | Evidence | Decision |
|---|---|---|---|
| Real ASR lexical errors on Vietnamese-accented spontaneous speech | **CONFIRMED (narrow)** | 92/364 edits across 4/4 Vietnamese-L1 spontaneous narratives | Record as an ASR reliability risk, not a population estimate |
| Every reading prompt difference is an ASR failure | **UNVERIFIED** | 109/435 prompt edits; 0/48 actual-spoken word references | Do not assign causal blame |
| Exact-match gate conservatively withholds on transcript/prompt differences | **CONFIRMED (predicate)** | 37/48 read clips | Retain safeguard without external actual-speech validation |
| Accent/domain, word-final consonant, short speech, VAD or temperature as specific causes | **UNVERIFIED** | No validated same-unit intervention isolating the factor | Do not guess or tune |
| Current normalization is broken | **NOT DEMONSTRATED** | Invariant tests pass; alternate references change labels, not ASR | Keep production normalization |
| ASR chiefly caused prior 22.72% Vietnamese GOPT phone coverage | **NOT VERIFIED** | Acoustic support was an independently measured major blocker in TASK-07.1 | Keep ASR and acoustic failure accounting separate |
| Pearson-level Content/Pronunciation/Speaking agreement | **NOT VERIFIED** | 0 independent numeric Pearson pairs | No equivalence claim |

**Production changes: none.** Added only `task_07_5_asr.py`, `tests/test_task_07_5_asr.py` and this evidence report. No new dependency, API, UI, infrastructure, model download, scorer settings or remote speech processing. Runner now records failure stage specifically; no production logic changed.

## H. Final engineering decision

**INSUFFICIENT_EVIDENCE.**

Measured ASR errors are real in a narrow human-annotated Vietnamese-English corpus. However, the Read Aloud gate-causality share and full GOPT consequences remain unverified. No controlled comparison has shown a justified smaller correction or an unavoidable need to replace ASR; scientific superiority and Pearson agreement remain unproved.

## I. Next smallest task (not executed)

Obtain independent human word-for-word **actual spoken** annotations for the **same frozen 48 read-speech recordings** (including learner omissions, substitutions and repetitions). Compare each actual transcript simultaneously to the intended prompt, the already recorded ASR output and the existing text-gate decision. This is the smallest direct way to adjudicate the **37/48** gate refusals and would remain a retrospective study, not independent population validation. Rights, human-review cost and authorization are for the Human Owner; **none was purchased or initiated**.

## J. Verification, reproduction, repository state

- One-time freeze: `.venv/bin/python task_07_5_asr.py freeze` — refuses to overwrite an existing manifest.
- Verify immutable source fingerprints: `.venv/bin/python task_07_5_asr.py verify`.
- One-time local inference: `.venv/bin/python task_07_5_asr.py evaluate` — refuses overwriting existing private per-record predictions.
- Regression: `.venv/bin/python -m unittest discover -s tests -q`.
- The manifest, full per-recording results, aggregate JSON and counterfactual JSON are stored at ignored `.local/research/task-07.5-asr/`. They cannot be recreated without the appropriately licensed source datasets and locally cached model snapshot.
- Verified regression observation during the task: **148 tests run, 147 pass, 1 skipped, 0 failures**; 11 task-specific tests pass. The manifest SHA and 54/54 inference success were checked. Final code verification and push state must be checked separately before declaring repository completion.
- Starting Git state: clean `main`, local and remote both `92eeecbc60e9548c00b58f928b1486c4822347d8`. Ending commit SHA/push and clean-worktree status are established by the subsequent publication verification, not anticipated.

## K. Pearson

**PEARSON_AGREEMENT_NOT_VERIFIED**. ASR accuracy, human phone annotations and test success cannot establish 80–90% Pearson agreement.
