# TASK-07.5.3 — Independent Human Read-Aloud Actual-Speech Ground Truth Acquisition & Validation

**Date:** 2026-11-12  
**Repository:** `phatnguyen03022001/pte-scoring-lab` · `main`  
**Starting HEAD / rollback:** `e4bf3c50522d6058164de824cec9b3dcb6a604de` (matched fresh `origin/main`; initially clean)  
**Final scientific verdict: INDEPENDENT_TRANSCRIPTS_UNAVAILABLE**  
**Production:** `PRODUCTION_MODEL_UNCHANGED` · **Pearson:** `PEARSON_AGREEMENT_NOT_VERIFIED`

## A. Acquisition, dataset identity and rights

The existing `.local/research/task-07.5-asr/manifest.json` has frozen fingerprint
`b52cafec4cfe23e49f602a36e5d2ae2c43ab89b9fd1e40f2a9ed9ba7d8ce497d`.
`read_manifest()` reverified canonical manifest identity, 54/54 source-audio SHA-256 hashes,
54/54 reference-source SHA-256 hashes, and the deterministic 48-read selection (12 each
from HQTV, PNV, THV and TLV; 150 annotated source files inventoried per speaker).
The full manifest also contains four spontaneous Suitcase and two Common Voice records;
neither is part of the frozen 48 Read Aloud human-ground-truth denominator.
All three existing cached recognizer results matched the manifest fingerprint with 54/54
unique successful recording IDs. No source or prediction was altered.

| Exact frozen Read Aloud evidence | Count | Scientific status |
|---|---:|---|
| Source audio and reference identities validated | 48/48 | Verified source integrity, **not** human ground truth |
| Manually corrected phone-annotation TextGrids | 48/48 | Word tiers agree with intended prompt; prompt-conditioned for actual-word purposes |
| Forced-aligned TextGrids | 48/48 | Not independent actual-speech word transcription |
| Unannotated records in existing handoff | 48/48 | Awaiting authorized human transcription |
| Submitted independently produced Read Aloud human transcripts discovered | **0/48** | None |
| Human actual-speech transcripts accepted | **0/48** | None |
| Rejected submitted human transcripts | **0** | No submissions exist to reject |
| Missing human actual-speech transcripts | **48/48** | Scientific blocker |

The previous 48 manual word tiers match the intended reading scripts after normalization.
Five word interval-label sequences differ from corresponding forced alignments, but zero
normalized lexical sequences differ. Manual **phone** annotation cannot be promoted into
independent *spoken-word* truth. The earlier independently annotated four Suitcase narratives
remain separate retrospective evidence; they are not eligible substitutes for the exact 48.

Rights: `.local/datasets/l2-arctic-v5.0/extracted/LICENSE` and the upstream PSI Lab
[L2-ARCTIC site](https://psi.engr.tamu.edu/l2-arctic-corpus/) identify **CC BY-NC 4.0**.
Noncommercial research use with the license's attribution and other conditions is
distinct from commercial deployment. The license does not itself provide a new
Owner approval for external audio disclosure, paid annotation, vendor terms or
commercial reuse. The official [corpus documentation](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/)
describes orthographic transcripts, forced-aligned `textgrid/`, manually reviewed
`annotation/` and the separate independently reviewed spontaneous Suitcase subset.
A targeted public-source review did not establish human actual-speech transcriptions
with independent provenance for the **exact 48** read recordings. This is not proof that
no such annotations exist anywhere.

No external audio disclosure, third-party service, payment, contract, new model download
or unapproved license decision occurred. Prior/ongoing Owner authorization for any of
those acts was not evidenced in this task.

## B. Handoff and provenance quality

**HANDOFF_NO_CHANGE_REQUIRED.** The preexisting ignored
`.local/research/task-07.5.2/handoff.json` was checked without rewriting it.
Its file SHA-256 is `4aba156d15fd661995faeec2988b4635c3fa73a4ef1bb769712e0229e6361ee8`.
All 48 IDs are unique and exactly match the frozen selection; all 48 audio hashes and
local paths match the manifest. All 48 entries are `UNANNOTATED` with null actual-speech
transcripts, null provenance and no invented uncertainty indices. The handoff contains
neither ASR hypotheses nor prompts; prompts are available separately in the private
manifest for **after** a locked audio-only first pass. No second handoff framework or UI
was created.

Missing for **each** record: actual audio-first lexical transcript, auditably independent
human origin, evidence of annotation pass and prompt-aware revisions, source-evidence
hash tied to a real artifact, external provenance verification, optional independent
review/adjudication evidence, and supported uncertain-span markings. False starts,
repetitions and actual substitutions/omissions cannot be recovered from prompt text.

`validate_handoff_submission()` verifies structural recording IDs/hashes, declared
blinding, human-review metadata and uncertainty-index shape. Critically it returns
`SCHEMA_VALID_INDEPENDENCE_REQUIRES_EXTERNAL_VERIFICATION` even for well-formed
two-annotator claims; a self-declared `annotator_id`, `reviewer_id` or evidence hash
is **not proof of actual human work**. No external provenance record was available to
exercise genuine acceptance. Initial human blind passes: 0; reviewed passes: 0;
adjudicated pairs: 0. Inter-annotator agreement and ambiguous spoken words: **N/A**.

## C. ASR evidence, with correct denominators

| Metric for 48 Read Aloud units | Whisper base.en | Whisper small.en | Parakeet Ultra |
|---|---:|---:|---:|
| Eligible independently verified actual-speech recordings | 0/48 | 0/48 | 0/48 |
| Actual-speech WER/CER | **N/A** | **N/A** | **N/A** |
| Actual-speech S/D/I, exact matches, per-speaker error rates | **N/A** | **N/A** | **N/A** |
| Historic prompt disagreement (not actual-speech WER), 435 prompt words | 109/435 (25.06%) | 98/435 (22.53%) | 73/435 (16.78%) |
| Historic exact intended-prompt matches | 11/48 | 14/48 | 17/48 |

The historically exposed **four Suitcase narratives** have separate actual-speech
word-edit counts of base **92/364 (25.27%)**, small **79/364 (21.70%)**, Ultra
**49/364 (13.46%)** and character-edit counts of 251/1804, 227/1804 and 121/1804
respectively. These numbers are unchanged historical controls, **not Read Aloud WER**,
not an independent evaluation of the 48 or evidence of Pearson agreement.
No actual-speech model improvement/regression, CER, lexical error taxonomy or
agreement statistics can be newly calculated on the read subset. Cached paired
intended-prompt comparisons were reconciled against TASK-07.5.2 diagnostics.
Model agreement cannot resolve the absent human reference.

## D. The 37 baseline text-gate refusals

All 37 cases are exactly the frozen Whisper base.en
`UNSUPPORTED_ASR_MISMATCH_OR_UNCERTAINTY` set; 11/48 have
`ASR_MATCH_UNVERIFIED`. Those are **text predicate states**, not confirmed
Pronunciation or GOPT failures.

| Cause / state | Confirmed cases | Eligible human denominator | Evidence |
|---|---:|---:|---|
| `ASR_ERROR_CONFIRMED` | 0 | 0/37 | No eligible actual speech |
| `SPEAKER_DEVIATION_CONFIRMED` | 0 | 0/37 | No eligible actual speech |
| `BOTH_CONFIRMED` | 0 | 0/37 | Co-occurrence cannot be adjudicated |
| `ALIGNMENT_AMBIGUITY` (human adjudicated) | 0 | 0/37 | No human word-level adjudication |
| `REFERENCE_ERROR_CONFIRMED` | 0 | 0/37 | No independent contradicting reference |
| `INSUFFICIENT_EVIDENCE` | **37 unresolved** | **0/37** | Cannot infer cause or rate |

| Speaker | Frozen Read Aloud | base.en refusals | Eligible refusals |
|---|---:|---:|---:|
| HQTV | 12 | 11 | 0 |
| PNV | 12 | 5 | 0 |
| THV | 12 | 9 | 0 |
| TLV | 12 | 12 | 0 |
| **Total** | **48** | **37** | **0** |

Speaker omissions, substitutions, repetitions, hallucinated words, mixed errors and
uncertain alignments may exist but their **confirmed counts are unknown**, not
established as zero events. Actual ASR, hypothetical independently transcribed speech,
and intended-prompt counterfactuals cannot be compared on the same eligible read
recording. Do not extrapolate 0/37 eligible into a causal percentage.

## E. Pronunciation and Parakeet implications

Correct ASR words are not equivalent to correctly articulated phonemes. Actual
prompt adherence, acoustic support, forced alignment, GOPT inference and legitimate
Pronunciation error attribution remain separate unverified conditions on these 48.
Parakeet's cached output lacks Whisper-equivalent native decoder-support diagnostics;
the existing compatibility function returned
`MISSING_DECODER_SUPPORT_FAIL_CLOSED`. Lower prompt disagreement is insufficient
to bypass decoder, VAD, acoustic or alignment protections. No model promotion,
threshold adjustment, downstream GOPT counterfactual, Content/Fluency formula change
or Pearson calibration occurred.

The same four Vietnamese-L1 speakers and CMU ARCTIC prompts have been repeatedly
exposed; this is retrospective causal preparation, **not** speaker-disjoint, blind,
population-level or prompt-generalization evaluation.

## F. Engineering verification and changes

- **No code or test changes:** existing `task_07_5_asr.py`,
  `task_07_5_2_attribution.py`, and `tests/test_task_07_5_2_attribution.py`
  already provide the frozen verification, structural submission checks and
  conditional synthetic three-way diagnostics. No real independent submission was
  available to justify or validate a new ingestion implementation.
- **Fresh executed checks:** frozen 54 hashes and references, 48 selected IDs/hashes,
  48 annotation/prompt identities, cached three-model manifest matching,
  baseline 37 refusals and aggregate diagnostics reconciliation: **PASS**.
- **Regression:** `.venv/bin/python -m unittest discover -s tests -v` —
  **172 run; 171 PASS, 1 SKIPPED, 0 FAIL**, including existing synthetic attribution
  tests. `.local/uv-tool/bin/uv lock --check --offline` — **PASS, 28 packages**.
  `pytest` was not installed in the project's virtualenv; the repository's
  supported unittest discovery was executed instead.
- **Unexercised gates:** genuine independent human provenance verification,
  multi-submission conflicts/revisions and real eligible-unit attribution
  (no human submissions exist). Passing synthetic tests does not prove these
  empirical scientific conditions.
- **Local private artifacts:** all existing handoff, diagnostics, row-level cached
  predictions, raw audio and annotation TextGrids remain under ignored `.local/`.
  No new private artifact was necessary.
- **Tracked change:** **only this aggregate scientific Markdown report**.
  No source audio, human identity, row-level transcript, checkpoint or private hash
  mapping is added to Git.

**Engineering assessment:** frozen integrity, handoff readiness and existing
regression are validated; end-to-end independently verified human ingestion and
eligible attribution are **NOT EXERCISED**. Do not claim full scientific Engineering
PASS on the basis of synthetic fixtures alone.

## G. Final verdict and acceptance gates

**INDEPENDENT_TRANSCRIPTS_UNAVAILABLE**

- Ground Truth PASS: **NO** — 0/48 independently verifiable Read Aloud actual-speech transcripts.
- Attribution PASS: **NO** — 0/37 baseline refusals have eligible human actual-speech evidence.
- Existing handoff: **HANDOFF_NO_CHANGE_REQUIRED**.
- Production: **PRODUCTION_MODEL_UNCHANGED**.
- Pearson: **PEARSON_AGREEMENT_NOT_VERIFIED**.
- Unavailable human evidence is a scientific input blocker, not proof of failed
  model implementation. No derived accuracy, agreement or causal percentage is fabricated.

## H. Git verification

Starting HEAD/rollback reference:
`e4bf3c50522d6058164de824cec9b3dcb6a604de`.
The exact ending commit SHA, observed origin/main after publication and final worktree
state are reported in the execution response (not self-referenced in this committed
report). Push only this validated aggregate report on `main` if publication succeeds.

## I. Exactly one next action — NOT EXECUTED

**Owner-authorized independent human annotation of the exact 48 frozen audio files,
using the existing handoff:** obtain real audio-only first passes, separate prompt-aware
review and independent adjudication with auditable local evidence, after explicit
Owner approval of any required external audio disclosure, service, terms and costs.
Only then resume human-grounded ASR-versus-speaker attribution; do not start TASK-07.6.
