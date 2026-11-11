# TASK-07.4 — Pearson Official Samples Ground Truth Audit

**Date:** 2026-11-11 · **Scope:** existing, local Pearson preparation assets only · **Branch:** `main`

## A. Verdict

- **Engineering: PASS** — full local file inventory, source/content audit, checksum/ZIP checks, audio metadata, response classification, source-to-response linking, score semantics, rights/exposure review, and existing regressions completed. No scorer or model was executed.
- **Primary dataset verdict: `PARTIAL_OFFICIAL_EVIDENCE`.** The resources are Pearson-issued teaching/preparation materials. They supply identifiable response examples and qualitative **Content** explanations but **zero verified numerical audio–score pairs**.
- **Scientific Evidence: NOT PASS.** No `EXACT_VERIFIED` audio-to-numerical-trait score exists.
- **Rights: `RIGHTS_UNVERIFIED` for automated local scoring/evaluation and commercial benchmarking; redistribution not authorized by reviewed terms.**
- **Independent Pearson evaluation: NOT READY.** The public samples and prior project inventory are not an untouched blind cohort.
- **Pearson agreement: `PEARSON_AGREEMENT_NOT_VERIFIED`.** This task did not perform an agreement experiment. Neither a Pearson 80–90% equivalence claim nor item/skill calibration is supported.

**Decision basis:** The absence of published response-specific numerical scores independently prevents quantitative comparison, even if the rights obstacle were cleared. Hence `PARTIAL_OFFICIAL_EVIDENCE` rather than declaring numerical pairs or treating access rights as the only blocker.

## B. Complete local inventory and integrity

Existing root: `.local/datasets/pearson-pte-academic-2025/`. Detailed per-file SHA-256, relative paths, source membership, format, duration, sampling, channel counts, and pairing metadata are retained **only** under ignored `.local/research/task-07.4-pearson/`; no copyrighted asset or response transcript is added to Git.

| Inventory measure | Verified count |
|---|---:|
| All materialized files, including 3 ZIP archives and unpacked members | **117** |
| MP3 extension / M4A extension / PDFs / ZIPs / metadata JSON | **104 / 4 / 5 / 3 / 1** |
| Extracted audio file instances | **108** |
| Binary-distinct audio files (SHA-256) | **106** |
| Prompt/reference audio instances | **84** (83 distinct) |
| Illustrative/test-taker sample-response instances | **24** (23 distinct) |
| Individually verified *operational test candidate* recordings | **0** |
| Source resources named in original manifest | **5** |
| Total materialized files size, archives included | **308,877,200 bytes** (about 294.6 MiB) |
| Audio metadata probe failures / bad source ZIP CRC checks | **0 / 0** |
| Raw-byte duplicate groups / redundant audio instances | **2 / 2** |

All **five originally acquired assets** match both the local acquisition manifest's recorded byte sizes and SHA-256 digests. All **111 files** extracted from the three archives match their original ZIP entries byte-for-byte by SHA-256 (41 + 61 + 9); no missing, divergent, or extra extracted files. Every one of the 108 audio instances yielded duration, codec, sample rate and channel metadata through `ffprobe`. Codec inspection found 99 MP3 streams and 9 AAC streams; **five files bearing a `.mp3` extension actually contain AAC**. Do not infer codec from extension during any later authorized use.

Identical SHA-256 pairs:

- Lesson Ideas `L6_Summarize_group_discussion_1.mp3` = Sample Answers `1. SGD_Prompt_1.mp3` (**same prompt**, not a new question).
- Lesson Ideas `L6_Summarize_group_discussion_1_Response_C1.mp3` = Sample Answers `1. SGD_Sample_1.mp3` (**same response**, not an additional speaker/observation).

No further byte-identical audio was found. Re-encoded perceptual near-duplicates and independent speaker identities cannot be ruled out by exact hashes and duration/codec metadata alone. There is **no justified claim** of 106 different speakers or 23 independent learners.

### Resource-by-resource breakdown

| Pearson resource | Local evidence | Audio classification | Score-bearing content |
|---|---|---|---|
| **Sample Answers with Explanations** (`sample-answers.zip`) | 28-page `Sampleanswer_Partners_and_Teachers.pdf`; table on PDF pp. 4 and 18, question-by-question correspondence on pp. 5–16 and 19–28 | 20 prompts, **20 recorded response exemplars** (10 SGD, 10 RTS) | Qualitative **Content** judgments and explanations; **no published numerical item/trait values** |
| **Example Speaking Questions** (`example-speaking-questions.zip`) | 61-page `NewSpeakingQuestions.pdf`; audio index on pp. 4–5 and 48–49 | **60 prompts** (30 SGD, 30 RTS); zero response exemplars | No score attached to an audio response |
| **Lesson Ideas** (`lesson-ideas-two-new-question-types.zip`) | 19-page teaching PDF, response/audio references on pp. 8–12 and 17–19 | 4 prompts and **4 response exemplars** (2 SGD, 2 RTS); first SGD prompt and response duplicate Sample Answers | Explanatory/qualitative commentary; **no numerical trait observations** |
| **Test Tips** (`test-tips-new-speaking-questions.pdf`) | 20-page PDF, marked **V4 July 2025** on p. 2; text-only answer examples on pp. 9–12 and 15–19 | No local audio from this resource | Qualitative instructional labels, not scores linked to an audio recording |
| **Test Taker Score Guide** (`PTE-Academic-Test-Taker-Score-Guide.pdf`) | 46-page official guide: overall/skills pp. 4–8, SGD/RTS rubrics pp. 25–31, Pronunciation/Oral Fluency pp. 45–46 | No local audio from this resource | **Numerical scoring rubrics** and illustrative Score Report, **not per-response audio labels** |

SGD = Summarize Group Discussion; RTS = Respond to a Situation. The 20 Sample Answers recordings are described by Pearson as **test-taker samples**; the local materials do not establish that they are independent operational exam responses, nor do they identify distinct candidate speakers. The four Lesson Ideas recordings are instructional exemplars; one is a byte-for-byte repeat.

Source authenticity: all five file names and purposes are corroborated by Pearson's [official 2025 teachers/partners resource page](https://www.pearsonpte.com/teachers/pte-academic-updates-partners-and-teachers/). The original local manifest records exact links under `pearsonpte.com`; download date is 2026-11-11. Pearson's [2025 test update notice](https://www.pearsonpte.com/pte-updates-2025/) makes the two question types applicable after **7 August 2025**. The Sample Answers document states it was created by the Pearson Test of English team. The instructional packs do not all state a distinct edition/publication date inside the PDFs: **do not invent an edition from a filename**. Score Guide bears Pearson's 2025 copyright; current site links may change after the recorded download.

## C. Audio-to-score pairing and provenance

The PDF directly references each of the **20 Sample Answers response recordings** by question identifier/file name and provides a corresponding transcript and narrative evaluation of **Content**. The Lesson Ideas PDF explicitly names the **four** recorded responses beside teaching explanations. This establishes the **source-to-response association**, not a numerical score.

| Pairing and label category | File instances | Distinct response audio |
|---|---:|---:|
| `EXACT_VERIFIED` **numeric audio–score** | **0** | **0** |
| `SOURCE_LINKED_NOT_NUMERIC` | **24** | **23** |
| `AMBIGUOUS` numeric score mapping | **0** | **0** |
| `UNPAIRED` response audio with no source relationship | **0** | **0** |
| Reference/prompt audio **not to be paired as candidate responses** | **84** | **83** |

Task distribution: **12 SGD and 12 RTS response file instances**, or **11 SGD and 12 RTS distinct exemplars** after removing the one repeated SGD response. No Read Aloud, Repeat Sentence, Describe Image, Retell Lecture, or other scored Speaking task response recording was present in these five materials.

| Numerical provenance type | Verified audio-linked numeric observations |
|---|---:|
| `PEARSON_MACHINE_SCORE` | **0** |
| `PEARSON_EXPERT_RATING` | **0** |
| `PEARSON_OFFICIAL_ILLUSTRATIVE_SCORE` | **0** |
| `THIRD_PARTY_RATING` | **0** |
| `UNVERIFIED_SCORE` submitted as numeric ground truth | **0** |
| `NO_SCORE` (published numerical value absent) | **24 response instances** |

Pearson's narrative descriptions such as “excellent”, “good”, and “average” are **not** measurements on the official numerical trait scale and must never be encoded as a guessed number. The Sample Answers PDF's SGD question 8 also has inconsistent qualitative wording between its short heading and longer explanatory paragraph (p. 13); this does not create a numerical label or justify choosing a rating. The Guide's example Score Report is illustrative **and has no matching audio identity**.

The local manifest deliberately leaves `published_numerical_score: null` for all 24 linked responses. There is no `EXACT_VERIFIED` benchmark manifest to freeze; **none was created**.

## D. Scoring meaning and engine comparability

The [official Test Taker Score Guide](https://www.pearsonpte.com/content/dam/ELL/pte/pearsonpte/resources/PTE-Academic-Test-Taker-Score-Guide.pdf) distinguishes:

- **Overall score and communicative skill scores:** **10–90**, integrated over test items, **not** an arithmetic average of illustrative traits (Guide p. 6).
- **Summarize Group Discussion Content:** integer rubric **0–6** (Guide pp. 25–27).
- **Respond to a Situation Content:** integer rubric **0–6** (Guide pp. 29–30).
- **Pronunciation:** rubric **0–5**, distinct from **Oral Fluency 0–5** (Guide pp. 28, 31 and 45–46).
- Actual test scoring includes machine scoring and human expert scoring/review; a publicly supplied **examiner explanation of Content** is not proof of a numerical Pearson machine rating for any response (Guide pp. 7, 13, 15).

Neither the 24 audio instances nor the 23 distinct response recordings come with numerical Content, Pronunciation, Oral Fluency, or overall Speaking labels. Existing local engine/GOPT phoneme-level unbounded predictions, and binary target-phone-error outputs, **are not on these Pearson item-trait scales**. A publicly visible rubric does not publish the mapping or aggregation needed to compare phone scores to item Pronunciation, fluency, Content or the 10–90 skill score. No transformation, proportional rescaling, label imputation, threshold adjustment or scorer inference was attempted.

## E. Rights, provenance and evaluation independence

**Rights disposition: `RIGHTS_UNVERIFIED` for the proposed automated evaluation.** Pearson publicly offers these resources to teachers and partners for preparation, but that does **not** establish a licence for model assessment, automated processing, commercial benchmarking, redistributing audio/transcripts, sending audio to third-party scoring APIs or publishing repackaged benchmarks.

The [Pearson PTE End User Licence Agreement](https://www.pearsonpte.com/policy-center/user-agreement/) reserves content rights, limits website usage to personal/non-commercial use and expressly restricts data mining/scraping, reproduction and systematic database creation without express permission. The local Test Tips PDF (p. 20) and Score Guide (p. 46) also prohibit unlicensed reproduction. These provisions must be reviewed for any subsequent permitted use; this research audit **does not grant or claim legal authorization**. Original files remain in ignored `.local/`, and this report reproduces no Pearson answers or audio. No data was sent to a third party.

**Independence disposition: NOT BLIND / NOT VERIFIED.** The audio examples are public preparation content, not a source-verified operational candidate cohort. `results/task_07_3_independent_ground_truth.md` already inventoried this Pearson collection, so the source has prior project exposure. Specific earlier use for scorer training, threshold selection or prompt-specific optimization is unverified. The audio files do not establish speaker identity, number of unique speakers, language backgrounds, candidate consent, test version per respondent, or speaker-disjointness. At least one response is exactly duplicated in a different resource, and some teaching prompts overlap other example material. Exposure to external model training cannot be ruled out. A newly created split or renamed response ID would not make this an untouched blind evaluation set.

## F. Scientific evaluation eligibility

| Gate | Status | Supported use |
|---|---|---|
| **Level 1 — Example verification** | **REFERENCE-ONLY; rights not cleared for scorer use** | Content-example identity, document/rubric inspection and qualitative reasoning. No inference executed. |
| **Level 2 — Trait agreement feasibility** | **FAIL** | Zero response-linked numerical Pearson or Pearson-published expert trait values; no compatible scale/link. |
| **Level 3 — Independent agreement evaluation** | **FAIL** | No numerical labels, frozen independent speakers/partitions, sufficiently diverse task coverage, or cleared benchmark rights. |
| **Pearson production equivalence 80–90%** | **NOT VERIFIED** | No experiment authorized or run; no statistical agreement estimate possible. |

A Pearson content descriptor can check source identity and reveal what the publisher regards as an illustrative strong/weak answer. It cannot act as independent **Pronunciation**, **Oral Fluency**, phoneme-error, overall Speaking, or Pearson AI-score ground truth. The existing engine evaluates different dimensions; no number in this material justifies converting those outputs into an “agreement percentage”.

## G. Next smallest action — **not executed**

Seek **written confirmation from Pearson** whether numerical per-response **Content, Pronunciation and Oral Fluency** ratings exist for the **23 distinct, locally held response examples**, together with explicit permission for local automated scorer evaluation and publication of aggregate research results. Request stable audio/response identifiers and score provenance (machine, expert or illustrative). **If Pearson supplies no numeric identity-linked labels or authorization, do not instantiate a Pearson-agreement benchmark from these resources.**

This is a single evidence/rights qualification request, **not** an authorization to contact Pearson, accept terms, pay for services or start TASK-07.5. None was performed.

## H. Verification and release scope

Local reproducible evidence artifacts (ignored by Git):

- `.local/research/task-07.4-pearson/inventory.json` — 117 per-file inventory records and SHA-256 checks.
- `.local/research/task-07.4-pearson/qualitative-pairs.json` — 24 source-linked responses with `null` numerical values.
- `.local/research/task-07.4-pearson/summary.json` — aggregate totals and duplicate groups.
- `.local/research/task-07.4-pearson/integrity.json` — five original-asset checksum matches and 111 ZIP-member-to-extraction matches.
- `.local/research/task-07.4-pearson/audit.py` and `integrity.py` — local-only, standard-library audit scripts using existing `ffprobe`.

Commands executed: `.venv/bin/python .local/research/task-07.4-pearson/audit.py`, `.venv/bin/python .local/research/task-07.4-pearson/integrity.py`, native macOS PDFKit page inspection, `.venv/bin/python -m unittest discover -s tests -q`.

**Regression result:** **137 tests, 136 passed, 1 deliberately skipped, 0 failures** (2026-11-11). No GOPT/ASR/scoring pipeline was run. No dependencies, models, new Pearson downloads, training, commercial APIs, cloud or spending. No production scoring file was edited.

**Git initial HEAD:** `8dd038ecdba4bfffa87b43f8eb59c02d3b4688f1` — verified equal to remote `origin/main`, clean worktree before task. Release artifact: this aggregate, non-copyrighted evidence report only; task-specific detailed evidence remains ignored locally. The final commit and remote-ref confirmation are to be read from Git history rather than self-referentially embedded into the commit content.

**Final decision: `PARTIAL_OFFICIAL_EVIDENCE`.**

**Pearson status: `PEARSON_AGREEMENT_NOT_VERIFIED`.**
