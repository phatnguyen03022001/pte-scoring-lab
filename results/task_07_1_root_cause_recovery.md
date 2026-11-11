# TASK-07.1 — Pronunciation Scoring Root-Cause Recovery and Controlled Validation

**Run date:** 2026-11-11
**Starting repository state:** `main`, HEAD `24d820311c3c6c5932ef3995e63e3d9f29b972d3`, clean and equal to `origin/main`
**Scope:** local, offline diagnosis and bounded supported-word recovery comparison. No Pearson data, vendor calls, GPU, LLM, cloud, new training, or infrastructure.

## Verdicts

- **Engineering: PASS.** The frozen benchmark has a reproducible failure taxonomy, exact-denominator recovery bounds, a small same-unit recovery experiment, and regression/failure-path tests. No production change was justified; the existing default remains unchanged.
- **Scientific improvement: NOT DEMONSTRATED.** Supported-word recovery increased coverage on four deliberately diagnostic failed recordings, while false-positive rates remained high. The sample cannot establish whole-corpus or population improvement.
- **Pearson: `PEARSON_AGREEMENT_NOT_VERIFIED`.** No Pearson-scored ground truth or agreement protocol exists in this repository.
- **Next decision: `INSUFFICIENT_EVIDENCE`.**

## Frozen baseline and denominator

The original TASK-07 cache was read without modification. Its 1,000 recordings yield **33,379 eligible human target-phone units** overall and **20,136** for Vietnamese L1. The positive label is a human substitution or deletion; human insertions have no target phone and remain excluded. All analyses retain those same units and labels. Abstentions count in the eligible denominator; ranking metrics are conditional on scored rows only. Scores remain native, unbounded regression outputs and are never clipped or treated as probabilities.

| Population | Eligible | Scored | Abstained | Coverage | Precision | Recall incl. abstentions | F1 | FPR on scored correct phones | Scored-only ROC / PR AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vietnamese L1 | 20,136 | 4,575 | 15,561 | 22.72% | 0.2726 (812/2,979) | 0.1720 (812/4,721) | 0.2109 | 59.08% (2,167/3,668) | 0.7600 / 0.4902 |
| All L1 | 33,379 | 10,044 | 23,335 | 30.09% | 0.2149 (1,361/6,334) | 0.2116 (1,361/6,432) | 0.2132 | 58.60% (4,973/8,487) | 0.7486 / 0.4078 |

Vietnamese confusion counts are TP 812, FP 2,167, scored FN 95, TN 1,501; 3,814 reference errors and 11,747 correct phones are abstained. The fixed threshold is 1.986756 from the exposed EpaDB development process, with errors flagged at scores **at or below** threshold. It is retrospective and not independently validated.

## Mutually exclusive failure accounting

The first accounting level assigns each eligible phone to its observed contract outcome. The second level refines the failed-record phones using the cached forced-phone and support diagnostics, with exact canonical phone identity alignment; it does not change the frozen baseline status or pretend withheld phones had GOPT scores.

| Baseline outcome, all L1 | Eligible phones | Interpretation |
|---|---:|---|
| Scored | 10,044 | Exact identity and available GOPT score |
| Exact identity, no score | 732 | Withheld/model abstention; no GOPT prediction |
| Target identity mismatch or gap in completed/partial records | 389 | Not positionally shifted or scored |
| Component failure in failed recordings | 22,214 | 649 recordings, all terminating at `alignment_support` |
| **Total** | **33,379** | Mutually exclusive; unchanged denominator |

Within the 22,214 component-failure units, the forced phone diagnostics permit a separate exact-identity refinement: **21,289** target units match an expected canonical phone and **925** are annotation-to-expected-sequence mismatches/gaps. Of the 21,289 exact identities, **957** phones are individually below the −6 support floor, **3,461** are individually supported but share a word with a below-floor phone, and **16,871** are in other individually supported words withheld because the whole planned GOPT window did not pass. These three support categories are mutually exclusive and sum to 21,289. Across the four Vietnamese speakers, the corresponding numbers are 762, 2,582, and 10,897 exact identities, plus 600 mismatch/gaps, totaling 14,841 eligible units in failed recordings.

No cached recording failed earlier than the support stage: reference normalization **0**, unsupported vocabulary **0**, missing lexicon entry **0**, unsupported phoneme/model mapping **0**, forced-alignment or feature-order validation **0**, support-stage failures **649**, GOPT inference/output failures **0**. The 732 exact-phone abstentions above are withheld predictions, not failed GOPT calls. The cached model-abstention diagnostic count of 764 includes predicted phone rows that do not all align to an eligible human identity; therefore it is not substituted for the 732 mutually exclusive eligible-unit count.

### Vietnamese speaker failure and recovery taxonomy

| Speaker | Failed / 150 recordings | Eligible phones in failed recordings | Below-floor | Same-word propagation | Other-word whole-window propagation | Identity gaps |
|---|---:|---:|---:|---:|---:|---:|
| HQTV | 117 | 4,022 | 245 | 792 | 2,831 | 154 |
| PNV | 82 | 2,754 | 54 | 332 | 2,241 | 127 |
| THV | 115 | 3,900 | 208 | 645 | 2,878 | 169 |
| TLV | 121 | 4,165 | 255 | 813 | 2,947 | 150 |
| **Total** | **435 / 600** | **14,841** | **762** | **2,582** | **10,897** | **600** |

The 10,897 other-word phones are an **upper bound** for the existing supported-word recovery policy on Vietnamese failed recordings. If every such phone yielded a valid, usable score, the theoretical Vietnamese coverage ceiling would be (4,575 + 10,897) / 20,136 = **76.83%**. A phone-only support policy could additionally expose the 2,582 supported siblings, for a theoretical ceiling of **89.65%**. Neither ceiling is measured candidate performance; identity gaps, GOPT errors, model mismatch, and the resulting scores still matter.

The −6 gate was also checked against manual phone boundaries on exact-identity rows, using both phone edges within 100 ms as the descriptive alignment reference. For Vietnamese, 19,360 exact-identity phone pairs were available: 18,112 supported and boundary-valid, 453 supported but boundary-invalid, 691 below-floor despite boundaries within 100 ms, and 104 below-floor and boundary-invalid. Thus the gate false-reject rate was 691/18,803 = **3.67%** among boundary-valid phones and the false-accept rate was 453/18,565 = **2.44%** among support-accepted phones. Only **61** of the 691 false-rejected boundary-valid rows were human-labeled unchanged/correct; the other 630 were human-labeled errors. This boundary proxy supports a conservative validity role, but the four-speaker sample cannot establish a population-valid support threshold.

## Hypotheses and controlled measurements

1. **Whole-window support rejection is a major coverage cause.** Supported: all 649 failed recordings reached the same support gate; 16,871 all-L1 exact-identity phones in otherwise supported words were blocked by a weak phone elsewhere in the window. Unsupported: that coverage bound alone does not show the missing scores would be accurate.
2. **Phone identity alignment and annotation differences also matter.** Supported: 925 failed-record target phones do not exactly match the expected sequence across all L1 (600 Vietnamese); completed/partial records contain another 389 baseline identity gaps/mismatches across all L1 (176 Vietnamese). These are kept as mismatch/gap units. The audit cannot attribute each mismatch to the human annotation, a legal pronunciation variant, alignment, or another source without adjudication.
3. **The operating threshold contributes substantial false positives independently of coverage.** Supported by fixed-denominator threshold sensitivity below. No threshold was selected or changed on this exposed test.
4. **First-variant lexicon and accent mismatch may affect alignment.** All 5,693 Vietnamese reference word tokens (648 unique word types) were found in the local Kaldi alignment lexicon; there were no vocabulary OOVs. The scorer takes the first listed variant. 143 reference word types have multiple listed variants, appearing in 2,437 tokens (42.8%). Of the listed alternatives for those types, 15 are identical to the first variant after canonicalization, while 152 have a distinct canonical phone sequence. Examples include `AND` with first `AE` versus alternate `AH`, `OUR` with `AA R` versus `AW ER`, and `WITHIN` with `DH` versus `TH`. Phone normalization strips ARPAbet stress digits and the documented `*` marker. No alternate variant was selected using labels or rescored, so their effect remains unmeasured; the local LibriSpeech lexicon is not a validated Vietnamese-English or UK-accent lexicon.
5. **Raw-score semantics or training-domain mismatch may limit discrimination.** Source inspection confirms the raw Kaldi chain node `output` supplies GOP features, while `output-xent` supplies forced alignment and acoustic support. GOP features are 42 acoustic-output phone masses plus 42 canonical-versus-alternative log-mass ratios, normalized using the released GOPT training constants and aligned to its training phone order. The acoustic chain and GOPT checkpoint are LibriSpeech-domain models applied to accented English read speech. No cached result indicates a node mix-up; the high false-positive rate and domain difference remain concerns, not proof of a code defect.

### One-factor score-threshold sensitivity (Vietnamese; same 20,136 units)

Coverage and all labels are fixed; only the threshold changes. FPR is FP / scored-correct, and recall includes every abstained reference error as a miss.

| Error threshold (`score <= threshold`) | Scored / eligible | Precision | Recall incl. abstentions | F1 | FPR |
|---:|---:|---:|---:|---:|---:|
| 1.50 | 4,575 / 20,136 | 0.8235 | 0.0030 | 0.0059 | 0.08% (3/3,668) |
| 1.75 | 4,575 / 20,136 | 0.8495 | 0.0167 | 0.0328 | 0.38% (14/3,668) |
| 1.986756, frozen | 4,575 / 20,136 | 0.2726 | 0.1720 | 0.2109 | 59.08% (2,167/3,668) |
| 2.25 | 4,575 / 20,136 | 0.1983 | 0.1921 | 0.1951 | 100% (3,668/3,668) |

Reversing the score direction at the frozen threshold gives precision 0.0595, recall 0.0201, F1 0.0301, and FPR 40.92%. Threshold sensitivity does not produce a defensible alternative operating point: lower thresholds collapse recall and F1; higher thresholds flag nearly every scored correct phone. ROC/PR ranking values do not change with threshold.

### Existing `--recover-supported-words` option: bounded paired diagnostic

One deterministic cached support-failure recording per Vietnamese speaker was rescored sequentially with only the existing `--recover-supported-words` flag changed. The 125 frozen eligible phone IDs and labels in those four recordings were preserved; baseline outputs were the cached `ALIGNMENT_INVALID` component failures. Candidate predictions came only from exact identity matches with available GOPT scores. These exposed test records are a small mechanism check, not an independently selected validation set.

| Same 125 units / 4 speakers | Scored | Coverage | Precision | Recall incl. abstentions | F1 | FPR on scored correct |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0 | 0% | null | 0% (0/27) | null | null (0 scored correct) |
| Supported-word recovery | 106 | 84.8% | 0.2778 (20/72) | 74.07% (20/27) | 0.4040 | 60.47% (52/86) |

The candidate abstained on 19 units (7 errors, 12 correct); it flagged 72 phones (20 errors, 52 correct). By speaker, candidate eligible/scored, precision, recall including abstentions, F1, FPR, and scored-only ROC/PR AUC were: HQTV 39/31, .3913, .8182, .5294, 63.64%, .9343/.8776; PNV 31/27, .1538, .5000, .2353, 44.00%, .8800/.2381; THV 16/13, .3333, .7500, .4615, 60.00%, .9000/.8167; TLV 39/35, .2222, .7500, .3429, 72.41%, .8276/.4219. These tiny speaker slices have unstable class support and are descriptive only.

The candidate therefore confirms that supported-word recovery can turn some whole-window failures into scored partial results. It does **not** demonstrate a good operating point: its paired FPR is still 60.47%, while the baseline's FPR is undefined because it has zero scored-correct units. No full Vietnamese or all-L1 candidate metrics can be inferred from four recordings.

`score_integrated.py` has a separate existing guard: supported-word recovery is enabled only when ASR matches the complete normalized reference in order without low-support or ambiguous tokens, and pronunciation attribution is then separately gated. TASK-07's frozen runs invoke `score_pronunciation.py` directly without that ASR precondition. The four-record direct diagnostic therefore measures only the acoustic support-recovery mechanism; it does not estimate integrated-pipeline coverage.

### Full-cohort cached-output taxonomy and cutoff check

As a separate cache-only check, the TASK-07 full-cohort row evaluation was mapped to the original eligible manual phone IDs. All 33,379 eligible IDs and labels were retained. All 1,000 selected rows matched their frozen audio and annotation hashes; 993 scorer outputs carry the declared parent protocol fingerprint and were reused only where the normalized references were exactly equivalent, while seven were rescored after the current reference-normalization correction (three completed and four remained alignment-invalid). This supplemental analysis did not score any audio.

The mutually exclusive baseline phone taxonomy is:

| Baseline phone outcome | All L1 | Vietnamese L1 |
|---|---:|---:|
| Finite native GOPT score | 10,044 | 4,575 |
| Exact identity, withheld for acoustic support | 732 | 544 |
| Identity mismatch/gap in completed or partial output | 389 | 176 |
| Recording-level component failure at acoustic gate | 22,214 | 14,841 |
| **Eligible total** | **33,379** | **20,136** |

This agrees with the TASK-06 all-eligible identity tally: 10,776 exact identities (10,044 scored plus 732 exact-identity abstentions) and 22,603 mismatch/gap units (22,214 component-failure phones plus 389 completed-output gaps). The 389 completed-output gaps and 22,214 failed-record units were not positionally shifted to create scores. On the 649 failed recordings, matching the per-phone support diagnostics against human target identities divides the 22,214 units into 20,332 phones with exact identity and individual support at or above −6, 957 exact identities below −6, and 925 identity mismatches/gaps. The label-linked split is 16,635 correct / 3,697 error in the individually supported group; 180 correct / 777 error below floor; and 744 correct / 181 error among identity mismatch/gap units. For Vietnamese these are respectively 10,654 / 2,825 of 13,479; 84 / 678 of 762; and 461 / 139 of 600. These associations describe agreement with human labels, not an independent acoustic judgment.

For each Vietnamese speaker, eligible phones partition into scored, exact-identity support abstention, completed-output identity gap, and failed-record component failure as follows:

| Speaker | Eligible | Scored | Support abstention | Identity gap | Component failure |
|---|---:|---:|---:|---:|---:|
| HQTV | 5,038 | 808 | 175 | 33 | 4,022 |
| PNV | 5,031 | 2,140 | 67 | 70 | 2,754 |
| THV | 5,030 | 918 | 174 | 38 | 3,900 |
| TLV | 5,037 | 709 | 128 | 35 | 4,165 |

Only cached scores were thresholded in this check; the same exact scored IDs are used at every cutoff, so coverage and conditional ROC-AUC/PR-AUC do not change. A fixed-seed paired bootstrap over 24 speaker clusters (500 resamples) gives these all-L1 coverage-aware F1 deltas from the frozen threshold: at 1.75, −0.1810 (95% CI −0.2169 to −0.1509); at 2.00, −0.0095 (−0.0184 to −0.0042); at 2.25, −0.0242 (−0.0418 to −0.0134). No threshold was chosen from this exposed test. Vietnamese remains four speakers and receives no population interval. Per-speaker threshold results are retained in the ignored experiment artifact below.

The −8 and −10 support-floor figures in the preceding bounded comparison remain eligibility bounds for failed windows only. Those windows have no cached GOPT predictions, so no paired precision, recall, F1, FPR, or conditional AUC can be computed for a support-floor change from existing data.

The cache-only whole-word floor counterfactual is also explicit: across the 649 failures, `-8` admits 486 windows containing 16,193 phones in 481 recordings (168 still have no passing window); `-10` admits 645 windows containing 21,835 phones in 636 recordings (13 still have none). At `-4`, no failed recording has a passing window. Among exact-identity failed phones in the `[-8, -6)` support band, 664/816 are human-labeled errors and 152 correct (Vietnamese: 578/645 errors, 67 correct). These are support-policy eligibility and label-consistency counts only; no new GOPT predictions exist for these phones.

The 10,044 scored native values remain unbounded. Of these, 2,091 exceed the nominal 2.0 upper target and the maximum is 2.04517; none were clipped. The raw chain `output` still supplies GOPT features, while `output-xent` supplies forced alignment and support validation. All 351 completed/partial outputs report compatible checkpoint inputs and the observed maximum absolute `output-xent` posterior log-sum-exp residual is 0.00018117.

## Error classes, scorer provenance, and limits

On the 4,575 scored Vietnamese phones, false positives are all human `unchanged_correct` phones: 2,167 total. By target class, vowels have 1,750 scored (1,478 correct, 272 errors), with 1,184 false positives / 1,478 scored-correct = **80.11% FPR**; consonants have 2,825 scored (2,190 correct, 635 errors), with 983 / 2,190 = **44.89% FPR**. The 812 true-positive error flags comprise 571 substitutions and 241 deletions. These descriptive class counts indicate the aggregate FPR is particularly high for vowels.

The scorer pipeline uses Kaldi MFCC/high-resolution features and i-vector extraction, then the M13 LibriSpeech `tdnn_1d_sp` chain acoustic model, forced alignment, GOP extraction, and the LibriSpeech GOPT checkpoint. The 84 GOP feature values comprise 42 phone-mass features and 42 canonical-versus-alternative log-mass ratios. Raw-output and `output-xent` branches are required to yield the same indexed phone keys; alignment and posterior frame counts are checked, with 30 ms frame shift and canonical phones verified in GOPT training order. Cached completed-run stage medians are preprocessing 0.082 s, Kaldi features/i-vector 0.744 s, posterior 0.667 s, alignment 0.356 s, and GOP extraction 0.060 s. Across 351 scored/partial runs, end-to-end processing median/p95/max was 2.991/3.508/5.571 s (excluding Python startup); peak process/child memory median/p95/max was 358/372/384 MiB. The four added recovery calls took 3.305–4.258 s wall each and measured peak memory median 359 MiB. Task-local cost was $0; vendor/API calls were zero.

L2-ARCTIC v5.0 is CC BY-NC 4.0 research-only data. The Vietnamese results represent only four speakers and read speech; historical project exposure means the benchmark is retrospective, not blind. L2-ARCTIC labels are manually annotated phones, not Pearson PTE outcomes. GOPT was trained for a different speech domain and returns uncalibrated regression outputs. Four speakers do not support Vietnamese population intervals. The 20 other speakers also do not create a Pearson or broad-domain validation set.

Useful next evidence would be a larger speaker-disjoint Vietnamese-English corpus with adjudicated phone labels and manual boundaries; full paired baseline/recovery outputs on the frozen benchmark under an agreed resource budget; alternate legal lexicon variants tested without label-guided selection; and genuinely Pearson-scored responses with a predefined agreement design. A vendor comparison remains subject to credentials, terms, consent/privacy, and cost authorization, none of which were invoked here.

## Reproduction and verification

Restricted row-level evaluation, cache data, the four candidate run files, and the paired TASK-06 contract inputs/report are in ignored `.local/research/task-07.1-root-cause/`. The original frozen TASK-07 files and run cache remain unchanged. The taxonomy is reproduced by loading the frozen protocol and run files, parsing only human `phones` labels with `l2arctic_benchmark.parse_manual_phones`, exact-aligning canonical targets to cached model phone identities with `quality_08_benchmark.edit_align`, and assigning each failed-record exact match once by precedence: below-floor phone; supported phone in a weak word; supported phone in another word withheld by the failed whole window. Non-matches remain identity gaps. No evaluation label selects an inference-time pronunciation or gate.

The four-record paired comparison was generated with `benchmark_evaluation.py --compare` from the exact baseline unit IDs and labels; the original evaluation JSON was not edited. Threshold ablations used only its cached predictions and fixed labels. The unchanged full test suite passed:

```text
.venv/bin/python -m unittest discover -s tests -v
Ran 137 tests in 7.637s — OK (skipped=1)
```

The skipped test is the explicitly gated real-audio integrated smoke test (`PTE_RUN_REAL_AUDIO=1`). Existing normalization, lexicon/OOV, phoneme mapping, support-gate, alignment-gap, GOPT failure, unbounded-score, paired-evaluation, and failure-isolation tests passed. No production source or test file was changed.

## Files, changes, and remaining blocks

- Tracked change: this aggregate report only.
- Private ignored artifacts: four small candidate run outputs, the 125-row paired baseline/candidate inputs, and their aggregate comparison under `.local/research/task-07.1-root-cause/`. No row-level values or protocol IDs are in this report.
- Private ignored full-cohort supplement: `.local/research/task-07-pronunciation-benchmark/task_07_1_row_evidence.json` and `task_07_1_experiments.json`.
- **Engineering:** root-cause accounting, bounded recovery execution, and regression suite verified; production behavior unchanged.
- **Scientific Improvement:** not demonstrated. Full-corpus recovery predictions and independent Vietnamese evidence remain blocked; lower false-positive performance is also unresolved.
- **Pearson:** remains unverified.

No TASK-08 work was started. No commit or push was made.
