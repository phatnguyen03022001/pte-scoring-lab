# TASK-07.2 — GOPT False-Positive Discrimination and Recovery

**Run date:** 2026-11-11  
**Starting state:** expected `main` HEAD `5d69eb2d267a12d39edd960b3822f08bb1900f41`, clean and equal to `origin/main`.  
**Scope:** cache-only analysis of existing TASK-07 / TASK-07.1 evidence; no audio rescoring, new downloads, model training, production changes, UI, or TASK-08 work.

## Verdicts

- **Engineering: PASS.** Exact eligible-unit labels and frozen predictions were reconciled by unique unit ID (33,379/33,379), with protocol fingerprint `72c08a102eed85c9ecb70f303f4db80ffc7978fa2cb45de9cd7f9cd5dc797c2b`. Threshold probes preserved the same eligible and scored IDs within each comparison; abstentions stayed misses in coverage-aware recall.
- **Scientific: NO IMPROVEMENT VALIDATED.** Existing GOPT outputs contain ranking information (Vietnamese scored-only ROC-AUC 0.7600), but the operating point has high false-positive rate and low coverage-aware recall. Retrospective cutoff probes do not establish a dependable recovery.
- **Pearson: `PEARSON_AGREEMENT_NOT_VERIFIED`.** No Pearson PTE labels or agreement study were found; no 80–90% agreement claim is made.
- **Production: `NO_CHANGE_REQUIRED`.** Preserve the current GOPT behavior, support gate, score direction, threshold `1.986756`, and no-clipping behavior. No new model or cutoff met the evidence bar for a production change.
- **Selected disposition:** **`KEEP_GOPT`** (keep the baseline pending stronger evidence; this is not an accuracy endorsement).

## Evidence and exposure

The analysis uses `.local/research/task-07-pronunciation-benchmark/{protocol.json,evaluation.json,report.json,runs/}` and TASK-07.1's `.local/research/task-07-pronunciation-benchmark/task_07_1_row_evidence.json`, plus its four cached diagnostic runs and the reports [TASK-06](task_06_benchmark_foundation.md), [TASK-07](task_07_pronunciation_scorer_benchmark.md), and [TASK-07.1](task_07_1_root_cause_recovery.md). Frozen scorer outputs were not edited or regenerated. Human substitution/deletion is positive; unchanged is negative; insertions have no target-phone unit and are excluded. Canonical phone identity mismatches/gaps remain abstentions, never positionally shifted scores.

L2-ARCTIC is an exposed retrospective corpus: historical reports, project iteration, and public release results make blind-test claims untenable even though the exploratory non-Vietnamese partitions below hold out speakers and prompts. Vietnamese evidence consists of only HQTV, PNV, THV, TLV, all reading a common prompt inventory; Vietnamese leave-one-speaker-out folds therefore cannot also be prompt-disjoint. Their results are explicitly exploratory. L2-ARCTIC is CC BY-NC 4.0, and the SpeechOcean checkpoint provenance is also exposed. The score is an unbounded native GOPT regression value; `score <= 1.986756` flags error. It is not a calibrated probability.

## Frozen baseline and score distributions

The contract reports the supplied historical Vietnamese baseline exactly: **20,136 eligible / 4,575 scored (22.72% coverage)**; TP=812, FP=2,167, scored FN=95, TN=1,501; 3,814 abstained reference errors; precision .2726, eligible-error recall .1720, F1 .2109, scored-correct FPR .5908, scored-only ROC-AUC .7600 and PR-AUC .4902. The last two metrics exclude abstentions. Conditional PR-AUC here follows `benchmark_evaluation.py`'s trapezoidal precision-recall definition. No phone was clipped: score scale remains unbounded. TASK-07's all-L1 baseline is 33,379 eligible / 10,044 scored, with precision .2149, eligible-error recall .2116, F1 .2132, FPR .5860, conditional ROC/PR AUC .7486/.4078.

Vietnamese score quantiles on scored eligible phones (5th / 25th / median / 75th / 95th percentile) show direct correct/error overlap:

| Human label | Eligible / scored | Error flags at frozen cutoff | Score quantiles |
|---|---:|---:|---|
| Unchanged/correct | 15,415 / 3,668 | 2,167 false positives | 1.8824 / 1.9536 / **1.9800** / 1.9983 / 2.0182 |
| Substitution | 3,155 / 647 | 571 true positives | 1.7448 / 1.8915 / **1.9407** / 1.9716 / 2.0034 |
| Deletion | 1,566 / 260 | 241 true positives | 1.5273 / 1.8078 / **1.9065** / 1.9576 / 2.0008 |

All three interquartile ranges overlap. The frozen cutoff is only 0.0068 above the correct-phone median; 75% of correct phones score at or below 1.9983, while the upper quartiles of substitution and deletion scores are 1.9716 and 1.9576. This makes many false positives expected at the existing cutoff while lowering it sharply trades away recall. The scored-only discrimination signal is real as a ranking association, but modest separation and support-based selection limit usable threshold performance.

The same quartile summary split by L1 confirms similar score location, but error prevalence, phone mix, and coverage differ. Each cell is scored-correct `Q25/median/Q75 ; scored-error Q25/median/Q75`:

| L1 | Correct score Q25 / median / Q75 | Error score Q25 / median / Q75 |
|---|---|---|
| Vietnamese | 1.9536 / 1.9800 / 1.9983 | 1.8696 / 1.9338 / 1.9684 |
| Arabic | 1.9536 / 1.9825 / 2.0014 | 1.8850 / 1.9364 / 1.9687 |
| Chinese | 1.9402 / 1.9706 / 1.9938 | 1.8728 / 1.9367 / 1.9768 |
| Hindi | 1.9513 / 1.9802 / 1.9987 | 1.8823 / 1.9358 / 1.9767 |
| Korean | 1.9625 / 1.9848 / 2.0022 | 1.8623 / 1.9386 / 1.9662 |
| Spanish | 1.9569 / 1.9820 / 1.9989 | 1.8904 / 1.9487 / 1.9793 |

Within Vietnamese, vowel scored-correct Q25/median/Q75 is 1.9411/1.9648/1.9829 versus error 1.8971/1.9370/1.9682; consonant correct is 1.9684/1.9902/2.0053 versus error 1.8645/1.9357/1.9730. Correct and error quantiles overlap in both classes. The ignored analysis JSON retains per-speaker and per-phone quantiles so the tracked report does not publish unit-level rows.

Vietnamese vowel/consonant pattern (eligible and scored supports are shown; FPR denominator is scored-correct phones only):

| Class | Eligible / scored | Scored errors / scored correct | FP / TN | FPR | Conditional ROC-AUC / PR-AUC |
|---|---:|---:|---:|---:|---:|
| Vowels | 7,843 / 1,750 | 272 / 1,478 | 1,184 / 294 | **80.11%** | .7138 / .3555 |
| Consonants | 12,293 / 2,825 | 635 / 2,190 | 983 / 1,207 | **44.89%** | .7984 / .5723 |

For context, the same fixed threshold yields vowel/consonant FPR 80.1%/44.2% among non-Vietnamese L1 scored-correct phones (n=1,881/2,938). Rates vary by L1: Vietnamese .5908 overall; Arabic .5607 (n=1,202 correct scored), Chinese .6802 (n=611), Hindi .5858 (n=1,002), Korean .5200 (n=779), Spanish .5781 (n=877). This common pattern argues against a Vietnamese-only code-path failure; it does not identify a unique acoustic cause. For all eligible units, the non-Vietnamese subset has 13,243 eligible / 5,469 scored, TP/FP/scored-FN/TN=549/2,806/101/2,013, plus 1,061 abstained errors; precision .1636, recall .3209, F1 .2167, FPR .5823.

Vietnamese per-target-phone false-positive rates are listed with the correct-phone denominator and scored support; rates with small supports are unstable and are descriptive only. `Eligible/scored; FP/scored-correct; FPR`:

| Phone | Eligible / scored | FP / correct support | FPR | Phone | Eligible / scored | FP / correct support | FPR |
|---|---:|---:|---:|---|---:|---:|---:|
| AA | 254 / 79 | 58 / 68 | 85.3% | AE | 533 / 143 | 101 / 104 | 97.1% |
| AH | 1,853 / 398 | 314 / 352 | 89.2% | AO | 366 / 57 | 36 / 51 | 70.6% |
| AW | 161 / 33 | 11 / 31 | 35.5% | AY | 362 / 81 | 18 / 77 | 23.4% |
| B | 340 / 88 | 13 / 83 | 15.7% | CH | 150 / 29 | 17 / 21 | 81.0% |
| D | 968 / 243 | 110 / 152 | 72.4% | DH | 599 / 130 | 40 / 46 | 87.0% |
| EH | 551 / 129 | 115 / 119 | 96.6% | ER | 510 / 109 | 34 / 36 | 94.4% |
| EY | 372 / 87 | 70 / 73 | 95.9% | F | 441 / 102 | 17 / 96 | 17.7% |
| G | 200 / 54 | 6 / 45 | 13.3% | HH | 542 / 128 | 17 / 124 | 13.7% |
| IH | 1,242 / 257 | 118 / 221 | 53.4% | IY | 813 / 192 | 177 / 187 | 94.7% |
| JH | 124 / 24 | 0 / 11 | 0.0% | K | 509 / 133 | 16 / 120 | 13.3% |
| L | 837 / 189 | 34 / 112 | 30.4% | M | 568 / 137 | 67 / 137 | 48.9% |
| N | 1,252 / 307 | 141 / 275 | 51.3% | NG | 196 / 46 | 15 / 36 | 41.7% |
| OW | 332 / 84 | 52 / 68 | 76.5% | OY | 28 / 7 | 2 / 6 | 33.3% |
| P | 428 / 100 | 24 / 77 | 31.2% | R | 871 / 199 | 103 / 129 | 79.8% |
| S | 830 / 175 | 41 / 151 | 27.2% | SH | 218 / 70 | 48 / 57 | 84.2% |
| T | 1,213 / 286 | 92 / 222 | 41.4% | TH | 148 / 29 | 20 / 20 | 100.0% |
| UH | 126 / 33 | 21 / 28 | 75.0% | UW | 340 / 61 | 57 / 57 | 100.0% |
| V | 452 / 86 | 57 / 57 | 100.0% | W | 555 / 104 | 18 / 104 | 17.3% |
| Y | 162 / 44 | 14 / 42 | 33.3% | Z | 666 / 118 | 69 / 69 | 100.0% |
| ZH | 24 / 4 | 4 / 4 | 100.0% |  |  |  |  |

Score overlap is not uniform across phones. For example, several vowel categories (AE, EH, EY, IY) have FPR above 94% with 73–187 scored-correct units, while B/F/G/HH/K have FPR below 18% with 45–124 scored-correct units. This points to phone-conditional score offsets or domain response differences as plausible contributors, but does not justify per-phone thresholds: the bins are small, exposed, and have no untouched population to tune against.

The four Vietnamese speakers also differ in coverage (14.08%–42.54%) and FPR (56.08%–62.19%). Per speaker at frozen cutoff:

| Speaker | Eligible / scored | TP / FP / scored FN / TN | Precision | Recall incl. abstentions | F1 | FPR | Conditional ROC / PR AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| HQTV | 5,038 / 808 | 154 / 394 / 13 / 247 | .2810 | .1198 | .1679 | 61.47% | .758 / .449 |
| PNV | 5,031 / 2,140 | 319 / 996 / 45 / 780 | .2426 | .3637 | .2911 | 56.08% | .749 / .426 |
| THV | 5,030 / 918 | 197 / 431 / 28 / 262 | .3137 | .1479 | .2010 | 62.19% | .744 / .553 |
| TLV | 5,037 / 709 | 142 / 346 / 9 / 212 | .2910 | .1158 | .1657 | 62.01% | .804 / .584 |

Across the 20 other speakers, coverage ranges 12.5%–70.7% and FPR 46.5%–80.1%; low scoring support drives some volatility. The frozen per-speaker eligible/scored, coverage, FP/TN, FPR, precision/recall/F1 are:

| Speaker (L1) | Eligible / scored | Coverage | FP / TN | FPR | Precision / recall / F1 | Conditional ROC / PR AUC |
|---|---:|---:|---:|---:|---:|---:|
| ABA (Arabic) | 622 / 440 | 70.7% | 229 / 175 | 56.7% | .123 / .696 / .208  | .739 / .259 |
| ASI (Hindi) | 740 / 322 | 43.5% | 141 / 128 | 52.4% | .234 / .358 / .283  | .752 / .383 |
| BWC (Chinese) | 654 / 196 | 30.0% | 112 / 43 | 72.3% | .243 / .263 / .253  | .698 / .417 |
| EBVS (Spanish) | 622 / 209 | 33.6% | 106 / 65 | 62.0% | .226 / .246 / .236  | .722 / .419 |
| ERMS (Spanish) | 713 / 254 | 35.6% | 138 / 75 | 64.8% | .207 / .261 / .231  | .717 / .338 |
| HJK (Korean) | 720 / 365 | 50.7% | 158 / 182 | 46.5% | .127 / .548 / .206  | .805 / .300 |
| HKK (Korean) | 694 / 87 | 12.5% | 46 / 34 | 57.5% | .132 / .121 / .126  | .909 / .619 |
| LXC (Chinese) | 650 / 192 | 29.5% | 121 / 30 | 80.1% | .239 / .311 / .270  | .679 / .420 |
| MBMPS (Spanish) | 563 / 282 | 50.1% | 142 / 114 | 55.5% | .118 / .292 / .168  | .695 / .239 |
| NCC (Chinese) | 688 / 247 | 35.9% | 142 / 78 | 64.5% | .134 / .256 / .176  | .676 / .239 |
| NJS (Spanish) | 656 / 269 | 41.0% | 121 / 116 | 51.1% | .154 / .216 / .180  | .674 / .274 |
| RRBI (Hindi) | 578 / 233 | 40.3% | 129 / 84 | 60.6% | .128 / .306 / .181  | .821 / .239 |
| SKA (Arabic) | 599 / 252 | 42.1% | 126 / 101 | 55.5% | .160 / .343 / .218  | .759 / .223 |
| SVBI (Hindi) | 693 / 327 | 47.2% | 161 / 106 | 60.3% | .244 / .437 / .313  | .701 / .367 |
| TNI (Hindi) | 634 / 277 | 43.7% | 156 / 97 | 61.7% | .093 / .225 / .132  | .623 / .323 |
| TXHC (Chinese) | 735 / 360 | 49.0% | 195 / 117 | 62.5% | .156 / .379 / .221  | .636 / .272 |
| YBAA (Arabic) | 711 / 334 | 47.0% | 188 / 124 | 60.3% | .100 / .362 / .157  | .787 / .282 |
| YDCK (Korean) | 650 / 208 | 32.0% | 104 / 81 | 56.2% | .181 / .343 / .237  | .876 / .587 |
| YKWK (Korean) | 635 / 335 | 52.8% | 160 / 135 | 54.2% | .167 / .451 / .243  | .746 / .295 |
| ZHAA (Arabic) | 686 / 280 | 40.8% | 131 / 128 | 50.6% | .115 / .304 / .167  | .768 / .311 |

Full per-phone conditional AUC and score quantile summaries remain in the ignored analysis output.

**Offsets:** the earlier TASK-04 report noted median absolute human-versus-forced phone start offset of 20 ms in both FP and TN groups, with no-overlap 3.94% vs 1.82%. That summary was from its earlier 2,159-FP run, not recalculated for this frozen 2,167-FP evaluation, so it is adjacent corroboration only. TASK-07.1's exact-identity boundary audit on Vietnamese had 19,360 pairs: 18,112 supported and boundary-valid; 453 supported but outside its 100 ms boundary proxy; 691 below-floor despite boundary-valid; 104 below-floor and boundary-invalid. Only 61/691 false-rejected boundary-valid rows were human-correct. These checks do not explain the false positives among the currently scored phones.

## Controlled threshold probes (retrospective, not validation)

Exploratory broad-L1 partition was deterministic SHA-256 assignment of speaker IDs and prompt/record IDs into train/development/heldout buckets; an eligible row was retained only when both assignments matched. Exact IDs are stored only at `.local/research/task-07.2-discrimination/analysis.json`. This creates disjoint sets of **speakers and prompts**, but discards most rows: train 3,163 eligible / 1,367 scored (9 speakers); dev 909 / 506 (4 speakers); heldout 1,025 / 459 (7 speakers). The heldout set has 139 reference errors, 72 of them abstained. The baseline row set and label denominator were unchanged for every candidate. Candidate cutoff(s) were selected on development rows only, maximizing F1 among observed cutoffs subject to recall >=10%; no heldout value was used for selection.

| Heldout candidate | Threshold | Eligible / scored | TP / FP / scored FN / TN | Precision | Recall incl. abstentions | F1 | FPR | Conditional ROC/PR AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen global control | 1.986756 | 1,025 / 459 | 64 / 259 / 3 / 133 | .198 | .460 | **.277** | 66.1% | .745 / .390 |
| Global threshold fit on dev | 1.874715 | 1,025 / 459 | 21 / 31 / 46 / 361 | .404 | .151 | .220 | 7.9% | .745 / .390 |
| Vowel/consonant thresholds fit on dev | 1.798219 / 1.918346 | 1,025 / 459 | 27 / 32 / 40 / 360 | .458 | .194 | .273 | 8.2% | .745 / .390 |

AUCs and coverage are identical because the scored IDs and continuous scores do not change. The tuned thresholds lower FPR by flagging fewer phones, but neither improves heldout F1; recall falls 27–31 absolute points. The classwise candidate is based on just 191 vowel and 315 consonant scored development units (16 and 24 errors respectively), so its apparent heldout precision is especially uncertain. It is a one-factor cutoff diagnostic, not a learned phone error classifier, and is not selected for production.

Vietnamese leave-one-speaker-out, classwise threshold fitting on the other three speakers reduces pooled fold FPR from 59.1% (frozen) to 22.9%, but also reduces pooled eligible-error recall from 17.2% to 12.4% and F1 from .2109 to .1907. Fold confusion TP/FP/scored-FN/TN and F1 are: HQTV frozen 154/394/13/247, .1679 vs fitted 117/169/50/472, .1489; PNV 319/996/45/780, .2911 vs 203/321/161/1,455, .2898; THV 197/431/28/262, .2010 vs 153/212/72/481, .1803; TLV 142/346/9/212, .1657 vs 113/138/38/420, .1530. The per-fold thresholds vary from 1.9346–1.9408 for vowels and 1.9616–1.9698 for consonants. By pooled class, frozen vs fitted FPR is 80.1% vs 20.9% for vowels (eligible/scored 7,843/1,750; TP/FP/FN/TN 253/1,184/19/294 vs 147/309/125/1,169) and 44.9% vs 24.2% for consonants (12,293/2,825; 559/983/76/1,207 vs 439/531/196/1,659). Candidate classwise F1 falls from .1748 to .1537 on vowels and .2326 to .2074 on consonants. The per-fold threshold-fitted heldout F1 values are HQTV .1489, PNV .2898, THV .1803, TLV .1530 (versus frozen .1679, .2911, .2010, .1657). The fold speakers share all 150 prompt IDs, so this analysis is speaker-disjoint but prompt-exposed and remains purely exploratory. No candidate survives as a justified recovery. We did not fit a multifeature model: existing row evidence contains the GOPT score and phone label, but no new legal per-phone GOP feature matrix on the exact eligible set was needed to establish that simple cutoff changes do not support deployment.

## Cause assessment

| Cause | Status | Evidence and limit |
|---|---|---|
| Frozen cutoff is poorly transferred to this corpus/domain and creates many false positives | **Confirmed as an operating-point fact; causal mechanism probable** | 2,167/3,668 scored-correct Vietnamese phones fall below cutoff; threshold-fit heldout probes reduce FPR while recall/F1 fall. EpaDB-development cutoff did not use L2-ARCTIC labels. This does not prove a single transfer mechanism. |
| Error and correct score distributions overlap around threshold | **Confirmed** | Quantiles above; score remains useful for ranking but not cleanly separable at the current cutoff. |
| Vowels and some phone classes have systematic score offsets / model weakness | **Probable** | Vietnamese vowel FPR 80.11%, consonant 44.89%, with large phone-to-phone variation and similar vowel/consonant pattern in other L1. Descriptive only; no causal or untouched-sample inference. |
| Acoustic-domain / accent mismatch (LibriSpeech ASR/GOPT features applied to L2-ARCTIC accented English) | **Probable** | Documented training/inference domain difference and cross-L1 score behavior; no matched native-domain causal experiment. |
| Whole-word support gate causes low coverage | **Confirmed for coverage, not for scored-phone FPs** | TASK-07.1 found all 649 failed recordings stop at `alignment_support`; Vietnamese coverage is 22.72%. Relaxing the support gate has no predictions in the full cached benchmark and cannot explain existing scored-phone FP labels. |
| Lexicon variant, stress stripping, or annotation/phone identity variant is the main FP cause | **Unsupported as primary cause** | TASK-07.1 found no Vietnamese OOVs, but first-listed variants differ for 152 canonical sequences across 2,437 tokens. Effect was not measured by label-independent alternative rescoring. Existing normalization strips ARPAbet stress digits as designed; no evidence justifies changing that semantics. |
| Kaldi output node was misused, scores clipped, or row mapping shifted | **Unsupported / contradicted by checks** | Code and regression tests require raw `output` for GOP feature mass/ratio features, `output-xent` for alignment/support, matching indexed phone keys, GOPT feature order/dimensions and finite unbounded outputs. TASK-07.1 validated all 351 completed/partial cached outputs; max output-xent log-sum-exp residual .00018117. No new contrary evidence found. |
| False positives mainly come from human-label noise or timing offsets | **Unsupported as primary cause** | L2-ARCTIC labels use manual unchanged/substitution/deletion/addition tiers and corrected boundaries. Earlier offset note had similar 20 ms medians among FP/TN; current exact run was not re-derived. Label disagreements cannot be resolved from score rows. |
| Support-gate selection explains the conditional FPR | **Unsupported as a sole explanation** | Gate affects which phones have scores and therefore conditions FPR, but the scored set itself still has high FP rates across classes, speakers and L1s. No score exists for most withheld units; counterfactual FP rate cannot be inferred. |

No causal attribution is claimed from exposed descriptive data. Current evidence supports a practical distinction: false positives among scored units are chiefly a threshold/separation and domain-transfer problem; abstention/coverage is separately dominated by the conservative alignment-support gate. They should not be conflated.

## Open-weight model review

| Candidate | Verified checkpoint / task fit | Rights | Size / CPU | Evidence and disposition |
|---|---|---|---|---|
| Existing GOPT (LibriSpeech) | Official [GOPT repository](https://github.com/YuanGongND/gopt) provides `pretrained_models/gopt_librispeech/best_audio_model.pth`, a phone-level pronunciation score head consuming 84 GOP features plus canonical phone sequence. The repository describes phone-level regression and its pretrained scores; model granularity fits. | Repository code is BSD-3-Clause; [SpeechOcean762](https://www.openslr.org/101/) and the [LibriSpeech corpus](https://www.openslr.org/12/) are CC BY 4.0, and Kaldi describes its [M13 LibriSpeech model](https://kaldi-asr.org/models.html) as available for any purpose. The public checkpoint is present in the BSD-3-Clause repository, but no separate checkpoint-specific grant was found; treat redistribution or commercial checkpoint use as unverified until that weight-level right is confirmed. | Official GitHub file metadata: **119 KB**; local checkpoint: 121,411 bytes. Current end-to-end pipeline's measured peak is 358 MiB median / 384 MiB max across 351 runs; this includes Kaldi/PyTorch and is not model-only RAM. A phone-level score is directly supported. | Authors report .616 phone-level PCC on SpeechOcean762 for this checkpoint family, but selection used test metrics and this is exposed/non-independent. Current exact-unit benchmark above is the relevant local evidence. Already present locally; **not downloaded in TASK-07.2 and not separately rebenchmarked**. [Official license](https://github.com/YuanGongND/gopt/blob/master/LICENSE), [checkpoint](https://github.com/YuanGongND/gopt/blob/master/pretrained_models/gopt_librispeech/best_audio_model.pth). |

The following additional models were checked against the task's size, licensing, output granularity, Apple Silicon/CPU, and scientific-comparability gates. Listed sizes are source metadata, not locally benchmarked measurements.

| Candidate | Task-specific capability and provenance | License and data rights | Size / local execution | Decision |
|---|---|---|---|---|
| [Whisper base.en phoneme-CTC + GOP regressor](https://huggingface.co/vb223/whisper-base-en-phoneme-ctc) | Published phoneme-CTC recognition head and separate trained GOP regressor, with expected-phone ARPAbet regression scores; not merely a generic encoder. Author-reported phone PCC 0.606 on SpeechOcean762, **not** binary mispronunciation F1 or Vietnamese-L1 results. | Model card declares MIT and uses SpeechOcean762 (CC BY 4.0); upstream model/checkpoint and mirror provenance need a full rights review before commercial reuse. | [File tree](https://huggingface.co/vb223/whisper-base-en-phoneme-ctc/tree/main): approximately 95.6 MB total, including 95.3 MB safetensors and a serialized Python joblib regressor. CPU should be technically possible but **peak RAM, Apple Silicon compatibility, latency not measured**. Deserializing untrusted joblib/pickle may execute code. | Size cap passes, but serialized regressor security review, exact phone mapping, dependency contract, independent error-discrimination evidence and fair local benchmark remain unresolved. **Not downloaded / not benchmarked**. |
| [OpenPronounce](https://github.com/Halleck45/OpenPronounce) | MIT-licensed application using phoneme CTC and IPA alignment/heuristics; phone/word diagnostics are not an independently validated exact-phone error classifier. | App MIT; component checkpoint and training-artifact licenses require separate verification. | Documents two Wav2Vec2 checkpoints of about **1.2 GB each**; CPU support advertised, Apple Silicon resource usage not measured. | Exceeds the sub-1-GB candidate download limit and task-specific comparability is unverified. **Not downloaded**. |
| [WavLM-Large phoneme scorer](https://huggingface.co/Jianshu001/wavlm-phoneme-scorer) | Task-specific phone scoring head evaluated in children learning English; no adult Vietnamese-English validation found. | Hub declares MIT; dataset rights/weight provenance need independent confirmation. | Model card reports approximately **1.2 GB** weights; memory, CPU/MPS performance unmeasured. | Exceeds download budget and has training-domain mismatch. **Not downloaded**. |
| [SSL-FT-PRON](https://huggingface.co/haeylee/ssl_ft_pron) | Collection of fine-tuned speech encoders and CTC recognition heads; no independently verified ready-to-use binary phone-error scoring head for this exact task. | Each checkpoint and dataset requires its own rights verification. | Specific checkpoint/download size and resource use unverified. | Not demonstrated to be a drop-in phone-error scorer. **Not downloaded**. |

A generic wav2vec2, WavLM, HuBERT, or XLS-R encoder is not a functioning phone-error grader without a validated scoring head. A compact **task-specific checkpoint does exist**, but no alternative has demonstrated a reliable F1/FPR advantage on identical eligible phoneme units. Accordingly no new model, dependency, or runtime was integrated, and no untrusted serialized weights were loaded.

## Operational constraints and limitations

Keep `score_pronunciation.py` production behavior as-is: `output` is the GOP feature branch; `output-xent` is for alignment/support; the support floor stays -6; threshold stays 1.986756; outputs stay unbounded; no human truth is available to inference. No label-specific threshold, selective mapping, or abstention accounting change is recommended. The cached model has phone-level granularity, but one overall target-phone classification cannot establish pronunciation trait or Pearson item scores.

The descriptive partitions and Vietnamese folds are retrospective because corpus, published outcomes, and prior reports are exposed. The broad holdout has only seven speakers, 139 error labels and 459 scored units; Vietnamese has four speakers. No confidence intervals were added to these small exploratory slices. Per-phone support and errors are uneven; rates with small denominators are unstable. The benchmark only scores exact identity rows; failures and support abstentions remain in the eligible denominator and missed-error count. Acoustic offset findings are partly from an earlier run and cannot establish the current FP mechanism.

## Next smallest task

**TASK-07.3: preregister one exact-ID, speaker- and prompt-disjoint Vietnamese-L1 phone-error evaluation on a legally cleared, previously unused corpus, freezing the baseline and decision rule before labels are opened.** Keep production unchanged until that independent result exists.

## Reproduction, verification, and files

Analysis command: `.venv/bin/python .local/research/task-07.2-discrimination/analyze.py`. Restricted aggregate tables, partition membership, candidate thresholds and metrics are stored in ignored `.local/research/task-07.2-discrimination/analysis.json`; the script reads only frozen local evidence and does not score audio. No unit-level labels, audio, row predictions, checkpoint binaries or trained model parameters were added to Git.

Requested verification: `.venv/bin/python -m unittest discover -s tests -q` — **PASS** — 137 tests ran, 1 skipped, 0 failures. No dependencies were added. Git publication is audited separately from this scientific report.

Tracked file list:

- `results/task_07_2_discrimination_recovery.md`
