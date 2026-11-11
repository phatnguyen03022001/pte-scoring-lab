# TASK-07 — Independent Pronunciation Scorer Benchmark

**Verified:** 2026-11-11
**Scope:** pronunciation scoring only
**Best candidate:** `INSUFFICIENT_EVIDENCE`

## Verdict

- **Engineering: PASS.** A fingerprinted L2-ARCTIC v5.0 protocol, traceable human phone labels, row-level local GOPT evaluation, TASK-06 contract report, vendor feasibility review, and regression tests are complete. The benchmark is retrospective and uses cached scorer outputs where inputs were exactly equivalent; seven changed-reference records were rescored with current code.
- **Scientific scorer comparison: BLOCKED.** GOPT is measured, but neither vendor could be evaluated under the authorized conditions. No paired multi-scorer metrics or uncertainty estimate exist.
- **Pearson validation: `PEARSON_AGREEMENT_NOT_VERIFIED`.** No independent Pearson-scored recordings or predefined Pearson agreement protocol are available.
- **Selection: `INSUFFICIENT_EVIDENCE`.** GOPT’s fixed operating point has low Vietnamese-L1 coverage and weak threshold metrics. Vendor feature lists and self-reported claims do not justify a replacement without valid paired results.

## Frozen benchmark and evidence

The official L2-ARCTIC v5.0 corpus is available locally. Its 7,515,454,814-byte archive SHA-256 is `490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda`. The release has 24 non-native English speakers across six L1 groups; the source describes approximately 150 manually annotated utterances per speaker and CC BY-NC 4.0 terms. The archive checksum, extraction manifest, all 24 speaker inventories, audio and annotation fingerprints were checked before scoring. [Official corpus and license](https://psi.engr.tamu.edu/l2-arctic-corpus/)

Selection was frozen before current-task rescoring: all 150 manually annotated recordings for each of the four Vietnamese-L1 speakers (HQTV, PNV, THV, TLV; 600 recordings), plus 20 deterministic SHA-256-selected recordings for each of the other 20 speakers (400; 1,000 total). Original human `phones` TextGrid annotations are ground truth; forced-alignment tiers are not. Human substitution (`s`) and deletion (`d`) labels are errors; unchanged labels are correct; insertion (`a`) events are counted but excluded from target-phone scoring because there is no target phone. Ambiguous `ERR` substitutions remain errors and are separately counted. Unlabeled/silence intervals are excluded. The evaluation aligns canonical target-phone identities exactly; mismatch/gap units abstain instead of being shifted positionally.

| Manual annotation count | All L1 | Vietnamese L1 |
|---|---:|---:|
| Correct target phones | 26,947 | 15,415 |
| Substitutions | 4,617 | 3,155 |
| Deletions | 1,815 | 1,566 |
| Eligible target phones | 33,379 | 20,136 |
| Insertions, excluded from target-phone metrics | 293 | 198 |
| Unlabeled/silence intervals, excluded | 4,607 | 2,927 |
| Ambiguous perceived phone (`ERR`, subset of substitutions) | 37 | 2 |

This is a retrospective test, not a blind holdout: prior project work used the corpus and historical reports included benchmark findings. The current threshold, 1.986756, remains the existing threshold selected on QUALITY-08 EpaDB development speakers; it was not fitted on L2-ARCTIC. No local project history records L2-ARCTIC model training/checkpoint selection, but external exposure cannot be disproved. The four Vietnamese speakers are too few to represent Vietnamese learners broadly. The corpus is noncommercial under CC BY-NC 4.0; audio, row labels, protocol IDs and row-level predictions remain in ignored `.local/` artifacts and are not included here.

Other inspected data: SpeechOcean762-derived GOPT feature/output artifacts are local, with CC BY 4.0 attribution, but the checkpoint was selected using its test metrics and its 2,500 utterance outputs use author-provided GOP features rather than this live audio pipeline. EpaDB QUALITY-08 (60 recordings / 30 speakers) supplied the development threshold; its historical holdout has since been exposed to project iteration. QUALITY-09 (76 recordings / 19 Argentinian Spanish-L1 speakers) is a retrospective cross-speaker diagnostic, not Vietnamese evidence. No Pearson-labeled response-level ground truth was found. See [TASK-06 data inventory](task_06_benchmark_foundation.md) and [dataset sources](SOURCES.md).

## GOPT results

The benchmark uses the frozen default GOPT pipeline and support floor (−6.0). It flags an error when the native phone score is at or below 1.986756; higher scores indicate more likely correctness. Scores are not calibrated probabilities. The contract now records the scorer’s documented unbounded regression behavior: 2,091 of 10,044 scored phones exceed its nominal 0–2 target range, and none were clipped. All denominators below are phone-level unless marked as recordings.

| Metric | All L1 | Vietnamese L1 |
|---|---:|---:|
| Eligible / scored / abstained | 33,379 / 10,044 / 23,335 | 20,136 / 4,575 / 15,561 |
| Coverage | 30.09% | 22.72% |
| Reference errors / correct | 6,432 / 26,947 | 4,721 / 15,415 |
| TP / FP / scored FN / TN | 1,361 / 4,973 / 196 / 3,514 | 812 / 2,167 / 95 / 1,501 |
| Abstained correct / errors (missed) | 18,460 / 4,875 | 11,747 / 3,814 |
| Precision (TP / predicted errors) | 0.2149 (1,361 / 6,334) | 0.2726 (812 / 2,979) |
| Recall incl. abstentions (TP / all reference errors) | 0.2116 (1,361 / 6,432) | 0.1720 (812 / 4,721) |
| F1 | 0.2132 | 0.2109 |
| False-positive rate on scored correct phones | 0.5860 (4,973 / 8,487) | 0.5908 (2,167 / 3,668) |
| Conditional ROC-AUC, scored phones only | 0.7486 (1,557 error / 8,487 correct) | 0.7600 (907 error / 3,668 correct) |
| Conditional PR-AUC, scored phones only | 0.4078 | 0.4902 |

All-L1 speaker-cluster percentile bootstrap, 2,000 resamples over 24 speakers: ROC-AUC 95% CI 0.7284–0.7668; PR-AUC 0.3290–0.4598; coverage-aware F1 0.1854–0.2455. These intervals describe this pooled, retrospective cross-L1 sample, not Vietnamese population performance. No Vietnamese confidence interval is reported: four speaker clusters are inadequate. Conditional AUCs exclude abstentions; coverage-aware recall counts every abstained reference error as missed.

| Vietnamese speaker | Eligible / scored | Coverage | Precision | Recall | F1 | FPR | Conditional ROC-AUC / PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| HQTV | 5,038 / 808 | 16.04% | 0.2810 | 0.1198 | 0.1679 | 0.6147 | 0.7582 / 0.4489 |
| PNV | 5,031 / 2,140 | 42.54% | 0.2426 | 0.3637 | 0.2911 | 0.5608 | 0.7493 / 0.4257 |
| THV | 5,030 / 918 | 18.25% | 0.3137 | 0.1479 | 0.2010 | 0.6219 | 0.7440 / 0.5527 |
| TLV | 5,037 / 709 | 14.08% | 0.2910 | 0.1158 | 0.1657 | 0.6201 | 0.8036 / 0.5836 |

The operational record contains 333 `SCORING_COMPLETED`, 18 `SCORING_COMPLETED_PARTIAL`, and 649 `ALIGNMENT_INVALID` recordings (64.9%); each of the 649 failed the unchanged whole-word acoustic support gate. The TASK-06 row contract records 22,214 target-phone units in failed utterances as component failures and another 1,121 as abstentions from alignment identity/support gaps. Across all units, 10,776 had exact phone-identity matches and 22,603 had mismatch/gap alignments. No threshold or support-floor changes were tried.

Recorded scorer processing time for 351 completed/partial utterances (excludes Python startup): median 2.991 s, p95 3.508 s, max 5.571 s. Stage medians: preprocessing 0.082 s; Kaldi features/i-vector 0.744 s; acoustic posterior 0.667 s; alignment 0.356 s; GOP extraction 0.060 s. Peak resident memory across process and children: median 375,062,528 bytes (358 MiB), p95 390,193,152 (372 MiB), max 402,554,880 (384 MiB), n=351. The current-task run reused 993 cache rows only where the old and current normalized reference tokens were identical, and rescored the seven changed references. Three now score; four still fail alignment. A successful Vietnamese rescore repeated with identical status, 31 phone rows and scores (maximum absolute difference 0.0). Local inference cost was $0; vendor calls were zero. These checks support scorer-output reproducibility for the repeated case, not accuracy.

Restricted row-level data and reproducible contract artifacts: `.local/research/task-07-pronunciation-benchmark/{protocol.json,evaluation.json,evaluation-report.json,report.json,metadata.json,runs/}`. The tracked evaluation report aggregates one row per eligible human target phone using `pte-scoring-evaluation-v1`.

## Vendor evidence and execution status

Documentation and published pricing were checked on 2026-11-11. Vendor claims are product descriptions, not independent accuracy evidence for this dataset or Pearson PTE.

| Candidate | Official capability and output fit | Cost, operational and privacy notes | Benchmark status |
|---|---|---|---|
| SpeechAce | Score Text handles known scripts up to 45 seconds and returns word, syllable and phoneme quality scores; fluency, lexical stress and intonation are optional. Score Word returns phoneme scores on a 0–100 scale with phone/extent fields. Published language choices are English US/UK, French and Spanish; no Vietnamese-English accent-specific validation was found. These outputs might support a phone-level comparison, but exact mapping to L2-ARCTIC phone identities and alignment is untested. [Score Text](https://api-docs.speechace.com/api-reference/score-text), [Score Word](https://api-docs.speechace.com/api-reference/score-word), [plans](https://www.speechace.com/api-plans/). | Basic: $40/month, 5,000 15-second requests, 30-second maximum; Pro: $80/month, 10,000 requests, 45-second maximum; Premium: $125/month, 10,000 requests, 2-minute maximum. Published overage is $0.008 per 15 seconds for Basic/Pro and $0.0125 for Premium. A free trial is advertised but requires an API key/account; no minimum duration was found in reviewed docs. Audio is not used to train their models; default retention is 90 days and processing/storage stays in the selected cloud region. | **BLOCKED.** No credentials. More decisively, current API terms bar use to test/evaluate another pronunciation system (section 202) and bar publishing benchmark/comparative results without prior written approval (section 203). No request was sent and no account/trial was created. [Privacy](https://www.speechace.com/api-privacy-policy/), [terms](https://www.speechace.com/api-terms-of-service/). |
| Azure Pronunciation Assessment | Scripted reference-text assessment returns phoneme-, word-, syllable- and utterance-level accuracy (FivePoint 0–5 or HundredMark 0–100); fluency/completeness and miscue are available, with prosody as an add-on. SDK and short-audio REST options exist. REST is limited to 30 seconds; longer audio uses continuous mode, where built-in miscue is unsupported and insertions/omissions require comparison with reference text. There are 33 target-language locales including English dialects and Vietnamese; `vi-VN` means Vietnamese is a target being learned, not a Vietnamese-accented-English model. Prosody is only available for `en-US`. [Assessment docs](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-pronunciation-assessment), [REST docs](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-speech-to-text-short), [locales](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support). | Baseline assessment is billed at Standard Speech-to-Text rates, per audio second (one-second increments); prosody costs extra. F0 lists 5 free hours/month for real-time Standard/Custom Speech-to-Text, shared; the pricing page does not clearly declare a distinct F0 pronunciation-assessment entitlement. Regional pay-as-you-go prices displayed as `$-`, so a monetary estimate cannot be verified without a region/offer calculator. Realtime pronunciation-assessment audio is not retained/stored by Microsoft; the customer remains responsible for permissions/consent. Product use is under Azure subscription terms; Microsoft’s online-services terms allow competitive benchmarking subject to a request for replication details/access. | **BLOCKED.** No Azure resource/key/region is configured, the SDK is absent, and TASK-07 forbids live calls without approval. No calls, accounts, trials or charges. Whether its phone symbols/alignments can be paired to the manual corpus is unverified. [Pricing](https://azure.microsoft.com/en-us/pricing/details/speech/), [privacy](https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/speech-service/speech-to-text/data-privacy-security), [Product Terms](https://www.microsoft.com/licensing/terms/product/ForallOnlineServices). |

The SpeechAce plans page advertises a 0.9 Pearson correlation/Kappa claim, but provides no public sample, method, confidence interval or PTE-scored recording evidence on that page. It is vendor-reported and is not Pearson agreement evidence for this project.

| Candidate | Same audio/labels/phone units | Scored / eligible | Paired metrics / latency / cost |
|---|---|---|---|
| Kaldi + GOP + GOPT | Yes; frozen retrospective L2-ARCTIC protocol | 10,044 / 33,379 phones overall; 4,575 / 20,136 Vietnamese | Measured above; local cost $0 |
| SpeechAce | No output | Not evaluated | None; blocked by terms and missing credentials |
| Azure | No output | Not evaluated | None; blocked by missing resource credentials and no live-call authorization |

Thus there is no valid paired comparison of at least two scorers, and no claim that either commercial service is more accurate, more robust to Vietnamese accents, or more cost-effective for this project.

## Root-cause findings

Confirmed observations are separated from causal explanations; no scorer threshold, support gate or model setting was tuned on this evaluation.

| Rank | Finding | Evidence and limit | Smallest useful follow-up |
|---|---|---|---|
| 1 | **Confirmed coverage bottleneck: whole-word acoustic support gate.** | 649/1,000 utterances fail at the fixed −6.0 support floor with `ALIGNMENT_INVALID`; Vietnamese speaker coverage ranges 14.08–42.54%. This establishes the failure mechanism, not that lowering the floor would improve validity. | On separate development speakers, audit rejected alignments against audio and labels; compare only a preregistered gate policy. |
| 2 | **Confirmed phone attribution bottleneck.** | 22,603/33,379 target units are mismatch/gap rather than exact phone identity; such units correctly abstain under the frozen rule. Cause may involve real mispronunciation, alignment, lexicon or phone mapping. | Adjudicate a bounded sample against the manual boundaries; do not positionally map. |
| 3 | **Vietnamese generalization is a concern, not a causal result.** | Only four Vietnamese speakers; coverage and threshold metrics are poor and vary widely. There is no controlled same-speaker/condition accent comparison. | Acquire speaker-disjoint Vietnamese learner recordings with independent labels. |
| 4 | **Acoustic-model/checkpoint domain mismatch is plausible, unverified.** | Current Kaldi chain model and GOPT checkpoint are documented for LibriSpeech ([pinned GOPT source](https://github.com/YuanGongND/gopt/tree/bed909daf8eca035095871e51642525acc5b9b55), [Kaldi GOP recipe](https://github.com/kaldi-asr/kaldi/blob/master/egs/gop_speechocean762/README.md)); L2-ARCTIC is non-native read English. High invalid-alignment rate is consistent with mismatch but does not identify its cause. | Compare a compatible model only on development data with all other pipeline components frozen. |
| 5 | **Lexicon variant selection is a plausible source.** | Existing code chooses the lexicon’s first pronunciation variant. Outputs warn for 1,251 word instances in 340 utterances; 592 instances in 157 Vietnamese utterances. No variant counterfactual was run. | Evaluate variant alternatives on a separate development set, without selecting variants from the final annotations. |
| 6 | **Threshold mismatch is plausible, not tested.** | The Q08-development threshold produces low precision, recall and high FPR; conditional ROC-AUC is higher but only among scored units. This does not establish a better threshold. | Tune a single operating point on new development speakers and freeze it before a new test. |
| 7 | **GOP feature compatibility and checkpoint limits remain uncertain.** | The pipeline uses unnormalized raw chain output for GOP features, documented as uncalibrated; the checkpoint is LibriSpeech-domain. No controlled feature-source/checkpoint ablation was run. | Check the upstream feature contract and test one factor at a time on separate development audio. |
| 8 | **Historical reference normalization defect was corrected before TASK-07.** | A leading-elision apostrophe regex changed normalized references in seven recordings; three now score and four still fail alignment. It explains a small set of prior invalid references, not overall weak performance. | Keep the normalization regression test; no broader normalization changes are justified here. |
| 9 | **ASR attribution was not a cause in this measurement.** | Scoring uses the known corpus prompt as reference, not production ASR text. Any ASR-dependent error contribution to the integrated pipeline was not measured. | Evaluate integrated ASR+scoring separately against the same independently labeled target set. |
| 10 | **Annotation/unit mismatch remains a limit.** | Manual target phones define the error labels; insertions are not target-phone units, ambiguous `ERR` is retained, and exact identity matching is required. No annotation defect was demonstrated. | Preserve these semantics and add independently adjudicated labels if broader error taxonomy is needed. |

## TASK-08 readiness and Pearson status

**TASK-08 Content Scoring: NO-GO for a scoring-quality experiment now.** The inspected corpora provide phone labels or exposed speech ratings, not an independent content-score rubric/response-level gold set suitable for validating an open-weight LLM. Reconsider only after defining the content construct, obtaining independent human ratings, and freezing a speaker/prompt-disjoint evaluation protocol. TASK-08 was not started.

**`PEARSON_AGREEMENT_NOT_VERIFIED`** — no independent Pearson-scored learner recordings or accepted agreement analysis were available.
