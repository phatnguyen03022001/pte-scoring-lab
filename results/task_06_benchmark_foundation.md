# TASK-06 — Benchmark foundation and acceptance gates

## Verdict

**Engineering: PASS for the shared evaluation contract. Scientific validation: NOT VERIFIED.** The contract computes reproducible descriptive metrics from row-level, licensed labels and scorer outputs. It does not alter scoring behavior, calibrate a score, or turn existing proxy results into Pearson PTE evidence.

**PEARSON_AGREEMENT_NOT_VERIFIED.** The tracked evidence has no licensed Pearson PTE outcomes. The project-proposed targets below are internal acceptance gates and are not Pearson standards.

## Reusable evidence and limits

| Evidence | Label and unit | Sample | Exposure / interpretation |
|---|---|---:|---|
| SpeechOcean762 / archived GOPT output | Expert aggregate scores, 0–10, utterance; separate phone and word arrays use their own source scales | 2,500 test utterances; corpus has 5,000 utterances / 250 speakers | GOPT checkpoint development/selection used test metrics. Reproduction and rubric compatibility only, not independent validation, Pearson agreement, or current audio-pipeline performance. See `results/SOURCES.md` and `results/gopt_speechocean762.json`. |
| QUALITY-08 EpaDB | Human target-phone correct/incorrect labels, binary, phoneme | 60 recordings / 30 speakers; 10 development and 20 holdout speakers; 210 timestamp-matched labels in integrity summary | CC BY-NC 4.0 research only. Used to develop/freeze a recovery policy and select a diagnostic threshold. Historical holdout is no longer blind to project iteration. |
| QUALITY-09 EpaDB | Source `+` / `-` target-phone labels, binary, phoneme | 76 recordings / 19 Argentinian Spanish-L1 speakers; 1,786 eligible labels (1,337 correct / 449 incorrect) | CC BY-NC 4.0, non-commercial research only. Speaker-disjoint from Q08; not tuned on Q09 in the recorded protocol, but subsequent project research used its results. Retrospective cross-speaker diagnostic, not an untouched final set or Vietnamese/PTE validation. |
| L2-ARCTIC v5.0 | Manual phone annotation; annotation tiers distinguish unchanged, substitution, deletion and insertion | Local dataset files are present; benchmark protocol selects all manual-annotation recordings for four Vietnamese-L1 speakers and a deterministic subset for the other 20 speakers | CC BY-NC 4.0. Local protocol requires exact phone identity alignment, excludes insertion from target-phone metrics, and records no local evidence of prior exposure. No TASK-06 result is asserted from this corpus. |
| Pearson PTE ground truth | None available in tracked evaluation evidence | 0 verified examples | No Pearson-referenced validation or score mapping. |

Historical Q09 pronunciation figures (not recomputed by TASK-06) are explicitly denominator-bearing: baseline coverage 1,008 / 1,786 = 56.44%; conditional ROC-AUC 0.719 on the 1,008 exact-identity scored phones; conditional error PR-AUC 0.366 on those same scored phones. Coverage-aware recall and false-negative rate count all 449 source error labels, including abstentions; precision uses predicted error flags as its denominator, and F1 is the harmonic mean of precision and recall. Recovery coverage is 1,334 / 1,786 = 74.69%; its conditional AUCs use a different scored subset. Its Q09 report provides 19-speaker cluster bootstrap intervals. These are not fresh TASK-06 test runs and do not establish scorer improvement or PTE agreement.

The historical Vietnamese-L1 pronunciation coverage of 22.72%, precision 0.2726, recall 0.1720, F1 0.2109, FPR 59.08% and conditional ROC-AUC 0.7600 were supplied as historical references. No current, traceable row-level artifact was found that would justify recomputing those values here; they remain unverified. The four Vietnamese speakers in L2-ARCTIC are not enough to claim broad Vietnamese population validity.

## Contract and use

`benchmark_evaluation.py` accepts `pte-scoring-evaluation-v1` JSON and emits a JSON report without modifying its input. Pronunciation inputs are one row per phoneme, word or utterance, with a binary reference, native numeric score, explicit score scale, threshold, and development-only threshold-selection declaration. Oral Fluency trait and speaking-skill inputs use utterance rows with numeric references and predictions on the same declared scale; outputs include MAE, RMSE, Pearson, Spearman and agreement within a declared tolerance. `prediction: null` is an explicit abstention by default; `prediction_status: "component_failure"` distinguishes pipeline failures. Percentages include scored/eligible denominators, and metrics return `null` when a denominator or class support is zero.

Every evaluation records dataset identity/version, label origin, license, applicability, exposure flags, source partition, and train/development/test speaker lists. Overlapping speaker partitions, undeclared speakers, mixed units, invalid scales, duplicate IDs, and out-of-scale scores are rejected. Paired comparison requires identical eligible unit IDs, labels, speakers, dataset, version, task, unit, partition and label definition. It permits scorer-specific score scales and development-selected thresholds while retaining each scale, threshold and coverage in its report. Speaker bootstrap intervals resample speakers with a fixed seed; fewer than two speakers yields no interval. Optional latency, peak memory and unit-cost fields stay explicitly attributed; estimates require declared assumptions and are never labeled measured.

Example: `.venv/bin/python benchmark_evaluation.py candidate.json --compare baseline.json --output comparison.json`. Keep restricted audio, labels and row-level outputs in ignored `.local/`; do not commit them. WER remains a transcript/reference content metric and is not pronunciation quality. Fluency timing/acoustic descriptors without reliable ratings are descriptive only. Operational latency, memory and cost must be measured on the target setup; the shared contract makes no cost estimate.

## Project-proposed acceptance gates (not Pearson-certified)

| Evaluation level | Project target |
|---|---|
| Pronunciation trait 0–5 | At least 85% within ±1; MAE ≤0.5 |
| Oral Fluency trait 0–5 | At least 85% within ±1; MAE ≤0.5 |
| Speaking skill 10–90 | At least 85% within ±5; MAE ≤4 |
| Scoring coverage | At least 95%, with abstention and denominator reported |
| Reliability | Report speaker-level variation and uncertainty where speaker sample supports it |

Correlation is association, not percentage accuracy or agreement. SpeechOcean762, CEFR/SANDI, vendor claims, and passing software tests do not establish Pearson PTE item agreement.

## TASK-07 readiness

The contract supports a fair paired comparison of Kaldi + GOPT, SpeechAce, or Azure Pronunciation Assessment **if** each scorer can produce comparable row-level outputs on the same licensed, frozen audio and labels, with a documented mapping to the same task/unit and development-only threshold selection. Engineering support is present. No API credentials or billable-request authorization are established by TASK-06. Pearson-compatible labels are unavailable. No third-party calls were made.
