"""Summarize TASK-03 evidence without rerunning model inference.

EpaDB-derived output is written below .local/ because the source sample is
licensed for non-commercial research only. The script consumes verified local
reports and the original five-rater SpeechOcean762 score file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / ".local" / "research" / "task-03-evaluation-report.md"
SPEECHOCEAN_ROOT = ROOT / ".local" / "datasets" / "speechocean762" / "speechocean762"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _icc_absolute_agreement(rows: list[list[float]]) -> float | None:
    """Two-way random-effects, absolute-agreement ICC(2,1)."""
    rows = [row for row in rows if len(row) >= 2 and all(math.isfinite(x) for x in row)]
    if len(rows) < 2:
        return None
    raters = len(rows[0])
    if any(len(row) != raters for row in rows):
        raise ValueError("ICC rows must have the same number of raters.")
    subjects = len(rows)
    grand = sum(sum(row) for row in rows) / (subjects * raters)
    subject_means = [sum(row) / raters for row in rows]
    rater_means = [sum(row[j] for row in rows) / subjects for j in range(raters)]
    ms_subject = raters * sum((mean - grand) ** 2 for mean in subject_means) / (subjects - 1)
    ms_rater = subjects * sum((mean - grand) ** 2 for mean in rater_means) / (raters - 1)
    residual = sum(
        (row[j] - subject_means[i] - rater_means[j] + grand) ** 2
        for i, row in enumerate(rows) for j in range(raters)
    )
    ms_error = residual / ((subjects - 1) * (raters - 1))
    denominator = ms_subject + (raters - 1) * ms_error + raters * (ms_rater - ms_error) / subjects
    return (ms_subject - ms_error) / denominator if denominator else None


def _cluster_bootstrap_icc(
    rows: list[tuple[str, list[float]]], *, seed: int, replicates: int = 1000,
) -> dict[str, Any]:
    groups: dict[str, list[list[float]]] = defaultdict(list)
    for speaker, ratings in rows:
        groups[speaker].append(ratings)
    speakers = sorted(groups)
    if len(speakers) < 2:
        return {"low_95": None, "high_95": None, "valid_replicates": 0, "clusters": len(speakers)}
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sample = []
        for _ in speakers:
            sample.extend(groups[rng.choice(speakers)])
        value = _icc_absolute_agreement(sample)
        if value is not None and math.isfinite(value):
            values.append(value)
    values.sort()
    return {
        "low_95": values[int(0.025 * (len(values) - 1))] if values else None,
        "high_95": values[int(0.975 * (len(values) - 1))] if values else None,
        "valid_replicates": len(values), "clusters": len(speakers),
    }


def _speaker_map() -> dict[str, str]:
    result = {}
    for split in ("train", "test"):
        with (SPEECHOCEAN_ROOT / split / "utt2spk").open(encoding="utf-8") as source:
            for line in source:
                utterance, speaker = line.split()
                result[utterance] = speaker
    return result


def _speechocean_rater_agreement() -> dict[str, Any]:
    detail = _load_json(SPEECHOCEAN_ROOT / "resource" / "scores-detail.json")
    speakers = _speaker_map()
    fields = ("accuracy", "fluency", "prosodic", "total")
    output = {}
    for field_index, field in enumerate(fields):
        rows = [(speakers[utterance], entry[field]) for utterance, entry in detail.items()]
        valid = [(speaker, values) for speaker, values in rows if len(values) == 5]
        output[field] = {
            "utterances": len(valid), "speakers": len({speaker for speaker, _ in valid}),
            "raters": 5,
            "icc_2_1_absolute_agreement": _icc_absolute_agreement([values for _, values in valid]),
            "speaker_cluster_bootstrap_95": _cluster_bootstrap_icc(valid, seed=3000 + field_index),
        }
    return output


def _q09_threshold_sensitivity(threshold: float) -> list[tuple[float, dict[str, Any]]]:
    """Evaluate a fixed neighborhood around the dev-selected threshold; do not select on Q09."""
    import quality_09_benchmark as q09

    manifest_path = q09.DATA_ROOT / "sample_manifest.json"
    manifest = _load_json(manifest_path)
    records = [{**record, "partition": "holdout"} for record in manifest["records"]]
    items = q09.collect_phone_items(q09.DATA_ROOT, records, "baseline")
    return [
        (point, q09.threshold_metrics(items, point)["coverage_aware"])
        for point in sorted({threshold - 1.0, threshold - 0.5, threshold, threshold + 0.5, threshold + 1.0})
    ]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report() -> str:
    so_path = ROOT / "results" / "gopt_speechocean762.json"
    q08 = _load_json(ROOT / "results" / "quality_08_report.json")
    q09 = _load_json(ROOT / "results" / "quality_09_report.json")
    so = _load_json(so_path)
    q08_parts = q08["results"]["partitions"]
    q08_holdout = q08_parts["holdout"]["frozen_baseline_vs_candidate"]
    q09_metrics = q09["results"]["pronunciation"]
    diagnostic_threshold = float(q09["evaluation_protocol"]["threshold"]["threshold"])
    paired_common = q09_metrics["paired_common_identity_sample"]
    q08_integrity = q08["verification"]["sample_integrity"]
    q09_inventory = q09["dataset_inventory"]["primary_evaluation"]
    q08_baseline = q08_holdout["frozen_baseline"]["phone_label_discrimination"]
    q08_candidate = q08_holdout["candidate"]["phone_label_discrimination"]
    q08_asr = q08_holdout["frozen_baseline"]["asr_vs_manual_spoken_word_tier"]
    q09_baseline_threshold = q09_metrics["baseline"]["threshold_metrics"]
    q09_confusion = q09_baseline_threshold["conditional_confusion_matrix"]
    q09_runtime = q09["results"]["runtime"]
    disagreement = q09["ground_truth_quality"]["interannotator_agreement"]
    baseline_speakers = q09_metrics["baseline"]["per_speaker"]
    speaker_aucs = [row["roc_auc_correct_vs_incorrect"] for row in baseline_speakers.values() if row["roc_auc_correct_vs_incorrect"] is not None]
    speaker_coverages = [row["threshold_metrics"]["coverage"] for row in baseline_speakers.values()]

    lines = [
        "# TASK-03 — Independent Benchmark, Generalization & Scoring Validity",
        "",
        "Generated from the local verified reports and source annotations by `python3 task_03_evaluation.py`. No scorer/model inference is run.",
        "",
        "## Verdict",
        "",
        "**EVALUATION ENGINEERING: PASS for retrospective analysis.** Existing metrics are reused with explicit scales, partitions, coverage, speaker-cluster uncertainty, and annotation disagreement. The SpeechOcean and EpaDB artifacts were verified in the TASK-02 acquisition report; restricted source audio and EpaDB-derived details stay local.",
        "",
        "**EXPERT SCORE AGREEMENT: PARTIAL.** GOPT outputs agree to varying degrees with SpeechOcean762's 0–10 expert dimensions on checkpoint-exposed test features. This is a reproduction/diagnostic, not independent generalization or PTE agreement.",
        "",
        "**INDEPENDENT GENERALIZATION: BLOCKED BY DATA AVAILABILITY.** EpaDB supports retrospective cross-speaker diagnostics for Argentinian Spanish-L1 English, but its outcomes were already used in project research/iteration. No genuinely untouched evaluation set or Vietnamese-L1 corpus is available.",
        "",
        "**PEARSON PTE AGREEMENT: NOT VERIFIED.** No Pearson-scored learner recordings or compatible 10–90 human outcomes exist.",
        "",
        "## Exposure classification",
        "",
        "| Dataset/evidence | Model training exposure | Model-selection exposure | Project development/evaluation exposure | Permitted interpretation |",
        "|---|---|---|---|---|",
        "| SpeechOcean762 (5,000 labeled utterances; 250 speakers) | Exposed: released GOPT checkpoint training used corpus data | Exposed: checkpoint selection used its test metrics | Extensively reproduced in QUALITY-05 and prior research | Compatibility and descriptive 0–10 rubric reproduction only; not independent final validation |",
        f"| EpaDB QUALITY-08 ({q08_integrity['recordings']} recordings; {q08_integrity['speakers']} speakers) | No evidence that these recordings trained GOPT | Q08 development ({q08_parts['development']['frozen_baseline_vs_candidate']['frozen_baseline']['speakers']} speakers) selected candidate behavior; Q08 holdout evaluated after freeze | Both partitions were subsequently used in project research/iteration | Bounded retrospective analysis; neither partition is a fresh TASK-03 blind set |",
        f"| EpaDB QUALITY-09 ({q09_inventory['recordings']} recordings; {q09_inventory['evaluation_speakers']} speakers) | No evidence that these recordings trained GOPT | One diagnostic threshold selected only on Q08 development | Q09 outcomes subsequently informed project research/iteration | Speaker-disjoint retrospective cross-speaker diagnostic only |",
        "| L2-ARCTIC / Cambridge Speak & Improve | Not acquired | Not used | Access remains blocked under TASK-02 | Do not bypass restrictions |",
        "| Pearson-scored learner audio | Unavailable | Unavailable | Unavailable | No Pearson PTE agreement evidence |",
        "",
        "Speaker and exact-audio duplication checks in the existing manifests found no overlap for the historical Q08/Q09 split construction. These checks do not establish project-level blindness, exclude near-duplicates/session overlap beyond recorded metadata, or make published checkpoint test data independent.",
        "",
        "## Existing model results",
        "",
        "### SpeechOcean762 (checkpoint-exposed features; 0–10 rubric)",
        "",
        "Reported utterance totals below are descriptive on 2,500 test utterances per model. The stored bootstrap CIs resample utterances, not speakers, because speaker IDs are absent from the supplied feature arrays; treat them as too-narrow if within-speaker dependence is material.",
        "",
        "| GOPT checkpoint | Component | n | Pearson r | Spearman ρ | MAE | RMSE | Existing 95% CI for total Pearson |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model in so["models"]:
        total = model["utterance"]["total"]
        ci = total.get("utterance_bootstrap_95pct_ci", {}).get("pearson_r")
        ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}] (utterance bootstrap)" if ci else "not stored"
        for component in ("accuracy", "fluency", "prosody", "total"):
            m = model["utterance"][component]
            lines.append(
                f"| {model['model']} | {component} | {model['test_utterances']} | "
                f"{_fmt(m.get('pearson_r'))} | {_fmt(m.get('spearman_r'))} | "
                f"{_fmt(m.get('mae_0_to_10'))} | {_fmt(m.get('rmse_0_to_10'))} | "
                f"{ci_text if component == 'total' else '—'} |"
            )
    lines.extend([
        "",
        "For the LibriSpeech checkpoint, stored phone-level correlation is 0.618 (n=47,369; 0–2 source phone scale); word accuracy correlation is 0.538 and stress correlation is 0.325 (n=15,967 words). These are separate label levels, not a combined pronunciation score. This feature-only reproduction does not validate the current end-to-end audio pipeline. Fluency/prosody comparisons are descriptive on the matching SpeechOcean 0–10 dimensions; applicability to PTE remains unverified.",
        "",
        "The SpeechOcean GOPT correlations above use the supplied processed `scores.json` aggregate, not the five original rater values directly. TASK-02 records a source-file variant/hash discrepancy for that aggregate; original rater arrays remain in `scores-detail.json` and are used separately for the ICC analysis. Treat the correlation table as the archived reproduction tied to its recorded source variant.",
        "",
        "### EpaDB retrospective pronunciation evidence (local, non-commercial research)",
        "",
        "Labels mean source target-phone `+`/`−`, evaluated only after exact phone-identity sequence alignment. Unscored phones remain abstentions. Baseline and recovery are different scoring coverages; point differences across their conditionally scored subsets do not establish improvement.",
        "",
        "| Policy | Eligible labels | Scored / coverage | Correct / error labels | Conditional ROC-AUC [speaker-cluster 95% CI] | Error PR-AUC [95% CI] | Coverage-aware precision / recall / F1 [F1 95% CI] | FPR / FNR incl. abstentions |",
        "|---|---:|---:|---:|---|---|---|---|",
    ])
    for policy in ("baseline", "recovery"):
        metric = q09_metrics[policy]
        auc_ci = metric["roc_auc_speaker_cluster_bootstrap_95"]
        pr_ci = metric["pr_auc_speaker_cluster_bootstrap_95"]
        f1_ci = metric["coverage_aware_f1_speaker_cluster_bootstrap_95"]
        cov = metric["threshold_metrics"]["coverage_aware"]
        lines.append(
            f"| {policy} | {metric['eligible_label_rows']} | {metric['exact_identity_scored_rows']} / {metric['coverage']:.1%} | "
            f"{metric['correct_label_rows']} / {metric['incorrect_label_rows']} | "
            f"{metric['roc_auc_correct_vs_incorrect_conditional']:.3f} [{auc_ci['low_95']:.3f}, {auc_ci['high_95']:.3f}] | "
            f"{metric['pr_auc_error_conditional']:.3f} [{pr_ci['low_95']:.3f}, {pr_ci['high_95']:.3f}] | "
            f"{cov['precision']:.3f} / {cov['recall_all_reference_errors']:.3f} / {cov['f1']:.3f} "
            f"[{f1_ci['low_95']:.3f}, {f1_ci['high_95']:.3f}] | "
            f"{cov['false_positive_rate_on_scored_correct_phones']:.3f} / {cov['false_negative_rate_including_abstained_errors']:.3f} |"
        )
    lines.extend([
        "",
        f"The single diagnostic threshold ({diagnostic_threshold:.6f}) was selected by maximum coverage-aware F1 using only {len(q09['evaluation_protocol']['threshold']['development_speakers'])} Q08 development speakers. On Q09, baseline confusion counts were {q09_confusion['true_error']} true-error flags, {q09_confusion['false_error']} false-error flags, {q09_confusion['missed_scored_error']} scored errors missed, and {q09_confusion['correct_rejection']} correct rejections; another {q09_baseline_threshold['abstained_error_counted_as_missed']} error labels were abstained and counted as misses. This indicates weak precision and incomplete coverage. The recovery policy increases coverage but its conditional AUC is on a different subset; paired AUC over the shared {paired_common['phone_count']} phone identities is {paired_common['recovery_roc_auc'] - paired_common['baseline_roc_auc']:.3f} (speaker-cluster 95% CI [{paired_common['roc_auc_delta_recovery_minus_baseline_speaker_cluster_bootstrap_95']['low_95']:.3f}, {paired_common['roc_auc_delta_recovery_minus_baseline_speaker_cluster_bootstrap_95']['high_95']:.3f}]). No model-quality improvement is claimed.",
        "",
        f"The Q09 source-label set is class imbalanced ({q09_metrics['baseline']['incorrect_label_rows']}/{q09_metrics['baseline']['eligible_label_rows']} error labels, {q09_metrics['baseline']['incorrect_label_rows'] / q09_metrics['baseline']['eligible_label_rows']:.1%}). Across {len(baseline_speakers)} speakers, baseline per-speaker conditional AUC ranged from {_fmt(min(speaker_aucs))} to {_fmt(max(speaker_aucs))}; exact-phone score coverage ranged from {_fmt(min(speaker_coverages) * 100, 1)}% to {_fmt(max(speaker_coverages) * 100, 1)}%. Speaker-level values and counts remain available in the local Q09 report.",
        "",
        "The following fixed ±1.0 native-score neighborhood is a post-hoc sensitivity display around the threshold selected on Q08 development. Q09 values are not used to pick or recommend a threshold; uncertainty on these operating points is not newly estimated.",
        "",
        "| Q09 baseline threshold | Precision | Recall (all eligible errors) | F1 | FPR | FNR incl. abstentions |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for point, metrics in _q09_threshold_sensitivity(diagnostic_threshold):
        lines.append(
            f"| {point:.6f} | {metrics['precision']:.3f} | {metrics['recall_all_reference_errors']:.3f} | "
            f"{metrics['f1']:.3f} | {metrics['false_positive_rate_on_scored_correct_phones']:.3f} | "
            f"{metrics['false_negative_rate_including_abstained_errors']:.3f} |"
        )
    lines.extend([
        "",
        f"Q08 holdout comprised {q08_holdout['frozen_baseline']['recordings']} recordings / {q08_holdout['frozen_baseline']['speakers']} speakers and {q08_holdout['frozen_baseline']['boundary_alignment']['matched_or_substituted_phone_pairs']} manually annotated phone boundaries. Historical baseline phone-label AUC was {q08_baseline['gopt_native_phone_score_auc']:.3f} ({q08_baseline['scored_exact_identity_label_rows']} exact-identity rows; 95% speaker-cluster CI [{q08_baseline['speaker_cluster_bootstrap_95']['low_95']:.3f}, {q08_baseline['speaker_cluster_bootstrap_95']['high_95']:.3f}]); candidate AUC was {q08_candidate['gopt_native_phone_score_auc']:.3f} ({q08_candidate['scored_exact_identity_label_rows']} rows; [{q08_candidate['speaker_cluster_bootstrap_95']['low_95']:.3f}, {q08_candidate['speaker_cluster_bootstrap_95']['high_95']:.3f}]). This previously used holdout is retrospective. On the same recordings, ASR had {q08_asr['substitutions']} substitutions, {q08_asr['deletions']} deletions, {q08_asr['insertions']} insertion{'s' if q08_asr['insertions'] != 1 else ''} over {q08_asr['reference_words']} manual spoken words: WER {q08_asr['wer']:.3f} (95% speaker-cluster CI [{q08_asr['speaker_cluster_bootstrap_95']['low_95']:.3f}, {q08_asr['speaker_cluster_bootstrap_95']['high_95']:.3f}]). This measures ASR transcript agreement, not pronunciation validity or prompt completion.",
        "",
        "## Human annotation reliability",
        "",
        "SpeechOcean762 retains five individual expert ratings in `scores-detail.json`; aggregate `scores.json` is not substituted for those original records. Two-way random-effects absolute-agreement ICC(2,1) is computed on each utterance-level 0–10 scale. Speaker-cluster percentile bootstrap (1,000 replicates; deterministic per dimension) accounts for repeated utterances within speaker. ICC assumes numeric score intervals and does not make raters infallible.",
        "",
        "| Dimension | Utterances | Speakers | ICC(2,1) | Speaker-cluster bootstrap 95% CI |",
        "|---|---:|---:|---:|---|",
    ])
    for dimension, stat in _speechocean_rater_agreement().items():
        ci = stat["speaker_cluster_bootstrap_95"]
        lines.append(f"| {dimension} | {stat['utterances']} | {stat['speakers']} | {stat['icc_2_1_absolute_agreement']:.3f} | [{ci['low_95']:.3f}, {ci['high_95']:.3f}] |")
    lines.extend([
        "",
        f"For the {disagreement['recordings_with_two_human_annotations']} Q09 recordings with two manual phone annotations, sequence-aligned raw exact-identity agreement was {disagreement['exact_phone_identity_agreement_rate']:.1%} over {disagreement['sequence_aligned_phone_operations']} alignment operations. Agreement is exact identity pairs divided by matches, substitutions, and gaps; it does not count each different full sequence as total disagreement. Among exact identity pairs, boundaries were within 30/60/100 ms in "
        + "/".join(f"{disagreement['exact_identity_pair_boundary_within_tolerance'][key]:.1%}" for key in ("30ms", "60ms", "100ms"))
        + ". These are raw agreement rates, not chance-corrected kappa, and no adjudicated gold was created. Only annotation_1 supplies the Q09 source +/- benchmark labels; annotation_2 is retained as disagreement evidence.",
        "",
        "## Component validity and gaps",
        "",
        "- **Phoneme correctness:** retrospective binary phone-error ranking exists for EpaDB and source phone scores exist for SpeechOcean762. These test different targets/scales and are not pooled. Exact-identity mapping excludes unsupported phones; EpaDB coverage and class imbalance are reported above.",
        "- **Word-level pronunciation:** SpeechOcean762 provides word accuracy/stress labels and correlations above. EpaDB has no compatible expert word-quality scale.",
        "- **Utterance-level pronunciation quality:** SpeechOcean762 0–10 agreement is available only on checkpoint-exposed features. There is no independent, untouched corpus with compatible holistic pronunciation labels.",
        "- **Fluency/prosody:** available as SpeechOcean762 descriptive dimensions only. QUALITY-08 fluency/silence descriptors have no compatible expert fluency reference; validity is not verified.",
        "- **ASR content:** Q08 WER uses manual spoken-word annotations; it is not prompt-reference WER and does not assign ASR errors to learner mispronunciations. Q06 documents decoding variability; no fresh ASR rerun was required for TASK-03.",
        "- **Sample-rate support:** the pronunciation path converts inputs to mono 16 kHz PCM before checking that normalized format; the verified SpeechOcean762 archive is mono 16 kHz. No rate-stratified human-validity estimate exists, and transcoding support is not evidence of scoring validity across source rates.",
        "- **Generalization:** available non-training-exposed EpaDB speakers are Spanish-L1 and project-outcome-exposed. No Vietnamese-L1 or untouched independent final holdout exists.",
        "- **Pearson PTE:** unavailable; all SpeechOcean utterance metrics use its 0–10 scales and cannot be transformed into Pearson 10–90 agreement without matched Pearson human labels and validated calibration.",
        "",
        "## Reproducibility and runtime",
        "",
        "This synthesis consumes existing files: `results/gopt_speechocean762.json`, `results/quality_08_report.json`, `results/quality_09_report.json`, and the local verified SpeechOcean762 original score records. The reported Q08/Q09 scorer runs are historical. Existing report protocols use 2,000 deterministic speaker-cluster replicates for EpaDB model metrics; this synthesis uses 1,000 seeded cluster replicates for SpeechOcean ICC. No full-corpus inference or model process was started.",
        f"Historical Q09 per-record runtime reported by the existing benchmark: baseline {q09_runtime['baseline']['successful_recordings']}/{q09_inventory['recordings']} successful (median {_fmt(q09_runtime['baseline']['processing_time_seconds_median'], 3)} s, p95 {_fmt(q09_runtime['baseline']['processing_time_seconds_p95'], 3)} s, median RSS {_fmt(q09_runtime['baseline']['peak_memory_bytes_median'] / (1024 * 1024), 1)} MiB); recovery {q09_runtime['recovery']['successful_recordings']}/{q09_inventory['recordings']} successful (median {_fmt(q09_runtime['recovery']['processing_time_seconds_median'], 3)} s, p95 {_fmt(q09_runtime['recovery']['processing_time_seconds_p95'], 3)} s, median RSS {_fmt(q09_runtime['recovery']['peak_memory_bytes_median'] / (1024 * 1024), 1)} MiB). These are stored historical measurements, not timings from this synthesis run.",
        "",
        "Input SHA-256:",
        "",
    ])
    input_paths = [
        ROOT / "results" / "gopt_speechocean762.json",
        ROOT / "results" / "quality_08_report.json",
        ROOT / "results" / "quality_09_report.json",
        SPEECHOCEAN_ROOT / "resource" / "scores-detail.json",
        ROOT / ".local" / "datasets" / "epadb-quality-08" / "sample_manifest.json",
        ROOT / ".local" / "datasets" / "quality-09-epadb-new-speakers" / "sample_manifest.json",
    ]
    for path in input_paths:
        lines.append(f"- `{path.relative_to(ROOT)}`: `{_sha256(path)}`")
    lines.extend([
        "",
        f"TASK-02 integrity evidence: SpeechOcean762 5,000 labeled/transcribed/speaker-linked utterances, 250 speakers, 2,500/2,500 official split; 245 unreferenced archive WAVs excluded. EpaDB Q08: {q08_integrity['recordings']} recordings/{q08_integrity['speakers']} speakers; Q09: {q09_inventory['recordings']} recordings/{q09_inventory['evaluation_speakers']} speakers and {q09_inventory['audio_and_annotations']['raw_files']} source/annotation files. EpaDB material and this derived report remain under local `.local/` storage for non-commercial research use.",
        "",
        "## Decision",
        "",
        "**TASK-03 evaluation engineering: PASS for authorized retrospective scope.** Existing scoring behavior was not changed. **Independent final validation: BLOCKED BY DATA AVAILABILITY.** **TASK-04 readiness: CONDITIONAL GO** only for a narrowly scoped plan that does not claim Pearson agreement or independent generalization and first secures an authorized, genuinely untouched evaluation source. L2-ARCTIC and Cambridge remain access blocked. Stop here; TASK-04–10 are not started.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote local TASK-03 report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
