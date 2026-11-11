"""Validated, scorer-neutral evaluation for pronunciation labels and trait scores.

Input is a JSON object following ``pte-scoring-evaluation-v1``. Predictions are
kept on their source scale; this module never calibrates or changes a scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "pte-scoring-evaluation-v1"
UNITS = {"phoneme", "word", "utterance"}
TASKS = {"pronunciation", "oral_fluency", "speaking_skill"}
EXPOSURE_FIELDS = ("training", "model_selection", "development", "test")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number.")
    return float(value)


def validate_evaluation(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and return a copy; malformed or incomparable metadata fails closed."""
    if not isinstance(data, dict):
        raise ValueError("Evaluation must be a JSON object.")
    data = json.loads(json.dumps(data))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}.")
    for key in ("evaluation_id", "scorer_id", "task", "unit", "dataset", "partition", "label", "rows"):
        if key not in data:
            raise ValueError(f"Missing required field: {key}.")
    if not isinstance(data["task"], str) or not isinstance(data["unit"], str) or data["task"] not in TASKS or data["unit"] not in UNITS:
        raise ValueError("task or unit is unsupported.")
    if data["task"] in {"oral_fluency", "speaking_skill"} and data["unit"] != "utterance":
        raise ValueError("oral_fluency and speaking_skill evaluations must use utterance units.")
    dataset = data["dataset"]
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be an object.")
    for key in ("name", "version", "license", "label_origin", "applicability", "exposure"):
        if key not in dataset or dataset[key] in (None, ""):
            raise ValueError(f"dataset.{key} is required.")
    exposure = dataset["exposure"]
    if not isinstance(exposure, dict) or any(not isinstance(exposure.get(field), bool) for field in EXPOSURE_FIELDS):
        raise ValueError("dataset.exposure must declare boolean training, model_selection, development and test fields.")
    label = data["label"]
    if not isinstance(label, dict):
        raise ValueError("label must be an object.")
    if label.get("kind") == "binary":
        if (data["task"] != "pronunciation" or "positive_label" not in label or "negative_label" not in label
                or isinstance(label["positive_label"], (list, dict)) or isinstance(label["negative_label"], (list, dict))
                or label["positive_label"] == label["negative_label"]):
            raise ValueError("Binary pronunciation labels must declare distinct positive_label and negative_label values.")
        label["scale"] = [0, 1]
        threshold = _finite_number(data.get("threshold"), "threshold")
        score_scale = data.get("prediction_scale")
        if not isinstance(score_scale, list) or len(score_scale) != 2:
            raise ValueError("prediction_scale must declare [minimum, maximum] for pronunciation scores.")
        score_low, score_high = (_finite_number(v, "prediction_scale") for v in score_scale)
        if score_low >= score_high or not score_low <= threshold <= score_high:
            raise ValueError("prediction_scale must be increasing and contain the threshold.")
        if data.get("threshold_source_partition") != "development":
            raise ValueError("Pronunciation threshold must be selected on the development partition.")
        if not isinstance(data.get("higher_score_is_positive"), bool):
            raise ValueError("higher_score_is_positive must be boolean for pronunciation evaluations.")
        data["threshold"] = threshold
        data["prediction_scale"] = [score_low, score_high]
    elif label.get("kind") == "continuous":
        if data["task"] not in {"oral_fluency", "speaking_skill"}:
            raise ValueError("Continuous ratings are supported for oral_fluency or speaking_skill evaluations.")
        scale = label.get("scale")
        if not isinstance(scale, list) or len(scale) != 2:
            raise ValueError("label.scale must contain [minimum, maximum].")
        low, high = (_finite_number(v, "label.scale") for v in scale)
        if low >= high:
            raise ValueError("label.scale minimum must be less than maximum.")
        tolerance = _finite_number(data.get("tolerance"), "tolerance")
        if tolerance < 0:
            raise ValueError("tolerance cannot be negative.")
        data["label"]["scale"] = [low, high]
        data["tolerance"] = tolerance
    else:
        raise ValueError("label.kind must be 'binary' or 'continuous'.")

    partitions = data.get("speaker_partitions")
    if partitions is None or not isinstance(partitions, dict):
        raise ValueError("speaker_partitions must explicitly declare train, development and test speakers.")
    speaker_sets = {}
    for part in ("train", "development", "test"):
        speakers = partitions.get(part)
        if not isinstance(speakers, list) or any(not isinstance(s, str) or not s for s in speakers):
            raise ValueError(f"speaker_partitions.{part} must be a list of speaker IDs.")
        if len(speakers) != len(set(speakers)):
            raise ValueError(f"Duplicate speaker in {part} partition.")
        speaker_sets[part] = set(speakers)
    for left, right in (("train", "development"), ("train", "test"), ("development", "test")):
        overlap = speaker_sets[left] & speaker_sets[right]
        if overlap:
            raise ValueError(f"Speaker leakage between {left} and {right}: {sorted(overlap)}.")
    if data["partition"] not in {"development", "test"}:
        raise ValueError("partition must be 'development' or 'test'.")
    rows = data["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list.")
    ids: set[str] = set()
    allowed_speakers = speaker_sets[data["partition"]]
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or not isinstance(row.get("unit_id"), str) or not row["unit_id"]
                or not isinstance(row.get("speaker_id"), str) or not row["speaker_id"]):
            raise ValueError(f"rows[{index}] requires unit_id and speaker_id.")
        if row["unit_id"] in ids:
            raise ValueError(f"Duplicate unit_id: {row['unit_id']}.")
        ids.add(row["unit_id"])
        if row["speaker_id"] not in allowed_speakers:
            raise ValueError(f"Speaker {row['speaker_id']} is not declared in the {data['partition']} partition.")
        reference = row.get("reference")
        prediction = row.get("prediction")
        status = row.get("prediction_status", "scored" if prediction is not None else "abstained")
        if status not in {"scored", "abstained", "component_failure"}:
            raise ValueError(f"rows[{index}].prediction_status is unsupported.")
        if (prediction is None) == (status == "scored"):
            raise ValueError(f"rows[{index}] prediction and prediction_status disagree.")
        row["prediction_status"] = status
        if label["kind"] == "binary":
            if reference not in (label["positive_label"], label["negative_label"]):
                raise ValueError(f"rows[{index}].reference is not a declared binary label.")
            if prediction is not None:
                row["prediction"] = _finite_number(prediction, f"rows[{index}].prediction")
                if not score_low <= row["prediction"] <= score_high:
                    raise ValueError(f"rows[{index}].prediction is outside prediction_scale.")
        else:
            low, high = label["scale"]
            ref = _finite_number(reference, f"rows[{index}].reference")
            if not low <= ref <= high:
                raise ValueError(f"rows[{index}].reference is outside label.scale.")
            row["reference"] = ref
            if prediction is not None:
                pred = _finite_number(prediction, f"rows[{index}].prediction")
                if not low <= pred <= high:
                    raise ValueError(f"rows[{index}].prediction is outside label.scale.")
                row["prediction"] = pred
    operational = data.get("operational", {})
    if not isinstance(operational, dict):
        raise ValueError("operational must be an object when supplied.")
    latency = operational.get("latency_ms")
    if latency is not None:
        if not isinstance(latency, dict) or "end_to_end" not in latency:
            raise ValueError("operational.latency_ms must include measured end_to_end latency.")
        for name, value in latency.items():
            if _finite_number(value, f"operational.latency_ms.{name}") < 0:
                raise ValueError("Operational latency cannot be negative.")
    memory = operational.get("peak_memory_bytes")
    if memory is not None and (isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0):
        raise ValueError("operational.peak_memory_bytes must be a positive measured integer.")
    cost = operational.get("unit_cost")
    if cost is not None:
        if not isinstance(cost, dict) or cost.get("kind") not in {"measured", "estimated"}:
            raise ValueError("operational.unit_cost must declare kind as measured or estimated.")
        if _finite_number(cost.get("amount"), "operational.unit_cost.amount") < 0 or not cost.get("currency"):
            raise ValueError("operational.unit_cost requires a non-negative amount and currency.")
        if cost["kind"] == "estimated" and (not isinstance(cost.get("assumptions"), list) or not cost["assumptions"]):
            raise ValueError("Estimated unit cost requires explicit assumptions.")
    return data


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels))
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(label for _, label in ordered[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _pr_auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    if not positives or positives == len(labels):
        return None
    ordered = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    previous_recall, previous_precision, area, index = 0.0, 1.0, 0.0, 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        tp += sum(int(label) for _, label in ordered[index:end])
        fp += end - index - sum(int(label) for _, label in ordered[index:end])
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * (precision + previous_precision) / 2
        previous_recall, previous_precision, index = recall, precision, end
    return area


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2
        for offset in range(index, end):
            ranks[order[offset]] = rank
        index = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    mean_l, mean_r = sum(left) / len(left), sum(right) / len(right)
    cov = sum((x - mean_l) * (y - mean_r) for x, y in zip(left, right))
    denom = math.sqrt(sum((x - mean_l) ** 2 for x in left) * sum((y - mean_r) ** 2 for y in right))
    return cov / denom if denom else None


def _bootstrap_speakers(rows: list[dict[str, Any]], metric, *, seed: int, replicates: int = 1000) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["speaker_id"]].append(row)
    speakers = sorted(groups)
    if len(speakers) < 2:
        return {"low_95": None, "high_95": None, "valid_replicates": 0, "clusters": len(speakers)}
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sample = [row for _ in speakers for row in groups[rng.choice(speakers)]]
        value = metric(sample)
        if value is not None and math.isfinite(value):
            values.append(value)
    values.sort()
    return {
        "low_95": values[int(0.025 * (len(values) - 1))] if values else None,
        "high_95": values[int(0.975 * (len(values) - 1))] if values else None,
        "valid_replicates": len(values), "clusters": len(speakers),
    }


def _pronunciation_metrics(data: dict[str, Any], *, include_per_speaker: bool = True) -> dict[str, Any]:
    rows = data["rows"]
    positive = data["label"]["positive_label"]
    factor = 1 if data["higher_score_is_positive"] else -1
    scored = [row for row in rows if row["prediction"] is not None]
    labels = [row["reference"] == positive for row in scored]
    scores = [factor * row["prediction"] for row in scored]
    tp = fp = fn = tn = abstained_positive = abstained_negative = 0
    for row in rows:
        if row["prediction"] is None:
            if row["reference"] == positive:
                abstained_positive += 1
            else:
                abstained_negative += 1
            continue
        pred_positive = factor * row["prediction"] >= factor * data["threshold"]
        actual_positive = row["reference"] == positive
        tp += int(pred_positive and actual_positive)
        fp += int(pred_positive and not actual_positive)
        fn += int(not pred_positive and actual_positive)
        tn += int(not pred_positive and not actual_positive)
    precision = tp / (tp + fp) if tp + fp else None
    recall_denominator = tp + fn + abstained_positive
    recall = tp / recall_denominator if recall_denominator else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0 if precision is not None and recall == 0 else None
    return {
        "eligible_n": len(rows), "scored_n": len(scored), "abstained_n": sum(row["prediction_status"] == "abstained" for row in rows),
        "component_failure_n": sum(row["prediction_status"] == "component_failure" for row in rows),
        "coverage": len(scored) / len(rows), "coverage_denominator_n": len(rows),
        "positive_n": sum(row["reference"] == positive for row in rows),
        "negative_n": sum(row["reference"] != positive for row in rows),
        "threshold": data["threshold"], "confusion_matrix": {
            "tp": tp, "fp": fp, "fn_scored": fn, "tn": tn,
            "abstained_positive": abstained_positive, "abstained_negative": abstained_negative,
        },
        "precision_denominator_n": tp + fp,
        "recall_including_abstentions_denominator_n": recall_denominator,
        "false_positive_rate_denominator_n": fp + tn,
        "false_negative_rate_including_abstentions_denominator_n": sum(row["reference"] == positive for row in rows),
        "roc_auc_scored_positive_n": sum(labels), "roc_auc_scored_negative_n": len(labels) - sum(labels),
        "pr_auc_denominator_n": len(scored),
        "precision": precision, "recall_including_abstentions": recall, "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate_including_abstentions": (fn + abstained_positive) / sum(row["reference"] == positive for row in rows) if any(row["reference"] == positive for row in rows) else None,
        "roc_auc_conditional": _auc(labels, scores),
        "pr_auc_conditional": _pr_auc(labels, scores),
        "roc_auc_speaker_bootstrap_95": _bootstrap_speakers(scored, lambda sample: _auc([r["reference"] == positive for r in sample], [factor * r["prediction"] for r in sample]), seed=6121),
        **({"per_speaker": _per_speaker(data, _pronunciation_metrics_single)} if include_per_speaker else {}),
    }


def _pronunciation_metrics_single(rows: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    subset = {**data, "rows": rows}
    return {key: value for key, value in _pronunciation_metrics(subset, include_per_speaker=False).items() if key != "roc_auc_speaker_bootstrap_95"}


def _per_speaker(data: dict[str, Any], metric) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data["rows"]:
        groups[row["speaker_id"]].append(row)
    return {speaker: metric(rows, data) for speaker, rows in sorted(groups.items())}


def _fluency_metrics(data: dict[str, Any], *, include_per_speaker: bool = True) -> dict[str, Any]:
    rows = data["rows"]
    scored = [row for row in rows if row["prediction"] is not None]

    def values(sample: list[dict[str, Any]]) -> dict[str, Any]:
        if not sample:
            return {"mae": None, "rmse": None, "within_tolerance_n": 0, "within_tolerance": None}
        errors = [row["prediction"] - row["reference"] for row in sample]
        close = sum(abs(error) <= data["tolerance"] for error in errors)
        return {"mae": sum(abs(e) for e in errors) / len(errors), "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)), "within_tolerance_n": close, "within_tolerance": close / len(errors)}

    point = values(scored)
    refs = [r["reference"] for r in scored]
    preds = [r["prediction"] for r in scored]
    return {
        "eligible_n": len(rows), "scored_n": len(scored), "abstained_n": sum(row["prediction_status"] == "abstained" for row in rows),
        "component_failure_n": sum(row["prediction_status"] == "component_failure" for row in rows),
        "coverage": len(scored) / len(rows), "coverage_denominator_n": len(rows),
        "score_scale": data["label"]["scale"], "tolerance": data["tolerance"],
        **point, "within_tolerance_denominator_n": len(scored),
        "pearson_r": _correlation(refs, preds), "spearman_rho": _correlation(_rank(refs), _rank(preds)),
        "correlation_denominator_n": len(scored),
        "mae_speaker_bootstrap_95": _bootstrap_speakers(scored, lambda sample: values(sample)["mae"], seed=6122),
        **({"per_speaker": _per_speaker(data, _fluency_metrics_single)} if include_per_speaker else {}),
    }


def _fluency_metrics_single(rows: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    subset = {**data, "rows": rows}
    result = _fluency_metrics(subset, include_per_speaker=False)
    return {key: value for key, value in result.items() if key != "mae_speaker_bootstrap_95"}


def evaluate(data: dict[str, Any]) -> dict[str, Any]:
    """Return reproducible metrics and provenance without mutating source data."""
    data = validate_evaluation(data)
    metrics = _pronunciation_metrics(data) if data["task"] == "pronunciation" else _fluency_metrics(data)
    failure_n = metrics["component_failure_n"]
    return {
        "schema_version": "pte-scoring-evaluation-report-v1",
        "evaluation_id": data["evaluation_id"], "scorer_id": data["scorer_id"],
        "task": data["task"], "unit": data["unit"], "dataset": data["dataset"],
        "partition": data["partition"], "label": data["label"],
        "prediction_scale": data.get("prediction_scale"),
        "threshold_source_partition": data.get("threshold_source_partition"),
        "higher_score_is_positive": data.get("higher_score_is_positive"),
        "operational": data.get("operational", {}),
        "exposure_limited": any(data["dataset"]["exposure"].values()),
        "independent_validation": not any(data["dataset"]["exposure"].values()),
        "project_targets": _project_targets(data, metrics), "metrics": metrics,
        "operational_summary": {"failure_n": failure_n, "failure_denominator_n": metrics["eligible_n"],
                                "failure_rate": failure_n / metrics["eligible_n"]},
    }


def _project_targets(data: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    coverage_ok = metrics["coverage"] >= 0.95
    if data["task"] == "pronunciation":
        return {"source": "project-proposed, not Pearson-certified", "scoring_coverage_min": 0.95,
                "scoring_coverage_met": coverage_ok,
                "reason": "No universal pronunciation accuracy target; declare a task-specific preregistered threshold."}
    scale = data["label"]["scale"]
    if scale == [0, 5]:
        tolerance, mae_max = 1, 0.5
    elif scale == [10, 90]:
        tolerance, mae_max = 5, 4
    else:
        return {"source": "project-proposed, not Pearson-certified", "scoring_coverage_min": 0.95,
                "scoring_coverage_met": coverage_ok, "acceptance_thresholds": None,
                "reason": "Declare project targets separately for this score scale."}
    return {"source": "project-proposed, not Pearson-certified", "scoring_coverage_min": 0.95,
            "scoring_coverage_met": coverage_ok, "within_tolerance_min": 0.85,
            "within_tolerance_tolerance": tolerance, "within_tolerance_met": metrics["within_tolerance"] is not None and metrics["within_tolerance"] >= 0.85,
            "mae_max": mae_max, "mae_met": metrics["mae"] is not None and metrics["mae"] <= mae_max}


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare same-label, same-unit, same-partition evaluations by unit identity."""
    base = validate_evaluation(baseline)
    cand = validate_evaluation(candidate)
    keys = ("task", "unit", "partition", "dataset", "label", "speaker_partitions", "tolerance")
    mismatches = [key for key in keys if base.get(key) != cand.get(key)]
    if base.get("threshold_source_partition") != cand.get("threshold_source_partition"):
        mismatches.append("threshold_source_partition")
    if mismatches:
        raise ValueError(f"Evaluations are not comparable; metadata differs: {', '.join(mismatches)}.")
    baseline_rows = {row["unit_id"]: row for row in base["rows"]}
    candidate_rows = {row["unit_id"]: row for row in cand["rows"]}
    if baseline_rows.keys() != candidate_rows.keys():
        raise ValueError("Evaluations must contain the same eligible unit IDs.")
    for unit_id in baseline_rows:
        left, right = baseline_rows[unit_id], candidate_rows[unit_id]
        if left["reference"] != right["reference"] or left["speaker_id"] != right["speaker_id"]:
            raise ValueError(f"Label or speaker differs for unit {unit_id}.")
    return {
        "schema_version": "pte-scoring-comparison-v1",
        "baseline": evaluate(base), "candidate": evaluate(cand),
        "paired_eligible_n": len(baseline_rows), "paired_speaker_n": len({r['speaker_id'] for r in baseline_rows.values()}),
        "interpretation": "Paired descriptive comparison on identical eligible units; candidate abstentions remain in coverage denominators.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path, help="JSON evaluation following pte-scoring-evaluation-v1")
    parser.add_argument("--compare", type=Path, help="Optional baseline JSON for a paired comparison")
    parser.add_argument("--output", type=Path, help="Write report JSON here; stdout when omitted")
    args = parser.parse_args()
    source = json.loads(args.evaluation.read_text(encoding="utf-8"))
    report = compare(json.loads(args.compare.read_text(encoding="utf-8")), source) if args.compare else evaluate(source)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
