from __future__ import annotations

import copy
import unittest

import benchmark_evaluation as evaluation


def pronunciation_fixture():
    return {
        "schema_version": evaluation.SCHEMA_VERSION,
        "evaluation_id": "synthetic-pron-v1", "scorer_id": "baseline",
        "task": "pronunciation", "unit": "phoneme", "partition": "test",
        "dataset": {
            "name": "synthetic-fixture", "version": "1", "license": "synthetic",
            "label_origin": "synthetic test labels", "applicability": "implementation test only",
            "exposure": {"training": False, "model_selection": False, "development": False, "test": False},
        },
        "label": {"kind": "binary", "name": "pronunciation_error", "positive_label": True, "negative_label": False},
        "threshold": 0.5, "prediction_scale": [0, 1], "threshold_source_partition": "development",
        "higher_score_is_positive": True,
        "speaker_partitions": {"train": ["train-1"], "development": ["dev-1"], "test": ["s1", "s2"]},
        "rows": [
            {"unit_id": "u1", "speaker_id": "s1", "reference": True, "prediction": 0.9},
            {"unit_id": "u2", "speaker_id": "s1", "reference": False, "prediction": 0.1},
            {"unit_id": "u3", "speaker_id": "s2", "reference": True, "prediction": 0.4},
            {"unit_id": "u4", "speaker_id": "s2", "reference": False, "prediction": None, "prediction_status": "component_failure"},
        ],
    }


class BenchmarkEvaluationTests(unittest.TestCase):
    def test_binary_metrics_and_denominators_match_hand_calculation(self):
        source = pronunciation_fixture()
        report = evaluation.evaluate(source)
        self.assertNotIn("prediction_status", source["rows"][0])
        result = report["metrics"]
        self.assertEqual(result["eligible_n"], 4)
        self.assertEqual(result["scored_n"], 3)
        self.assertEqual(result["abstained_n"], 0)
        self.assertEqual(result["component_failure_n"], 1)
        self.assertEqual(result["confusion_matrix"], {
            "tp": 1, "fp": 0, "fn_scored": 1, "tn": 1,
            "abstained_positive": 0, "abstained_negative": 1,
        })
        self.assertEqual(result["false_positive_rate_denominator_n"], 1)
        self.assertEqual(result["false_negative_rate_including_abstentions_denominator_n"], 2)
        self.assertEqual(result["roc_auc_scored_negative_n"], 1)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall_including_abstentions"], 0.5)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(result["roc_auc_conditional"], 1.0)
        self.assertAlmostEqual(result["pr_auc_conditional"], 1.0)
        self.assertEqual(result["false_positive_rate"], 0.0)

    def test_zero_denominators_are_null_and_abstentions_remain_explicit(self):
        data = pronunciation_fixture()
        data["rows"] = [{"unit_id": "u1", "speaker_id": "s1", "reference": True, "prediction": None}]
        result = evaluation.evaluate(data)["metrics"]
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["false_positive_rate"])
        self.assertIsNone(result["roc_auc_conditional"])
        self.assertEqual(result["recall_including_abstentions"], 0.0)
        self.assertEqual(result["coverage"], 0.0)

    def test_invalid_score_scale_and_test_selected_threshold_fail_closed(self):
        data = pronunciation_fixture()
        data["prediction_scale"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "prediction_scale"):
            evaluation.validate_evaluation(data)
        data = pronunciation_fixture()
        data["threshold_source_partition"] = "test"
        with self.assertRaisesRegex(ValueError, "development"):
            evaluation.validate_evaluation(data)

    def test_unbounded_native_regression_scores_are_explicit_and_not_clipped(self):
        data = pronunciation_fixture()
        data["rows"][0]["prediction"] = 2.04
        with self.assertRaisesRegex(ValueError, "outside prediction_scale"):
            evaluation.validate_evaluation(data)
        data["prediction_scale_behavior"] = "unbounded_regression"
        report = evaluation.evaluate(data)
        self.assertEqual(report["prediction_scale"], [0.0, 1.0])
        self.assertEqual(report["prediction_scale_behavior"], "unbounded_regression")
        self.assertEqual(data["rows"][0]["prediction"], 2.04)

    def test_speaker_overlap_and_undeclared_speaker_are_rejected(self):
        data = pronunciation_fixture()
        data["speaker_partitions"]["train"].append("s1")
        with self.assertRaisesRegex(ValueError, "leakage"):
            evaluation.validate_evaluation(data)
        data = pronunciation_fixture()
        data["rows"][0]["speaker_id"] = "not-in-test"
        with self.assertRaisesRegex(ValueError, "not declared"):
            evaluation.validate_evaluation(data)

    def test_comparison_requires_identical_units_labels_dataset_and_partition(self):
        baseline = pronunciation_fixture()
        candidate = copy.deepcopy(baseline)
        candidate["scorer_id"] = "candidate"
        candidate["threshold"] = 0.4
        candidate["rows"][0]["prediction"] = None
        report = evaluation.compare(baseline, candidate)
        self.assertEqual(report["paired_eligible_n"], 4)
        self.assertEqual(report["candidate"]["metrics"]["coverage"], 0.5)
        candidate["dataset"]["version"] = "2"
        with self.assertRaisesRegex(ValueError, "not comparable"):
            evaluation.compare(baseline, candidate)
        candidate = copy.deepcopy(baseline)
        candidate["rows"].pop()
        with self.assertRaisesRegex(ValueError, "unit IDs"):
            evaluation.compare(baseline, candidate)

    def test_speaker_cluster_interval_and_report_are_reproducible(self):
        data = pronunciation_fixture()
        data["rows"] = [
            {"unit_id": "a", "speaker_id": "s1", "reference": True, "prediction": 0.9},
            {"unit_id": "b", "speaker_id": "s1", "reference": False, "prediction": 0.1},
            {"unit_id": "c", "speaker_id": "s2", "reference": True, "prediction": 0.8},
            {"unit_id": "d", "speaker_id": "s2", "reference": False, "prediction": 0.2},
        ]
        first, second = evaluation.evaluate(data), evaluation.evaluate(data)
        self.assertEqual(first, second)
        interval = first["metrics"]["roc_auc_speaker_bootstrap_95"]
        self.assertEqual(interval["clusters"], 2)
        self.assertEqual(interval["valid_replicates"], 1000)

    def test_estimated_cost_requires_assumptions_and_is_not_reported_as_measured(self):
        data = pronunciation_fixture()
        data["operational"] = {"latency_ms": {"end_to_end": 120, "asr": 70}, "peak_memory_bytes": 1024,
                               "unit_cost": {"kind": "estimated", "amount": 0.01, "currency": "USD", "assumptions": ["provider list price", "one request per utterance"]}}
        report = evaluation.evaluate(data)
        self.assertEqual(report["operational"]["unit_cost"]["kind"], "estimated")
        self.assertEqual(report["operational_summary"]["failure_denominator_n"], 4)
        data["operational"]["unit_cost"].pop("assumptions")
        with self.assertRaisesRegex(ValueError, "assumptions"):
            evaluation.validate_evaluation(data)

    def test_fluency_statistics_obey_declared_scale_and_tolerance(self):
        data = pronunciation_fixture()
        data.update({"task": "oral_fluency", "unit": "utterance"})
        data.pop("threshold")
        data.pop("threshold_source_partition")
        data.pop("prediction_scale")
        data.pop("higher_score_is_positive")
        data["label"] = {"kind": "continuous", "name": "oral_fluency", "scale": [0, 5]}
        data["tolerance"] = 1
        data["rows"] = [
            {"unit_id": "u1", "speaker_id": "s1", "reference": 1, "prediction": 1.5},
            {"unit_id": "u2", "speaker_id": "s1", "reference": 3, "prediction": 2.5},
            {"unit_id": "u3", "speaker_id": "s2", "reference": 5, "prediction": 4},
        ]
        report = evaluation.evaluate(data)
        metrics = report["metrics"]
        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertAlmostEqual(metrics["rmse"], (1.5 / 3) ** 0.5)
        self.assertEqual(metrics["within_tolerance_n"], 3)
        self.assertAlmostEqual(metrics["pearson_r"], 0.9933992678)
        self.assertEqual(report["project_targets"]["source"], "project-proposed, not Pearson-certified")

    def test_speaking_skill_gate_uses_10_to_90_scale(self):
        data = pronunciation_fixture()
        data.update({"task": "speaking_skill", "unit": "utterance"})
        for key in ("threshold", "threshold_source_partition", "prediction_scale", "higher_score_is_positive"):
            data.pop(key)
        data["label"] = {"kind": "continuous", "name": "speaking_skill", "scale": [10, 90]}
        data["tolerance"] = 5
        data["rows"] = [{"unit_id": "u1", "speaker_id": "s1", "reference": 60, "prediction": 64}]
        targets = evaluation.evaluate(data)["project_targets"]
        self.assertEqual(targets["within_tolerance_tolerance"], 5)
        self.assertEqual(targets["mae_max"], 4)


if __name__ == "__main__":
    unittest.main()
