import json
import unittest

import quality_08_benchmark as q08
import quality_09_benchmark as q09


class Quality09BenchmarkTests(unittest.TestCase):
    def test_source_label_direction_matches_author_rule(self):
        labels = [
            {"source_phone_id": "a", "target_phone": "K", "observed_phone": "K", "label": "+"},
            {"source_phone_id": "b", "target_phone": "T", "observed_phone": "D", "label": "-"},
        ]
        self.assertEqual(q09._source_label_semantics_conflicts(labels), [])
        labels[0]["label"] = "-"
        self.assertEqual(q09._source_label_semantics_conflicts(labels), ["a"])

    def test_gopt_score_direction_and_pr_auc_treat_low_scores_as_errors(self):
        items = [
            {"correct": True, "score": 0.9},
            {"correct": True, "score": 0.8},
            {"correct": False, "score": 0.2},
            {"correct": False, "score": 0.1},
        ]
        self.assertEqual(q09._auc_error(items), 1.0)
        self.assertEqual(q09.pr_auc_error(items), 1.0)
        inverted = [{**row, "score": 1 - row["score"]} for row in items]
        self.assertEqual(q09._auc_error(inverted), 0.0)

    def test_pr_auc_is_unavailable_without_both_label_classes(self):
        self.assertIsNone(q09.pr_auc_error([{"correct": True, "score": 0.5}]))
        self.assertIsNone(q09.pr_auc_error([{"correct": False, "score": 0.5}]))

    def test_pr_auc_groups_tied_scores_before_updating_precision_and_recall(self):
        items = [
            {"correct": True, "score": 0.1},
            {"correct": False, "score": 0.2},
            {"correct": True, "score": 0.2},
            {"correct": False, "score": 0.8},
        ]
        self.assertAlmostEqual(q09.pr_auc_error(items), 7 / 24)

    def test_abstained_errors_count_as_misses_in_coverage_aware_metrics(self):
        items = [
            {"correct": True, "score": 0.9},
            {"correct": True, "score": 0.8},
            {"correct": False, "score": 0.2},
            {"correct": False, "score": None},
        ]
        result = q09.threshold_metrics(items, threshold=0.5)
        self.assertEqual(result["coverage"], 0.75)
        self.assertEqual(result["conditional_confusion_matrix"], {
            "true_error": 1, "false_error": 0, "missed_scored_error": 0, "correct_rejection": 2,
        })
        self.assertEqual(result["abstained_error_counted_as_missed"], 1)
        self.assertEqual(result["coverage_aware"]["recall_all_reference_errors"], 0.5)
        self.assertEqual(result["coverage_aware"]["accuracy_abstentions_count_wrong"], 0.75)

    def test_threshold_is_selected_only_from_supplied_development_items(self):
        development = [
            {"speaker_id": "dev-a", "correct": True, "score": 0.9},
            {"speaker_id": "dev-a", "correct": False, "score": 0.1},
        ]
        selected = q09.choose_threshold(development)
        self.assertEqual(selected["threshold"], 0.1)
        self.assertEqual(selected["development_speakers"], ["dev-a"])

    def test_paired_comparison_uses_only_common_scored_phone_identities(self):
        baseline = [
            {"record_id": "r1", "source_phone_id": "0", "speaker_id": "s1", "score": 0.8},
            {"record_id": "r2", "source_phone_id": "1", "speaker_id": "s2", "score": 0.2},
            {"record_id": "r3", "source_phone_id": "2", "speaker_id": "s2", "score": None},
        ]
        candidate = [
            {"record_id": "r1", "source_phone_id": "0", "speaker_id": "s1", "score": 0.7},
            {"record_id": "r2", "source_phone_id": "1", "speaker_id": "s2", "score": 0.3},
            {"record_id": "r4", "source_phone_id": "3", "speaker_id": "s3", "score": 0.9},
        ]
        base_by_id = q09._paired_items(baseline)
        candidate_by_id = q09._paired_items(candidate)
        self.assertEqual(set(base_by_id) & set(candidate_by_id), {"r1:0", "r2:1"})
        interval = q09._paired_delta_ci(
            base_by_id, candidate_by_id,
            lambda rows: sum(row["score"] for row in rows) / len(rows),
            reps=50,
        )
        self.assertEqual(interval["paired_phone_count"], 2)
        self.assertEqual(interval["paired_speaker_count"], 2)
        self.assertEqual(interval["valid_replicates"], 50)

    def test_new_speaker_sample_is_disjoint_from_quality08(self):
        old_manifest = json.loads((q09.OLD_DATA_ROOT / "sample_manifest.json").read_text(encoding="utf-8"))
        old_speakers = {row["speaker_id"] for row in old_manifest["records"]}
        self.assertEqual(len(q09.SPEAKERS), len(set(q09.SPEAKERS)))
        self.assertFalse(old_speakers.intersection(q09.SPEAKERS))
        self.assertEqual(q09.PROMPTS, (18, 33, 46, 50))

    def test_phone_alignment_preserves_gaps_and_never_positionally_shifts(self):
        operations = q08.edit_align(["AW", "R", "N", "AY"], ["AW", "N", "AY", "S"])
        self.assertEqual(
            [row["operation"] for row in operations],
            ["MATCH", "DELETION", "MATCH", "MATCH", "INSERTION"],
        )
        self.assertEqual(operations[2]["reference_index"], 2)
        self.assertEqual(operations[2]["hypothesis_index"], 1)

    def test_interannotator_reader_accepts_singular_and_plural_phone_tier_names(self):
        singular = [{"text": "AA", "start_seconds": 0.0, "end_seconds": 0.1}]
        plural = [{"text": "A", "start_seconds": 0.0, "end_seconds": 0.1}]
        uppercase = [{"text": "AX", "start_seconds": 0.0, "end_seconds": 0.1}]
        self.assertEqual(q09._annotation_phone_tier({"tiers": {"annotation": singular}}), singular)
        self.assertEqual(q09._annotation_phone_tier({"tiers": {"annotations": plural}}), plural)
        self.assertEqual(q09._annotation_phone_tier({"tiers": {"Annotation": uppercase, "Transcription": []}}), uppercase)
        with self.assertRaises(ValueError):
            q09._annotation_phone_tier({"tiers": {}})


if __name__ == "__main__":
    unittest.main()
