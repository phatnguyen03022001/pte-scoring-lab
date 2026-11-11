import unittest
from collections import Counter

import l2arctic_benchmark as l2


TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 1
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "test"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1
        intervals: size = 5
        intervals [1]:
            xmin = 0
            xmax = 0.2
            text = "T"
        intervals [2]:
            xmin = 0.2
            xmax = 0.4
            text = "EH1,AE0,S"
        intervals [3]:
            xmin = 0.4
            xmax = 0.6
            text = "SIL,AH0,A"
        intervals [4]:
            xmin = 0.6
            xmax = 0.8
            text = "S, SIL, D "
        intervals [5]:
            xmin = 0.8
            xmax = 1
            text = ""
'''


class L2ArcticBenchmarkTests(unittest.TestCase):
    def test_manual_labels_keep_error_polarity_and_exclude_insertions(self):
        rows, counts = l2.parse_manual_phones(TEXTGRID)
        self.assertEqual([(row["target_phone"], row["category"], row["correct"]) for row in rows], [
            ("T", "unchanged_correct", True),
            ("EH", "substitution", False),
            ("S", "deletion", False),
        ])
        self.assertEqual(counts["insertion_unscored"], 1)
        self.assertEqual(counts["unlabeled_or_silence_intervals"], 1)
        self.assertEqual(counts["substitution"], 1)
        self.assertEqual(counts["deletion"], 1)

    def test_phone_normalization_handles_stress_and_deviation(self):
        self.assertEqual(l2.canonical_phone(" AH1* "), "AH")
        self.assertEqual(l2.canonical_phone("dh0"), "DH")

    def test_unknown_tag_semantics_fail_closed(self):
        text = TEXTGRID.replace('text = "EH1,AE0,S"', 'text = "EH1,AE0,?"')
        with self.assertRaisesRegex(ValueError, "Unrecognized manual phone annotation"):
            l2.parse_manual_phones(text)

    def test_insertion_and_deletion_silence_conventions_are_checked(self):
        bad_insertion = TEXTGRID.replace('text = "SIL,AH0,A"', 'text = "AH0,SIL,A"')
        with self.assertRaisesRegex(ValueError, "Insertion tag must use a silence target"):
            l2.parse_manual_phones(bad_insertion)
        bad_deletion = TEXTGRID.replace('text = "S, SIL, D "', 'text = "S,Z,D"')
        with self.assertRaisesRegex(ValueError, "Deletion tag must use a silence observation"):
            l2.parse_manual_phones(bad_deletion)

    def test_selection_is_deterministic_and_score_independent(self):
        ids = [f"utterance-{index:03d}" for index in range(150)]
        self.assertEqual(l2._deterministic_sample("BWC", ids, 20), l2._deterministic_sample("BWC", ids, 20))
        self.assertEqual(len(l2._deterministic_sample("BWC", ids, 20)), 20)
        self.assertNotEqual(l2._deterministic_sample("BWC", ids, 20), l2._deterministic_sample("LXC", ids, 20))

    def test_metrics_use_low_score_for_error_and_count_abstained_errors_as_misses(self):
        metrics = l2._metrics([
            {"speaker_id": "s1", "correct": True, "score": 2.1},
            {"speaker_id": "s1", "correct": False, "score": 1.8},
            {"speaker_id": "s2", "correct": False, "score": None},
        ])
        self.assertEqual(metrics["coverage"], 2 / 3)
        self.assertEqual(metrics["threshold_metrics"]["coverage_aware"]["recall_all_reference_errors"], 0.5)
        self.assertEqual(metrics["roc_auc_conditional"], 1.0)
        self.assertEqual(metrics["pr_auc_conditional"], 1.0)
        self.assertEqual(metrics["speaker_cluster_bootstrap_95"]["status"], "NOT_REPORTED")


if __name__ == "__main__":
    unittest.main()
