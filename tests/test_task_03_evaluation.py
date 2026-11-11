from __future__ import annotations

import unittest

import task_03_evaluation as task03


class Task03EvaluationTests(unittest.TestCase):
    def test_absolute_agreement_icc_is_one_for_identical_rater_scores(self):
        rows = [[1.0, 1.0, 1.0], [4.0, 4.0, 4.0], [8.0, 8.0, 8.0]]
        self.assertAlmostEqual(task03._icc_absolute_agreement(rows), 1.0)

    def test_absolute_agreement_icc_penalizes_systematic_rater_offset(self):
        consistent = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        exact = [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [8.0, 8.0, 8.0]]
        self.assertLess(task03._icc_absolute_agreement(consistent), task03._icc_absolute_agreement(exact))

    def test_cluster_bootstrap_is_deterministic_and_counts_speakers(self):
        rows = [
            ("s1", [1.0, 1.0]), ("s1", [2.0, 2.0]),
            ("s2", [4.0, 4.0]), ("s2", [5.0, 5.0]),
            ("s3", [8.0, 8.0]), ("s3", [9.0, 9.0]),
        ]
        first = task03._cluster_bootstrap_icc(rows, seed=17, replicates=100)
        second = task03._cluster_bootstrap_icc(rows, seed=17, replicates=100)
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"], 3)
        self.assertEqual(first["valid_replicates"], 100)


if __name__ == "__main__":
    unittest.main()
