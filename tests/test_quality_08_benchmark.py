import unittest

import quality_08_benchmark as quality


TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 2
item [1]:
    class = "IntervalTier"
    name = "words"
    xmin = 0
    xmax = 1.0
    intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "our"
        intervals [2]:
            xmin = 0.5
            xmax = 1.0
            text = "nice"
item [2]:
    class = "IntervalTier"
    name = "annotation"
    xmin = 0
    xmax = 1.0
    intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "AW"
        intervals [2]:
            xmin = 0.5
            xmax = 1.0
            text = "N"
'''


class Quality08BenchmarkTests(unittest.TestCase):
    def test_parses_long_textgrid_word_and_phone_tiers(self):
        parsed = quality.parse_textgrid(TEXTGRID)
        self.assertEqual(parsed["duration_seconds"], 1.0)
        self.assertEqual([x["text"] for x in parsed["tiers"]["words"]], ["our", "nice"])
        self.assertEqual([x["text"] for x in parsed["tiers"]["annotation"]], ["AW", "N"])

    def test_source_label_parser_keeps_missing_times_explicit(self):
        rows = quality.parse_source_labels(
            "utt_0 AX AX +\nutt_1 Kh K - 0.25 0.41\n"
        )
        self.assertIsNone(rows[0]["start_seconds"])
        self.assertEqual(rows[0]["label"], "+")
        self.assertEqual(rows[1]["start_seconds"], 0.25)
        self.assertEqual(rows[1]["observed_phone"], "K")

    def test_phone_aliases_are_explicit_and_unknowns_are_not_guessed(self):
        self.assertEqual(quality.normalize_phone("Th"), "T")
        self.assertEqual(quality.normalize_phone("Kh"), "K")
        self.assertEqual(quality.normalize_phone("EDX"), "EDX")

    def test_sequence_alignment_does_not_shift_after_deletion_and_insertion(self):
        operations = quality.edit_align(
            ["AW", "R", "N", "AY"], ["AW", "N", "AY", "S"]
        )
        self.assertEqual(
            [item["operation"] for item in operations],
            ["MATCH", "DELETION", "MATCH", "MATCH", "INSERTION"],
        )
        self.assertEqual(operations[2]["reference"], "N")
        self.assertEqual(operations[2]["hypothesis"], "N")

    def test_sequence_alignment_marks_phone_substitutions(self):
        operations = quality.edit_align(["DH", "EH", "D"], ["D", "EH", "D"])
        self.assertEqual(operations[0]["operation"], "SUBSTITUTION")
        self.assertEqual(operations[0]["reference"], "DH")
        self.assertEqual(operations[0]["hypothesis"], "D")

    def test_auc_uses_correct_direction_and_requires_both_classes(self):
        self.assertEqual(quality._auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]), 1.0)
        self.assertIsNone(quality._auc([1, 1], [0.9, 0.8]))

    def test_integrity_summary_detects_speaker_leakage_and_duplicate_audio(self):
        records = [
            {"record_id": "a1", "partition": "development", "speaker_id": "speaker-a",
             "files": {"audio": {"sha256": "same"}}},
            {"record_id": "a2", "partition": "development", "speaker_id": "speaker-b",
             "files": {"audio": {"sha256": "same"}}},
            {"record_id": "b1", "partition": "holdout", "speaker_id": "speaker-a",
             "files": {"audio": {"sha256": "other"}}},
        ]
        integrity = quality.summarize_integrity(records)
        self.assertEqual(integrity["speaker_overlap"], ["speaker-a"])
        self.assertEqual(integrity["audio_duplicate_groups"], [["a1", "a2"]])

    def test_support_gate_reports_false_acceptance_and_rejection(self):
        rows = [
            {"supported": True, "manual_valid_100ms": True, "boundary_available": True},
            {"supported": True, "manual_valid_100ms": False, "boundary_available": True},
            {"supported": False, "manual_valid_100ms": True, "boundary_available": True},
            {"supported": False, "manual_valid_100ms": False, "boundary_available": True},
        ]
        result = quality.compare_support_to_boundaries(rows, 100)
        self.assertEqual((result["true_accept"], result["false_accept"]), (1, 1))
        self.assertEqual((result["false_reject"], result["true_reject"]), (1, 1))
        self.assertEqual(result["false_accept_rate_among_accepted"], 0.5)
        self.assertEqual(result["false_reject_rate_among_valid"], 0.5)


if __name__ == "__main__":
    unittest.main()
