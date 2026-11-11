"""Integrity and token-counterfactual tests; synthetic fixtures are not ASR evidence."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import task_07_5_asr as benchmark


class ReferenceIntegrityTests(unittest.TestCase):
    def test_human_word_tier_extracts_spoken_labels_only(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "manual.TextGrid"
            path.write_text('name = "words"\n'
                            'intervals [1]:\n text = ""\n'
                            'intervals [2]:\n text = "I"\n'
                            'intervals [3]:\n text = "I"\n'
                            'intervals [4]:\n text = "spoke"\n'
                            'item [2]:\n name = "phones"\n text = "S"\n')
            self.assertEqual(benchmark.manual_words(path), "I I spoke")

    def test_missing_human_words_tier_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "invalid.TextGrid"
            path.write_text('name = "phones"\n text = "AH"\n')
            with self.assertRaises(ValueError):
                benchmark.manual_words(path)

    def test_missing_human_transcript_cannot_be_promoted_to_ground_truth(self):
        with self.assertRaises(ValueError):
            benchmark.comparison("", "any words")

    def test_duplicate_units_are_rejected_by_frozen_manifest_verifier(self):
        source = benchmark.ROOT / "task_07_5_asr.py"
        row = {"id": "duplicate", "reference_type": "INTENDED_REFERENCE_ONLY",
               "audio": source.name, "audio_sha256": benchmark.digest(source),
               "reference_source": None, "reference_source_sha256": None}
        manifest = {"config": benchmark.ASR, "rows": [row, row]}
        manifest["fingerprint"] = hashlib.sha256(benchmark.canonical(manifest)).hexdigest()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            benchmark.verify_manifest(manifest)

    def test_manifest_fingerprint_tampering_is_rejected(self):
        manifest = {"config": benchmark.ASR, "rows": [], "fingerprint": "0" * 64}
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            benchmark.verify_manifest(manifest)


class ScoringPropagationTests(unittest.TestCase):
    def test_substitution_deletion_insertion_are_distinct(self):
        values = benchmark.comparison("a b c d", "a x c d z")
        self.assertEqual((values["substitutions"], values["deletions"], values["insertions"]), (1, 0, 1))
        self.assertFalse(values["exact"])

    def test_number_and_typographic_normalization(self):
        result = benchmark.comparison("TWENTY-ONE, don't!", "21 don’t.")
        self.assertTrue(result["exact"])
        self.assertEqual(result["reference_words"], 3)

    def test_repeated_words_are_not_silently_removed(self):
        value = benchmark.comparison("go go now", "go now")
        self.assertEqual((value["substitutions"], value["deletions"], value["insertions"]), (0, 1, 0))

    def test_content_is_not_attributed_to_speaker_from_asr_alone(self):
        case = benchmark._content("I like blue cars", "I like cars",
                                  segments=[], vad=[{"start_seconds": 0, "end_seconds": 1}])
        self.assertEqual(case["status"], "AVAILABLE_UNVERIFIED_ASR_PREDICTION")
        self.assertIn("UNCERTAIN_ASR_ONLY", [op.get("speaker_attribution") for op in
                                              case["measurements"]["alignment"]])
        self.assertEqual(benchmark._gate_result(case), "UNSUPPORTED_ASR_MISMATCH_OR_UNCERTAINTY")

    def test_exact_verified_text_counterfactual_passes_only_text_gate(self):
        case = benchmark._content("I like blue cars", "I like blue cars",
                                  segments=[], vad=[{"start_seconds": 0, "end_seconds": 1}])
        self.assertEqual(benchmark._gate_result(case), "ASR_MATCH_UNVERIFIED")

    def test_no_vad_does_not_become_speaker_omission(self):
        case = benchmark._content("go away", "", segments=[], vad=[])
        self.assertEqual(case["status"], "UNCERTAIN")
        self.assertTrue(all(op["classification"] == "UNCERTAIN"
                            for op in case["measurements"]["alignment"]))
        self.assertEqual(benchmark._gate_result(case), "UNSUPPORTED_ASR_MISMATCH_OR_UNCERTAINTY")


if __name__ == "__main__":
    unittest.main()
