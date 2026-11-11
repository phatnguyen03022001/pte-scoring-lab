"""Synthetic invariants for TASK-07.5.2; not human-label accuracy evidence."""
import tempfile
import unittest
from pathlib import Path

from task_07_5_2_attribution import (
    decoder_support_compatibility,
    paired_prompt_changes,
    three_way_diagnostic,
    validate_handoff_submission,
    word_interval_labels,
)


class ThreeWayEditTests(unittest.TestCase):
    def call(self, prompt, actual, asr, **kwargs):
        return three_way_diagnostic(
            prompt, actual, asr, independently_verified=True, **kwargs)

    def test_no_independent_label_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "independently verified"):
            three_way_diagnostic("a blue car", "a blue car", "a blue car")

    def test_speaker_correct_and_asr_wrong(self):
        result = self.call("the red car", "the red car", "the blue car")
        self.assertFalse(result["speaker_deviation_present"])
        self.assertTrue(result["asr_error_present"])
        self.assertFalse(result["both_present"])
        self.assertFalse(result["asr_vs_prompt_text_gate"])

    def test_speaker_omission_asr_faithful(self):
        result = self.call("the red car", "the car", "the car")
        self.assertTrue(result["speaker_deviation_present"])
        self.assertFalse(result["asr_error_present"])
        self.assertFalse(result["actual_vs_prompt_text_gate"])

    def test_speaker_omission_asr_hallucinates_prompt_word(self):
        result = self.call("the red car", "the car", "the red car")
        self.assertTrue(result["both_present"])
        self.assertTrue(result["asr_vs_prompt_text_gate"])
        self.assertFalse(result["actual_vs_prompt_text_gate"])
        self.assertTrue(any(e["classification"] == "INSERTION"
                            for e in result["asr_edit_events"]))

    def test_speaker_and_asr_substitute_same_span(self):
        result = self.call("the red car", "the blue car", "the green car")
        self.assertTrue(result["both_present"])
        self.assertTrue(any(e["classification"] == "SUBSTITUTION"
                            for e in result["speaker_edit_events"]))
        self.assertTrue(any(e["classification"] == "SUBSTITUTION"
                            for e in result["asr_edit_events"]))

    def test_speaker_inserts_and_asr_is_faithful(self):
        result = self.call("we drive home", "we really drive home",
                           "we really drive home")
        self.assertTrue(result["speaker_deviation_present"])
        self.assertFalse(result["asr_error_present"])

    def test_asr_inserts_extra_word(self):
        result = self.call("we drive home", "we drive home",
                           "we really drive home")
        self.assertFalse(result["speaker_deviation_present"])
        self.assertTrue(result["asr_error_present"])

    def test_repeated_word_alignment_flagged(self):
        result = self.call("go now", "go go now", "go now")
        self.assertTrue(result["possible_repeated_token_alignment_ambiguity"])
        self.assertTrue(result["both_present"])

    def test_independent_actual_uncertain_excluded(self):
        result = self.call("a car", "a car", "a car",
                           uncertain_token_indices=(1,))
        self.assertEqual(result["status"], "AMBIGUOUS_EXCLUDE_FROM_UNQUALIFIED_WER")
        self.assertIsNone(result["actual_vs_asr"])

    def test_invalid_uncertainty_indices_rejected(self):
        for bad in ((-1,), (2,), (True,)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.call("a car", "a car", "a car",
                              uncertain_token_indices=bad)

    def test_exact_transcript_not_pronunciation_proof(self):
        result = self.call("a car", "a car", "a car")
        self.assertTrue(result["oracle_actual_vs_itself_text_gate"])
        self.assertIsNone(result["decoder_confidence_eligible"])
        self.assertIsNone(result["acoustic_support_eligible"])
        self.assertIsNone(result["gopt_success"])
        self.assertFalse(result["pronunciation_attribution_verified"])


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.manifest_row = {"id": "l2_read_intended:HQTV:arctic_a0052",
                             "audio_sha256": "a" * 64}
        self.entry = {
            "recording_id": self.manifest_row["id"],
            "audio_sha256": self.manifest_row["audio_sha256"],
            "status": "HUMAN_REVIEW_SUBMITTED",
            "actual_speech_transcript": "we said the word",
            "uncertain_token_indices": [],
            "provenance": {
                "kind": "INDEPENDENT_HUMAN",
                "annotator_id": "annotator-A",
                "reviewer_id": "reviewer-B",
                "audio_only_first_pass": True,
                "model_predictions_blinded": True,
                "prompt_aware_adjudication": True,
                "rights_confirmed": True,
                "source_evidence_sha256": "b" * 64,
            },
        }

    def test_valid_format_is_not_verified_human_truth(self):
        self.assertEqual(
            validate_handoff_submission(self.entry, self.manifest_row),
            "SCHEMA_VALID_INDEPENDENCE_REQUIRES_EXTERNAL_VERIFICATION",
        )

    def test_single_qualified_annotator_is_not_automatically_rejected(self):
        self.entry["provenance"]["reviewer_id"] = None
        self.assertEqual(
            validate_handoff_submission(self.entry, self.manifest_row),
            "SCHEMA_VALID_SINGLE_ANNOTATOR_UNVERIFIED",
        )

    def test_missing_recording_or_transcript_fails(self):
        self.entry["audio_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_handoff_submission(self.entry, self.manifest_row)
        self.entry["audio_sha256"] = "a" * 64
        self.entry["actual_speech_transcript"] = None
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_handoff_submission(self.entry, self.manifest_row)

    def test_no_human_review_and_model_transcript_fails(self):
        self.entry["status"] = "UNANNOTATED"
        with self.assertRaisesRegex(ValueError, "cannot become ground truth"):
            validate_handoff_submission(self.entry, self.manifest_row)
        self.entry["status"] = "HUMAN_REVIEW_SUBMITTED"
        self.entry["provenance"]["kind"] = "MODEL_GENERATED"
        with self.assertRaisesRegex(ValueError, "not human"):
            validate_handoff_submission(self.entry, self.manifest_row)

    def test_no_blinding_or_same_reviewer_fails(self):
        self.entry["provenance"]["model_predictions_blinded"] = False
        with self.assertRaisesRegex(ValueError, "Unverified"):
            validate_handoff_submission(self.entry, self.manifest_row)
        self.entry["provenance"]["model_predictions_blinded"] = True
        self.entry["provenance"]["reviewer_id"] = "annotator-A"
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_handoff_submission(self.entry, self.manifest_row)

    def test_bad_external_evidence_hash_and_ambiguity_fails(self):
        self.entry["provenance"]["source_evidence_sha256"] = "unknown"
        with self.assertRaisesRegex(ValueError, "evidence hash"):
            validate_handoff_submission(self.entry, self.manifest_row)
        self.entry["provenance"]["source_evidence_sha256"] = "b" * 64
        for indices in ([9], [0, 0], ["0"]):
            with self.subTest(indices=indices):
                self.entry["uncertain_token_indices"] = indices
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    validate_handoff_submission(self.entry, self.manifest_row)

    def test_word_interval_empty_spans_are_kept(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.TextGrid"
            path.write_text('name = "words"\n'
                            'intervals [1]:\n text = ""\n'
                            'intervals [2]:\n text = "go"\n'
                            'intervals [3]:\n text = ""\n'
                            'item [2]:\nname = "phones"\n')
            self.assertEqual(word_interval_labels(path), ["", "go", ""])


class CompatibilityTests(unittest.TestCase):
    def test_no_parakeet_decoder_features_fails_closed(self):
        segments = [{"text": "go", "start": 0.0, "end": 0.5,
                     "word_count": 1}]
        self.assertEqual(decoder_support_compatibility(segments),
                         "MISSING_DECODER_SUPPORT_FAIL_CLOSED")
        self.assertEqual(decoder_support_compatibility([]),
                         "MISSING_DECODER_SUPPORT_FAIL_CLOSED")

    def test_familiar_field_names_are_not_validated_semantics(self):
        segments = [{"avg_logprob": -0.1, "no_speech_prob": 0.1,
                     "compression_ratio": 1.1}]
        self.assertEqual(decoder_support_compatibility(segments),
                         "UNCALIBRATED_DIAGNOSTIC_SEMANTICS_FAIL_CLOSED")

    def test_paired_prompt_improvements_do_not_imply_actual_wer(self):
        cases = [{"id": "x", "reference": "a b", "speaker": "HQTV"},
                 {"id": "y", "reference": "go now", "speaker": "PNV"}]
        models = {
            "base": {"x": {"transcript": "a b"}, "y": {"transcript": "go"}},
            "small": {"x": {"transcript": "a"}, "y": {"transcript": "go now"}},
            "ultra": {"x": {"transcript": "a"}, "y": {"transcript": "go now"}},
        }
        counts = paired_prompt_changes(cases, models)["base_vs_ultra"]
        self.assertEqual(counts["baseline_refusal_to_prompt_exact"], 1)
        self.assertEqual(counts["baseline_prompt_exact_to_mismatch"], 1)
        self.assertEqual(counts["more_prompt_edits"], 1)
        self.assertEqual(counts["fewer_prompt_edits"], 1)


if __name__ == "__main__":
    unittest.main()
