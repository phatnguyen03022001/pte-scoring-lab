"""Contract tests for deterministic integration and failure isolation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import score_integrated as integrated


ROOT = Path(__file__).resolve().parents[1]
# Local-only SpeechOcean762 fixture, CC BY 4.0 (OpenSLR SLR101); audio remains ignored.
SPEECHOCEAN_MANIFEST = ROOT / ".local" / "datasets" / "speechocean762-reconstruction" / "sample_manifest.jsonl"
SPEECHOCEAN_AUDIO = ROOT / ".local" / "datasets" / "speechocean762-reconstruction" / "corpus" / "WAVE" / "SPEAKER0112" / "001120097.WAV"


class NormalizationTests(unittest.TestCase):
    def test_case_punctuation_whitespace_and_typographic_apostrophe(self):
        self.assertEqual(
            integrated.normalize_tokens("  I’m, HERE!\n"),
            (["i'm", "here"], [False, False]),
        )

    def test_contractions_are_preserved_as_spoken_tokens(self):
        self.assertEqual(integrated.normalize_tokens("he's isn't"), (["he's", "isn't"], [False, False]))
        self.assertNotEqual(integrated.normalize_tokens("he's")[0], integrated.normalize_tokens("he is")[0])

    def test_small_integer_digits_match_cardinal_words(self):
        self.assertEqual(integrated.normalize_tokens("21, TWENTY-ONE"),
                         (["twenty", "one", "twenty", "one"], [False] * 4))

    def test_large_and_decimal_numbers_remain_literal_and_are_flagged(self):
        self.assertEqual(integrated.normalize_tokens("2,025 and 3.5"),
                         (["2025", "and", "3.5"], [True, False, True]))

    def test_long_digit_strings_are_flagged_without_integer_conversion(self):
        value = "9" * 5000
        self.assertEqual(integrated.normalize_tokens(value), ([value], [True]))

    def test_comma_grouped_number_is_never_cardinalized(self):
        self.assertEqual(integrated.normalize_tokens("0,001"), (["0001"], [True]))

    def test_unicode_words_are_preserved_for_explicit_support_handling(self):
        self.assertEqual(integrated.normalize_tokens("read café 你好"),
                         (["read", "café", "你好"], [False, False, False]))

    def test_repeated_tokens_are_not_normalized_away(self):
        self.assertEqual(integrated.normalize_tokens("go, go"), (["go", "go"], [False, False]))

    def test_unsupported_symbols_are_reported(self):
        self.assertEqual(integrated.unsupported_characters("hello @world"), ["@"])


class TokenAlignmentTests(unittest.TestCase):
    def classifications(self, reference: str, hypothesis: str) -> list[str]:
        ref, ref_flags = integrated.normalize_tokens(reference)
        hyp, hyp_flags = integrated.normalize_tokens(hypothesis)
        return [
            item["classification"] for item in integrated.classify_alignment(
                ref, hyp,
                reference_unsupported_numbers=ref_flags,
                hypothesis_unsupported_numbers=hyp_flags,
            )
        ]

    def test_match_and_punctuation_normalization(self):
        self.assertEqual(self.classifications("Hello, WORLD!", "hello world"), ["MATCH", "MATCH"])

    def test_omission_category_and_asr_only_attribution(self):
        aligned = integrated.classify_alignment(["the", "team"], ["the"])
        self.assertEqual(aligned[1]["classification"], "OMISSION")
        self.assertEqual(aligned[1]["speaker_attribution"], "UNCERTAIN_ASR_ONLY")

    def test_insertion_category(self):
        self.assertEqual(self.classifications("the team", "the research team"), ["MATCH", "INSERTION", "MATCH"])

    def test_substitution_category(self):
        self.assertEqual(self.classifications("red car", "blue car"), ["SUBSTITUTION", "MATCH"])

    def test_partial_pronunciation_recovery_requires_exact_asr_reference_match(self):
        matched = {
            "status": "AVAILABLE_UNVERIFIED_ASR_PREDICTION",
            "measurements": {"alignment": [{"classification": "MATCH"}, {"classification": "MATCH"}]},
        }
        mismatched = {
            "status": "AVAILABLE_UNVERIFIED_ASR_PREDICTION",
            "measurements": {"alignment": [{"classification": "MATCH"}, {"classification": "OMISSION"}]},
        }
        uncertain = {**matched, "status": "UNCERTAIN"}
        self.assertTrue(integrated.has_exact_asr_reference_match(matched))
        self.assertFalse(integrated.has_exact_asr_reference_match(mismatched))
        self.assertFalse(integrated.has_exact_asr_reference_match(uncertain))

    def test_repeated_word_category_preserves_wer_insertion(self):
        aligned = integrated.classify_alignment(["go"], ["go", "go"])
        self.assertEqual([item["classification"] for item in aligned].count("REPETITION"), 1)
        counts = integrated._word_counts(["go"], ["go", "go"])
        self.assertEqual(counts["insertions"], 1)
        self.assertEqual(counts["word_error_rate"], 1.0)

    def test_adjacent_transposition_is_order_mismatch_with_two_edit_units(self):
        aligned = integrated.classify_alignment(["a", "b"], ["b", "a"])
        self.assertEqual([item["classification"] for item in aligned], ["ORDER_MISMATCH", "ORDER_MISMATCH"])
        self.assertEqual(integrated._word_counts(["a", "b"], ["b", "a"])["word_error_rate"], 1.0)

    def test_nonadjacent_unique_word_reordering_is_reported_explicitly(self):
        aligned = integrated.classify_alignment(["a", "b", "c"], ["c", "a", "b"])
        self.assertEqual(
            [item["classification"] for item in aligned],
            ["ORDER_MISMATCH", "MATCH", "MATCH", "ORDER_MISMATCH"],
        )

    def test_repeated_phrase_is_reported_without_duplicate_reference_credit(self):
        spans = integrated.repeated_response_spans(["we", "read", "we", "read", "the", "book"])
        self.assertEqual(spans[0]["tokens"], ["we", "read"])
        self.assertEqual(spans[0]["repeat_recognized_indices"], [2, 3])

    def test_rs_ordered_match_diagnostic_preserves_global_reference_indices(self):
        diagnostic = integrated.rs_sequence_diagnostic(
            ["the", "research", "team", "will", "publish", "final", "results", "tomorrow"],
            ["publish", "final", "results", "tomorrow"],
        )
        self.assertEqual(diagnostic["ordered_match_count"], 4)
        self.assertEqual(diagnostic["ordered_match_reference_indices"], [4, 5, 6, 7])
        self.assertEqual(diagnostic["rubric_inspired_band"], "AT_LEAST_HALF")
        self.assertIsNone(diagnostic["official_pearson_item_score"])

    def test_rs_content_component_exposes_task_specific_sequence_diagnostic(self):
        transcription = {
            "raw_transcript": "publish final results tomorrow",
            "normalized_tokens": ["publish", "final", "results", "tomorrow"],
            "unsupported_numeric_tokens": [False] * 4,
            "vad_intervals": [{"start_seconds": 0.2, "end_seconds": 2.0}],
            "recognized_token_segments": {},
            "diagnostic_thresholds": {"avg_logprob_threshold": -1.0, "no_speech_threshold": 0.6,
                                      "compression_ratio_threshold": 2.4},
            "segments": [], "model": {}, "decode_configuration": {},
            "vad_interval_duration_seconds": 1.8, "vad_interval_complement_duration_seconds": 0.2,
            "duration_after_vad_seconds": 1.8, "model_load_seconds": 0.1,
            "inference_seconds_excluding_model_load": 0.2, "elapsed_seconds_including_model_load": 0.3,
        }
        content = integrated._build_content(
            "the research team will publish final results tomorrow", transcription, task="rs",
        )
        self.assertEqual(content["task"], "REPEAT_SENTENCE")
        self.assertEqual(content["measurements"]["sequence_diagnostic"]["ordered_match_count"], 4)

    def test_alignment_budget_blocks_pathological_quadratic_matrices(self):
        self.assertTrue(integrated.content_alignment_within_budget(900, 1000))
        self.assertFalse(integrated.content_alignment_within_budget(1000, 1000))

    def test_no_speech_returns_uncertain_candidate_omissions(self):
        aligned = integrated.classify_alignment(["expected", "words"], [], no_speech=True)
        self.assertEqual([item["classification"] for item in aligned], ["UNCERTAIN", "UNCERTAIN"])
        self.assertEqual([item["candidate_classification"] for item in aligned], ["OMISSION", "OMISSION"])

    def test_unsupported_numeric_difference_is_uncertain(self):
        aligned = integrated.classify_alignment(
            ["2025"], ["2024"], reference_unsupported_numbers=[True], hypothesis_unsupported_numbers=[True]
        )
        self.assertEqual(aligned[0]["classification"], "UNCERTAIN")
        self.assertEqual(aligned[0]["candidate_classification"], "SUBSTITUTION")


class DiagnosticAndTimingTests(unittest.TestCase):
    def test_native_decoder_thresholds_turn_error_classification_uncertain(self):
        class FakeModel:
            feature_extractor = SimpleNamespace(chunk_length=30)

            def transcribe(self, audio, **kwargs):
                self.kwargs = kwargs
                segment = SimpleNamespace(
                    text=" wrong", start=0.1, end=0.8, avg_logprob=-1.2,
                    no_speech_prob=0.7, compression_ratio=2.5, temperature=0.6,
                )
                return iter([segment]), SimpleNamespace(duration_after_vad=1.0)

        model = FakeModel()
        with patch.object(integrated, "WhisperModel", return_value=model), \
             patch.object(integrated, "_vad_intervals", return_value=[{"start_seconds": 0.0, "end_seconds": 1.0}]):
            transcription = integrated.transcribe_audio(Path("audio.wav"), np.ones(16000, dtype=np.float32), threads=2)
        self.assertNotIn("temperature", model.kwargs)
        self.assertEqual(model.kwargs["language"], "en")
        self.assertEqual(model.kwargs["beam_size"], 5)
        self.assertFalse(model.kwargs["condition_on_previous_text"])
        self.assertNotIn("reference", model.kwargs)
        self.assertEqual(transcription["segments"][0]["diagnostic_flags"], [
            "below_decoder_log_probability_threshold",
            "at_or_above_decoder_no_speech_threshold",
            "above_decoder_compression_ratio_threshold",
            "temperature_fallback_used",
        ])
        content = integrated._build_content("right", transcription)
        operation = content["measurements"]["alignment"][0]
        self.assertEqual(operation["classification"], "UNCERTAIN")
        self.assertEqual(operation["candidate_classification"], "SUBSTITUTION")

    def test_builtin_vad_keeps_original_timeline_and_compressed_duration_separate(self):
        class FakeModel:
            feature_extractor = SimpleNamespace(chunk_length=30)

            def transcribe(self, audio, **kwargs):
                self.kwargs = kwargs
                segment = SimpleNamespace(text=" hello", start=2.0, end=2.5,
                                          avg_logprob=-0.2, no_speech_prob=0.01,
                                          compression_ratio=0.5, temperature=0.0)
                return iter([segment]), SimpleNamespace(duration_after_vad=3.0)

        model = FakeModel()
        intervals = [{"start_seconds": 2.0, "end_seconds": 3.0}]
        with patch.object(integrated, "WhisperModel", return_value=model), \
             patch.object(integrated, "_vad_intervals", return_value=intervals):
            result = integrated.transcribe_audio(Path("audio.wav"), np.ones(5 * 16000, dtype=np.float32), threads=2)
        self.assertTrue(model.kwargs["vad_filter"])
        self.assertNotIn("clip_timestamps", model.kwargs)
        self.assertAlmostEqual(result["segments"][0]["start_seconds_original_timeline"], 2.0)
        self.assertEqual(result["duration_seconds"], 5.0)
        self.assertEqual(result["vad_interval_duration_seconds"], 1.0)
        self.assertEqual(result["duration_after_vad_seconds"], 3.0)

    def test_silence_log_tracks_intervals_on_original_timeline(self):
        intervals = integrated.parse_silence_intervals(
            "silence_start: 0.0\nsilence_end: 0.4\nsilence_start: 2.0\nsilence_end: 3.5\nsilence_start: 4.0",
            5.0,
        )
        self.assertEqual([item["duration_seconds"] for item in intervals], [0.4, 1.5, 1.0])

    def test_fluency_reports_descriptors_without_calling_candidates_hesitations(self):
        samples = np.ones(5 * 16000, dtype=np.float32) * 0.1
        transcription = {"normalized_tokens": ["um", "hello"], "vad_intervals": [{"start_seconds": 0.2, "end_seconds": 4.8}],
                         "vad_interval_duration_seconds": 4.6, "duration_after_vad_seconds": 4.6}
        silence = [{"start_seconds": 2.0, "end_seconds": 3.2, "duration_seconds": 1.2}]
        result = integrated._build_fluency(samples, transcription, silence, {"predictions": {"fluency": 1.2, "native_scale": "GOPT units"}})
        self.assertEqual(result["measurements"]["silence_candidates"]["internal_pause_count"], 1)
        self.assertEqual(result["measurements"]["silence_candidates"]["long_pause_candidates_ge_1_second"], silence)
        self.assertIsNone(result["measurements"]["active_speech_duration_seconds"])
        self.assertEqual(result["measurements"]["hesitation_evidence"]["asr_filler_token_candidates"], ["um"])
        self.assertEqual(result["predictions"]["gopt_fluency"], 1.2)

    def test_missing_asr_preserves_recording_timing(self):
        samples = np.zeros(16000, dtype=np.float32)
        result = integrated._build_fluency(samples, None, [], {"predictions": None})
        self.assertEqual(result["measurements"]["total_recording_duration_seconds"], 1.0)
        self.assertIsNone(result["measurements"]["speech_rate"]["recognized_asr_tokens_per_original_recording_minute"])
        self.assertEqual(result["status"], "PARTIAL_TIMING_MEASUREMENTS")

    def test_energy_descriptor_is_finite_for_silence_and_signal(self):
        silence = integrated._signal_energy(np.zeros(16000, dtype=np.float32))
        signal = integrated._signal_energy(np.ones(16000, dtype=np.float32) * 0.1)
        self.assertTrue(math.isfinite(silence["peak_dbfs"]))
        self.assertGreater(signal["peak_dbfs"], silence["peak_dbfs"])
        self.assertEqual(signal["unit"], "dBFS")


class FailureIsolationTests(unittest.TestCase):
    def test_audio_duration_preflight_happens_before_waveform_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "long.wav"
            audio.touch()
            with patch.object(integrated, "_probe_audio_duration", return_value=121.0), \
                 patch.object(integrated, "decode_audio") as decode:
                result = integrated.assess(audio, "hello")
        decode.assert_not_called()
        self.assertEqual(result["error"]["status"], "INVALID_AUDIO_DURATION")
        self.assertEqual(result["input"]["duration_seconds"], 121.0)

    def test_cli_accepts_explicit_repeat_sentence_task(self):
        process = subprocess.run(
            [sys.executable, str(ROOT / "score_integrated.py"), "--task", "rs",
             "--audio", "/missing.wav", "--reference", "hello"],
            capture_output=True, text=True, timeout=30, cwd=ROOT,
        )
        self.assertEqual(process.returncode, 2)
        result = json.loads(process.stdout)
        self.assertEqual(result["task"], "REPEAT_SENTENCE")
        self.assertEqual(result["long_form"]["aggregation_status"], "NOT_RUN")
        self.assertIsNone(result["pearson_pte_score"])

    def test_mismatched_asr_withholds_reference_phone_attribution_but_keeps_diagnostics(self):
        pronunciation = {
            "status": "AVAILABLE_UNCALIBRATED_MODEL_PREDICTIONS",
            "measurements": {
                "phone_count": 1, "word_count": 1,
                "phonemes": [{"index": 0, "word_index": 0, "word": "hello", "gopt_phone_score": 1.0}],
                "words": [{"word_index": 0, "word": "hello"}],
                "phone_timestamps": [{"index": 0, "start_seconds": 0.0, "end_seconds": 0.1}],
                "word_to_phone_relationships": [{"word_index": 0, "phone_indices": [0]}],
                "units": "uncalibrated GOPT units",
            },
            "predictions": {"utterance_accuracy": 1.0},
            "provenance": {"alignment_valid": True}, "warnings": [],
        }
        content = {
            "status": "AVAILABLE_UNVERIFIED_ASR_PREDICTION",
            "measurements": {
                "reference_normalized_tokens": ["hello", "there"],
                "alignment": [
                    {"classification": "MATCH", "reference_index": 0, "recognized_index": 0},
                    {"classification": "OMISSION", "reference_index": 1, "recognized_index": None},
                ],
            },
        }
        result = integrated.gate_pronunciation_attribution(pronunciation, content)
        self.assertEqual(result["status"], "PARTIAL_UNATTRIBUTED_REFERENCE_CONDITIONED_DIAGNOSTICS")
        self.assertEqual(result["measurements"]["phonemes"], [])
        self.assertEqual(result["measurements"]["coverage"]["supported_word_count"], 0)
        self.assertEqual(result["research_diagnostics"]["phonemes"][0]["word"], "hello")
        self.assertIsNone(result["predictions"])

    def test_exact_asr_sequence_keeps_unverified_pronunciation_coverage_explicit(self):
        pronunciation = {
            "status": "AVAILABLE_UNCALIBRATED_MODEL_PREDICTIONS",
            "measurements": {"phone_count": 1, "word_count": 1, "phonemes": [{"index": 0}], "words": [{"word_index": 0}],
                             "phone_timestamps": [], "word_to_phone_relationships": [], "units": "GOPT units"},
            "predictions": {"utterance_accuracy": 1.0}, "provenance": {}, "warnings": [],
        }
        content = {
            "status": "AVAILABLE_UNVERIFIED_ASR_PREDICTION",
            "measurements": {
                "reference_normalized_tokens": ["hello"],
                "alignment": [{"classification": "MATCH", "recognition_support": "NO_THRESHOLD_CROSSED"}],
            },
        }
        result = integrated.gate_pronunciation_attribution(pronunciation, content)
        self.assertEqual(result["status"], "AVAILABLE_UNCALIBRATED_MODEL_PREDICTIONS")
        self.assertEqual(result["speaker_attribution"]["status"], "ASR_MATCH_UNVERIFIED")
        self.assertEqual(result["measurements"]["coverage"]["supported_word_count"], 1)

    def test_asr_failure_does_not_erase_pronunciation_or_timing(self):
        pronunciation = {
            "status": "AVAILABLE_UNCALIBRATED_MODEL_PREDICTIONS",
            "measurements": {"words": [], "phonemes": [], "units": "GOPT units"},
            "predictions": {"utterance_accuracy": 1.0, "prosody": 1.0, "fluency": 1.0, "native_scale": "GOPT units"},
            "provenance": {}, "warnings": [], "error": None,
        }
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "valid.wav"
            audio.touch()
            with patch.object(integrated, "_probe_audio_duration", return_value=1.0), \
                 patch.object(integrated, "decode_audio", return_value=np.ones(16000, dtype=np.float32)), \
                 patch.object(integrated, "transcribe_audio", side_effect=RuntimeError("simulated ASR failure")), \
                 patch.object(integrated, "_build_pronunciation", return_value=pronunciation), \
                 patch.object(integrated.subprocess, "run", return_value=SimpleNamespace(stderr="", stdout="", returncode=0)):
                result = integrated.assess(audio, "Hello world", threads=2)
        self.assertEqual(result["components"]["content"]["status"], "FAILED")
        self.assertEqual(result["components"]["pronunciation"]["status"], "PARTIAL_UNATTRIBUTED_REFERENCE_CONDITIONED_DIAGNOSTICS")
        self.assertEqual(result["components"]["pronunciation"]["research_diagnostics"]["predictions"]["utterance_accuracy"], 1.0)
        self.assertIsNone(result["components"]["pronunciation"]["predictions"])
        self.assertEqual(result["components"]["oral_fluency"]["status"], "PARTIAL_TIMING_MEASUREMENTS")
        self.assertIsNone(result["overall_pte_score"])
        self.assertIsNone(result["pearson_pte_score"])
        json.dumps(result, allow_nan=False)

    def test_invalid_audio_fails_closed_before_components_run(self):
        result = integrated.assess(Path("/path/that/does/not/exist.wav"), "Hello")
        self.assertEqual(result["status"], "FAILED")
        self.assertTrue(all(item["status"] == "FAILED" for item in result["components"].values()))
        self.assertIsNone(result["overall_pte_score"])
        self.assertIsNone(result["pearson_pte_score"])

    def test_corrupted_audio_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            corrupt = Path(temp) / "corrupt.wav"
            corrupt.write_bytes(b"not a wave file")
            result = integrated.assess(corrupt, "Hello")
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error"]["status"], "AUDIO_DECODE_FAILED")
        self.assertTrue(all(item["status"] == "FAILED" for item in result["components"].values()))

    def test_reference_without_supported_tokens_fails_closed(self):
        result = integrated.assess(Path("/missing.wav"), "...!!!")
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error"]["status"], "INVALID_REFERENCE")

    def test_invalid_thread_count_returns_structured_json(self):
        process = subprocess.run(
            [sys.executable, str(ROOT / "score_integrated.py"), "--audio", "/missing.wav",
             "--reference", "Hello", "--threads", "0"],
            capture_output=True, text=True, timeout=30, cwd=ROOT,
        )
        self.assertEqual(process.returncode, 2)
        result = json.loads(process.stdout)
        self.assertEqual(result["status"], "INVALID_INPUT")
        self.assertEqual(result["error"]["status"], "INVALID_THREADS")
        self.assertEqual(set(result["components"]), {"content", "pronunciation", "oral_fluency", "prosody", "word_stress"})

    def test_pronunciation_failure_does_not_erase_content_or_temporal_metrics(self):
        transcription = {
            "raw_transcript": "Hello",
            "normalized_tokens": ["hello"],
            "unsupported_numeric_tokens": [False],
            "vad_intervals": [{"start_seconds": 0.0, "end_seconds": 1.0}],
            "vad_interval_duration_seconds": 1.0,
            "vad_interval_complement_duration_seconds": 0.0,
            "segments": [],
            "recognized_token_segments": {},
            "diagnostic_thresholds": {"avg_logprob_threshold": -1.0, "no_speech_threshold": 0.6, "compression_ratio_threshold": 2.4},
            "model": {"name": "test", "device": "cpu"},
            "decode_configuration": {},
            "model_load_seconds": 0.1,
            "inference_seconds_excluding_model_load": 0.1,
            "elapsed_seconds_including_model_load": 0.2,
            "duration_after_vad_seconds": 1.0,
        }
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "valid.wav"
            audio.touch()
            with patch.object(integrated, "_probe_audio_duration", return_value=1.0), \
                 patch.object(integrated, "decode_audio", return_value=np.ones(16000, dtype=np.float32)), \
                 patch.object(integrated, "transcribe_audio", return_value=transcription), \
                 patch.object(integrated, "_build_pronunciation", return_value=integrated._component_failure("ALIGNMENT_INVALID", "unsupported reference alignment")), \
                 patch.object(integrated.subprocess, "run", return_value=SimpleNamespace(stderr="", stdout="", returncode=0)):
                result = integrated.assess(audio, "Hello", threads=2)
        self.assertEqual(result["components"]["content"]["status"], "AVAILABLE_UNVERIFIED_ASR_PREDICTION")
        self.assertEqual(result["components"]["pronunciation"]["status"], "FAILED")
        self.assertEqual(result["components"]["oral_fluency"]["status"], "PARTIAL_DESCRIPTIVE_MEASUREMENTS")

    def test_silent_recording_is_uncertain_content_and_keeps_timing(self):
        if not (ROOT / ".local" / "gopt-venv" / "bin" / "python").is_file():
            self.skipTest("local GOPT runtime is unavailable")
        with tempfile.TemporaryDirectory() as temp:
            silent = Path(temp) / "silence.wav"
            with wave.open(str(silent), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 48000)
            process = subprocess.run(
                [sys.executable, str(ROOT / "score_integrated.py"), "--audio", str(silent),
                 "--reference", "Hello there", "--threads", "2"],
                capture_output=True, text=True, timeout=180, cwd=ROOT,
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["components"]["content"]["status"], "UNCERTAIN")
        self.assertTrue(all(item["classification"] == "UNCERTAIN" for item in result["components"]["content"]["measurements"]["alignment"]))
        self.assertEqual(result["components"]["oral_fluency"]["measurements"]["total_recording_duration_seconds"], 3.0)
        self.assertEqual(result["components"]["pronunciation"]["status"], "FAILED")
        self.assertIsNone(result["overall_pte_score"])
        self.assertIsNone(result["pearson_pte_score"])

    def test_nonfinite_values_are_serialized_as_null(self):
        safe, changed = integrated._json_safe({"value": float("nan"), "array": [float("inf"), 1.0]})
        self.assertTrue(changed)
        self.assertEqual(safe, {"value": None, "array": [None, 1.0]})
        json.dumps(safe, allow_nan=False)

    def test_real_audio_integrated_cli_emits_all_components_and_null_pte(self):
        if not SPEECHOCEAN_AUDIO.is_file() or not SPEECHOCEAN_MANIFEST.is_file():
            self.skipTest("verified local SpeechOcean762 fixture is unavailable")
        records = [json.loads(line) for line in SPEECHOCEAN_MANIFEST.read_text(encoding="utf-8").splitlines()]
        record = next((item for item in records if item["utterance_id"] == "001120097"), None)
        if record is None:
            self.skipTest("SpeechOcean762 manifest does not contain the selected fixture")
        process = subprocess.run(
            [sys.executable, str(ROOT / "score_integrated.py"), "--audio", str(SPEECHOCEAN_AUDIO),
             "--reference", record["reference_text"], "--threads", "2"],
            capture_output=True, text=True, timeout=180, cwd=ROOT,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(set(result["components"]), {"content", "pronunciation", "oral_fluency", "prosody", "word_stress"})
        self.assertIsInstance(result["components"]["content"]["predictions"]["transcript_raw"], str)
        self.assertIsNone(result["overall_pte_score"])
        self.assertIsNone(result["pearson_pte_score"])
        json.dumps(result, allow_nan=False)

    def test_speechocean_learner_audio_engineering_integration_schema(self):
        """Exercise the public pipeline on licensed, checkpoint-exposed learner audio."""
        if not SPEECHOCEAN_AUDIO.is_file() or not SPEECHOCEAN_MANIFEST.is_file():
            self.skipTest("verified SpeechOcean762 learner-audio fixture is unavailable")

        records = [json.loads(line) for line in SPEECHOCEAN_MANIFEST.read_text(encoding="utf-8").splitlines()]
        record = next((item for item in records if item["utterance_id"] == "001120097"), None)
        if record is None:
            self.skipTest("SpeechOcean762 manifest does not contain the selected fixture")
        expected_hash = record["audio"]["sha256"]
        actual_hash = hashlib.sha256(SPEECHOCEAN_AUDIO.read_bytes()).hexdigest()
        self.assertEqual(actual_hash, expected_hash)

        environment = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        process = subprocess.run(
            [sys.executable, str(ROOT / "score_integrated.py"), "--audio", str(SPEECHOCEAN_AUDIO),
             "--reference", record["reference_text"], "--threads", "2"],
            capture_output=True, text=True, timeout=180, cwd=ROOT, env=environment,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(set(result["components"]), {"content", "pronunciation", "oral_fluency", "prosody", "word_stress"})
        self.assertIsNone(result["overall_pte_score"])
        self.assertIsNone(result["pearson_pte_score"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
