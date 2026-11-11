"""Local contract tests for the Kaldi GOP + GOPT audio scorer."""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

import score_pronunciation as scoring


ROOT = Path(__file__).resolve().parents[1]
GOPT_PYTHON = ROOT / ".local" / "gopt-venv" / "bin" / "python"


class ReferenceAndPhoneTests(unittest.TestCase):
    def test_transcript_normalization(self):
        self.assertEqual(scoring.normalize_reference("  Maple, grows QUIETLY-near the stream! "),
                         ["MAPLE", "GROWS", "QUIETLY", "NEAR", "THE", "STREAM"])

    def test_leading_elision_apostrophe_does_not_invalidate_reference(self):
        self.assertEqual(
            scoring.normalize_reference("Give 'em what they don't have."),
            ["GIVE", "EM", "WHAT", "THEY", "DON'T", "HAVE"],
        )
        self.assertEqual(scoring.normalize_reference("Give ’em."), ["GIVE", "EM"])

    def test_transcript_empty_digits_and_unsupported_characters_fail_closed(self):
        for reference in ("...", "yellow 12", "hello @world", "don''t", "''em"):
            with self.subTest(reference=reference), self.assertRaises(scoring.ScoreError):
                scoring.normalize_reference(reference)

    def test_lexicon_oov_and_unsupported_phone(self):
        with self.assertRaisesRegex(scoring.ScoreError, "absent"):
            scoring.expected_phones(["MISSING"], {}, {"AH": 4})
        with self.assertRaisesRegex(scoring.ScoreError, "absent from the acoustic model"):
            scoring.expected_phones(["WORD"], {"WORD": [["XYZ_B"]]}, {"AH": 4})

    def test_lexicon_phone_inventory_keeps_complete_long_reference(self):
        lexicon = {"WORD": [["AH_B"]]}
        expected, _ = scoring.expected_phones(["WORD"] * 51, lexicon, {"AH": 4})
        self.assertEqual(len(expected), 51)
        self.assertEqual([phone["word_index"] for phone in expected], list(range(51)))

    def test_source_phone_inventory_keeps_complete_long_reference(self):
        expected = scoring.expected_phones_from_source(
            ["WORD"] * 51, [["AH_B"] for _ in range(51)], {"AH": 4},
        )
        self.assertEqual(len(expected), 51)
        self.assertEqual([phone["word_index"] for phone in expected], list(range(51)))

    def test_gopt_windows_cover_global_phones_once_at_whole_word_boundaries(self):
        expected = [
            {"word_index": word_index, "phone": f"P{phone_index}"}
            for word_index, phone_count in enumerate((25, 25, 3, 47))
            for phone_index in range(phone_count)
        ]
        windows = scoring.plan_gopt_windows(expected)
        self.assertEqual(
            windows,
            [
                {"phone_start_index": 0, "phone_end_index": 50,
                 "word_start_index": 0, "word_end_index": 2},
                {"phone_start_index": 50, "phone_end_index": 100,
                 "word_start_index": 2, "word_end_index": 4},
            ],
        )
        self.assertTrue(all(window["phone_end_index"] - window["phone_start_index"] <= 50 for window in windows))

    def test_gopt_window_boundaries_cover_short_exact_and_long_sequences(self):
        for count in (1, 49, 50, 51, 60, 100, 101, 151):
            with self.subTest(phone_count=count):
                expected = [{"word_index": index} for index in range(count)]
                windows = scoring.plan_gopt_windows(expected)
                self.assertEqual(sum(w["phone_end_index"] - w["phone_start_index"] for w in windows), count)
                self.assertEqual(windows[0]["phone_start_index"], 0)
                self.assertEqual(windows[-1]["phone_end_index"], count)
                self.assertTrue(all(0 < w["phone_end_index"] - w["phone_start_index"] <= 50 for w in windows))

    def test_gopt_window_planner_rejects_missing_global_word_indexes(self):
        expected = [{"word_index": 0}, {"word_index": 2}]
        with self.assertRaisesRegex(scoring.ScoreError, "consecutive global word indices"):
            scoring.plan_gopt_windows(expected)

    def test_gopt_window_count_is_bounded(self):
        expected = [{"word_index": index} for index in range(scoring.MAX_GOPT_WINDOWS * 50 + 1)]
        with self.assertRaisesRegex(scoring.ScoreError, "bounded GOPT window count"):
            scoring.plan_gopt_windows(expected)

    def test_gopt_window_does_not_split_a_word_larger_than_model_capacity(self):
        expected = [{"word_index": 0, "phone": f"P{index}"} for index in range(51)]
        with self.assertRaisesRegex(scoring.ScoreError, "word exceeds the 50-phone GOPT window"):
            scoring.plan_gopt_windows(expected)

    def test_window_predictions_reconstruct_global_phone_order_exactly_once(self):
        windows = [
            {"phone_start_index": 0, "phone_end_index": 2,
             "word_start_index": 0, "word_end_index": 1},
            {"phone_start_index": 2, "phone_end_index": 3,
             "word_start_index": 1, "word_end_index": 2},
        ]
        scores = scoring.stitch_gopt_phone_scores([[0.1, 0.2], [0.3]], windows, 3)
        self.assertEqual(scores, [0.1, 0.2, 0.3])

    def test_window_predictions_reject_duplicate_or_missing_global_phone_ownership(self):
        overlapping = [
            {"phone_start_index": 0, "phone_end_index": 2,
             "word_start_index": 0, "word_end_index": 1},
            {"phone_start_index": 1, "phone_end_index": 3,
             "word_start_index": 1, "word_end_index": 2},
        ]
        with self.assertRaisesRegex(scoring.ScoreError, "more than one GOPT window"):
            scoring.stitch_gopt_phone_scores([[0.1, 0.2], [0.2, 0.3]], overlapping, 3)
        with self.assertRaisesRegex(scoring.ScoreError, "not covered by a GOPT window"):
            scoring.stitch_gopt_phone_scores([[0.1]], windows := [
                {"phone_start_index": 0, "phone_end_index": 1,
                 "word_start_index": 0, "word_end_index": 1},
            ], 2)

    def test_partial_window_predictions_preserve_global_indices_and_null_gaps(self):
        windows = [
            {"phone_start_index": 0, "phone_end_index": 2,
             "word_start_index": 0, "word_end_index": 1},
            {"phone_start_index": 4, "phone_end_index": 6,
             "word_start_index": 2, "word_end_index": 3},
        ]
        scores = scoring.stitch_gopt_phone_scores(
            [[0.1, 0.2], [0.5, 0.6]], windows, 6, allow_partial=True,
        )
        self.assertEqual(scores, [0.1, 0.2, None, None, 0.5, 0.6])
        with self.assertRaisesRegex(scoring.ScoreError, "not covered by a GOPT window"):
            scoring.stitch_gopt_phone_scores(
                [[0.1, 0.2], [0.5, 0.6]], windows, 6,
            )

    def test_alignment_support_withholds_a_whole_window_and_reports_global_failures(self):
        windows = [
            {"phone_start_index": 0, "phone_end_index": 2,
             "word_start_index": 0, "word_end_index": 1},
            {"phone_start_index": 2, "phone_end_index": 4,
             "word_start_index": 1, "word_end_index": 2},
        ]
        active, diagnostics = scoring.partition_gopt_windows_by_support(
            windows, [-1.0, -8.0, -1.0, -2.0],
        )
        self.assertEqual([window["window_index"] for window in active], [1])
        self.assertEqual(diagnostics[0]["status"], "WITHHELD_ALIGNMENT_SUPPORT")
        self.assertEqual(diagnostics[0]["unsupported_phone_indices"], [1])
        self.assertEqual(diagnostics[1]["status"], "READY_FOR_GOPT")

    def test_alignment_support_isolates_weak_whole_word_regions(self):
        windows = [{
            "phone_start_index": 0, "phone_end_index": 6,
            "word_start_index": 0, "word_end_index": 4,
        }]
        expected = [
            {"word_index": 0},
            {"word_index": 1}, {"word_index": 1},
            {"word_index": 2},
            {"word_index": 3}, {"word_index": 3},
        ]
        active, diagnostics = scoring.partition_gopt_windows_by_support(
            windows, [-1.0, -1.0, -8.0, -1.0, -1.0, -1.0], expected=expected,
        )
        self.assertEqual(
            [(item["phone_start_index"], item["phone_end_index"]) for item in active],
            [(0, 1), (3, 6)],
        )
        self.assertEqual(
            [item["status"] for item in diagnostics],
            ["READY_FOR_GOPT", "WITHHELD_ALIGNMENT_SUPPORT", "READY_FOR_GOPT"],
        )
        self.assertEqual(diagnostics[1]["unsupported_phone_indices"], [2])

    def test_source_phone_transcript_preserves_canonical_word_order(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "text-phone"
            path.write_text("other.0 W_B\nutt.1 K_B EH0_I T_E\nutt.0 M_B AA0_I R_I K_E\n")
            self.assertEqual(
                scoring.read_source_phone_transcript(path, "utt"),
                [["M_B", "AA0_I", "R_I", "K_E"], ["K_B", "EH0_I", "T_E"]],
            )

    def test_source_phone_sequence_is_mapped_without_lexicon_substitution(self):
        expected = scoring.expected_phones_from_source(
            ["MARK"], [["M_B", "AA0_I", "R_I", "K_E"]],
            {"M": 3, "AA": 4, "R": 5, "K": 6},
        )
        self.assertEqual([phone["phone"] for phone in expected], ["M", "AA", "R", "K"])
        self.assertEqual([phone["marked_phone"] for phone in expected], ["M_B", "AA0_I", "R_I", "K_E"])
        with self.assertRaisesRegex(scoring.ScoreError, "word count differs"):
            scoring.expected_phones_from_source(["MARK", "IS"], [["M_B"]], {"M": 3})

    def test_feature_reconstruction_mode_requires_source_canonical_phones(self):
        with patch.object(scoring, "require_assets"):
            with self.assertRaisesRegex(scoring.ScoreError, "requires a source-native canonical phone sequence"):
                scoring.score(Path("missing.wav"), "MARK", feature_reconstruction_only=True)

    def test_training_phone_order_is_explicit_and_matches_inventory_size(self):
        self.assertEqual(len(scoring.GOPT_PHONE_ORDER), 39)
        self.assertEqual(len(set(scoring.GOPT_PHONE_ORDER)), 39)


class FeatureContractTests(unittest.TestCase):
    def test_acoustic_output_branches_have_separate_feature_and_support_roles(self):
        self.assertEqual(scoring.FEATURE_ACOUSTIC_OUTPUT_NODE, "output")
        self.assertEqual(scoring.ALIGNMENT_SUPPORT_OUTPUT_NODE, "output-xent")
        rows = np.zeros((1, 85), dtype=np.float64)
        rows[0, 0] = 5
        expected = [{"pure_phone_id": 5}]
        scoring.validate_gop_phone_order(rows, [5], expected)
        rows[0, 0] = 6
        with self.assertRaisesRegex(scoring.ScoreError, "canonical aligned phone order"):
            scoring.validate_gop_phone_order(rows, [5], expected)

    def test_gop_dimensions_and_phone_ids(self):
        features = np.zeros((2, scoring.FEATURE_DIM), dtype=np.float32)
        scoring.verify_gopt_inputs(features, [0, 38])
        with self.assertRaisesRegex(scoring.ScoreError, "shape"):
            scoring.verify_gopt_inputs(np.zeros((2, 83), dtype=np.float32), [0, 1])
        with self.assertRaisesRegex(scoring.ScoreError, "39-class"):
            scoring.verify_gopt_inputs(features, [39, 1])
        with self.assertRaisesRegex(scoring.ScoreError, "integer class IDs"):
            scoring.verify_gopt_inputs(np.zeros((1, scoring.FEATURE_DIM), dtype=np.float32), [0.5])

    def test_full_gop_sequence_validation_allows_more_than_one_model_window(self):
        scoring.verify_gopt_sequence_inputs(
            np.zeros((51, scoring.FEATURE_DIM), dtype=np.float32), [0] * 51,
        )

    def test_gopt_output_validation_accepts_multiple_windows(self):
        outputs = [np.zeros((2, 1), dtype=np.float32) for _ in range(5)]
        outputs.extend(np.zeros((2, 50, 1), dtype=np.float32) for _ in range(4))
        self.assertEqual(len(scoring.validate_gopt_outputs(outputs, batch_size=2)), 9)

    def test_nonfinite_gop_features_fail_closed(self):
        features = np.zeros((1, scoring.FEATURE_DIM), dtype=np.float32)
        features[0, 17] = math.nan
        with self.assertRaisesRegex(scoring.ScoreError, "NaN or infinity"):
            scoring.verify_gopt_inputs(features, [0])

    def test_training_normalization_and_zero_padding(self):
        raw = np.full((2, scoring.FEATURE_DIM), scoring.GOPT_MEAN, dtype=np.float32)
        feats, ids = scoring.prepare_gopt_batch(raw, [2, 8])
        self.assertEqual(feats.shape, (1, 50, 84))
        self.assertEqual(ids.shape, (1, 50))
        np.testing.assert_array_equal(feats[0, :2], np.zeros((2, 84), dtype=np.float32))
        np.testing.assert_array_equal(feats[0, 2:], np.zeros((48, 84), dtype=np.float32))
        np.testing.assert_array_equal(ids[0], np.array([2, 8] + [-1] * 48))

    def test_acoustic_support_floor_rejects_bad_forced_alignment(self):
        matrix = np.zeros((1, 43), dtype=np.float64)
        matrix[0, 1 + 5] = -8.0
        expected = [{"word": "YELLOW", "phone": "Z"}]
        with self.assertRaisesRegex(scoring.ScoreError, "negligible acoustic posterior"):
            scoring.validate_phone_acoustic_support(matrix, [5], expected)

    def test_alignment_and_posterior_frame_counts_must_match(self):
        scoring.validate_alignment_frame_count(100, 100)
        with self.assertRaisesRegex(scoring.ScoreError, "frame counts differ"):
            scoring.validate_alignment_frame_count(100, 99)

    def test_malformed_gopt_outputs_fail_closed(self):
        valid = [np.zeros((1, 1), dtype=np.float32) for _ in range(5)]
        valid.extend(np.zeros((1, 50, 1), dtype=np.float32) for _ in range(4))
        self.assertEqual(len(scoring.validate_gopt_outputs(valid)), 9)
        with self.assertRaisesRegex(scoring.ScoreError, "unexpected output tuple"):
            scoring.validate_gopt_outputs(valid[:-1])
        valid[-1][0, 0, 0] = np.inf
        with self.assertRaisesRegex(scoring.ScoreError, "invalid shape or non-finite"):
            scoring.validate_gopt_outputs(valid)

    def test_missing_model_assets_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_file = Path(directory) / "missing.mdl"
            missing_dir = Path(directory) / "missing-bin"
            self.assertEqual(scoring.missing_assets([missing_file], [missing_dir]),
                             [str(missing_file), str(missing_dir)])


class AudioInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise unittest.SkipTest("ffmpeg/ffprobe unavailable")

    def test_empty_corrupt_and_silent_inputs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty.wav"
            empty.touch()
            corrupt = root / "corrupt.wav"
            corrupt.write_bytes(b"not a wave file")
            silent = root / "silent.wav"
            with wave.open(str(silent), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(struct.pack("<h", 0) * 16000)
            for audio in (empty, corrupt, silent):
                with self.subTest(audio=audio.name), self.assertRaises(scoring.ScoreError):
                    with tempfile.TemporaryDirectory() as work:
                        scoring.validate_audio(audio, Path(work))

    def test_duration_limit_is_enforced_before_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "valid.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(struct.pack("<h", 1000) * 3200)
            with patch.object(scoring, "MAX_AUDIO_SECONDS", 0.1):
                with self.assertRaisesRegex(scoring.ScoreError, "120-second"):
                    with tempfile.TemporaryDirectory() as work:
                        scoring.validate_audio(audio, Path(work))

    def test_cli_returns_structured_error_for_invalid_audio(self):
        if not GOPT_PYTHON.is_file():
            self.skipTest("local GOPT CPU runtime is not installed")
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "empty.wav"
            invalid.touch()
            process = subprocess.run(
                [str(GOPT_PYTHON), str(ROOT / "score_pronunciation.py"),
                 "--audio", str(invalid), "--reference", "Hello world"],
                capture_output=True, text=True, timeout=60, cwd=ROOT,
            )
            self.assertEqual(process.returncode, 2, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(result["status"], "AUDIO_DECODE_FAILED")
            self.assertIsNone(result["pronunciation"])
            self.assertIsNone(result["pearson_pte_score"])


@unittest.skipUnless(GOPT_PYTHON.is_file(), "local GOPT CPU runtime is not installed")
class CheckpointFailureTests(unittest.TestCase):
    def test_production_single_pass_matches_explicit_reproducibility_validation(self):
        script = """
import json, pathlib, sys, types, numpy as np, torch, score_pronunciation as s
class FakeGOPT:
    calls = 0
    def __init__(self, **kwargs): pass
    def load_state_dict(self, state, strict=True): pass
    def eval(self): pass
    def __call__(self, inputs, phone_ids):
        self.__class__.calls += 1
        batch = inputs.shape[0]
        utterance = [torch.full((batch, 1), float(index)) for index in range(5)]
        phone = torch.full((batch, 50, 1), 0.5)
        words = [torch.full((batch, 50, 1), float(index + 1)) for index in range(3)]
        return tuple(utterance + [phone] + words)
models = types.ModuleType('models'); models.__path__ = []
models.gopt = types.ModuleType('models.gopt'); models.gopt.GOPT = FakeGOPT
sys.modules['models'] = models; sys.modules['models.gopt'] = models.gopt
torch.load = lambda *args, **kwargs: {}
s.CHECKPOINT = pathlib.Path('/dev/null')
raw = np.zeros((2, s.FEATURE_DIM), dtype=np.float32)
phones = [0, 1]
production, _ = s.load_and_run_gopt(raw, phones, 1)
production_calls = FakeGOPT.calls
FakeGOPT.calls = 0
validated, _ = s.load_and_run_gopt(raw, phones, 1, validate_reproducibility=True)
print(json.dumps({
    'production_calls': production_calls,
    'validation_calls': FakeGOPT.calls,
    'production_scores': production,
    'validated_scores': validated,
}))
"""
        process = subprocess.run(
            [str(GOPT_PYTHON), "-c", script], capture_output=True, text=True,
            timeout=30, cwd=ROOT,
        )
        self.assertEqual(process.returncode, 0, process.stderr or process.stdout)
        result = json.loads(process.stdout)
        self.assertEqual(result["production_calls"], 1)
        self.assertEqual(result["validation_calls"], 2)
        # Timing metadata intentionally differs; all returned score fields match.
        production, validated = result["production_scores"], result["validated_scores"]
        for key in ("window_predictions", "phoneme_scores", "word_phone_scores", "utterance"):
            self.assertEqual(production[key], validated[key])

    def test_nonfinite_repeat_inference_fails_closed(self):
        script = """
import pathlib, sys, types, numpy as np, torch, score_pronunciation as s
class FakeGOPT:
    def __init__(self, **kwargs): self.calls = 0
    def load_state_dict(self, state, strict=True): pass
    def eval(self): pass
    def __call__(self, inputs, phone_ids):
        self.calls += 1
        value = 0.0 if self.calls == 1 else float("nan")
        outputs = [torch.full((inputs.shape[0], 1), value) for _ in range(5)]
        outputs.extend(torch.full((inputs.shape[0], 50, 1), value) for _ in range(4))
        return tuple(outputs)
models = types.ModuleType("models")
models.__path__ = []
models.gopt = types.ModuleType("models.gopt")
models.gopt.GOPT = FakeGOPT
sys.modules["models"] = models
sys.modules["models.gopt"] = models.gopt
torch.load = lambda *args, **kwargs: {}
s.CHECKPOINT = pathlib.Path("/dev/null")
try:
    s.load_and_run_gopt(np.zeros((1, s.FEATURE_DIM), dtype=np.float32), [0], 1, validate_reproducibility=True)
except s.ScoreError as error:
    print(error.status, error.stage)
else:
    raise SystemExit("non-finite repeated inference unexpectedly succeeded")
"""
        process = subprocess.run(
            [str(GOPT_PYTHON), "-c", script], capture_output=True, text=True,
            timeout=30, cwd=ROOT,
        )
        self.assertEqual(process.returncode, 0, process.stderr or process.stdout)
        self.assertEqual(process.stdout.strip(), "GOPT_INFERENCE_FAILED gopt_output_validation")

    def test_gopt_inference_scores_long_sequences_in_separate_fixed_windows(self):
        script = """
import json, numpy as np, score_pronunciation as s
raw = np.zeros((51, s.FEATURE_DIM), dtype=np.float32)
phones = [0] * 51
windows = [
    {"phone_start_index": 0, "phone_end_index": 26, "word_start_index": 0, "word_end_index": 1},
    {"phone_start_index": 26, "phone_end_index": 51, "word_start_index": 1, "word_end_index": 2},
]
result, _ = s.load_and_run_gopt(raw, phones, 1, windows=windows)
print(json.dumps({
    "phone_count": len(result["phoneme_scores"]),
    "word_phone_count": len(result["word_phone_scores"]),
    "window_count": len(result["window_predictions"]),
    "window_phone_counts": [item["phone_count"] for item in result["window_predictions"]],
    "aggregation_status": result["utterance"]["aggregation_status"],
    "utterance_overall": result["utterance"]["overall"],
}))
"""
        process = subprocess.run(
            [str(GOPT_PYTHON), "-c", script], capture_output=True, text=True,
            timeout=60, cwd=ROOT,
        )
        self.assertEqual(process.returncode, 0, process.stderr or process.stdout)
        result = json.loads(process.stdout)
        self.assertEqual(result["phone_count"], 51)
        self.assertEqual(result["word_phone_count"], 51)
        self.assertEqual(result["window_count"], 2)
        self.assertEqual(result["window_phone_counts"], [26, 25])
        self.assertEqual(result["aggregation_status"], "NOT_VALIDATED")
        self.assertIsNone(result["utterance_overall"])

    def test_partial_gopt_inference_preserves_windows_and_null_global_gap(self):
        script = """
import json, sys, types, numpy as np, torch, score_pronunciation as s
class FakeGOPT:
    def __init__(self, **kwargs): pass
    def load_state_dict(self, state, strict=True): pass
    def eval(self): pass
    def __call__(self, inputs, phone_ids):
        batch = inputs.shape[0]
        values = torch.arange(batch, dtype=torch.float32)
        utterance = [values.reshape(batch, 1) + 1 for _ in range(5)]
        phone = values.reshape(batch, 1, 1).expand(batch, 50, 1)
        words = [phone.clone() for _ in range(3)]
        return tuple(utterance + [phone] + words)
models = types.ModuleType('models'); models.__path__ = []
models.gopt = types.ModuleType('models.gopt'); models.gopt.GOPT = FakeGOPT
sys.modules['models'] = models; sys.modules['models.gopt'] = models.gopt
torch.load = lambda *args, **kwargs: {}
s.CHECKPOINT = __import__('pathlib').Path('/dev/null')
windows = [
    {'phone_start_index': 0, 'phone_end_index': 2, 'word_start_index': 0, 'word_end_index': 1, 'window_index': 3},
    {'phone_start_index': 4, 'phone_end_index': 6, 'word_start_index': 2, 'word_end_index': 3, 'window_index': 7},
]
result, _ = s.load_and_run_gopt(
    np.zeros((6, s.FEATURE_DIM), dtype=np.float32), [0] * 6, 1,
    windows=windows, allow_partial=True,
)
print(json.dumps({
    'phone_scores': result['phoneme_scores'],
    'word_scores': result['word_phone_scores'],
    'window_indices': [item['window_index'] for item in result['window_predictions']],
    'aggregation_status': result['utterance']['aggregation_status'],
    'utterance_overall': result['utterance']['overall'],
}))
"""
        process = subprocess.run(
            [str(GOPT_PYTHON), "-c", script], capture_output=True, text=True,
            timeout=30, cwd=ROOT,
        )
        self.assertEqual(process.returncode, 0, process.stderr or process.stdout)
        result = json.loads(process.stdout)
        self.assertEqual(result["phone_scores"][2:4], [None, None])
        self.assertEqual(result["phone_scores"][:2], [0.0, 0.0])
        self.assertEqual(result["phone_scores"][4:], [1.0, 1.0])
        self.assertEqual(result["word_scores"][2:4], [None, None])
        self.assertEqual(result["window_indices"], [3, 7])
        self.assertEqual(result["aggregation_status"], "NOT_VALIDATED")
        self.assertIsNone(result["utterance_overall"])

    def test_incompatible_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            bad_checkpoint = Path(directory) / "bad-checkpoint.pth"
            bad_checkpoint.write_text("not a torch checkpoint", encoding="utf-8")
            script = f"""
import json, pathlib, numpy as np, score_pronunciation as s
s.CHECKPOINT = pathlib.Path({str(bad_checkpoint)!r})
try:
    s.load_and_run_gopt(np.zeros((1, 84), dtype=np.float32), [0], 1)
except s.ScoreError as error:
    print(json.dumps({{'status': error.status, 'stage': error.stage}}))
else:
    raise SystemExit('incompatible checkpoint unexpectedly scored')
"""
            process = subprocess.run(
                [str(GOPT_PYTHON), "-c", script], capture_output=True, text=True,
                timeout=60, cwd=ROOT,
            )
            self.assertEqual(process.returncode, 0, process.stderr or process.stdout)
            self.assertEqual(json.loads(process.stdout), {
                "status": "MODEL_INCOMPATIBLE", "stage": "gopt_model_loading",
            })


if __name__ == "__main__":
    unittest.main()
