"""Integrity and eligibility contract tests for the SpeechOcean inventory tool."""

from __future__ import annotations

import json
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from dataset_foundation import build_inventory, main


def _ratings(value: float) -> list[float]:
    return [value] * 5


def _detail(text: str = "HELLO") -> dict:
    return {
        "text": text,
        "accuracy": _ratings(8),
        "completeness": _ratings(1),
        "fluency": _ratings(8),
        "prosodic": _ratings(8),
        "total": _ratings(8),
        "words": [{
            "accuracy": _ratings(8),
            "stress": _ratings(10),
            "total": _ratings(8),
            "text": "HELLO",
            "ref-phones": "HH AH0 L OW1",
            "phones": ["HH AH0 L OW1"] * 5,
        }],
    }


def _aggregate(text: str = "HELLO") -> dict:
    return {
        "accuracy": 8,
        "completeness": 10,
        "fluency": 8,
        "prosodic": 8,
        "total": 8,
        "text": text,
        "words": [{
            "accuracy": 8,
            "phones": ["HH", "AH0", "L", "OW1"],
            "phones-accuracy": [2, 2, 2, 2],
            "stress": 10,
            "text": "HELLO",
            "total": 8,
        }],
    }


class DatasetFoundationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "resource").mkdir()
        self.rows = {
            "train": [("000010001", "speaker0001", "HELLO", 1000, 16000)],
            "test": [("000010002", "speaker0002", "HELLO", 2000, 16000)],
        }
        self._write_corpus()

    def tearDown(self):
        self.temp.cleanup()

    def _write_corpus(self):
        details = {}
        aggregates = {}
        for split, rows in self.rows.items():
            folder = self.root / split
            folder.mkdir(exist_ok=True)
            (folder / "utt2spk").write_text("".join(f"{utt} {speaker}\n" for utt, speaker, *_ in rows))
            (folder / "text").write_text("".join(f"{utt} {text}\n" for utt, _, text, *_ in rows))
            (folder / "wav.scp").write_text("".join(f"{utt} WAVE/SPEAKER{speaker[-4:]}/{utt}.WAV\n" for utt, speaker, *_ in rows))
            speakers = sorted({speaker for _, speaker, *_ in rows})
            (folder / "spk2age").write_text("".join(f"{speaker} 18\n" for speaker in speakers))
            (folder / "spk2gender").write_text("".join(f"{speaker} f\n" for speaker in speakers))
            for utt, speaker, text, sample, sample_rate in rows:
                speaker_dir = self.root / "WAVE" / f"SPEAKER{speaker[-4:]}"
                speaker_dir.mkdir(parents=True, exist_ok=True)
                audio_path = speaker_dir / f"{utt}.WAV"
                with wave.open(str(audio_path), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(sample_rate)
                    audio.writeframes(struct.pack("<h", sample) * 1000)
                details[utt] = _detail(text)
                aggregates[utt] = _aggregate(text)
        (self.root / "resource" / "scores-detail.json").write_text(json.dumps(details))
        (self.root / "resource" / "scores.json").write_text(json.dumps(aggregates))

    def test_valid_records_preserve_labels_and_are_never_independent(self):
        records, report = build_inventory(self.root, expected_utterances=2, expected_speakers=2)
        self.assertEqual(report["integrity_status"], "PASS")
        self.assertEqual(report["counts"]["utterances"], 2)
        self.assertEqual(report["counts"]["valid_audio"], 2)
        self.assertEqual(report["counts"]["speakers"], 2)
        self.assertEqual(report["split_checks"]["speaker_overlap_count"], 0)
        self.assertEqual(records[0]["evaluation_eligibility"], "checkpoint_exposed_not_independent")
        self.assertEqual(len(records[0]["annotations"]["phone"][0]["rater_phone_sequences"]), 5)
        self.assertEqual(records[0]["annotations"]["aggregate"]["total"], 8)
        self.assertIn("audio_sha256", records[0]["audio"])
        self.assertFalse(report["evaluation"]["independent_final_evaluation_eligible"])

    def test_duplicate_audio_and_speaker_leakage_are_reported(self):
        train_audio = self.root / "WAVE" / "SPEAKER0001" / "000010001.WAV"
        test_audio = self.root / "WAVE" / "SPEAKER0002" / "000010002.WAV"
        shutil.copyfile(train_audio, test_audio)
        (self.root / "test" / "utt2spk").write_text("000010002 speaker0001\n")
        records, report = build_inventory(self.root, expected_utterances=2, expected_speakers=2)
        self.assertEqual(report["integrity_status"], "FAIL")
        self.assertEqual(report["split_checks"]["speaker_overlap_ids"], ["speaker0001"])
        self.assertEqual(report["counts"]["duplicate_audio_utterances"], 2)
        self.assertTrue(all(record["evaluation_eligibility"] == "review_required" for record in records))

    def test_missing_transcript_invalid_rating_and_wrong_sample_rate_are_not_dropped(self):
        test_dir = self.root / "test"
        test_dir.joinpath("text").write_text("\n")
        test_dir.joinpath("utt2spk").write_text("000010002 speaker0001\n")
        details_path = self.root / "resource" / "scores-detail.json"
        details = json.loads(details_path.read_text())
        details["000010002"]["accuracy"] = [11, 8, 8, 8, 8]
        details_path.write_text(json.dumps(details))
        wrong_rate = self.root / "WAVE" / "SPEAKER0002" / "000010002.WAV"
        with wave.open(str(wrong_rate), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(struct.pack("<h", 2000) * 1000)
        records, report = build_inventory(self.root, expected_utterances=2, expected_speakers=2)
        test_record = next(record for record in records if record["utterance_id"] == "000010002")
        codes = {issue["code"] for issue in test_record["issues"]}
        self.assertIn("missing_transcript", codes)
        self.assertIn("rating_out_of_range", codes)
        self.assertIn("unexpected_sample_rate", codes)
        self.assertEqual(report["counts"]["utterances"], 2)
        self.assertEqual(test_record["evaluation_eligibility"], "review_required")

    def test_corrupt_audio_is_retained_with_an_exclusion_reason(self):
        audio_path = self.root / "WAVE" / "SPEAKER0002" / "000010002.WAV"
        audio_path.write_bytes(b"broken")
        records, report = build_inventory(self.root, expected_utterances=2, expected_speakers=2)
        test_record = next(record for record in records if record["utterance_id"] == "000010002")
        self.assertIn("corrupt_audio", {issue["code"] for issue in test_record["issues"]})
        self.assertEqual(test_record["evaluation_eligibility"], "review_required")
        self.assertEqual(report["counts"]["utterances"], 2)

    def test_transcript_punctuation_differences_preserve_both_source_forms(self):
        details_path = self.root / "resource" / "scores-detail.json"
        details = json.loads(details_path.read_text())
        details["000010001"]["text"] = "HELLO."
        details_path.write_text(json.dumps(details))
        records, report = build_inventory(self.root, expected_utterances=2, expected_speakers=2)
        record = next(value for value in records if value["utterance_id"] == "000010001")
        self.assertEqual(report["integrity_status"], "PASS")
        self.assertEqual(record["reference_text"], "HELLO")
        self.assertEqual(record["annotations"]["source_native"]["text"], "HELLO.")

    def test_cli_report_points_to_the_written_manifest(self):
        manifest = self.root / "inventory.jsonl"
        report_path = self.root / "report.json"
        arguments = [
            "dataset_foundation.py", "--root", str(self.root),
            "--manifest", str(manifest), "--report", str(report_path),
            "--expected-utterances", "2", "--expected-speakers", "2",
        ]
        with patch("sys.argv", arguments):
            self.assertEqual(main(), 0)
        report = json.loads(report_path.read_text())
        self.assertEqual(report["records_file"], str(manifest.resolve()))
        self.assertTrue(manifest.is_file())


if __name__ == "__main__":
    unittest.main()
