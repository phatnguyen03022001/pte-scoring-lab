#!/usr/bin/env python3
"""Inventory and integrity checks for an extracted SpeechOcean762 corpus.

The tool keeps every source utterance in its JSONL manifest. Problems are
recorded on the utterance instead of silently dropping it. The corpus is
checkpoint-selection-exposed and is therefore never eligible as an
independent final evaluation set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any


SOURCE_URL = "https://github.com/jimbozhang/speechocean762"
LICENSE = "CC BY 4.0"
RATER_COUNT = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _issue(code: str, detail: str, source: str | None = None) -> dict[str, str]:
    value = {"code": code, "detail": detail}
    if source is not None:
        value["source"] = source
    return value


def _read_key_value(path: Path, global_issues: list[dict[str, str]]) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        global_issues.append(_issue("missing_metadata_file", f"Required metadata file is missing: {path.name}", str(path)))
        return values
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            global_issues.append(_issue("malformed_metadata_line", f"{path.name}:{line_number} must contain a key and value", str(path)))
            continue
        key, value = fields
        if key in values:
            global_issues.append(_issue("duplicate_utterance_id", f"{key} occurs more than once in {path.name}", str(path)))
            continue
        values[key] = value
    return values


def _numeric_scores(
    values: Any,
    *,
    field: str,
    low: float,
    high: float,
    issues: list[dict[str, str]],
    expected_count: int = RATER_COUNT,
) -> None:
    if not isinstance(values, list) or len(values) != expected_count:
        issues.append(_issue("invalid_rating_coverage", f"{field} must contain {expected_count} independent ratings"))
        return
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            issues.append(_issue("invalid_rating_value", f"{field}[{index}] is not a finite number"))
        elif not low <= float(value) <= high:
            issues.append(_issue("rating_out_of_range", f"{field}[{index}]={value} is outside [{low}, {high}]"))


def _validate_annotations(detail: Any, issues: list[dict[str, str]]) -> None:
    if not isinstance(detail, dict):
        issues.append(_issue("missing_or_invalid_labels", "scores-detail entry is not an object"))
        return
    for field in ("accuracy", "fluency", "prosodic", "total"):
        _numeric_scores(detail.get(field), field=field, low=0, high=10, issues=issues)
    _numeric_scores(detail.get("completeness"), field="completeness", low=0, high=1, issues=issues)
    words = detail.get("words")
    if not isinstance(words, list) or not words:
        issues.append(_issue("broken_phone_annotations", "words must be a non-empty list"))
        return
    for word_index, word in enumerate(words):
        prefix = f"words[{word_index}]"
        if not isinstance(word, dict):
            issues.append(_issue("broken_phone_annotations", f"{prefix} is not an object"))
            continue
        if not isinstance(word.get("text"), str) or not word["text"].strip():
            issues.append(_issue("missing_word_text", f"{prefix}.text is missing"))
        if not isinstance(word.get("ref-phones"), str) or not word["ref-phones"].strip():
            issues.append(_issue("broken_phone_annotations", f"{prefix}.ref-phones is missing"))
        rater_phones = word.get("phones")
        if not isinstance(rater_phones, list) or len(rater_phones) != RATER_COUNT or any(not isinstance(item, str) or not item.strip() for item in rater_phones):
            issues.append(_issue("broken_phone_annotations", f"{prefix}.phones must preserve {RATER_COUNT} non-empty rater sequences"))
        for field in ("accuracy", "total"):
            _numeric_scores(word.get(field), field=f"{prefix}.{field}", low=0, high=10, issues=issues)
        _numeric_scores(word.get("stress"), field=f"{prefix}.stress", low=5, high=10, issues=issues)
        if isinstance(word.get("stress"), list) and any(value not in (5, 10, 5.0, 10.0) for value in word["stress"]):
            issues.append(_issue("rating_out_of_range", f"{prefix}.stress must use the source scale {{5, 10}}"))


def _validate_aggregate_scores(aggregate: Any, issues: list[dict[str, str]]) -> None:
    if not isinstance(aggregate, dict):
        issues.append(_issue("missing_or_invalid_aggregate_scores", "scores.json entry is not an object"))
        return
    for field in ("accuracy", "completeness", "fluency", "prosodic", "total"):
        value = aggregate.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 10:
            issues.append(_issue("invalid_aggregate_score", f"{field} must be a finite scalar in [0, 10]"))
    words = aggregate.get("words")
    if not isinstance(words, list) or not words:
        issues.append(_issue("broken_aggregate_phone_scores", "scores.json words must be a non-empty list"))
        return
    for word_index, word in enumerate(words):
        prefix = f"scores.json words[{word_index}]"
        if not isinstance(word, dict) or not isinstance(word.get("text"), str) or not word["text"].strip():
            issues.append(_issue("broken_aggregate_phone_scores", f"{prefix}.text is missing"))
            continue
        for field in ("accuracy", "stress", "total"):
            value = word.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 10:
                issues.append(_issue("invalid_aggregate_word_score", f"{prefix}.{field} must be a finite scalar in [0, 10]"))
        phones = word.get("phones")
        values = word.get("phones-accuracy")
        if not isinstance(phones, list) or not phones or any(not isinstance(phone, str) or not phone for phone in phones):
            issues.append(_issue("broken_aggregate_phone_scores", f"{prefix}.phones must contain source phone symbols"))
        if not isinstance(values, list) or not isinstance(phones, list) or len(values) != len(phones):
            issues.append(_issue("broken_aggregate_phone_scores", f"{prefix}.phones-accuracy must align with its phone sequence"))
        elif any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 2 for value in values):
            issues.append(_issue("invalid_aggregate_phone_score", f"{prefix}.phones-accuracy must use the source scale [0, 2]"))


def _normalized_transcript_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Z0-9]+(?:'[A-Z0-9]+)?", value.upper()))


def _audio_metadata(root: Path, raw_path: str, expected_sample_rate: int) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    if "|" in raw_path:
        return None, [_issue("unsupported_audio_specifier", "wav.scp command pipelines are not executed by this tool")]
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None, [_issue("unsupported_audio_path", "audio path resolves outside the supplied corpus root")]
    if resolved.suffix.lower() != ".wav":
        return None, [_issue("unsupported_audio_file", f"unsupported audio extension: {resolved.suffix or '(none)'}")]
    if not resolved.is_file():
        return None, [_issue("missing_audio", f"audio file does not exist: {resolved.relative_to(root.resolve())}")]
    try:
        container_hash = _sha256(resolved)
        pcm_digest = hashlib.sha256()
        with wave.open(str(resolved), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            if channels <= 0 or sample_width not in (1, 2, 3, 4) or sample_rate <= 0 or frame_count <= 0:
                issues.append(_issue("invalid_audio_metadata", "WAV has invalid channels, sample width, sample rate, or frame count"))
            if sample_rate != expected_sample_rate:
                issues.append(_issue("unexpected_sample_rate", f"expected {expected_sample_rate} Hz, observed {sample_rate} Hz"))
            if channels != 1:
                issues.append(_issue("unexpected_channel_count", f"expected mono audio, observed {channels} channels"))
            remaining = frame_count
            while remaining:
                frames = audio.readframes(min(remaining, 65536))
                if not frames:
                    raise EOFError("audio ended before declared frame count")
                pcm_digest.update(frames)
                remaining -= len(frames) // (channels * sample_width)
            metadata = {
                "audio_path": resolved.relative_to(root.resolve()).as_posix(),
                "audio_sha256": container_hash,
                "decoded_pcm_sha256": pcm_digest.hexdigest(),
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width_bytes": sample_width,
                "frame_count": frame_count,
                "duration_seconds": frame_count / sample_rate if sample_rate > 0 else None,
            }
            return metadata, issues
    except (OSError, EOFError, wave.Error) as exc:
        return None, [_issue("corrupt_audio", f"WAV decode failed: {exc}")]


def _load_json(path: Path, global_issues: list[dict[str, str]], code: str) -> dict[str, Any]:
    if not path.is_file():
        global_issues.append(_issue("missing_annotation_file", f"Required annotation file is missing: {path}", str(path)))
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("top-level value must be an object")
        return value
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        global_issues.append(_issue(code, f"Could not load {path}: {exc}", str(path)))
        return {}


def build_inventory(
    root: Path,
    expected_utterances: int | None = 5000,
    expected_sample_rate: int = 16000,
    expected_speakers: int | None = 250,
    source_version: str = "unverified_local_source_version",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = root.resolve()
    global_issues: list[dict[str, str]] = []
    split_maps: dict[str, dict[str, dict[str, str]]] = {}
    split_speakers: dict[str, set[str]] = {}
    split_ids: dict[str, set[str]] = {}
    metadata_paths: list[Path] = []

    for split in ("train", "test"):
        split_dir = root / split
        maps = {}
        for name in ("utt2spk", "text", "wav.scp"):
            path = split_dir / name
            maps[name] = _read_key_value(path, global_issues)
            if path.is_file():
                metadata_paths.append(path)
        for name in ("spk2age", "spk2gender"):
            path = split_dir / name
            maps[name] = _read_key_value(path, global_issues)
            if path.is_file():
                metadata_paths.append(path)
        split_maps[split] = maps
        split_ids[split] = set(maps["utt2spk"]) | set(maps["text"]) | set(maps["wav.scp"])
        split_speakers[split] = set(maps["utt2spk"].values())

    labels_path = root / "resource" / "scores-detail.json"
    labels = _load_json(labels_path, global_issues, "invalid_annotation_json")
    if labels_path.is_file():
        metadata_paths.append(labels_path)
    aggregate_path = root / "resource" / "scores.json"
    aggregate_scores = _load_json(aggregate_path, global_issues, "invalid_aggregate_json")
    if aggregate_path.is_file():
        metadata_paths.append(aggregate_path)

    overlap = sorted(split_speakers.get("train", set()) & split_speakers.get("test", set()))
    if overlap:
        global_issues.append(_issue("speaker_split_leakage", f"{len(overlap)} speaker IDs occur in both source splits"))
    id_overlap = sorted(split_ids.get("train", set()) & split_ids.get("test", set()))
    if id_overlap:
        global_issues.append(_issue("utterance_split_overlap", f"{len(id_overlap)} utterance IDs occur in both source splits"))

    assignment: dict[str, str] = {}
    for split, ids in split_ids.items():
        for utterance_id in ids:
            assignment.setdefault(utterance_id, split)
    for utterance_id in labels:
        assignment.setdefault(utterance_id, "unassigned")
    all_ids = sorted(assignment)

    records: list[dict[str, Any]] = []
    container_hash_to_ids: dict[str, list[str]] = defaultdict(list)
    pcm_hash_to_ids: dict[str, list[str]] = defaultdict(list)
    label_coverage = transcript_coverage = speaker_coverage = valid_audio_count = 0
    per_split: dict[str, dict[str, int]] = defaultdict(lambda: {"utterances": 0, "valid_audio": 0, "labelled": 0, "transcribed": 0, "speaker_identified": 0})

    for utterance_id in all_ids:
        split = assignment[utterance_id]
        maps = split_maps.get(split, {})
        issues: list[dict[str, str]] = []
        speaker_id = maps.get("utt2spk", {}).get(utterance_id)
        raw_audio_path = maps.get("wav.scp", {}).get(utterance_id)
        metadata = None
        if not speaker_id:
            issues.append(_issue("missing_speaker_id", "utterance has no source utt2spk entry"))
        else:
            speaker_coverage += 1
            if speaker_id not in maps.get("spk2age", {}):
                issues.append(_issue("missing_speaker_metadata", "speaker has no source age entry"))
            else:
                try:
                    if int(maps["spk2age"][speaker_id]) <= 0:
                        raise ValueError("age must be positive")
                except ValueError:
                    issues.append(_issue("invalid_speaker_metadata", "speaker age is not a positive integer"))
            if maps.get("spk2gender", {}).get(speaker_id, "").lower() not in {"m", "f"}:
                issues.append(_issue("missing_or_invalid_speaker_metadata", "speaker gender is missing or outside source m/f codes"))
        detail = labels.get(utterance_id)
        aggregate = aggregate_scores.get(utterance_id)
        if detail is None:
            issues.append(_issue("missing_labels", "utterance is absent from resource/scores-detail.json"))
        else:
            label_coverage += 1
            _validate_annotations(detail, issues)
        if aggregate is None:
            issues.append(_issue("missing_aggregate_scores", "utterance is absent from resource/scores.json"))
        else:
            _validate_aggregate_scores(aggregate, issues)
        text_file_value = maps.get("text", {}).get(utterance_id)
        detail_text = detail.get("text") if isinstance(detail, dict) else None
        if not isinstance(text_file_value, str) or not text_file_value.strip():
            issues.append(_issue("missing_transcript", "utterance has no source text entry"))
        else:
            transcript_coverage += 1
        if isinstance(detail_text, str) and text_file_value and _normalized_transcript_tokens(detail_text) != _normalized_transcript_tokens(text_file_value):
            issues.append(_issue("transcript_mismatch", "resource/scores-detail.json and split text file disagree"))
        reference_text = text_file_value if isinstance(text_file_value, str) else detail_text
        if not raw_audio_path:
            issues.append(_issue("missing_audio_path", "utterance has no source wav.scp entry"))
            audio_valid = False
        else:
            metadata, audio_issues = _audio_metadata(root, raw_audio_path, expected_sample_rate)
            issues.extend(audio_issues)
            if metadata is not None:
                container_hash_to_ids[metadata["audio_sha256"]].append(utterance_id)
                pcm_hash_to_ids[metadata["decoded_pcm_sha256"]].append(utterance_id)
                audio_parts = Path(metadata["audio_path"])
                if audio_parts.stem != utterance_id:
                    issues.append(_issue("utterance_audio_id_mismatch", "WAV filename stem differs from utterance ID"))
                expected_speaker_folder = f"SPEAKER{speaker_id.removeprefix('speaker')}".upper() if speaker_id else None
                if expected_speaker_folder and audio_parts.parent.name.upper() != expected_speaker_folder:
                    issues.append(_issue("speaker_audio_id_mismatch", "WAV speaker folder differs from utt2spk speaker ID"))
            audio_valid = metadata is not None and not audio_issues
        valid_audio_count += int(audio_valid)
        speaker_meta = {}
        if speaker_id:
            for key in ("spk2age", "spk2gender"):
                if speaker_id in maps.get(key, {}):
                    speaker_meta[key.removeprefix("spk2")] = maps[key][speaker_id]
        if split not in ("train", "test"):
            issues.append(_issue("missing_split_membership", "utterance has labels but no train/test metadata"))
        per_split[split]["utterances"] += 1
        per_split[split]["valid_audio"] += int(audio_valid)
        per_split[split]["labelled"] += int(detail is not None)
        per_split[split]["transcribed"] += int(bool(text_file_value and text_file_value.strip()))
        per_split[split]["speaker_identified"] += int(bool(speaker_id))
        annotations = None
        if isinstance(detail, dict):
            words = detail.get("words")
            utterance_labels = {key: value for key, value in detail.items() if key not in ("text", "words")}
            annotations = {
                "phone": [
                    {"word": word.get("text"), "reference_phone_sequence": word.get("ref-phones"), "rater_phone_sequences": word.get("phones")}
                    for word in words if isinstance(word, dict)
                ] if isinstance(words, list) else None,
                "word": words,
                "utterance": utterance_labels,
                "source_native": detail,
                "aggregate": aggregate,
            }
        eligibility = "checkpoint_exposed_not_independent"
        reasons = ["SpeechOcean762 was used for GOPT training and test-set checkpoint selection"]
        if issues:
            eligibility = "review_required"
            reasons.extend(issue["code"] for issue in issues)
        record = {
            "dataset": "SpeechOcean762",
            "dataset_version": source_version,
            "split": split,
            "utterance_id": utterance_id,
            "speaker_id": speaker_id,
            "speaker_metadata": speaker_meta,
            "audio_path": metadata["audio_path"] if metadata else raw_audio_path,
            "audio": metadata,
            "reference_text": reference_text,
            "annotations": annotations,
            "provenance": {
                "source": SOURCE_URL,
                "license": LICENSE,
                "label_source": "resource/scores-detail.json; five independent expert records retained source-natively",
                "aggregate_score_source": "resource/scores.json; source phone-accuracy values retained on the [0, 2] scale",
                "checkpoint_exposed": True,
            },
            "evaluation_eligibility": eligibility,
            "eligibility_reasons": reasons,
            "issues": issues,
        }
        records.append(record)

    for record in records:
        if record["speaker_id"] in overlap:
            record["issues"].append(_issue("speaker_split_leakage", "speaker occurs in both source train and test splits"))
            record["evaluation_eligibility"] = "review_required"
            record["eligibility_reasons"].append("speaker_split_leakage")
        if record["utterance_id"] in id_overlap:
            record["issues"].append(_issue("utterance_split_overlap", "utterance ID occurs in both source splits"))
            record["evaluation_eligibility"] = "review_required"
            record["eligibility_reasons"].append("utterance_split_overlap")

    duplicate_groups = []
    duplicate_ids: set[str] = set()
    for hash_kind, groups in (("audio_sha256", container_hash_to_ids), ("decoded_pcm_sha256", pcm_hash_to_ids)):
        for digest, utterance_ids in sorted(groups.items()):
            unique_ids = sorted(set(utterance_ids))
            if len(unique_ids) > 1:
                duplicate_groups.append({"kind": hash_kind, "sha256": digest, "utterance_ids": unique_ids})
                duplicate_ids.update(unique_ids)
    for record in records:
        if record["utterance_id"] in duplicate_ids:
            record["issues"].append(_issue("duplicate_audio", "audio has an exact file-byte or decoded-PCM match under another utterance ID"))
            record["evaluation_eligibility"] = "review_required"
            record["eligibility_reasons"].append("duplicate_audio")

    corpus_count = len(all_ids)
    if expected_utterances is not None and corpus_count != expected_utterances:
        global_issues.append(_issue("dataset_count_mismatch", f"expected {expected_utterances} utterances, indexed {corpus_count}"))
    speaker_count = len({record["speaker_id"] for record in records if record["speaker_id"]})
    if expected_speakers is not None and speaker_count != expected_speakers:
        global_issues.append(_issue("speaker_count_mismatch", f"expected {expected_speakers} speakers, indexed {speaker_count}"))
    issue_counts: dict[str, int] = defaultdict(int)
    for issue in global_issues:
        issue_counts[issue["code"]] += 1
    for record in records:
        for issue in record["issues"]:
            issue_counts[issue["code"]] += 1
    any_errors = bool(global_issues or any(record["issues"] for record in records))
    report = {
        "manifest_version": 1,
        "dataset": "SpeechOcean762",
        "source": SOURCE_URL,
        "license": LICENSE,
        "source_version": source_version,
        "source_file_sha256": {path.relative_to(root).as_posix(): _sha256(path) for path in metadata_paths if path.is_file()},
        "expected_sample_rate": expected_sample_rate,
        "counts": {
            "utterances": corpus_count,
            "valid_audio": valid_audio_count,
            "speakers": speaker_count,
            "labelled": label_coverage,
            "transcribed": transcript_coverage,
            "speaker_identified": speaker_coverage,
            "duplicate_audio_utterances": len(duplicate_ids),
            "exclusion_or_review_records": sum(record["evaluation_eligibility"] == "review_required" for record in records),
        },
        "split_counts": dict(sorted(per_split.items())),
        "split_checks": {
            "speaker_overlap_count": len(overlap),
            "speaker_overlap_ids": overlap,
            "utterance_overlap_count": len(id_overlap),
            "utterance_overlap_ids": id_overlap,
        },
        "duplicate_audio_groups": duplicate_groups,
        "near_duplicate_detection": "not run; exact container SHA-256 and decoded PCM SHA-256 checked",
        "issue_counts": dict(sorted(issue_counts.items())),
        "global_issues": global_issues,
        "integrity_status": "FAIL" if any_errors else "PASS",
        "evaluation": {
            "speech_ocean762_checkpoint_exposed": True,
            "development_or_feature_reconstruction_only": True,
            "independent_final_evaluation_eligible": False,
            "reason": "Released GOPT checkpoint selection used this corpus's test split; exposure cannot be undone by re-splitting.",
        },
        "records_file": "manifest.jsonl",
    }
    return records, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Extracted corpus root containing train/, test/, resource/, and WAVE/")
    parser.add_argument("--manifest", type=Path, required=True, help="Output JSONL path; all IDs are retained, including excluded/review records")
    parser.add_argument("--report", type=Path, required=True, help="Output integrity summary JSON path")
    parser.add_argument("--expected-utterances", type=int, default=5000)
    parser.add_argument("--expected-sample-rate", type=int, default=16000)
    parser.add_argument("--expected-speakers", type=int, default=250)
    parser.add_argument("--source-version", default="unverified_local_source_version")
    args = parser.parse_args()
    records, report = build_inventory(args.root, args.expected_utterances, args.expected_sample_rate, args.expected_speakers, args.source_version)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.resolve()
    try:
        report["records_file"] = manifest_path.relative_to(Path(__file__).resolve().parent).as_posix()
    except ValueError:
        report["records_file"] = manifest_path.as_posix()
    with args.manifest.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"integrity_status": report["integrity_status"], "counts": report["counts"], "issue_counts": report["issue_counts"]}, indent=2))
    return 0 if report["integrity_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
