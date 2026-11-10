"""Reproducible integrity checks and benchmark helpers for QUALITY-08.

The EpaDB sample is research-only (CC BY-NC 4.0). This utility keeps source
audio and generated measurements under .local/ and never fits score mappings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / ".local" / "datasets" / "epadb-quality-08"
SOURCE_COMMIT = "c697e2994d227e8fd7d48bc014e12777a0292424"
SOURCE_REPOSITORY = "https://github.com/donggangj/EpaDB"
LICENSE = "CC BY-NC 4.0; research/non-commercial use only"
SUPPORT_FLOOR = -6.0

# EpaDB's label export uses these source-specific spellings for broad phones.
# Unknown symbols are retained verbatim and reported; they are never guessed.
EPADB_PHONE_ALIASES = {
    "ph": "P",
    "th": "T",
    "sh": "SH",
    "kh": "K",
}
SILENCE_LABELS = {"", "sil", "sp", "pau", "<sil>"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unquote(text: str) -> str:
    return text.replace('""', '"')


def parse_textgrid(text: str) -> dict[str, Any]:
    """Parse Praat long-format interval tiers used in the EpaDB sample."""
    global_end = re.search(r"^xmax\s*=\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)
    if not global_end:
        raise ValueError("TextGrid has no global xmax.")
    tier_blocks = re.split(r"^\s*item\s+\[\d+\]:\s*$", text, flags=re.MULTILINE)
    if len(tier_blocks) < 2:
        raise ValueError("TextGrid has no interval tiers.")
    tiers: dict[str, list[dict[str, Any]]] = {}
    for block in tier_blocks[1:]:
        name_match = re.search(r'^\s*name\s*=\s*"(.*)"\s*$', block, re.MULTILINE)
        if not name_match:
            continue
        name = _unquote(name_match.group(1))
        lines = block.splitlines()
        intervals: list[dict[str, Any]] = []
        index = 0
        while index < len(lines):
            if not re.match(r"^\s*intervals\s+\[\d+\]:\s*$", lines[index]):
                index += 1
                continue
            fields: dict[str, str] = {}
            index += 1
            while index < len(lines) and len(fields) < 3:
                line = lines[index]
                index += 1
                if match := re.match(r"^\s*xmin\s*=\s*([0-9.eE+-]+)\s*$", line):
                    fields["xmin"] = match.group(1)
                elif match := re.match(r"^\s*xmax\s*=\s*([0-9.eE+-]+)\s*$", line):
                    fields["xmax"] = match.group(1)
                elif match := re.match(r'^\s*text\s*=\s*"(.*)"\s*$', line):
                    fields["text"] = match.group(1)
            if set(fields) != {"xmin", "xmax", "text"}:
                raise ValueError(f"Malformed interval in TextGrid tier {name!r}.")
            start, end = float(fields["xmin"]), float(fields["xmax"])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
                raise ValueError(f"Invalid interval times in TextGrid tier {name!r}.")
            intervals.append({"start_seconds": start, "end_seconds": end, "text": _unquote(fields["text"])})
        tiers[name] = intervals
    return {"duration_seconds": float(global_end.group(1)), "tiers": tiers}


def parse_source_labels(text: str) -> list[dict[str, Any]]:
    """Read the author's target/observed-phone +/- rows without relabeling."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) not in {4, 6}:
            raise ValueError(f"Source label line {line_number} must have four or six fields.")
        start = end = None
        if len(fields) == 6:
            try:
                start, end = float(fields[4]), float(fields[5])
            except ValueError as exc:
                raise ValueError(f"Invalid source label times on line {line_number}.") from exc
        if fields[3] not in {"+", "-"}:
            raise ValueError(f"Invalid source label class on line {line_number}.")
        if start is not None and not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
            raise ValueError(f"Invalid source label interval on line {line_number}.")
        rows.append({
            "source_phone_id": fields[0], "target_phone": fields[1],
            "observed_phone": fields[2], "label": fields[3],
            "start_seconds": start, "end_seconds": end,
        })
    return rows


def normalize_phone(phone: str) -> str:
    stripped = phone.strip()
    if stripped.casefold() in EPADB_PHONE_ALIASES:
        return EPADB_PHONE_ALIASES[stripped.casefold()]
    return stripped.upper()


def edit_align(reference: list[str], hypothesis: list[str]) -> list[dict[str, Any]]:
    """Unit-cost Levenshtein alignment; diagonal substitutions precede gaps."""
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    back: list[list[str | None]] = [[None] * columns for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], back[i][0] = i, "DELETION"
    for j in range(1, columns):
        costs[0][j], back[0][j] = j, "INSERTION"
    for i in range(1, rows):
        for j in range(1, columns):
            if reference[i - 1] == hypothesis[j - 1]:
                costs[i][j], back[i][j] = costs[i - 1][j - 1], "MATCH"
            else:
                candidates = (
                    (costs[i - 1][j - 1] + 1, 0, "SUBSTITUTION"),
                    (costs[i - 1][j] + 1, 1, "DELETION"),
                    (costs[i][j - 1] + 1, 2, "INSERTION"),
                )
                costs[i][j], _, back[i][j] = min(candidates)
    operations: list[dict[str, Any]] = []
    i, j = len(reference), len(hypothesis)
    while i or j:
        operation = back[i][j]
        if operation in {"MATCH", "SUBSTITUTION"}:
            operations.append({"reference_index": i - 1, "hypothesis_index": j - 1,
                               "reference": reference[i - 1], "hypothesis": hypothesis[j - 1],
                               "operation": operation})
            i -= 1
            j -= 1
        elif operation == "DELETION":
            operations.append({"reference_index": i - 1, "hypothesis_index": None,
                               "reference": reference[i - 1], "hypothesis": None,
                               "operation": operation})
            i -= 1
        elif operation == "INSERTION":
            operations.append({"reference_index": None, "hypothesis_index": j - 1,
                               "reference": None, "hypothesis": hypothesis[j - 1],
                               "operation": operation})
            j -= 1
        else:
            raise ValueError("Incomplete sequence-alignment traceback.")
    operations.reverse()
    return operations


def parse_record(wav_path: Path) -> dict[str, Any]:
    relative = wav_path.relative_to(DATA_ROOT)
    parts = relative.parts
    partition, source_split, speaker = parts[:3]
    record_id = wav_path.stem
    transcript_path = wav_path.parents[1] / "transcriptions" / f"{record_id}.lab"
    textgrid_path = wav_path.parents[1] / "annotations_1" / f"{record_id}.TextGrid"
    labels_path = wav_path.parents[1] / "labels" / f"{record_id}.csv"
    missing = [str(path.relative_to(DATA_ROOT)) for path in (transcript_path, textgrid_path, labels_path) if not path.is_file()]
    with wave.open(str(wav_path), "rb") as audio:
        frames, rate = audio.getnframes(), audio.getframerate()
        channels, width = audio.getnchannels(), audio.getsampwidth()
    duration = frames / rate if rate else 0
    transcript = transcript_path.read_text(encoding="utf-8").strip() if transcript_path.is_file() else ""
    grid = parse_textgrid(textgrid_path.read_text(encoding="utf-8")) if textgrid_path.is_file() else {"duration_seconds": None, "tiers": {}}
    tiers = grid["tiers"]
    labels = parse_source_labels(labels_path.read_text(encoding="utf-8")) if labels_path.is_file() else []
    word_intervals = tiers.get("words", [])
    phone_intervals = tiers.get("annotation", [])
    manual_labels = [
        interval["text"].strip() for interval in word_intervals
        if interval["text"].strip().casefold() not in SILENCE_LABELS
    ]
    manual_transcript_raw = " ".join(manual_labels)
    # EpaDB uses underscores for compound interval labels such as hot_tea;
    # source .lab transcripts preserve the orthographic word boundary.
    manual_transcript = " ".join(label.replace("_", " ") for label in manual_labels)
    tokenize_words = lambda value: re.findall(
        r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+", value.lower(), re.UNICODE,
    )
    utterance_ratings = [
        interval["text"].strip() for tier_name, intervals in tiers.items()
        if "score" in tier_name.casefold() or "rating" in tier_name.casefold()
        for interval in intervals if interval["text"].strip()
    ]
    phone_segments = [
        interval for interval in phone_intervals
        if interval["text"].strip().casefold() not in SILENCE_LABELS
    ]
    timed_labels = [row for row in labels if row["start_seconds"] is not None]
    annotation_match_count = sum(
        any(abs(row["start_seconds"] - segment["start_seconds"]) <= 0.00002
            and abs(row["end_seconds"] - segment["end_seconds"]) <= 0.00002
            for segment in phone_segments)
        for row in timed_labels
    )
    paths = {"audio": wav_path, "transcript": transcript_path,
             "textgrid": textgrid_path, "labels": labels_path}
    file_records = {
        key: {
            "path": str(path.relative_to(DATA_ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for key, path in paths.items() if path.is_file()
    }
    problems: list[str] = []
    warnings: list[str] = []
    if missing:
        problems.append("missing_files")
    if channels != 1 or width != 2:
        problems.append("unsupported_audio_format")
    if rate not in {44100, 48000}:
        problems.append("unexpected_sample_rate")
    if not tiers.get("words") or not tiers.get("annotation"):
        problems.append("required_textgrid_tier_missing")
    if any(interval["end_seconds"] > duration + 2 / rate for interval in phone_segments):
        problems.append("manual_phone_interval_exceeds_audio")
    if any(
        interval["text"].strip().casefold() not in SILENCE_LABELS
        and interval["end_seconds"] > duration + 2 / rate
        for interval in word_intervals
    ):
        problems.append("manual_word_interval_exceeds_audio")
    grid_delta = grid["duration_seconds"] - duration if grid["duration_seconds"] is not None else None
    if grid_delta is not None and abs(grid_delta) > 2 / rate:
        warnings.append("TextGrid global/tier extent differs from WAV duration; non-speech tier extents are checked separately.")
    if not labels or annotation_match_count != len(timed_labels):
        problems.append("label_time_not_found_in_manual_phone_tier")
    if any(row["end_seconds"] is not None and row["end_seconds"] > duration + 2 / rate for row in labels):
        problems.append("label_time_exceeds_audio")
    return {
        "record_id": record_id, "partition": partition, "source_split": source_split,
        "speaker_id": speaker, "reference_transcript": transcript,
        "manual_word_tier_transcript": manual_transcript,
        "manual_word_tier_transcript_raw": manual_transcript_raw,
        "manual_word_tier_labels": manual_labels,
        "manual_word_tier_matches_reference": tokenize_words(transcript) == tokenize_words(manual_transcript),
        "audio": {"duration_seconds": duration, "sample_rate_hz": rate,
                  "channels": channels, "sample_width_bytes": width},
        "textgrid_duration_seconds": grid["duration_seconds"],
        "textgrid_minus_audio_duration_seconds": grid_delta,
        "manual_phone_count": len(phone_segments),
        "manual_phone_symbols": sorted({segment["text"].strip() for segment in phone_segments}),
        "source_phone_label_count": len(labels),
        "source_phone_label_time_count": len(timed_labels),
        "source_phone_label_classes": dict(Counter(row["label"] for row in labels)),
        "source_label_intervals_found_in_manual_tier": annotation_match_count,
        "utterance_rating_values_uninterpreted": utterance_ratings,
        "files": file_records,
        "integrity_problems": problems,
        "integrity_warnings": warnings,
    }


def summarize_integrity(records: list[dict[str, Any]]) -> dict[str, Any]:
    speakers_by_partition: dict[str, set[str]] = defaultdict(set)
    duplicate_hashes: dict[str, list[str]] = defaultdict(list)
    for record in records:
        speakers_by_partition[record["partition"]].add(record["speaker_id"])
        audio = (record.get("files") or {}).get("audio") or {}
        if audio.get("sha256"):
            duplicate_hashes[audio["sha256"]].append(record["record_id"])
    development = speakers_by_partition["development"]
    holdout = speakers_by_partition["holdout"]
    duplicate_groups = [ids for ids in duplicate_hashes.values() if len(ids) > 1]
    return {
        "recording_count": len(records),
        "speaker_count": len(development | holdout),
        "development_speaker_count": len(development),
        "holdout_speaker_count": len(holdout),
        "speaker_overlap": sorted(development & holdout),
        "audio_duplicate_groups": duplicate_groups,
        "records_with_integrity_problems": [r["record_id"] for r in records if r.get("integrity_problems")],
        "all_records_have_four_source_files": all(len(r.get("files", {})) == 4 for r in records),
        "all_timestamped_source_label_intervals_match_manual_phone_tier": all(
            r.get("source_phone_label_time_count", 0) == r.get("source_label_intervals_found_in_manual_tier", 0)
            for r in records
        ),
        "timestamped_label_rows": sum(r.get("source_phone_label_time_count", 0) for r in records),
        "untimestamped_label_rows": sum(
            r.get("source_phone_label_count", 0) - r.get("source_phone_label_time_count", 0) for r in records
        ),
        "recordings_with_untimestamped_source_labels": sum(
            r.get("source_phone_label_count", 0) > r.get("source_phone_label_time_count", 0) for r in records
        ),
        "records_with_textgrid_extent_warning": [r["record_id"] for r in records if r.get("integrity_warnings")],
    }


def build_manifest() -> dict[str, Any]:
    audio_paths = sorted(DATA_ROOT.glob("development/**/waveforms/*.wav")) + sorted(DATA_ROOT.glob("holdout/**/waveforms/*.wav"))
    records = [parse_record(path) for path in audio_paths]
    integrity = summarize_integrity(records)
    reference_files = {
        path.name: {"path": str(path.relative_to(DATA_ROOT)), "sha256": sha256_file(path)}
        for path in sorted((DATA_ROOT / "source").glob("*.txt"))
    }
    baseline_path = DATA_ROOT / "frozen_baseline" / "fingerprint.json"
    frozen_baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else None
    gopt_phones = {phone.upper() for phone in (
        "W IY K AO L IH T B EH R Z OW TH F AY V AH N UW S G AA M P NG HH EY SH AE D UH AW DH ER Y JH CH OY ZH"
    ).split()}
    manual_symbols = sorted({symbol for record in records for symbol in record["manual_phone_symbols"]})
    unsupported_symbols = sorted({
        symbol for symbol in manual_symbols if normalize_phone(symbol) not in gopt_phones
    })
    return {
        "schema_version": "quality-08-sample-manifest-v1",
        "created_utc_date": "2026-11-10",
        "source": {
            "name": "EpaDB English pronunciation database",
            "repository": SOURCE_REPOSITORY, "pinned_commit": SOURCE_COMMIT,
            "license": LICENSE,
            "official_paper": "https://www.isca-archive.org/interspeech_2019/vidal19_interspeech.pdf",
            "usage_note": "Acquired examples and annotations remain under the source non-commercial research license. No commercial training or deployment rights are claimed.",
            "acquisition_method": "Bounded files fetched from the public author repository at the pinned commit; source files kept unchanged.",
            "source_reference_transcription_files": reference_files,
        },
        "frozen_baseline": frozen_baseline,
        "selection": {
            "development": "10 source train speakers; prompts 1 and 2 where present; prompt 4 replaced a missing prompt for spkr105 and spkr51.",
            "holdout": "All 20 source test speakers; prompts 1 and 2 where present; prompt 4 replaced a missing prompt for spkr105 and spkr51.",
            "holdout_was_not_used_for_model_selection": True,
            "partition_basis": "Source train/test speaker folders; speaker IDs checked for disjointness.",
        },
        "integrity": {
            **integrity,
            "manual_phone_inventory": manual_symbols,
            "manual_phone_symbols_without_gopt_class_mapping": unsupported_symbols,
        },
        "records": records,
    }


def _load_run(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _direct_phone_rows(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not result:
        return []
    if isinstance(result.get("phonemes"), list):
        return result["phonemes"]
    details = result.get("details") or {}
    return details.get("phone_support_diagnostics") or []


def _grid_phone_rows(record_id: str, partition: str) -> list[dict[str, Any]]:
    candidates = list((DATA_ROOT / partition).glob(f"**/annotations_1/{record_id}.TextGrid"))
    if len(candidates) != 1:
        return []
    intervals = parse_textgrid(candidates[0].read_text(encoding="utf-8"))["tiers"].get("annotation", [])
    return [row for row in intervals if row["text"].strip().casefold() not in SILENCE_LABELS]


def _source_label_rows(record_id: str, partition: str) -> list[dict[str, Any]]:
    candidates = list((DATA_ROOT / partition).glob(f"**/labels/{record_id}.csv"))
    return parse_source_labels(candidates[0].read_text(encoding="utf-8")) if len(candidates) == 1 else []


def _as_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _auc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) != len(scores) or not labels or len(set(labels)) < 2:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    positives = sum(labels)
    negatives = len(labels) - positives
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _cluster_interval(items: list[dict[str, Any]], metric, seed: int = 8026, replicates: int = 2000) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item["speaker_id"]].append(item)
    speakers = sorted(grouped)
    if len(speakers) < 2:
        return {"low_95": None, "high_95": None, "valid_replicates": 0}
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sample: list[dict[str, Any]] = []
        for _ in speakers:
            sample.extend(grouped[rng.choice(speakers)])
        value = metric(sample)
        if value is not None and math.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return {"low_95": None, "high_95": None, "valid_replicates": 0}
    estimates.sort()
    return {
        "low_95": estimates[int(0.025 * (len(estimates) - 1))],
        "high_95": estimates[int(0.975 * (len(estimates) - 1))],
        "valid_replicates": len(estimates),
    }


def compare_support_to_boundaries(rows: list[dict[str, Any]], tolerance_ms: int) -> dict[str, Any]:
    key = f"manual_valid_{tolerance_ms}ms"
    matched = [row for row in rows if row.get("boundary_available")]
    accepted = [row for row in matched if row.get("supported")]
    valid = [row for row in matched if row.get(key)]
    tp = sum(bool(row.get(key)) and bool(row.get("supported")) for row in matched)
    fp = sum(not bool(row.get(key)) and bool(row.get("supported")) for row in matched)
    fn = sum(bool(row.get(key)) and not bool(row.get("supported")) for row in matched)
    tn = sum(not bool(row.get(key)) and not bool(row.get("supported")) for row in matched)
    return {
        "matched_phones": len(matched), "valid_boundary_phones": len(valid),
        "support_passed": len(accepted), "true_accept": tp, "false_accept": fp,
        "false_reject": fn, "true_reject": tn,
        "false_accept_rate_among_accepted": fp / len(accepted) if accepted else None,
        "false_reject_rate_among_valid": fn / len(valid) if valid else None,
        "support_accuracy": (tp + tn) / len(matched) if matched else None,
    }


def _partition_metrics(manifest: dict[str, Any], partition: str, include_integrated: bool, run_name: str) -> dict[str, Any]:
    boundary_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    discrimination_rows: list[dict[str, Any]] = []
    asr_rows: list[dict[str, Any]] = []
    asr_unavailable = 0
    coverage_total = 0
    phone_total = 0
    windows_total = windows_withheld = direct_failures = 0
    manual_phone_deletions = scorer_phone_insertions = 0
    exact_phone_matches = source_label_rows_total = 0
    per_speaker: dict[str, dict[str, Any]] = {}
    for record in [r for r in manifest["records"] if r["partition"] == partition]:
        record_id = record["record_id"]
        direct = _load_run(DATA_ROOT / "runs" / run_name / "direct" / f"{record_id}.json")
        integrated = _load_run(DATA_ROOT / "runs" / run_name / "integrated" / f"{record_id}.json") if include_integrated else None
        if direct is None:
            continue
        if direct.get("status") not in {"SCORING_COMPLETED", "SCORING_COMPLETED_PARTIAL"}:
            direct_failures += 1
        rows = _direct_phone_rows(direct)
        manual = _grid_phone_rows(record_id, partition)
        manual_symbols = [normalize_phone(row["text"]) for row in manual]
        scorer_symbols = [normalize_phone(row.get("phone", "")) for row in rows]
        operations = edit_align(manual_symbols, scorer_symbols)
        speaker_id = record["speaker_id"]
        speaker_bucket = per_speaker.setdefault(speaker_id, {"recordings": 0, "direct_failures": 0, "phones": 0, "covered_phones": 0})
        speaker_bucket["recordings"] += 1
        speaker_bucket["direct_failures"] += int(direct.get("status") not in {"SCORING_COMPLETED", "SCORING_COMPLETED_PARTIAL"})
        for operation in operations:
            mi, pi = operation["reference_index"], operation["hypothesis_index"]
            manual_phone_deletions += operation["operation"] == "DELETION"
            scorer_phone_insertions += operation["operation"] == "INSERTION"
            exact_phone_matches += operation["operation"] == "MATCH"
            if mi is None:
                continue
            if pi is None:
                support_rows.append({"speaker_id": speaker_id, "supported": False, "manual_valid_30ms": False,
                                    "manual_valid_60ms": False, "manual_valid_100ms": False, "boundary_available": False})
                continue
            source = manual[mi]
            predicted = rows[pi]
            start_error = abs(float(predicted.get("start_seconds", 0.0)) - source["start_seconds"])
            end_error = abs(float(predicted.get("end_seconds", 0.0)) - source["end_seconds"])
            boundary = {"speaker_id": speaker_id, "record_id": record_id,
                        "manual_phone": source["text"], "scorer_phone": predicted.get("phone"),
                        "operation": operation["operation"], "start_error_seconds": start_error,
                        "end_error_seconds": end_error, "max_edge_error_seconds": max(start_error, end_error)}
            boundary_rows.append(boundary)
            support = _as_number(predicted.get("canonical_phone_mean_log_posterior"))
            supported = support is not None and support >= SUPPORT_FLOOR
            for threshold_ms in (30, 60, 100):
                boundary[f"boundary_valid_{threshold_ms}ms"] = max(start_error, end_error) <= threshold_ms / 1000
            support_rows.append({"speaker_id": speaker_id, "record_id": record_id, "supported": supported,
                                 "support": support, "manual_valid_30ms": boundary["boundary_valid_30ms"],
                                 "manual_valid_60ms": boundary["boundary_valid_60ms"],
                                 "manual_valid_100ms": boundary["boundary_valid_100ms"], "boundary_available": True})
        phones = int(direct.get("long_form", {}).get("reference_phone_count") or len(rows) or record["source_phone_label_count"])
        covered = sum(_as_number(row.get("gopt_phone_score")) is not None for row in rows)
        phone_total += phones
        coverage_total += covered
        speaker_bucket["phones"] += phones
        speaker_bucket["covered_phones"] += covered
        if isinstance(direct.get("long_form"), dict):
            windows = direct["long_form"].get("windows", [])
            windows_total += len(windows)
            windows_withheld += sum(window.get("status") != "SCORING_COMPLETED" for window in windows)
        else:
            details = direct.get("details") or {}
            windows_total += int(details.get("withheld_window_count", 0))
            windows_withheld += int(details.get("withheld_window_count", 0))
        labels = _source_label_rows(record_id, partition)
        source_label_rows_total += len(labels)
        label_alignment = edit_align(
            [normalize_phone(row["target_phone"]) for row in labels], scorer_symbols,
        )
        for pair in label_alignment:
            li, pi = pair["reference_index"], pair["hypothesis_index"]
            if li is None or pi is None or pair["operation"] != "MATCH":
                continue
            score = _as_number(rows[pi].get("gopt_phone_score"))
            if score is None:
                continue
            discrimination_rows.append({"speaker_id": speaker_id, "label": int(labels[li]["label"] == "+"),
                                        "score": score, "record_id": record_id})
        if include_integrated and integrated:
            content = integrated.get("components", {}).get("content", {})
            asr_raw = (content.get("predictions") or {}).get("transcript_raw")
            if asr_raw is None:
                asr_unavailable += 1
                continue
            manual_words = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+", record["manual_word_tier_transcript"].lower(), re.UNICODE)
            asr_words = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+", str(asr_raw or "").lower(), re.UNICODE)
            word_ops = edit_align(manual_words, asr_words)
            substitutions = sum(op["operation"] == "SUBSTITUTION" for op in word_ops)
            deletions = sum(op["operation"] == "DELETION" for op in word_ops)
            insertions = sum(op["operation"] == "INSERTION" for op in word_ops)
            ref_count = len(manual_words)
            asr_rows.append({"speaker_id": speaker_id, "record_id": record_id,
                             "reference_words": ref_count, "substitutions": substitutions,
                             "deletions": deletions, "insertions": insertions,
                             "error_units": substitutions + deletions + insertions,
                             "wer": (substitutions + deletions + insertions) / ref_count if ref_count else None,
                             "manual_transcript": " ".join(manual_words), "asr_transcript": " ".join(asr_words)})
    boundary_errors = [row["max_edge_error_seconds"] for row in boundary_rows]
    support_eval: dict[str, Any] = {}
    for tolerance in (30, 60, 100):
        support_eval[f"{tolerance}ms"] = compare_support_to_boundaries(support_rows, tolerance)
    speaker_interval = _cluster_interval(
        [row for row in support_rows if row["boundary_available"]],
        lambda sample: sum(row["supported"] == row["manual_valid_100ms"] for row in sample) / len(sample) if sample else None,
    )
    per_phone_items = [{"speaker_id": row["speaker_id"], "label": row["label"], "score": row["score"]} for row in discrimination_rows]
    discrimination_ci = _cluster_interval(per_phone_items, lambda sample: _auc([r["label"] for r in sample], [r["score"] for r in sample]))
    labels = [row["label"] for row in discrimination_rows]
    scores = [row["score"] for row in discrimination_rows]
    asr_errors = sum(row["error_units"] for row in asr_rows)
    asr_reference_words = sum(row["reference_words"] for row in asr_rows)
    return {
        "partition": partition,
        "recordings_scored": sum(value["recordings"] for value in per_speaker.values()),
        "speaker_count_scored": len(per_speaker),
        "per_speaker": per_speaker,
        "direct_failed_recordings": direct_failures,
        "gopt_phone_coverage": coverage_total / phone_total if phone_total else None,
        "scored_gopt_phones": coverage_total, "reference_phones": phone_total,
        "gopt_windows": windows_total, "withheld_gopt_windows": windows_withheld,
        "alignment": {
            "matched_or_substituted_phone_pairs": len(boundary_rows),
            "exact_phone_identity_matches": exact_phone_matches,
            "manual_phone_deletions": manual_phone_deletions,
            "scorer_phone_insertions": scorer_phone_insertions,
            "boundary_max_edge_error_mae_seconds": sum(boundary_errors) / len(boundary_errors) if boundary_errors else None,
            "boundary_max_edge_error_median_seconds": sorted(boundary_errors)[len(boundary_errors) // 2] if boundary_errors else None,
            "boundary_within_tolerance": {
                f"{threshold}ms": sum(error <= threshold / 1000 for error in boundary_errors) / len(boundary_errors) if boundary_errors else None
                for threshold in (30, 60, 100)
            },
            "boundary_error_interval_100ms_speaker_cluster_bootstrap": speaker_interval,
            "acoustic_support_vs_manual_boundary": support_eval,
        },
        "pronunciation": {
            "label_rows_scored_on_exact_phone_matches": len(discrimination_rows),
            "source_label_rows_total": source_label_rows_total,
            "source_label_exact_phone_match_fraction": len(discrimination_rows) / source_label_rows_total if source_label_rows_total else None,
            "correct_label_count": sum(labels), "incorrect_label_count": len(labels) - sum(labels),
            "gopt_native_phone_score_auc_correct_vs_incorrect": _auc(labels, scores),
            "gopt_auc_speaker_cluster_bootstrap_95": discrimination_ci,
            "interpretation": "Descriptive discrimination only; no calibrated mispronunciation threshold or Pearson mapping is fitted.",
        },
        "asr_against_manual_spoken_word_tier": {
            "recordings_scored": len(asr_rows), "reference_words": asr_reference_words,
            "recordings_without_asr_result": asr_unavailable,
            "substitutions": sum(row["substitutions"] for row in asr_rows),
            "deletions": sum(row["deletions"] for row in asr_rows),
            "insertions": sum(row["insertions"] for row in asr_rows),
            "wer": asr_errors / asr_reference_words if asr_reference_words else None,
            "hallucination_insertions_per_recording": sum(row["insertions"] for row in asr_rows) / len(asr_rows) if asr_rows else None,
            "speaker_cluster_bootstrap_95": _cluster_interval(asr_rows, lambda sample: sum(r["error_units"] for r in sample) / sum(r["reference_words"] for r in sample) if sample and sum(r["reference_words"] for r in sample) else None),
            "scope_note": "ASR text is compared with the manually annotated spoken word tier. Prompt-versus-ASR differences are not treated as learner errors.",
            "per_record": asr_rows,
        },
    }


def run_models(manifest: dict[str, Any], component: str, partition: str, threads: int, model: str, run_name: str) -> None:
    direct_python = ROOT / ".local" / "gopt-venv" / "bin" / "python"
    integrated_python = ROOT / ".venv" / "bin" / "python"
    records = [record for record in manifest["records"] if partition == "all" or record["partition"] == partition]
    commands: list[tuple[str, Path, list[str]]] = []
    for record in records:
        base = DATA_ROOT / record["files"]["audio"]["path"]
        for which in (["direct", "integrated"] if component == "both" else [component]):
            destination = DATA_ROOT / "runs" / run_name / which / f"{record['record_id']}.json"
            if destination.is_file():
                continue
            reference = record["reference_transcript"]
            if which == "direct":
                cmd = [str(direct_python), str(ROOT / "score_pronunciation.py"), "--audio", str(base),
                       "--reference", reference, "--threads", str(threads)]
                if run_name == "candidate":
                    cmd.append("--recover-supported-words")
            else:
                cmd = [str(integrated_python), str(ROOT / "score_integrated.py"), "--task", "ra",
                       "--audio", str(base), "--reference", reference, "--model", model,
                       "--threads", str(threads)]
                if run_name == "baseline":
                    cmd.append("--legacy-whole-window-policy")
            commands.append((which, destination, cmd))
    for index, (which, destination, command) in enumerate(commands, 1):
        destination.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.run(command, capture_output=True, text=True, timeout=900)
        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError:
            result = {"status": "BENCHMARK_RUNNER_FAILED", "error": "Scorer did not return JSON.",
                      "stdout_tail": process.stdout[-2000:], "stderr_tail": process.stderr[-2000:]}
        result["_benchmark"] = {"component": which, "exit_code": process.returncode,
                                 "command": command, "stderr": process.stderr[-2000:] if process.returncode else ""}
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[{index}/{len(commands)}] {which} {destination.stem}: {result.get('status')}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true", help="Validate and write the pinned sample manifest.")
    parser.add_argument("--run", choices=("direct", "integrated", "both"), help="Run selected baseline scorers and cache per-record JSON under .local/.")
    parser.add_argument("--partition", choices=("development", "holdout", "all"), default="all")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--run-name", default="baseline", help="Cache namespace under .local/.../runs/ (baseline or candidate).")
    parser.add_argument("--summarize", action="store_true", help="Summarize cached direct/integrated runs by the frozen split.")
    args = parser.parse_args()
    manifest = build_manifest()
    if args.write_manifest:
        (DATA_ROOT / "sample_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest["integrity"], indent=2))
        if manifest["integrity"]["records_with_integrity_problems"] or manifest["integrity"]["speaker_overlap"] or manifest["integrity"]["audio_duplicate_groups"]:
            return 1
    if args.run:
        if args.threads < 1 or args.threads > 8:
            parser.error("--threads must be between 1 and 8")
        if args.run in {"direct", "both"} and not (ROOT / ".local/gopt-venv/bin/python").is_file():
            parser.error("Local GOPT Python is not installed at .local/gopt-venv/bin/python")
        if args.run in {"integrated", "both"} and not (ROOT / ".venv/bin/python").is_file():
            parser.error("Project Python is not installed at .venv/bin/python")
        run_models(manifest, args.run, args.partition, args.threads, args.model, args.run_name)
    if args.summarize:
        summary = {
            "schema_version": "quality-08-baseline-summary-v1",
            "dataset_manifest_sha256": sha256_file(DATA_ROOT / "sample_manifest.json") if (DATA_ROOT / "sample_manifest.json").is_file() else None,
            "support_floor": SUPPORT_FLOOR,
            "native_scale_reporting": "No Pearson conversion; source utterance ratings remain uninterpreted pending rating-direction verification.",
            "partitions": {
                name: _partition_metrics(manifest, name, include_integrated=True, run_name=args.run_name)
                for name in ("development", "holdout")
            },
        }
        output = DATA_ROOT / "runs" / f"{args.run_name}_summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({name: {
            "recordings": part["recordings_scored"], "speakers": part["speaker_count_scored"],
            "coverage": part["gopt_phone_coverage"], "boundary_pairs": part["alignment"]["matched_or_substituted_phone_pairs"],
            "support_100ms": part["alignment"]["acoustic_support_vs_manual_boundary"]["100ms"],
            "gopt_auc": part["pronunciation"]["gopt_native_phone_score_auc_correct_vs_incorrect"],
            "asr_wer": part["asr_against_manual_spoken_word_tier"]["wer"],
        } for name, part in summary["partitions"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
