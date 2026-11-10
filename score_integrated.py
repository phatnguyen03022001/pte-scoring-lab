"""Local descriptive speech assessment with independent component failures."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
import sys
import subprocess
import shutil
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import score_pronunciation
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps


SAMPLE_RATE = 16_000
MAX_CONTENT_ALIGNMENT_CELLS = 1_000_000
TOKEN_PATTERN = re.compile(r"\d+(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|[^\W\d_]+(?:['’‘`][^\W\d_]+)*", re.UNICODE)
SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)")
SMALL_NUMBERS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four "
    "twenty-five twenty-six twenty-seven twenty-eight twenty-nine thirty thirty-one thirty-two "
    "thirty-three thirty-four thirty-five thirty-six thirty-seven thirty-eight thirty-nine forty "
    "forty-one forty-two forty-three forty-four forty-five forty-six forty-seven forty-eight forty-nine "
    "fifty fifty-one fifty-two fifty-three fifty-four fifty-five fifty-six fifty-seven fifty-eight "
    "fifty-nine sixty sixty-one sixty-two sixty-three sixty-four sixty-five sixty-six sixty-seven "
    "sixty-eight sixty-nine seventy seventy-one seventy-two seventy-three seventy-four seventy-five "
    "seventy-six seventy-seven seventy-eight seventy-nine eighty eighty-one eighty-two eighty-three "
    "eighty-four eighty-five eighty-six eighty-seven eighty-eight eighty-nine ninety ninety-one "
    "ninety-two ninety-three ninety-four ninety-five ninety-six ninety-seven ninety-eight ninety-nine"
).split()
FILLER_TOKENS = {"um", "uh", "erm", "er"}
ALLOWED_PUNCTUATION = set(".,!?;:-–—()[]{}\"“”/'’‘`…")


def normalize_tokens(text: str) -> tuple[list[str], list[bool]]:
    """Normalize case/punctuation/spacing; preserve contractions and repetitions.

    Arabic integers 0..99 normalize to their English cardinal words. Larger,
    comma-grouped, and decimal numbers remain literal and are flagged because
    speech has multiple valid renderings (for example, a year can be spoken in
    more than one way).
    """
    normalized: list[str] = []
    unsupported_number: list[bool] = []
    normalized_text = unicodedata.normalize(
        "NFC", text.replace("’", "'").replace("‘", "'").replace("`", "'"),
    )
    for raw in TOKEN_PATTERN.findall(normalized_text):
        token = raw.lower().replace(",", "")
        if token.isdigit():
            if "," in raw or len(token) > 2:
                normalized.append(token)
                unsupported_number.append(True)
                continue
            value = int(token)
            if 0 <= value <= 99:
                normalized.extend(SMALL_NUMBERS[value].split("-"))
                unsupported_number.extend([False] * len(SMALL_NUMBERS[value].split("-")))
        elif token[0].isdigit():
            normalized.append(token)
            unsupported_number.append(True)
        else:
            normalized.append(token)
            unsupported_number.append(False)
    return normalized, unsupported_number


def unsupported_characters(text: str) -> list[str]:
    return sorted({
        char for char in text
        if not char.isspace()
        and (unicodedata.category(char)[0] not in ("L", "N") and char not in ALLOWED_PUNCTUATION)
    })


def _operation_stream(reference: list[str], hypothesis: list[str]) -> list[dict[str, Any]]:
    """Deterministic unit-cost Levenshtein alignment with stable tie-breaking."""
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    back: list[list[str | None]] = [[None] * columns for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], back[i][0] = i, "OMISSION"
    for j in range(1, columns):
        costs[0][j], back[0][j] = j, "INSERTION"
    for i in range(1, rows):
        for j in range(1, columns):
            if reference[i - 1] == hypothesis[j - 1]:
                costs[i][j], back[i][j] = costs[i - 1][j - 1], "MATCH"
                continue
            candidates = (
                (costs[i - 1][j - 1] + 1, 0, "SUBSTITUTION"),
                (costs[i - 1][j] + 1, 1, "OMISSION"),
                (costs[i][j - 1] + 1, 2, "INSERTION"),
            )
            costs[i][j], _, back[i][j] = min(candidates)

    operations: list[dict[str, Any]] = []
    i, j = len(reference), len(hypothesis)
    while i or j:
        operation = back[i][j]
        if operation == "MATCH" or operation == "SUBSTITUTION":
            operations.append({
                "reference_index": i - 1,
                "recognized_index": j - 1,
                "reference_token": reference[i - 1],
                "recognized_token": hypothesis[j - 1],
                "classification": operation,
            })
            i, j = i - 1, j - 1
        elif operation == "OMISSION":
            operations.append({
                "reference_index": i - 1,
                "recognized_index": None,
                "reference_token": reference[i - 1],
                "recognized_token": None,
                "classification": "OMISSION",
            })
            i -= 1
        elif operation == "INSERTION":
            operations.append({
                "reference_index": None,
                "recognized_index": j - 1,
                "reference_token": None,
                "recognized_token": hypothesis[j - 1],
                "classification": "INSERTION",
            })
            j -= 1
        else:
            raise ValueError("Incomplete token-alignment traceback.")
    operations.reverse()

    # Label adjacent transpositions distinctly while preserving two edit units.
    index = 0
    while index + 1 < len(operations):
        first, second = operations[index:index + 2]
        if (
            first["classification"] == second["classification"] == "SUBSTITUTION"
            and first["reference_token"] == second["recognized_token"]
            and second["reference_token"] == first["recognized_token"]
        ):
            first["classification"] = second["classification"] = "ORDER_MISMATCH"
            index += 2
            continue
        index += 1

    for index, item in enumerate(operations):
        if item["classification"] != "INSERTION":
            continue
        recognized_index = item["recognized_index"]
        is_adjacent_repeat = (
            (recognized_index > 0 and hypothesis[recognized_index] == hypothesis[recognized_index - 1])
            or (recognized_index + 1 < len(hypothesis) and hypothesis[recognized_index] == hypothesis[recognized_index + 1])
        )
        if is_adjacent_repeat:
            item["classification"] = "REPETITION"

    reference_counts, hypothesis_counts = Counter(reference), Counter(hypothesis)
    for first_index, first in enumerate(operations):
        if first["classification"] in ("INSERTION", "OMISSION"):
            first_token = first["recognized_token"] if first["classification"] == "INSERTION" else first["reference_token"]
            first_kind = first["classification"]
            opposite_kind = "OMISSION" if first_kind == "INSERTION" else "INSERTION"
            if reference_counts[first_token] != 1 or hypothesis_counts[first_token] != 1:
                continue
            for second in operations[first_index + 1:]:
                if second["classification"] != opposite_kind:
                    continue
                second_token = second["reference_token"] if opposite_kind == "OMISSION" else second["recognized_token"]
                if second_token == first_token:
                    first["classification"] = second["classification"] = "ORDER_MISMATCH"
                    break
            continue
        if first["classification"] != "SUBSTITUTION":
            continue
        for second in operations[first_index + 1:]:
            if second["classification"] != "SUBSTITUTION":
                continue
            if (
                first["reference_token"] == second["recognized_token"]
                and second["reference_token"] == first["recognized_token"]
                and reference_counts[first["reference_token"]] == hypothesis_counts[first["reference_token"]] == 1
                and reference_counts[second["reference_token"]] == hypothesis_counts[second["reference_token"]] == 1
            ):
                first["classification"] = second["classification"] = "ORDER_MISMATCH"
                break
    return operations


def classify_alignment(
    reference: list[str],
    hypothesis: list[str],
    *,
    reference_unsupported_numbers: list[bool] | None = None,
    hypothesis_unsupported_numbers: list[bool] | None = None,
    no_speech: bool = False,
) -> list[dict[str, Any]]:
    operations = _operation_stream(reference, hypothesis)
    reference_counts, hypothesis_counts = Counter(reference), Counter(hypothesis)
    reference_unsupported_numbers = reference_unsupported_numbers or [False] * len(reference)
    hypothesis_unsupported_numbers = hypothesis_unsupported_numbers or [False] * len(hypothesis)
    if no_speech:
        for item in operations:
            if item["classification"] != "MATCH":
                item["candidate_classification"] = item["classification"]
                item["classification"] = "UNCERTAIN"
                item["uncertainty_reason"] = "No VAD speech interval was found; absence from ASR is not treated as a speaker omission."
        return operations
    for item in operations:
        repeated_tokens = {
            token for token in (item.get("reference_token"), item.get("recognized_token"))
            if token is not None and (reference_counts[token] > 1 or hypothesis_counts[token] > 1)
            and reference_counts[token] != hypothesis_counts[token]
        }
        if repeated_tokens:
            item["ambiguous_occurrence"] = True
            item["ambiguity_reason"] = "Repeated token counts do not identify a unique spoken occurrence."
        if item["classification"] == "MATCH":
            continue
        ref_index, hyp_index = item["reference_index"], item["recognized_index"]
        touches_unsupported_number = (
            ref_index is not None and reference_unsupported_numbers[ref_index]
        ) or (
            hyp_index is not None and hypothesis_unsupported_numbers[hyp_index]
        )
        if touches_unsupported_number:
            item["candidate_classification"] = item["classification"]
            item["classification"] = "UNCERTAIN"
            item["uncertainty_reason"] = "The numeric expression is outside the documented normalization range."
        item["speaker_attribution"] = "UNCERTAIN_ASR_ONLY"
    return operations


def repeated_response_spans(hypothesis: list[str], max_phrase_tokens: int = 8) -> list[dict[str, Any]]:
    """Return adjacent repeated token spans as restart/repetition candidates."""
    results: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    longest = min(max_phrase_tokens, len(hypothesis) // 2)
    for width in range(longest, 0, -1):
        for start in range(0, len(hypothesis) - 2 * width + 1):
            repeat_start = start + width
            if hypothesis[start:repeat_start] != hypothesis[repeat_start:repeat_start + width]:
                continue
            key = (start, repeat_start, width)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "tokens": hypothesis[start:repeat_start],
                "first_recognized_indices": list(range(start, repeat_start)),
                "repeat_recognized_indices": list(range(repeat_start, repeat_start + width)),
                "classification": "POSSIBLE_REPEATED_PHRASE_OR_RESTART",
            })
    return results


def rs_sequence_diagnostic(reference: list[str], hypothesis: list[str]) -> dict[str, Any]:
    """Describe monotonic ordered matches without mapping them to Pearson points."""
    operations = _operation_stream(reference, hypothesis)
    matches = [item for item in operations if item["classification"] == "MATCH"]
    match_count = len(matches)
    reference_count = len(reference)
    coverage = None if reference_count == 0 else match_count / reference_count
    if reference_count and match_count == reference_count and all(item["classification"] == "MATCH" for item in operations):
        band = "ALL_WORDS_IN_CORRECT_SEQUENCE"
    elif match_count == 0:
        band = "ALMOST_NONE_CANDIDATE"
    elif coverage is not None and coverage >= 0.5:
        band = "AT_LEAST_HALF"
    else:
        band = "LESS_THAN_HALF"
    return {
        "ordered_match_count": match_count,
        "reference_word_count": reference_count,
        "ordered_match_coverage": coverage,
        "ordered_match_reference_indices": [item["reference_index"] for item in matches],
        "ordered_match_recognized_indices": [item["recognized_index"] for item in matches],
        "rubric_inspired_band": band,
        "official_pearson_item_score": None,
        "operations": operations,
        "method": "Deterministic Levenshtein MATCH operations; each recognized word matches at most one reference position in sequence.",
        "rubric_limit": "The official 'almost nothing' category has no published numeric threshold; this local band is descriptive only.",
    }


def _word_counts(reference: list[str], hypothesis: list[str]) -> dict[str, Any]:
    operations = _operation_stream(reference, hypothesis)
    matches = sum(item["classification"] == "MATCH" for item in operations)
    substitutions = sum(item["classification"] == "SUBSTITUTION" for item in operations)
    substitutions += sum(item["classification"] == "ORDER_MISMATCH" for item in operations)
    omissions = sum(item["classification"] == "OMISSION" for item in operations)
    insertions = sum(item["classification"] in ("INSERTION", "REPETITION") for item in operations)
    error_units = substitutions + omissions + insertions
    return {
        "word_error_rate": error_units / len(reference),
        "reference_token_count": len(reference),
        "recognized_token_count": len(hypothesis),
        "matches": matches,
        "substitutions": substitutions,
        "omissions": omissions,
        "insertions": insertions,
        "error_units": error_units,
        "unit": "dimensionless WER; token counts are words after documented normalization",
    }


def content_alignment_within_budget(reference_count: int, hypothesis_count: int) -> bool:
    return (
        reference_count >= 0 and hypothesis_count >= 0
        and (reference_count + 1) * (hypothesis_count + 1) <= MAX_CONTENT_ALIGNMENT_CELLS
    )


def parse_silence_intervals(log: str, duration: float) -> list[dict[str, float]]:
    starts: list[float] = []
    intervals: list[dict[str, float]] = []
    for line in log.splitlines():
        match = SILENCE_START.search(line)
        if match:
            starts.append(float(match.group(1)))
        match = SILENCE_END.search(line)
        if match and starts:
            start = starts.pop(0)
            end = min(duration, float(match.group(1)))
            if end > start:
                intervals.append({"start_seconds": start, "end_seconds": end, "duration_seconds": end - start})
    for start in starts:
        if start < duration:
            intervals.append({"start_seconds": start, "end_seconds": duration, "duration_seconds": duration - start})
    return intervals


def _signal_energy(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict[str, Any]:
    frame = int(sample_rate * 0.025)
    hop = int(sample_rate * 0.010)
    if samples.size == 0:
        raise ValueError("Decoded audio is empty.")
    if samples.size < frame:
        frames = samples.reshape(1, -1)
    else:
        windows = np.lib.stride_tricks.sliding_window_view(samples, frame)[::hop]
        frames = windows
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    dbfs = 20.0 * np.log10(np.maximum(rms, 1e-9))
    p10, median, p90 = np.percentile(dbfs, [10, 50, 90])
    peak = float(np.max(np.abs(samples)))
    values = [float(p10), float(median), float(p90), 20.0 * math.log10(max(peak, 1e-9))]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Decoded audio energy measurements are not finite.")
    return {
        "frame_rms_dbfs_p10_median_p90": [round(values[0], 3), round(values[1], 3), round(values[2], 3)],
        "peak_dbfs": round(values[3], 3),
        "frame_length_ms": 25,
        "hop_ms": 10,
        "unit": "dBFS",
        "interpretation": "Signal-level descriptor; microphone gain and noise affect values; not a prosody or loudness grade.",
    }


def _vad_intervals(samples: np.ndarray) -> list[dict[str, float]]:
    options = VadOptions()
    detected = get_speech_timestamps(samples, options, sampling_rate=SAMPLE_RATE)
    return [
        {"start_seconds": segment["start"] / SAMPLE_RATE, "end_seconds": segment["end"] / SAMPLE_RATE}
        for segment in detected
    ]


def transcribe_audio(
    audio: Path,
    samples: np.ndarray,
    model_size: str = "base.en",
    threads: int = 4,
) -> dict[str, Any]:
    """Decode using score.py's current settings and expose native diagnostics."""
    started = time.perf_counter()
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=threads,
        num_workers=1,
        local_files_only=True,
    )
    model_load_seconds = time.perf_counter() - started
    vad_intervals = _vad_intervals(samples)
    duration = samples.size / SAMPLE_RATE
    vad_interval_duration = sum(item["end_seconds"] - item["start_seconds"] for item in vad_intervals)
    if not vad_intervals:
        segments: list[Any] = []
        info = None
        inference_seconds = time.perf_counter() - started - model_load_seconds
    else:
        segment_iterator, info = model.transcribe(
            samples,
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        segments = list(segment_iterator)
        inference_seconds = time.perf_counter() - started - model_load_seconds

    raw_transcript = " ".join(segment.text.strip() for segment in segments).strip()
    hyp_tokens, hyp_unsupported = normalize_tokens(raw_transcript)
    recognized_segment_token_ranges: list[tuple[int, int, Any]] = []
    cursor = 0
    for segment in segments:
        segment_tokens, _ = normalize_tokens(segment.text)
        recognized_segment_token_ranges.append((cursor, cursor + len(segment_tokens), segment))
        cursor += len(segment_tokens)
    segment_by_token: dict[int, Any] = {}
    for start, end, segment in recognized_segment_token_ranges:
        for index in range(start, end):
            segment_by_token[index] = segment

    diagnostic_thresholds = {
        "avg_logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "compression_ratio_threshold": 2.4,
        "source": "faster-whisper 1.2.1 decoder defaults; diagnostics are not calibrated correctness probabilities",
    }
    segment_results = []
    for index, segment in enumerate(segments):
        flags = []
        if segment.avg_logprob < diagnostic_thresholds["avg_logprob_threshold"]:
            flags.append("below_decoder_log_probability_threshold")
        if segment.no_speech_prob >= diagnostic_thresholds["no_speech_threshold"]:
            flags.append("at_or_above_decoder_no_speech_threshold")
        if segment.compression_ratio > diagnostic_thresholds["compression_ratio_threshold"]:
            flags.append("above_decoder_compression_ratio_threshold")
        if segment.temperature is not None and segment.temperature > 0:
            flags.append("temperature_fallback_used")
        segment_results.append({
            "index": index,
            "start_seconds_original_timeline": float(segment.start),
            "end_seconds_original_timeline": float(segment.end),
            "raw_text": segment.text,
            "avg_logprob": float(segment.avg_logprob),
            "no_speech_prob": float(segment.no_speech_prob),
            "compression_ratio": float(segment.compression_ratio),
            "final_temperature": None if segment.temperature is None else float(segment.temperature),
            "diagnostic_flags": flags,
        })
    return {
        "raw_transcript": raw_transcript,
        "normalized_tokens": hyp_tokens,
        "unsupported_numeric_tokens": hyp_unsupported,
        "vad_intervals": vad_intervals,
        "vad_interval_duration_seconds": vad_interval_duration,
        "vad_interval_complement_duration_seconds": max(0.0, duration - vad_interval_duration),
        "duration_after_vad_seconds": 0.0 if not vad_intervals else float(info.duration_after_vad),
        "segments": segment_results,
        "segment_objects": segments,
        "recognized_token_segments": segment_by_token,
        "duration_seconds": duration,
        "model_load_seconds": model_load_seconds,
        "inference_seconds_excluding_model_load": inference_seconds,
        "model": {
            "name": model_size,
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": threads,
            "faster_whisper_version": importlib.metadata.version("faster-whisper"),
        },
        "decode_configuration": {
            "language": "en",
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": {"threshold": 0.5, "neg_threshold": 0.35, "min_speech_duration_ms": 0,
                                "min_silence_duration_ms": 2000, "speech_pad_ms": 400,
                                "max_speech_duration_seconds": "unbounded (installed faster-whisper default)"},
            "condition_on_previous_text": False,
            "temperature": "not explicitly set; installed faster-whisper default retry sequence retained",
            "reference_prompt": None,
        },
        "diagnostic_thresholds": diagnostic_thresholds,
        "elapsed_seconds_including_model_load": time.perf_counter() - started,
    }


def _component_failure(status: str, message: str, stage: str | None = None, **details: Any) -> dict[str, Any]:
    error: dict[str, Any] = {"status": status, "message": message}
    if stage is not None:
        error["stage"] = stage
    if details:
        error["details"] = details
    return {"status": "FAILED", "measurements": None, "predictions": None, "provenance": None, "warnings": [], "error": error}


def _probe_audio_duration(audio: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required for bounded audio processing.")
    process = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
        capture_output=True, text=True, timeout=30,
    )
    if process.returncode != 0:
        raise ValueError(process.stderr[-1000:] or "ffprobe could not inspect the audio stream.")
    duration = float(process.stdout.strip())
    if not math.isfinite(duration):
        raise ValueError("Audio duration is not finite.")
    return duration


def _build_content(reference: str, transcription: dict[str, Any], task: str = "ra") -> dict[str, Any]:
    task_name = "READ_ALOUD" if task == "ra" else "REPEAT_SENTENCE"
    ref_tokens, ref_unsupported = normalize_tokens(reference)
    hyp_tokens = transcription["normalized_tokens"]
    hyp_unsupported = transcription["unsupported_numeric_tokens"]
    if not content_alignment_within_budget(len(ref_tokens), len(hyp_tokens)):
        failure = _component_failure(
            "CONTENT_ALIGNMENT_LIMIT", "Reference and transcript exceed the bounded content-alignment cell budget.",
            "content_alignment", reference_token_count=len(ref_tokens), recognized_token_count=len(hyp_tokens),
            max_alignment_cells=MAX_CONTENT_ALIGNMENT_CELLS,
        )
        failure["task"] = task_name
        return failure
    no_speech = not transcription["vad_intervals"]
    alignment = classify_alignment(
        ref_tokens,
        hyp_tokens,
        reference_unsupported_numbers=ref_unsupported,
        hypothesis_unsupported_numbers=hyp_unsupported,
        no_speech=no_speech,
    )
    for operation in alignment:
        recognized_index = operation.get("recognized_index")
        segment = transcription["recognized_token_segments"].get(recognized_index)
        if segment is not None:
            low_support = (
                segment.avg_logprob < transcription["diagnostic_thresholds"]["avg_logprob_threshold"]
                or segment.no_speech_prob >= transcription["diagnostic_thresholds"]["no_speech_threshold"]
                or segment.compression_ratio > transcription["diagnostic_thresholds"]["compression_ratio_threshold"]
            )
            operation["recognition_support"] = "LOW" if low_support else "NO_THRESHOLD_CROSSED"
            operation["decoder_diagnostic_flags"] = list(
                next((item["diagnostic_flags"] for item in transcription["segments"]
                      if item["start_seconds_original_timeline"] == segment.start
                      and item["end_seconds_original_timeline"] == segment.end), [])
            )
        else:
            low_support = False
        if segment is not None and operation["classification"] not in ("MATCH", "UNCERTAIN"):
            if low_support:
                operation["candidate_classification"] = operation["classification"]
                operation["classification"] = "UNCERTAIN"
                operation["uncertainty_reason"] = "The recognized token belongs to a segment that crossed a native decoder diagnostic threshold."
                operation["speaker_attribution"] = "UNCERTAIN_ASR_ONLY"

    counts = _word_counts(ref_tokens, hyp_tokens)
    warnings = [
        "Alignment categories describe differences between the reference and ASR transcript, not verified speaker omissions, insertions, substitutions, or word-order errors.",
        "avg_logprob, no_speech_prob, compression_ratio, and temperature are decoder diagnostics, not calibrated probabilities of transcript correctness.",
        "The reference is not supplied to ASR as a prompt; transcription is not reference-conditioned.",
    ]
    if no_speech:
        warnings.append("No speech interval was detected by the ASR VAD; reference differences are UNCERTAIN rather than speaker omissions.")
    if any(ref_unsupported) or any(hyp_unsupported):
        warnings.append("Numbers outside integer 0–99 remain literal; non-matching numeric expressions are marked UNCERTAIN.")
    unsupported_text_chars = sorted(set(unsupported_characters(reference) + unsupported_characters(transcription["raw_transcript"])))
    if unsupported_text_chars:
        warnings.append("Unsupported non-word symbols were not used as tokens: " + " ".join(unsupported_text_chars) + ".")
    flags = sorted({flag for segment in transcription["segments"] for flag in segment["diagnostic_flags"]})
    if flags:
        warnings.append("One or more segments crossed native decoder diagnostic thresholds: " + ", ".join(flags) + ".")
    return {
        "task": task_name,
        "status": "UNCERTAIN" if no_speech else "AVAILABLE_UNVERIFIED_ASR_PREDICTION",
        "measurements": {
            **counts,
            "reference_normalized_tokens": ref_tokens,
            "alignment": alignment,
            "alignment_semantics": "ASR/reference token operation; not an acoustically confirmed learner content error",
            "repeated_response_spans": repeated_response_spans(hyp_tokens),
            **({"sequence_diagnostic": rs_sequence_diagnostic(ref_tokens, hyp_tokens)} if task == "rs" else {}),
            "task_content_semantics": (
                "Correct reference words and only those words; ASR agreement remains unverified."
                if task == "ra" else
                "Correct reference words in order; the sequence diagnostic is not a Pearson item score."
            ),
        },
        "predictions": {
            "transcript_raw": transcription["raw_transcript"],
            "transcript_normalized_tokens": hyp_tokens,
        },
        "provenance": {
            **transcription["model"],
            "decoding": transcription["decode_configuration"],
            "segment_diagnostics": transcription["segments"],
            "diagnostic_thresholds": transcription["diagnostic_thresholds"],
            "vad_intervals_original_timeline": transcription["vad_intervals"],
            "vad_interval_duration_seconds": transcription["vad_interval_duration_seconds"],
            "vad_interval_complement_duration_seconds": transcription["vad_interval_complement_duration_seconds"],
            "duration_after_vad_seconds": transcription["duration_after_vad_seconds"],
            "model_load_seconds": transcription["model_load_seconds"],
            "inference_seconds_excluding_model_load": transcription["inference_seconds_excluding_model_load"],
            "elapsed_seconds_including_model_load": transcription["elapsed_seconds_including_model_load"],
        },
        "warnings": warnings,
        "error": None,
    }


def gate_pronunciation_attribution(
    pronunciation: dict[str, Any], content: dict[str, Any],
) -> dict[str, Any]:
    """Keep forced-reference predictions diagnostic unless ASR supports the full ordered text."""
    if pronunciation.get("status") != "AVAILABLE_UNCALIBRATED_MODEL_PREDICTIONS":
        return pronunciation
    content_measurements = content.get("measurements") or {}
    reference_tokens = content_measurements.get("reference_normalized_tokens") or []
    alignment = content_measurements.get("alignment") or []
    exact_asr_match = (
        content.get("status") == "AVAILABLE_UNVERIFIED_ASR_PREDICTION"
        and bool(reference_tokens)
        and len(alignment) == len(reference_tokens)
        and all(item.get("classification") == "MATCH" for item in alignment)
        and all(item.get("recognition_support") != "LOW" for item in alignment)
        and not any(item.get("ambiguous_occurrence") for item in alignment)
    )
    measurements = pronunciation.get("measurements") or {}
    reference_word_count = len(reference_tokens)
    if exact_asr_match:
        measurements["coverage"] = {
            "reference_word_count": reference_word_count,
            "supported_word_count": len(measurements.get("words") or []),
            "supported_phone_count": len(measurements.get("phonemes") or []),
            "coverage_basis": "Full normalized ASR/reference sequence match plus validated reference-conditioned acoustic alignment.",
            "speaker_error_attribution": "NOT_VERIFIED",
        }
        pronunciation["measurements"] = measurements
        pronunciation["speaker_attribution"] = {
            "status": "ASR_MATCH_UNVERIFIED",
            "confirmed_speaker_errors": False,
            "reason": "ASR sequence agreement is evidence for alignment selection, not independent verification of spoken content.",
        }
        pronunciation.setdefault("warnings", []).append(
            "ASR matched the complete reference, but it does not independently verify that every reference word was spoken."
        )
        return pronunciation

    raw_phones = measurements.get("phonemes") or []
    raw_words = measurements.get("words") or []
    raw_predictions = pronunciation.get("predictions")
    reason = (
        "ASR failed or was unavailable."
        if content.get("status") == "FAILED" else
        "ASR/reference differences, low-support tokens, or ambiguous repeated-token alignment prevent safe word attribution."
    )
    pronunciation["status"] = "PARTIAL_UNATTRIBUTED_REFERENCE_CONDITIONED_DIAGNOSTICS"
    pronunciation["measurements"] = {
        "phone_count": 0,
        "word_count": 0,
        "phonemes": [],
        "words": [],
        "phone_timestamps": [],
        "word_to_phone_relationships": [],
        "units": measurements.get("units"),
        "coverage": {
            "reference_word_count": reference_word_count,
            "supported_word_count": 0,
            "supported_phone_count": 0,
            "unsupported_reference_word_indices": list(range(reference_word_count)),
            "reason": reason,
        },
    }
    if isinstance(pronunciation.get("long_form"), dict):
        pronunciation["long_form"] = {
            **pronunciation["long_form"],
            "supported_phone_count": 0,
            "coverage": 0.0,
        }
    pronunciation["research_diagnostics"] = {
        "source": "Full-reference-conditioned forced alignment and GOPT predictions; not attributed to spoken words.",
        "phonemes": raw_phones,
        "words": raw_words,
        "predictions": raw_predictions,
    }
    pronunciation["predictions"] = None
    pronunciation["speaker_attribution"] = {
        "status": "UNSUPPORTED_ASR_MISMATCH_OR_UNCERTAINTY",
        "confirmed_speaker_errors": False,
        "reason": reason,
    }
    pronunciation.setdefault("warnings", []).append(
        "Reference-conditioned predictions are withheld from learner attribution because ASR did not support the complete ordered reference; raw model outputs remain available as research diagnostics."
    )
    return pronunciation


def has_exact_asr_reference_match(content: dict[str, Any]) -> bool:
    """Recovery requires exact ordered ASR/reference agreement, never a guess."""
    measurements = content.get("measurements") or {}
    alignment = measurements.get("alignment")
    return (
        content.get("status") == "AVAILABLE_UNVERIFIED_ASR_PREDICTION"
        and isinstance(alignment, list)
        and bool(alignment)
        and all(item.get("classification") == "MATCH" for item in alignment)
    )


def _build_pronunciation(
    audio: Path, reference: str, threads: int, *, recover_supported_words: bool = False,
) -> dict[str, Any]:
    try:
        gopt_python = Path(__file__).resolve().parent / ".local" / "gopt-venv" / "bin" / "python"
        if not gopt_python.is_file():
            return _component_failure("MISSING_DEPENDENCY", "Local GOPT Python runtime is missing.", "gopt_runtime", python=str(gopt_python))
        command = [
            str(gopt_python), str(Path(__file__).resolve().parent / "score_pronunciation.py"),
            "--audio", str(audio), "--reference", reference, "--threads", str(threads),
        ]
        if recover_supported_words:
            command.append("--recover-supported-words")
        process = subprocess.run(
            command,
            capture_output=True, text=True, timeout=900,
        )
        result = json.loads(process.stdout)
    except subprocess.TimeoutExpired as exc:
        return _component_failure("STAGE_TIMEOUT", "Local Kaldi/GOPT pronunciation processing timed out.", "pronunciation", timeout_seconds=exc.timeout)
    except (OSError, json.JSONDecodeError) as exc:
        return _component_failure("PRONUNCIATION_FAILED", str(exc), "pronunciation")
    except Exception as exc:
        return _component_failure("PRONUNCIATION_FAILED", str(exc), "pronunciation")
    if result.get("status") not in ("SCORING_COMPLETED", "SCORING_COMPLETED_PARTIAL"):
        return _component_failure(
            result.get("status", "PRONUNCIATION_FAILED"), result.get("error", "Pronunciation scoring did not complete."),
            result.get("stage"), details=result.get("details"), stderr=process.stderr[-2000:]
        )
    pronunciation = result["pronunciation"]
    scale = pronunciation["native_scale"]
    partial = result.get("status") == "SCORING_COMPLETED_PARTIAL"
    warnings = list(result.get("warnings", []))
    warnings.extend([
        "GOPT phone, word, and utterance values are uncalibrated model predictions, not PTE scores or probabilities of mispronunciation.",
        "Reference-conditioned alignment does not independently prove that all reference words were spoken.",
        "Individual phones are not classified as mispronounced because no validated error threshold is available.",
    ])
    return {
        "status": "PARTIAL_UNCALIBRATED_MODEL_PREDICTIONS" if partial else "AVAILABLE_UNCALIBRATED_MODEL_PREDICTIONS",
        "measurements": {
            "phone_count": result["long_form"]["processed_phone_count"],
            "reference_phone_count": result["long_form"]["reference_phone_count"],
            "supported_phone_count": result["long_form"]["supported_phone_count"],
            "coverage": result["long_form"]["coverage"],
            "word_count": sum(word.get("gopt_scores") is not None for word in result["words"]),
            "reference_word_count": len(result["words"]),
            "phonemes": result["phonemes"],
            "words": result["words"],
            "phone_timestamps": [
                {"index": item["index"], "word": item["word"], "phone": item["phone"],
                 "start_seconds": item["start_seconds"], "end_seconds": item["end_seconds"]}
                for item in result["phonemes"]
            ],
            "word_to_phone_relationships": [
                {"word_index": word["word_index"], "word": word["word"],
                 "phone_indices": [phone["index"] for phone in result["phonemes"] if phone["word_index"] == word["word_index"]]}
                for word in result["words"]
            ],
            "units": scale,
        },
        "predictions": pronunciation,
        "long_form": result.get("long_form") or {
            "reference_phone_count": len(result["phonemes"]),
            "processed_phone_count": len(result["phonemes"]),
            "supported_phone_count": len(result["phonemes"]),
            "chunk_count": 1,
            "coverage": 1.0,
            "aggregation_status": "SINGLE_WINDOW_DIRECT",
        },
        "provenance": {
            "pipeline": "Kaldi M13 LibriSpeech chain → forced alignment on output-xent → raw output GOP features → official GOPT LibriSpeech checkpoint",
            "feature_acoustic_output_node": result["evidence"]["feature_acoustic_output_node"],
            "alignment_and_support_output_node": result["evidence"]["alignment_and_support_output_node"],
            "alignment_valid": result["validation"]["alignment_valid"],
            "transcript_independently_verified": result["validation"]["transcript_independently_verified"],
            "phone_order_source": result["evidence"]["phone_order_source"],
            "canonical_phone_sequence_source": result["evidence"]["canonical_phone_sequence_source"],
            "support_recovery_policy": "whole-word regions after exact ASR/reference match" if recover_supported_words else "whole-window rejection",
            "acoustic_model": result["evidence"]["model"]["acoustic"],
            "gopt_checkpoint_sha256": result["evidence"]["model"]["gopt_checkpoint_sha256"],
            "native_scale": scale,
            "max_phone_positions": score_pronunciation.MAX_PHONES,
            "performance": result["performance"],
            "processing_time_seconds": result["processing_time_seconds"],
            "evidence": result["evidence"],
        },
        "warnings": warnings,
        "error": None,
    }


def _build_fluency(
    samples: np.ndarray,
    transcription: dict[str, Any] | None,
    silence_intervals: list[dict[str, float]],
    pronunciation: dict[str, Any],
) -> dict[str, Any]:
    duration = samples.size / SAMPLE_RATE
    internal_pauses = [
        item for item in silence_intervals
        if item["start_seconds"] > 0.01 and item["end_seconds"] < duration - 0.01
    ]
    leading = [item for item in silence_intervals if item["start_seconds"] <= 0.01]
    trailing = [item for item in silence_intervals if item["end_seconds"] >= duration - 0.01]
    normalized_count = None if transcription is None else len(transcription["normalized_tokens"])
    vad_intervals = [] if transcription is None else transcription["vad_intervals"]
    return {
        "status": "PARTIAL_TIMING_MEASUREMENTS" if transcription is None else "PARTIAL_DESCRIPTIVE_MEASUREMENTS",
        "measurements": {
            "total_recording_duration_seconds": duration,
            "vad_clip_count": None if transcription is None else len(vad_intervals),
            "vad_interval_duration_seconds": None if transcription is None else transcription["vad_interval_duration_seconds"],
            "asr_vad_compressed_input_duration_seconds": None if transcription is None else transcription["duration_after_vad_seconds"],
            "active_speech_duration_seconds": None,
            "active_speech_duration_reason": "VAD intervals include padding and can merge pauses; neither interval duration nor ASR-compressed duration is labeled active speech time.",
            "speech_rate": {
                "recognized_asr_tokens_per_original_recording_minute": None if normalized_count is None else normalized_count * 60 / duration,
                "unit": "normalized ASR tokens/minute",
                "not_articulation_rate": True,
                "reason": None if normalized_count is not None else "ASR transcript unavailable.",
            },
            "articulation_rate": {"value": None, "unit": "syllables/active-speech minute", "reason": "No validated syllable counter and active-speech interval measurement are available."},
            "silence_candidates": {
                "internal_pause_count": len(internal_pauses),
                "internal_pause_total_seconds": sum(item["duration_seconds"] for item in internal_pauses),
                "internal_pause_median_seconds": None if not internal_pauses else float(np.median([item["duration_seconds"] for item in internal_pauses])),
                "internal_pause_max_seconds": None if not internal_pauses else max(item["duration_seconds"] for item in internal_pauses),
                "long_pause_candidates_ge_1_second": [item for item in internal_pauses if item["duration_seconds"] >= 1.0],
                "leading_silence_candidates": leading,
                "trailing_silence_candidates": trailing,
                "all_candidates": silence_intervals,
                "method": "FFmpeg silencedetect noise=-35dB, duration>=0.25s; candidate intervals only",
                "unit": "seconds",
            },
            "hesitation_evidence": {
                "asr_filler_token_candidates": [] if transcription is None else [token for token in transcription["normalized_tokens"] if token in FILLER_TOKENS],
                "status": "ASR-derived candidates only; no independently validated hesitation detector",
            },
        },
        "predictions": None if pronunciation.get("predictions") is None else {
            "gopt_fluency": pronunciation["predictions"].get("fluency"),
            "native_scale": pronunciation["predictions"].get("native_scale"),
        },
        "provenance": {
            "recording_timing_basis": "decoded original audio at 16 kHz mono; timestamps remain on the original recording timeline",
            "vad_intervals_original_timeline": vad_intervals,
            "ffmpeg_silence_detector": "ffmpeg silencedetect noise=-35dB, d=0.25",
        },
        "warnings": [
            "Silence candidates include recording boundaries and may be breathing, noise-floor effects, or natural pauses; no candidate is labeled a hesitation.",
            "ASR token rate can be wrong when ASR omits, inserts, or misrecognizes words and divides by original recording duration.",
            "No calibrated fluency grade or combined fluency score is produced.",
        ],
        "error": None,
    }


def _build_prosody(pronunciation: dict[str, Any], energy: dict[str, Any]) -> dict[str, Any]:
    prediction = pronunciation.get("predictions")
    gopt = None if prediction is None else prediction.get("prosody")
    status = "PARTIAL_MODEL_PREDICTION_AND_ENERGY_DESCRIPTOR" if gopt is not None else "PARTIAL_ENERGY_DESCRIPTOR_ONLY"
    return {
        "status": status,
        "measurements": {
            "energy_level_distribution": energy,
            "f0_contour": {"value": None, "status": "UNAVAILABLE", "reason": "No validated F0 tracker is installed in the local dependency set."},
            "intonation_contour": {"value": None, "status": "UNAVAILABLE", "reason": "No validated pitch/intonation measurement is available."},
            "rhythm_measurement": {"value": None, "status": "UNAVAILABLE", "reason": "No validated rhythm or syllable-timing method is available."},
        },
        "predictions": None if gopt is None else {"gopt_utterance_prosody": gopt, "native_scale": prediction["native_scale"]},
        "provenance": {
            "energy": "decoded waveform RMS distribution; acoustic signal descriptor, not a prosody assessment",
            "gopt": None if gopt is None else "official GOPT LibriSpeech prosody head; uncalibrated model prediction",
        },
        "warnings": [
            "Energy is affected by microphone distance, recording gain, and noise; it does not establish stress or intonation correctness.",
            "GOPT prosody is a native-scale model prediction and is not a calibrated learner grade.",
        ],
        "error": None,
    }


def _build_stress(pronunciation: dict[str, Any]) -> dict[str, Any]:
    measurements = pronunciation.get("measurements")
    words = [] if measurements is None else measurements.get("words", [])
    predictions = [
        {"word_index": word["word_index"], "word": word["word"],
         "gopt_word_stress": None if word.get("gopt_scores") is None else word["gopt_scores"]["stress"],
         "status": word.get("pronunciation_evidence_status", "UNAVAILABLE")}
        for word in words
    ]
    available = any(item["gopt_word_stress"] is not None for item in predictions)
    return {
        "status": "PARTIAL_MODEL_PREDICTION_LEXICAL_STRESS_UNAVAILABLE" if available else "UNAVAILABLE",
        "measurements": {
            "lexical_stress_measurement": None,
            "lexical_stress_status": "UNAVAILABLE: the live pronunciation lexicon does not provide a validated lexical-stress reference for each word.",
            "word_predictions": predictions,
            "native_scale": None if not available else measurements["units"],
        },
        "predictions": None if not available else {"source": "GOPT word-phone stress head averaged over aligned phones", "values": predictions},
        "provenance": None if not available else pronunciation.get("provenance"),
        "warnings": [
            "GOPT stress outputs are uncalibrated model predictions, not measured or expert-validated lexical stress correctness.",
            "No lexical-stress correctness label or phone threshold is generated.",
        ],
        "error": None,
    }


def _json_safe(value: Any) -> tuple[Any, bool]:
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        changed = False
        for key, item in value.items():
            safe[key], item_changed = _json_safe(item)
            changed = changed or item_changed
        return safe, changed
    if isinstance(value, (list, tuple)):
        result = []
        changed = False
        for item in value:
            safe, item_changed = _json_safe(item)
            result.append(safe)
            changed = changed or item_changed
        return result, changed
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None, True
    if isinstance(value, np.integer):
        return int(value), False
    return value, False


def _unavailable_component(reason: str) -> dict[str, Any]:
    return {"status": "UNAVAILABLE", "measurements": None, "predictions": None, "provenance": None,
            "warnings": [], "error": {"status": "COMPONENT_UNAVAILABLE", "message": reason}}


def assess(
    audio: Path, reference: str, model_size: str = "base.en", threads: int = 4, task: str = "ra",
    recover_supported_words: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    audio = audio.expanduser().resolve()
    task = task.lower() if isinstance(task, str) else ""
    task_name = "READ_ALOUD" if task == "ra" else "REPEAT_SENTENCE" if task == "rs" else None
    result: dict[str, Any] = {
        "schema_version": "quality-07-integrated-v1",
        "task": task_name,
        "status": "FAILED",
        "input": {"audio": str(audio), "reference": reference, "task": task_name},
        "overall_pte_score": None,
        "pearson_pte_score": None,
        "components": {},
        "long_form": {
            "reference_phone_count": None,
            "processed_phone_count": 0,
            "supported_phone_count": 0,
            "chunk_count": 0,
            "coverage": 0.0,
            "aggregation_status": "NOT_RUN",
        },
        "assessment_validity": {"empirical_assessment_quality": "NOT_VERIFIED", "pearson_agreement": "NOT_VERIFIED"},
        "warnings": [],
        "performance": {},
    }
    if task_name is None:
        failure = _component_failure("INVALID_TASK", "--task must be ra or rs.", "cli_arguments")
        result["status"] = "INVALID_INPUT"
        result["error"] = failure["error"]
        return result
    if not isinstance(reference, str) or not reference.strip():
        failure = _component_failure("INVALID_REFERENCE", "Reference must contain words.", "reference")
        result["components"] = {name: failure for name in ("content", "pronunciation", "oral_fluency", "prosody", "word_stress")}
        result["error"] = failure["error"]
        return result
    if not normalize_tokens(reference)[0]:
        failure = _component_failure("INVALID_REFERENCE", "Reference must contain supported English words or numbers.", "reference")
        result["components"] = {name: failure for name in ("content", "pronunciation", "oral_fluency", "prosody", "word_stress")}
        result["error"] = failure["error"]
        return result
    if not audio.is_file():
        failure = _component_failure("AUDIO_NOT_FOUND", f"Audio file not found: {audio}", "audio_input")
        result["components"] = {name: failure for name in ("content", "pronunciation", "oral_fluency", "prosody", "word_stress")}
        result["error"] = failure["error"]
        return result
    try:
        probed_duration = _probe_audio_duration(audio)
    except Exception as exc:
        failure = _component_failure("AUDIO_DECODE_FAILED", str(exc), "audio_probe")
        result["components"] = {name: failure for name in ("content", "pronunciation", "oral_fluency", "prosody", "word_stress")}
        result["error"] = failure["error"]
        return result
    result["input"]["duration_seconds"] = probed_duration
    if probed_duration <= 0 or probed_duration > score_pronunciation.MAX_AUDIO_SECONDS:
        failure = _component_failure(
            "INVALID_AUDIO_DURATION", "Audio duration is outside the supported finite 0–120 second range.",
            "audio_probe", duration_seconds=probed_duration,
        )
        result["error"] = failure["error"]
        return result
    try:
        samples = decode_audio(str(audio), sampling_rate=SAMPLE_RATE)
    except Exception as exc:
        failure = _component_failure("AUDIO_DECODE_FAILED", str(exc), "audio_decode")
        result["components"] = {name: failure for name in ("content", "pronunciation", "oral_fluency", "prosody", "word_stress")}
        result["error"] = failure["error"]
        return result
    if samples.size == 0 or not np.isfinite(samples).all():
        failure = _component_failure("AUDIO_DECODE_FAILED", "Decoded audio is empty or contains non-finite samples.", "audio_decode")
        result["components"] = {name: failure for name in ("content", "pronunciation", "oral_fluency", "prosody", "word_stress")}
        result["error"] = failure["error"]
        return result
    duration = samples.size / SAMPLE_RATE
    if not math.isfinite(duration) or duration <= 0 or duration > score_pronunciation.MAX_AUDIO_SECONDS:
        failure = _component_failure("INVALID_AUDIO_DURATION", "Audio duration is outside the supported finite 0–120 second range.", "audio_decode", duration_seconds=duration)
        result["components"] = {name: failure for name in ("content", "pronunciation", "oral_fluency", "prosody", "word_stress")}
        result["error"] = failure["error"]
        return result
    result["input"]["duration_seconds"] = duration
    try:
        transcription = transcribe_audio(audio, samples, model_size=model_size, threads=threads)
        content = _build_content(reference, transcription, task)
    except Exception as exc:
        transcription = None
        content = _component_failure("ASR_FAILED", str(exc), "asr_transcription")

    try:
        pronunciation = _build_pronunciation(
            audio, reference, threads,
            recover_supported_words=(
                recover_supported_words and has_exact_asr_reference_match(content)
            ),
        )
    except Exception as exc:
        pronunciation = _component_failure("PRONUNCIATION_FAILED", str(exc), "pronunciation")
    pronunciation = gate_pronunciation_attribution(pronunciation, content)
    if isinstance(pronunciation.get("long_form"), dict):
        result["long_form"] = pronunciation["long_form"]

    silence_intervals: list[dict[str, float]] = []
    energy: dict[str, Any] | None = None
    fluency: dict[str, Any]
    try:
        import subprocess
        process = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", str(audio),
             "-af", "silencedetect=noise=-35dB:d=0.25", "-f", "null", "-"],
            capture_output=True, text=True, check=True,
        )
        silence_intervals = parse_silence_intervals(process.stderr, duration)
        energy = _signal_energy(samples)
        fluency = _build_fluency(samples, transcription, silence_intervals, pronunciation)
    except Exception as exc:
        fluency = _component_failure("FLUENCY_MEASUREMENT_FAILED", str(exc), "temporal_analysis")
    if energy is None:
        energy = {"status": "UNAVAILABLE", "reason": "Waveform energy calculation failed."}

    prosody = _build_prosody(pronunciation, energy)
    stress = _build_stress(pronunciation)
    result["components"] = {
        "content": content,
        "pronunciation": pronunciation,
        "oral_fluency": fluency,
        "prosody": prosody,
        "word_stress": stress,
    }
    result["performance"] = {
        "total_processing_seconds": time.perf_counter() - started,
        "asr_model_load_seconds": None if transcription is None else transcription["model_load_seconds"],
        "asr_inference_seconds_excluding_model_load": None if transcription is None else transcription["inference_seconds_excluding_model_load"],
        "pronunciation_processing_seconds": None if pronunciation.get("provenance") is None else pronunciation["provenance"].get("processing_time_seconds"),
    }
    if any(
        component["status"] in ("FAILED", "UNAVAILABLE", "UNCERTAIN")
        or component["status"].startswith("PARTIAL")
        for component in result["components"].values()
    ):
        result["status"] = "PARTIAL_SUCCESS"
    else:
        result["status"] = "COMPONENTS_AVAILABLE_UNCALIBRATED"
    safe, changed = _json_safe(result)
    if changed:
        safe["warnings"].append("One or more non-finite measurements were serialized as null.")
        safe["status"] = "PARTIAL_SUCCESS"
    json.dumps(safe, allow_nan=False)
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description="Local integrated PTE-style speech measurements; not Pearson calibrated.")
    parser.add_argument("--task", choices=("ra", "rs"), default="ra")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--legacy-whole-window-policy", action="store_true",
        help="Use the frozen QUALITY-07 whole-window rejection policy for baseline comparisons.",
    )
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        failure = _component_failure("INVALID_THREADS", "--threads must be between 1 and 8.", "cli_arguments")
        print(json.dumps({
            "schema_version": "quality-07-integrated-v1",
            "task": "READ_ALOUD" if args.task == "ra" else "REPEAT_SENTENCE",
            "status": "INVALID_INPUT",
            "error": failure["error"],
            "components": {name: failure for name in ("content", "pronunciation", "oral_fluency", "prosody", "word_stress")},
            "overall_pte_score": None,
            "pearson_pte_score": None,
            "long_form": {
                "reference_phone_count": None, "processed_phone_count": 0, "supported_phone_count": 0,
                "chunk_count": 0, "coverage": 0.0, "aggregation_status": "NOT_RUN",
            },
        }, indent=2))
        return 2
    result = assess(
        args.audio, args.reference, args.model, args.threads, args.task,
        recover_supported_words=not args.legacy_whole_window_policy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] == "FAILED":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
