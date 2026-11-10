"""Read-aloud / repeat-sentence baseline. Not a Pearson PTE score."""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import jiwer

NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)
SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)")


def compare_words(reference: str, transcript: str) -> dict:
    if not reference.strip() or not re.search(r"\w", reference):
        raise ValueError("Reference must contain words.")
    result = jiwer.process_words(
        reference,
        transcript,
        reference_transform=NORMALIZE,
        hypothesis_transform=NORMALIZE,
    )
    # This is a transcription agreement proxy, not pronunciation or Pearson PTE.
    return {
        "word_error_rate": round(result.wer, 4),
        "word_match_percent": round(max(0.0, 100.0 * (1.0 - result.wer)), 1),
        "correct_words": result.hits,
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
    }


def audio_duration(audio: Path) -> float:
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
        capture_output=True, text=True, check=True,
    )
    duration = float(process.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio must have a positive, finite duration.")
    return duration


def parse_silence_log(log: str, duration: float) -> tuple[int, float]:
    """Acoustic silence heuristic; NOT a validated fluency measure."""
    starts = []
    count = 0
    seconds = 0.0
    for line in log.splitlines():
        m = SILENCE_START.search(line)
        if m:
            starts.append(float(m.group(1)))
        m = SILENCE_END.search(line)
        if m and starts:
            start = starts.pop(0)
            end = min(duration, float(m.group(1)))
            if end > start:
                seconds += end - start
                count += 1
    for start in starts:
        if start < duration:
            seconds += duration - start
            count += 1
    return count, min(duration, max(0.0, seconds))


def acoustic_pauses(audio: Path, duration: float) -> dict:
    process = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-nostats",
         "-i", str(audio),
         "-af", "silencedetect=noise=-35dB:d=0.25",
         "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    count, total = parse_silence_log(process.stderr, duration)
    return {
        "detected_silences": count,
        "silence_seconds": round(total, 2),
        "silence_ratio_percent": round(100 * total / duration, 1),
        "method": "ffmpeg silencedetect -35dB, >=0.25s (heuristic)",
    }


def transcribe(audio: Path, model_size: str, threads: int) -> tuple[str, float]:
    # Import lazily so word-scoring tests do not load speech models.
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=threads,
        num_workers=1,
    )
    segments, _ = model.transcribe(
        str(audio),
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, round(time.perf_counter() - started, 2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local PTE-style RA/RS scoring baseline; NOT Pearson calibrated."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", required=True, help="Expected English sentence.")
    parser.add_argument("--model", default="base.en", help="Whisper model, default base.en.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file not found: {args.audio}")
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    try:
        compare_words(args.reference, "")
        duration = audio_duration(args.audio)
        transcript, elapsed = transcribe(args.audio, args.model, args.threads)
        word_result = compare_words(args.reference, transcript)
        pauses = acoustic_pauses(args.audio, duration)
        detected_words = len(NORMALIZE(transcript)[0])
        result = {
            "status": "UNCALIBRATED_LOCAL_BASELINE",
            "input": str(args.audio.resolve()),
            "model": args.model,
            "reference": args.reference,
            "transcript": transcript,
            "audio_duration_seconds": round(duration, 2),
            "inference_seconds_including_model_load": elapsed,
            "real_time_factor": round(elapsed / duration, 2),
            "content_proxy": word_result,
            "fluency_descriptors": {
                "recognized_words_per_minute": round(detected_words * 60 / duration, 1),
                **pauses,
            },
            "pronunciation_score": None,
            "pearson_pte_score": None,
            "limitations": [
                "Word agreement measures ASR/reference match, not phoneme quality.",
                "Fluency descriptors are uncalibrated acoustic heuristics.",
                "No PTE accuracy, pronunciation or score correlation has been validated.",
            ],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, subprocess.CalledProcessError, RuntimeError, OSError) as exc:
        print(f"score failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
