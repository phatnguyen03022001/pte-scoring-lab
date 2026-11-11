"""Local, reference-conditioned Kaldi GOP + GOPT pronunciation assessment."""

from __future__ import annotations

import argparse
import array
import csv
import contextlib
import hashlib
import io
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import wave
import zipfile
from pathlib import Path
from typing import Any

from process_safety import DEADLINE_ENV, deadline_from_env, run_bounded


ROOT = Path(__file__).resolve().parent
KALDI_ROOT = ROOT / ".local" / "kaldi"
GOPT_ROOT = ROOT / ".local" / "gopt"
LIBRI_ROOT = KALDI_ROOT / "egs" / "librispeech" / "s5"
GOP_RECIPE = KALDI_ROOT / "egs" / "gop_speechocean762" / "s5"
ACOUSTIC_MODEL = LIBRI_ROOT / "exp" / "chain_cleaned" / "tdnn_1d_sp"
IVECTOR_EXTRACTOR = LIBRI_ROOT / "exp" / "nnet3_cleaned" / "extractor"
LANG = LIBRI_ROOT / "data" / "lang_test_tgsmall"
CHECKPOINT = GOPT_ROOT / "pretrained_models" / "gopt_librispeech" / "best_audio_model.pth"
GOPT_DATA = GOPT_ROOT / "data.zip"

MAX_AUDIO_SECONDS = 120.0
MAX_PHONES = 50
MAX_GOPT_WINDOWS = 128
FEATURE_DIM = 84
PURE_PHONE_COUNT = 42
GOPT_MEAN = 3.203
GOPT_STD = 4.045
FRAME_SHIFT_SECONDS = 0.03
MIN_PHONE_LOG_POSTERIOR = -6.0
FEATURE_ACOUSTIC_OUTPUT_NODE = "output"
ALIGNMENT_SUPPORT_OUTPUT_NODE = "output-xent"

# Reconstructed from the first-seen phone labels in the official GOPT
# LibriSpeech training archive, not from the local inference utterance.
GOPT_PHONE_ORDER = (
    "W", "IY", "K", "AO", "L", "IH", "T", "B", "EH", "R", "Z", "OW",
    "TH", "F", "AY", "V", "AH", "N", "UW", "S", "G", "AA", "M", "P",
    "NG", "HH", "EY", "SH", "AE", "D", "UH", "AW", "DH", "ER", "Y",
    "JH", "CH", "OY", "ZH",
)


class ScoreError(Exception):
    def __init__(self, status: str, stage: str, message: str, **details: Any):
        super().__init__(message)
        self.status = status
        self.stage = stage
        self.message = message
        self.details = details


def normalize_reference(reference: str) -> list[str]:
    text = reference.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.upper()
    # A single leading apostrophe can mark a spoken elision (e.g. 'em).
    # Keep internal contractions intact and reject doubled/malformed apostrophes.
    text = re.sub(r"(?<![A-Z'])'(?=[A-Z])", "", text)
    words = re.findall(r"[A-Z]+(?:'[A-Z]+)*|[0-9]+", text)
    if not words:
        raise ScoreError("INVALID_REFERENCE", "reference", "Reference must contain English words.")
    if any(word.isdigit() for word in words):
        raise ScoreError(
            "INVALID_REFERENCE", "reference",
            "Spell out numbers in the reference so the pronunciation lexicon can resolve them.",
        )
    residue = re.sub(r"[A-Z]+(?:'[A-Z]+)*|[0-9]+|[\s.,!?;:\-–—()\[\]{}\"“”/…]+", "", text)
    if residue:
        raise ScoreError(
            "INVALID_REFERENCE", "reference", "Reference contains unsupported characters.",
            characters=sorted(set(residue)),
        )
    return words


def read_symbol_table(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    symbol_to_id: dict[str, int] = {}
    id_to_symbol: dict[int, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 2:
                continue
            symbol, raw_id = fields[:2]
            if symbol.startswith("#"):
                continue
            index = int(raw_id)
            if symbol in symbol_to_id or index in id_to_symbol:
                raise ValueError(f"duplicate symbol or ID in {path}: {line}")
            symbol_to_id[symbol] = index
            id_to_symbol[index] = symbol
    except (OSError, ValueError) as exc:
        raise ScoreError("MODEL_INCOMPATIBLE", "phone_inventory", f"Cannot read symbol table: {path}", error=str(exc)) from exc
    return symbol_to_id, id_to_symbol


def read_alignment_lexicon(path: Path) -> dict[str, list[list[str]]]:
    lexicon: dict[str, list[list[str]]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            word, output_word, *phones = fields
            if word != output_word or word.startswith("<"):
                continue
            lexicon.setdefault(word, []).append(phones)
    except OSError as exc:
        raise ScoreError("MODEL_INCOMPATIBLE", "lexicon", f"Cannot read pronunciation lexicon: {path}", error=str(exc)) from exc
    return lexicon


def strip_phone_markers(phone: str) -> str:
    phone = re.sub(r"_[BIES]$", "", phone)
    return re.sub(r"[0-9]+$", "", phone)


def expected_phones(words: list[str], lexicon: dict[str, list[list[str]]], pure_symbol_to_id: dict[str, int]) -> tuple[list[dict[str, Any]], list[str]]:
    missing = sorted(set(words) - set(lexicon))
    if missing:
        raise ScoreError(
            "UNSUPPORTED_VOCABULARY", "lexicon", "One or more reference words are absent from the LibriSpeech pronunciation lexicon.",
            words=missing,
        )

    expected: list[dict[str, Any]] = []
    warnings: list[str] = []
    for word_index, word in enumerate(words):
        pronunciations = lexicon[word]
        if len(pronunciations) > 1:
            warnings.append(f"{word}: selected the lexicon's first pronunciation from {len(pronunciations)} listed variants.")
        for marked_phone in pronunciations[0]:
            base_phone = strip_phone_markers(marked_phone)
            if base_phone not in pure_symbol_to_id:
                raise ScoreError(
                    "UNSUPPORTED_PHONEME", "phone_inventory", "Pronunciation lexicon phone is absent from the acoustic model inventory.",
                    word=word, phone=marked_phone, base_phone=base_phone,
                )
            pure_id = pure_symbol_to_id[base_phone]
            if pure_id < 3:
                raise ScoreError(
                    "UNSUPPORTED_PHONEME", "phone_inventory", "A lexical phone maps to a reserved silence/noise class.",
                    word=word, phone=marked_phone,
                )
            expected.append({"word": word, "word_index": word_index, "phone": base_phone, "pure_phone_id": pure_id})

    if not expected:
        raise ScoreError("INVALID_REFERENCE", "reference", "Reference has no lexical phones.")
    return expected, warnings


def read_source_phone_transcript(path: Path, utterance_id: str) -> list[list[str]]:
    """Read one source-native word-to-phone sequence from Kaldi text-phone."""
    rows: dict[int, list[str]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                continue
            key, phones = fields
            prefix, separator, raw_index = key.rpartition(".")
            if not separator or prefix != utterance_id:
                continue
            index = int(raw_index)
            if index in rows:
                raise ValueError(f"duplicate word index {index} for {utterance_id}")
            rows[index] = phones.split()
    except (OSError, ValueError) as exc:
        raise ScoreError("INVALID_INPUT", "source_phone_transcript", "Cannot read the requested source phone transcript.", path=str(path), utterance_id=utterance_id, error=str(exc)) from exc
    if not rows or sorted(rows) != list(range(len(rows))) or any(not phones for phones in rows.values()):
        raise ScoreError("INVALID_INPUT", "source_phone_transcript", "Source phone transcript is missing words or has non-contiguous word indexes.", path=str(path), utterance_id=utterance_id, word_indexes=sorted(rows))
    return [rows[index] for index in range(len(rows))]


def expected_phones_from_source(
    words: list[str],
    marked_phones_by_word: list[list[str]],
    pure_symbol_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    """Validate source-native canonical phones without substituting lexicon variants."""
    if len(words) != len(marked_phones_by_word):
        raise ScoreError(
            "ALIGNMENT_INVALID", "source_phone_transcript",
            "Source phone transcript word count differs from the reference transcript.",
            reference_words=len(words), source_words=len(marked_phones_by_word),
        )
    expected: list[dict[str, Any]] = []
    for word_index, (word, marked_phones) in enumerate(zip(words, marked_phones_by_word)):
        for marked_phone in marked_phones:
            base_phone = strip_phone_markers(marked_phone)
            if base_phone not in pure_symbol_to_id:
                raise ScoreError(
                    "UNSUPPORTED_PHONEME", "source_phone_transcript",
                    "Source canonical phone is absent from the local acoustic model inventory.",
                    word=word, marked_phone=marked_phone, base_phone=base_phone,
                )
            pure_id = pure_symbol_to_id[base_phone]
            if pure_id < 3:
                raise ScoreError(
                    "UNSUPPORTED_PHONEME", "source_phone_transcript",
                    "Source canonical phone maps to a reserved silence/noise class.",
                    word=word, marked_phone=marked_phone, pure_phone_id=pure_id,
                )
            expected.append({
                "word": word, "word_index": word_index, "phone": base_phone,
                "pure_phone_id": pure_id, "marked_phone": marked_phone,
            })
    if not expected:
        raise ScoreError(
            "INVALID_REFERENCE", "source_phone_transcript",
            "Source phone sequence is empty.", phone_count=len(expected),
        )
    return expected


def plan_gopt_windows(
    expected: list[dict[str, Any]], max_phones: int = MAX_PHONES,
) -> list[dict[str, int]]:
    """Partition a global phone sequence into contiguous, whole-word windows."""
    if not expected:
        raise ScoreError("INVALID_REFERENCE", "gopt_windowing", "Cannot window an empty phone sequence.")
    if not 1 <= max_phones <= MAX_PHONES:
        raise ScoreError(
            "INVALID_INPUT", "gopt_windowing", "Window size must be between 1 and the model's 50-phone limit.",
            max_phones=max_phones,
        )

    word_runs: list[tuple[int, int, int]] = []
    seen_words: set[int] = set()
    run_start = 0
    current_word = expected[0].get("word_index")
    if not isinstance(current_word, int) or isinstance(current_word, bool) or current_word < 0:
        raise ScoreError("ALIGNMENT_INVALID", "gopt_windowing", "Reference word indices must be non-negative integers.")
    if current_word != 0:
        raise ScoreError(
            "ALIGNMENT_INVALID", "gopt_windowing",
            "Reference word indices must start at zero and remain consecutive.",
            first_word_index=current_word,
        )
    seen_words.add(current_word)
    for index, phone in enumerate(expected[1:], start=1):
        word_index = phone.get("word_index")
        if not isinstance(word_index, int) or isinstance(word_index, bool) or word_index < 0:
            raise ScoreError("ALIGNMENT_INVALID", "gopt_windowing", "Reference word indices must be non-negative integers.", phone_index=index)
        if word_index == current_word:
            continue
        if word_index in seen_words or word_index <= current_word:
            raise ScoreError(
                "ALIGNMENT_INVALID", "gopt_windowing", "Phones are not grouped in strictly increasing word order.",
                phone_index=index, previous_word_index=current_word, word_index=word_index,
            )
        if word_index != current_word + 1:
            raise ScoreError(
                "ALIGNMENT_INVALID", "gopt_windowing",
                "Reference word indices must be consecutive global word indices.",
                phone_index=index, previous_word_index=current_word, word_index=word_index,
            )
        word_runs.append((current_word, run_start, index))
        current_word, run_start = word_index, index
        seen_words.add(word_index)
    word_runs.append((current_word, run_start, len(expected)))

    windows: list[dict[str, int]] = []
    phone_start = 0
    word_start = word_runs[0][0]
    phone_end = 0
    word_end = word_start
    for word_index, word_phone_start, word_phone_end in word_runs:
        word_phone_count = word_phone_end - word_phone_start
        if word_phone_count > max_phones:
            raise ScoreError(
                "UNSUPPORTED_SEQUENCE_LENGTH", "gopt_windowing",
                f"A word exceeds the {max_phones}-phone GOPT window and cannot be split safely.",
                word_index=word_index, phone_count=word_phone_count, max_phones=max_phones,
            )
        if phone_end > phone_start and phone_end - phone_start + word_phone_count > max_phones:
            windows.append({
                "phone_start_index": phone_start, "phone_end_index": phone_end,
                "word_start_index": word_start, "word_end_index": word_end,
            })
            phone_start, word_start = word_phone_start, word_index
        phone_end = word_phone_end
        word_end = word_index + 1
    windows.append({
        "phone_start_index": phone_start, "phone_end_index": phone_end,
        "word_start_index": word_start, "word_end_index": word_end,
    })
    if len(windows) > MAX_GOPT_WINDOWS:
        raise ScoreError(
            "UNSUPPORTED_SEQUENCE_LENGTH", "gopt_windowing",
            "Reference exceeds the bounded GOPT window count.",
            window_count=len(windows), max_windows=MAX_GOPT_WINDOWS,
        )
    return windows


def load_training_phone_order(data_zip: Path, pure_id_to_symbol: dict[int, str]) -> dict[int, int]:
    member = "raw_kaldi_gop/librispeech/tr_labels_phn.csv"
    if not data_zip.is_file():
        raise ScoreError("MODEL_INCOMPATIBLE", "phone_mapping", "Official GOPT training phone labels are unavailable; refusing to guess the learned phone order.", path=str(data_zip))
    first_seen: list[int] = []
    seen: set[int] = set()
    try:
        with zipfile.ZipFile(data_zip) as archive:
            with archive.open(member) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))
                for row in reader:
                    if not row:
                        continue
                    pure_id = int(float(row[0]))
                    if pure_id not in seen:
                        seen.add(pure_id)
                        first_seen.append(pure_id)
    except (OSError, KeyError, zipfile.BadZipFile, ValueError) as exc:
        raise ScoreError("MODEL_INCOMPATIBLE", "phone_mapping", "Cannot reconstruct the GOPT training phone order.", error=str(exc)) from exc

    mapped_symbols = [pure_id_to_symbol.get(pure_id) for pure_id in first_seen]
    if mapped_symbols != list(GOPT_PHONE_ORDER):
        raise ScoreError(
            "MODEL_INCOMPATIBLE", "phone_mapping", "Local LibriSpeech phone symbols do not match the GOPT checkpoint's training phone order.",
            observed=mapped_symbols, expected=list(GOPT_PHONE_ORDER),
        )
    if len(first_seen) != 39 or set(first_seen) != set(range(3, 42)):
        raise ScoreError(
            "MODEL_INCOMPATIBLE", "phone_mapping", "Unexpected GOPT LibriSpeech phone inventory.",
            pure_phone_ids=first_seen,
        )
    return {pure_id: class_id for class_id, pure_id in enumerate(first_seen)}


def validate_audio(audio: Path, work_dir: Path, deadline: float | None = None) -> tuple[Path, float, dict[str, Any]]:
    if not audio.is_file():
        raise ScoreError("INVALID_INPUT", "audio", "Audio file does not exist or is not a regular file.", path=str(audio))
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise ScoreError("MISSING_DEPENDENCY", "audio", "ffmpeg and ffprobe must be available on PATH.")
    probe = run_scoring_command("audio_probe", [
        ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
        timeout=30, deadline=deadline,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError as exc:
        raise ScoreError("AUDIO_DECODE_FAILED", "audio", "ffprobe could not read a finite audio duration.", stderr=probe.stderr[-1000:]) from exc
    if probe.returncode != 0 or not math.isfinite(duration) or duration <= 0:
        raise ScoreError("AUDIO_DECODE_FAILED", "audio", "Audio is empty, malformed, or has no decodable audio stream.", stderr=probe.stderr[-1000:])
    if duration > MAX_AUDIO_SECONDS:
        raise ScoreError("UNSUPPORTED_AUDIO_DURATION", "audio", "Audio exceeds the 120-second local processing limit.", duration_seconds=duration, max_seconds=MAX_AUDIO_SECONDS)

    normalized = work_dir / "audio_16khz_mono.wav"
    decoded = run_scoring_command(
        "audio_decode",
        [ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-y", "-i", str(audio), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(normalized)],
        timeout=max(30, int(duration * 4)), deadline=deadline,
    )
    if decoded.returncode != 0 or not normalized.is_file():
        raise ScoreError("AUDIO_DECODE_FAILED", "audio", "ffmpeg could not decode and normalize the input audio.", stderr=decoded.stderr[-1500:])
    try:
        with wave.open(str(normalized), "rb") as wav:
            channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
            samples = array.array("h")
            samples.frombytes(wav.readframes(frames))
    except (OSError, wave.Error) as exc:
        raise ScoreError("AUDIO_DECODE_FAILED", "audio", "Normalized audio is not a valid PCM WAV file.", error=str(exc)) from exc
    if channels != 1 or width != 2 or rate != 16000 or frames <= 0 or not samples:
        raise ScoreError("AUDIO_DECODE_FAILED", "audio", "Audio preprocessing did not produce mono 16 kHz PCM16.")
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    peak = max(abs(sample) for sample in samples)
    if rms < 50 or peak < 100:
        raise ScoreError("INVALID_AUDIO", "audio", "Recording is silent or below the minimum signal level.", rms_pcm16=round(rms, 3), peak_pcm16=peak)
    return normalized, frames / rate, {
        "sample_rate_hz": rate,
        "channels": channels,
        "sample_format": "signed 16-bit PCM",
        "rms_pcm16": round(rms, 3),
        "peak_pcm16": int(peak),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ScoreError("MODEL_INCOMPATIBLE", "model_metadata", f"Cannot hash required model file: {path}", error=str(exc)) from exc
    return digest.hexdigest()


def require_assets() -> None:
    required = required_asset_paths()
    missing = missing_assets(required, [KALDI_ROOT / "src" / name for name in ("bin", "featbin", "nnet3bin", "ivectorbin")])
    if missing:
        raise ScoreError("MISSING_MODEL", "model_loading", "Required Kaldi/GOPT assets are missing.", missing=missing)


def required_asset_paths() -> list[Path]:
    return [
        ACOUSTIC_MODEL / "final.mdl", ACOUSTIC_MODEL / "tree", ACOUSTIC_MODEL / "phones.txt",
        ACOUSTIC_MODEL / "frame_subsampling_factor", IVECTOR_EXTRACTOR / "final.ie",
        LANG / "phones.txt", LANG / "words.txt", LANG / "phones" / "disambig.int",
        LANG / "phones" / "align_lexicon.txt", GOP_RECIPE / "conf" / "mfcc_hires.conf",
        GOP_RECIPE / "local" / "remove_phone_markers.pl", CHECKPOINT, GOPT_DATA,
    ]


def missing_assets(required: list[Path], kaldi_bin_dirs: list[Path]) -> list[str]:
    missing = [str(path) for path in required if not path.is_file()]
    missing.extend(str(path) for path in kaldi_bin_dirs if not path.is_dir())
    return missing


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_work_area(work_dir: Path, threads: int = 4) -> dict[str, Any]:
    for name in ("steps", "utils", "conf", "local"):
        (work_dir / name).symlink_to(GOP_RECIPE / name, target_is_directory=True)
    (work_dir / "cmd.sh").symlink_to(GOP_RECIPE / "cmd.sh")
    (work_dir / "parse_options.sh").symlink_to(GOP_RECIPE / "utils" / "parse_options.sh")
    custom_path = work_dir / "path.sh"
    _write(
        custom_path,
        "export KALDI_ROOT=" + str(KALDI_ROOT) + "\n"
        + ". \"$KALDI_ROOT/tools/config/common_path.sh\"\n"
        + "export PATH=\"$PWD/utils:$PWD:$PATH\"\n"
        + "export LC_ALL=C\n",
    )
    env = os.environ.copy()
    env["KALDI_ROOT"] = str(KALDI_ROOT)
    kaldi_bins = [str(path) for path in sorted((KALDI_ROOT / "src").glob("*bin")) if path.is_dir()]
    env["PATH"] = os.pathsep.join(
        [str(GOP_RECIPE / "utils"), str(GOP_RECIPE), *kaldi_bins, str(KALDI_ROOT / "tools" / "openfst" / "bin"), env.get("PATH", "")]
    )
    env["LC_ALL"] = "C"
    env["OMP_NUM_THREADS"] = str(threads)
    return {"env_path": work_dir / "path.sh", "env_marker": work_dir / "cmd.sh", "_env": env}


def run_scoring_command(stage: str, command: list[str], *, timeout: int, cwd: Path | None = None, env: dict[str, str] | None = None, deadline: float | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return run_bounded(command, cwd=cwd, env=env, capture_output=True, text=True,
                           timeout=timeout, deadline=deadline if deadline is not None else deadline_from_env(env))
    except subprocess.TimeoutExpired as exc:
        raise ScoreError("STAGE_TIMEOUT", stage, f"{stage} exceeded the remaining execution deadline.", timeout_seconds=exc.timeout) from exc


def run_stage(stage: str, command: list[str], cwd: Path, env: dict[str, str], logs: Path, timeout: int = 180) -> float:
    start = time.perf_counter()
    try:
        proc = run_scoring_command(stage, command, cwd=cwd, env=env, timeout=timeout)
    except ScoreError as exc:
        if exc.status == "STAGE_TIMEOUT":
            exc.message = f"{stage} exceeded its {timeout}-second stage limit or the overall execution deadline."
        raise
    elapsed = time.perf_counter() - start
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{stage}.log"
    log_path.write_text("$ " + " ".join(command) + "\n\n" + proc.stdout + "\n" + proc.stderr, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stdout + "\n" + proc.stderr)[-4000:]
        raise ScoreError("PIPELINE_STAGE_FAILED", stage, f"Kaldi stage {stage} failed with exit code {proc.returncode}.", log=str(log_path), output_tail=tail)
    return elapsed


def parse_text_vector_archive(path: Path, expected_key: str) -> list[int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ScoreError("ALIGNMENT_FAILED", "alignment_validation", "Cannot read the per-frame phone alignment.", error=str(exc)) from exc
    found = False
    values: list[int] = []
    collecting = False
    for line in lines:
        fields = line.strip().split()
        if not found:
            if not fields or fields[0] != expected_key:
                continue
            found = True
            tail = fields[1:]
            collecting = "[" in tail and "]" not in tail
            values.extend(int(token) for token in tail if re.fullmatch(r"-?\d+", token))
            if not collecting:
                break
            continue
        if collecting:
            collecting = "]" not in fields
            values.extend(int(token) for token in fields if re.fullmatch(r"-?\d+", token))
            if not collecting:
                break
    if not found or collecting or not values:
        raise ScoreError(
            "ALIGNMENT_FAILED", "alignment_validation", "Per-frame phone alignment is missing or empty.",
            key=expected_key, path=str(path), excerpt=lines[:8],
        )
    return values


def read_ctm(path: Path, expected_key: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5 or fields[0] != expected_key:
                continue
            start, duration, phone_id = float(fields[2]), float(fields[3]), int(fields[4])
            if not all(math.isfinite(v) for v in (start, duration)) or start < 0 or duration <= 0:
                raise ValueError(f"invalid CTM segment: {line}")
            segments.append({"start_seconds": start, "end_seconds": start + duration, "phone_id": phone_id})
    except (OSError, ValueError) as exc:
        raise ScoreError("ALIGNMENT_FAILED", "alignment_validation", "Cannot read or validate phone boundaries.", error=str(exc)) from exc
    if not segments:
        raise ScoreError("ALIGNMENT_FAILED", "alignment_validation", "Forced alignment returned no phone boundaries.", key=expected_key)
    return segments


def verify_gopt_sequence_inputs(raw_features: Any, phone_ids: list[int]) -> None:
    if getattr(raw_features, "ndim", None) != 2 or tuple(raw_features.shape) != (len(phone_ids), FEATURE_DIM):
        raise ScoreError(
            "MODEL_INCOMPATIBLE", "gop_validation", "GOP matrix shape does not match the LibriSpeech GOPT contract.",
            observed_shape=list(getattr(raw_features, "shape", ())), expected_shape=[len(phone_ids), FEATURE_DIM],
        )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in phone_ids):
        raise ScoreError("MODEL_INCOMPATIBLE", "gop_validation", "GOPT phone IDs must be integer class IDs.")
    if len(phone_ids) == 0 or any(not 0 <= value < 39 for value in phone_ids):
        raise ScoreError("MODEL_INCOMPATIBLE", "gop_validation", "Phone IDs fall outside the GOPT 39-class/50-token contract.", phone_count=len(phone_ids), phone_ids=phone_ids)
    if not all(math.isfinite(float(value)) for value in raw_features.flat):
        raise ScoreError("MODEL_INCOMPATIBLE", "gop_validation", "GOP matrix contains NaN or infinity.")


def verify_gopt_inputs(raw_features: Any, phone_ids: list[int]) -> None:
    verify_gopt_sequence_inputs(raw_features, phone_ids)
    if len(phone_ids) > MAX_PHONES:
        raise ScoreError(
            "MODEL_INCOMPATIBLE", "gop_validation", "A GOPT inference window exceeds the 50-phone model contract.",
            phone_count=len(phone_ids), max_phones=MAX_PHONES,
        )


def stitch_gopt_phone_scores(
    window_scores: list[list[Any]], windows: list[dict[str, int]], phone_count: int,
    allow_partial: bool = False,
) -> list[Any | None]:
    """Restore global phone order, rejecting overlap and optionally marking gaps."""
    if phone_count < 1 or len(window_scores) != len(windows):
        raise ScoreError("GOPT_INFERENCE_FAILED", "gopt_mapping", "Window outputs do not match the global phone sequence.")
    missing = object()
    global_scores: list[Any] = [missing] * phone_count
    for window_index, (scores, window) in enumerate(zip(window_scores, windows)):
        start, end = window["phone_start_index"], window["phone_end_index"]
        if not 0 <= start < end <= phone_count or len(scores) != end - start:
            raise ScoreError(
                "GOPT_INFERENCE_FAILED", "gopt_mapping", "A GOPT window has invalid global bounds or output length.",
                window_index=window_index, phone_start_index=start, phone_end_index=end,
                output_phone_count=len(scores),
            )
        for offset, value in enumerate(scores):
            global_index = start + offset
            if global_scores[global_index] is not missing:
                raise ScoreError(
                    "GOPT_INFERENCE_FAILED", "gopt_mapping", "A phone is assigned by more than one GOPT window.",
                    phone_index=global_index,
                )
            global_scores[global_index] = value
    uncovered = [index for index, value in enumerate(global_scores) if value is missing]
    if uncovered and not allow_partial:
        raise ScoreError(
            "GOPT_INFERENCE_FAILED", "gopt_mapping", "One or more phones are not covered by a GOPT window.",
            uncovered_phone_indices=uncovered,
        )
    return [None if value is missing else value for value in global_scores]


def partition_gopt_windows_by_support(
    windows: list[dict[str, int]], phone_log_posteriors: list[float],
    expected: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
    """Split at whole-word boundaries so weak support cannot erase neighbors."""
    if not phone_log_posteriors or not all(math.isfinite(value) for value in phone_log_posteriors):
        raise ScoreError("ALIGNMENT_INVALID", "alignment_support", "Phone acoustic support must be a non-empty finite sequence.")
    # The planned windows must account for every aligned phone exactly once,
    # even when one of them will later be withheld from GOPT inference.
    stitch_gopt_phone_scores(
        [[0.0] * (window["phone_end_index"] - window["phone_start_index"]) for window in windows],
        windows, len(phone_log_posteriors),
    )
    if expected is not None and len(expected) != len(phone_log_posteriors):
        raise ScoreError(
            "ALIGNMENT_INVALID", "alignment_support",
            "Canonical phone rows and support measurements have different lengths.",
            expected_phone_count=len(expected), support_phone_count=len(phone_log_posteriors),
        )
    active: list[dict[str, int]] = []
    diagnostics: list[dict[str, Any]] = []
    next_window_index = 0
    for parent_window_index, window in enumerate(windows):
        start, end = window["phone_start_index"], window["phone_end_index"]
        if expected is None:
            word_ranges = [(window["word_start_index"], window["word_end_index"], start, end)]
        else:
            word_ranges: list[tuple[int, int, int, int]] = []
            word_start = start
            while word_start < end:
                word_index = expected[word_start]["word_index"]
                word_end = word_start + 1
                while word_end < end and expected[word_end]["word_index"] == word_index:
                    word_end += 1
                word_ranges.append((word_index, word_index + 1, word_start, word_end))
                word_start = word_end

        active_run: tuple[int, int, int, int] | None = None

        def record_range(word_start: int, word_end: int, phone_start: int, phone_end: int) -> None:
            nonlocal next_window_index
            unsupported = [
                phone_index for phone_index in range(phone_start, phone_end)
                if phone_log_posteriors[phone_index] < MIN_PHONE_LOG_POSTERIOR
            ]
            status = "WITHHELD_ALIGNMENT_SUPPORT" if unsupported else "READY_FOR_GOPT"
            diagnostic = {
                "window_index": next_window_index,
                "parent_window_index": parent_window_index,
                "phone_start_index": phone_start,
                "phone_end_index": phone_end,
                "word_start_index": word_start,
                "word_end_index": word_end,
                "phone_count": phone_end - phone_start,
                "status": status,
                "unsupported_phone_indices": unsupported,
            }
            diagnostics.append(diagnostic)
            if not unsupported:
                active.append({key: value for key, value in diagnostic.items() if key not in ("status", "unsupported_phone_indices")})
            next_window_index += 1

        for word_start, word_end, phone_start, phone_end in word_ranges:
            word_is_supported = all(
                phone_log_posteriors[index] >= MIN_PHONE_LOG_POSTERIOR
                for index in range(phone_start, phone_end)
            )
            if word_is_supported:
                if active_run is None:
                    active_run = (word_start, word_end, phone_start, phone_end)
                else:
                    active_run = (active_run[0], word_end, active_run[2], phone_end)
                continue
            if active_run is not None:
                record_range(*active_run)
                active_run = None
            record_range(word_start, word_end, phone_start, phone_end)
        if active_run is not None:
            record_range(*active_run)
    return active, diagnostics


def prepare_gopt_batch(raw_features: Any, phone_ids: list[int]) -> tuple[Any, Any]:
    import numpy as np

    verify_gopt_inputs(raw_features, phone_ids)
    padded = np.zeros((1, MAX_PHONES, FEATURE_DIM), dtype=np.float32)
    padded[0, :len(phone_ids), :] = (raw_features.astype(np.float32) - GOPT_MEAN) / GOPT_STD
    padded_phone_ids = np.full((1, MAX_PHONES), -1, dtype=np.int64)
    padded_phone_ids[0, :len(phone_ids)] = phone_ids
    if not np.isfinite(padded).all():
        raise ScoreError("MODEL_INCOMPATIBLE", "gopt_normalization", "Normalized GOP features contain NaN or infinity.")
    # Padding remains zero after normalization, matching the released GoPDataset.
    if np.any(padded[0, len(phone_ids):, :] != 0) or np.any(padded_phone_ids[0, len(phone_ids):] != -1):
        raise ScoreError("MODEL_INCOMPATIBLE", "gopt_normalization", "GOPT padding does not match the released training convention.")
    return padded, padded_phone_ids


def validate_gopt_outputs(outputs: Any, batch_size: int = 1) -> list[Any]:
    import numpy as np

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 9:
        raise ScoreError("GOPT_INFERENCE_FAILED", "gopt_output_validation", "GOPT returned an unexpected output tuple.")
    if batch_size < 1:
        raise ScoreError("GOPT_INFERENCE_FAILED", "gopt_output_validation", "GOPT output batch size must be positive.")
    expected_shapes = [(batch_size, 1)] * 5 + [(batch_size, MAX_PHONES, 1)] * 4
    arrays = []
    for index, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
        try:
            array_value = output.detach().cpu().numpy()
        except AttributeError:
            array_value = np.asarray(output)
        if tuple(array_value.shape) != expected_shape or not np.isfinite(array_value).all():
            raise ScoreError(
                "GOPT_INFERENCE_FAILED", "gopt_output_validation", "GOPT output has an invalid shape or non-finite value.",
                output_index=index, observed_shape=list(array_value.shape), expected_shape=list(expected_shape),
            )
        arrays.append(array_value)
    return arrays


def validate_alignment_frame_count(alignment_frames: int, posterior_frames: int) -> None:
    if alignment_frames != posterior_frames:
        raise ScoreError(
            "ALIGNMENT_INVALID", "alignment_validation", "Forced-alignment and acoustic-posterior frame counts differ; GOP computation is blocked.",
            alignment_frames=alignment_frames, posterior_frames=posterior_frames,
        )


def phone_acoustic_support_values(
    raw_features: Any, pure_phone_ids: list[int], expected: list[dict[str, Any]],
) -> list[float]:
    """Return canonical-phone log posteriors from the output-xent feature branch."""
    if len(raw_features) != len(pure_phone_ids) or len(expected) != len(pure_phone_ids):
        raise ScoreError("ALIGNMENT_INVALID", "alignment_support", "Phone features, IDs and reference phones have different lengths.")
    lpp = raw_features[:, 1:1 + PURE_PHONE_COUNT]
    support: list[float] = []
    for index, pure_id in enumerate(pure_phone_ids):
        if not 3 <= pure_id < PURE_PHONE_COUNT:
            raise ScoreError("ALIGNMENT_INVALID", "alignment_support", "Aligned phone ID is outside the supported speech-phone inventory.", pure_phone_id=pure_id)
        value = float(lpp[index, pure_id])
        if not math.isfinite(value):
            raise ScoreError("ALIGNMENT_INVALID", "alignment_support", "Canonical phone posterior is not finite.", index=index)
        support.append(value)
    return support


def validate_phone_acoustic_support(raw_features: Any, pure_phone_ids: list[int], expected: list[dict[str, Any]]) -> list[float]:
    """Reject the utterance if any forced phone falls below the alignment floor.

    The floor is an alignment sanity gate, not a pronunciation-quality cutoff.
    """
    support = phone_acoustic_support_values(raw_features, pure_phone_ids, expected)
    for index, value in enumerate(support):
        if value < MIN_PHONE_LOG_POSTERIOR:
            raise ScoreError(
                "ALIGNMENT_INVALID", "alignment_support", "Forced alignment assigned a reference phone with negligible acoustic posterior support.",
                index=index, word=expected[index]["word"], phone=expected[index]["phone"],
                mean_log_posterior=value, minimum_log_posterior=MIN_PHONE_LOG_POSTERIOR,
                minimum_geometric_posterior=round(math.exp(MIN_PHONE_LOG_POSTERIOR), 8),
            )
    return support


def validate_gop_phone_order(feature_rows: Any, pure_phone_ids: list[int], expected: list[dict[str, Any]]) -> None:
    """Require identical phone identities/order across acoustic output branches."""
    if getattr(feature_rows, "ndim", None) != 2 or feature_rows.shape != (len(pure_phone_ids), 85) or len(expected) != len(pure_phone_ids):
        raise ScoreError("ALIGNMENT_INVALID", "alignment_support", "Feature branches and aligned reference phones have different lengths.")
    if not all(math.isfinite(float(value)) for value in feature_rows.flat):
        raise ScoreError("ALIGNMENT_INVALID", "alignment_support", "Output-xent support features contain NaN or infinity.")
    for index, (row, pure_id, phone) in enumerate(zip(feature_rows, pure_phone_ids, expected)):
        row_id = int(round(float(row[0])))
        if abs(float(row[0]) - row_id) > 1e-5 or row_id != pure_id or pure_id != phone["pure_phone_id"]:
            raise ScoreError(
                "ALIGNMENT_INVALID", "alignment_support", "GOP feature branches do not share the canonical aligned phone order.",
                index=index, feature_phone_id=row_id, expected_phone_id=phone["pure_phone_id"],
            )


def load_and_run_gopt(
    raw_features: Any, phone_ids: list[int], threads: int,
    windows: list[dict[str, int]] | None = None,
    allow_partial: bool = False,
    validate_reproducibility: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise ScoreError("MISSING_DEPENDENCY", "gopt_inference", "Run this CLI with .local/gopt-venv/bin/python (PyTorch and NumPy are required).", error=str(exc)) from exc
    sys.path.insert(0, str(GOPT_ROOT / "src"))
    try:
        from models.gopt import GOPT
    except ImportError as exc:
        raise ScoreError("MISSING_DEPENDENCY", "gopt_inference", "Could not import the official local GOPT model implementation.", error=str(exc)) from exc

    start = time.perf_counter()
    torch.set_num_threads(threads)
    try:
        model = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=FEATURE_DIM)
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
    except Exception as exc:
        raise ScoreError("MODEL_INCOMPATIBLE", "gopt_model_loading", "GOPT checkpoint failed strict loading against the official LibriSpeech model.", error=str(exc)) from exc
    load_seconds = time.perf_counter() - start

    verify_gopt_sequence_inputs(raw_features, phone_ids)
    if windows is None:
        if len(phone_ids) > MAX_PHONES:
            raise ScoreError(
                "UNSUPPORTED_SEQUENCE_LENGTH", "gopt_windowing",
                "Long GOPT inference requires explicit whole-word window boundaries.",
                phone_count=len(phone_ids), max_phones=MAX_PHONES,
            )
        windows = [{
            "phone_start_index": 0, "phone_end_index": len(phone_ids),
            "word_start_index": 0, "word_end_index": 0,
        }]
    # Validate exact-once global ownership before paying for model inference.
    stitch_gopt_phone_scores(
        [[0.0] * (window["phone_end_index"] - window["phone_start_index"]) for window in windows],
        windows, len(phone_ids), allow_partial=allow_partial,
    )
    if not windows:
        raise ScoreError("GOPT_INFERENCE_FAILED", "gopt_mapping", "No supported GOPT windows remain for inference.")
    batches = []
    phone_batches = []
    for batch_index, window in enumerate(windows):
        window_index = window.get("window_index", batch_index)
        start_index, end_index = window["phone_start_index"], window["phone_end_index"]
        if end_index - start_index > MAX_PHONES:
            raise ScoreError(
                "MODEL_INCOMPATIBLE", "gopt_windowing", "A GOPT inference window exceeds the 50-phone model contract.",
                window_index=window_index, phone_start_index=start_index, phone_end_index=end_index,
            )
        padded, padded_phone_ids = prepare_gopt_batch(
            raw_features[start_index:end_index], phone_ids[start_index:end_index],
        )
        batches.append(padded)
        phone_batches.append(padded_phone_ids)
    inputs = torch.from_numpy(np.concatenate(batches, axis=0))
    phone_tensor = torch.from_numpy(np.concatenate(phone_batches, axis=0))
    try:
        with torch.inference_mode():
            first_start = time.perf_counter()
            first = model(inputs, phone_tensor)
            first_seconds = time.perf_counter() - first_start
            warm_seconds = None
            max_delta = None
            if validate_reproducibility:
                warm_start = time.perf_counter()
                second = model(inputs, phone_tensor)
                warm_seconds = time.perf_counter() - warm_start
    except Exception as exc:
        raise ScoreError("GOPT_INFERENCE_FAILED", "gopt_inference", "GOPT forward inference failed.", error=str(exc)) from exc

    arrays = validate_gopt_outputs(first, batch_size=len(windows))
    if validate_reproducibility:
        validate_gopt_outputs(second, batch_size=len(windows))
        max_delta = 0.0
        for left, right in zip(first, second):
            delta = float(torch.max(torch.abs(left - right)).item())
            if not math.isfinite(delta):
                raise ScoreError("GOPT_INFERENCE_FAILED", "reproducibility", "Repeated CPU inference returned a non-finite prediction delta.")
            max_delta = max(max_delta, delta)
        if not math.isfinite(max_delta) or max_delta > 1e-6:
            raise ScoreError("GOPT_INFERENCE_FAILED", "reproducibility", "Repeated CPU inference exceeded the 1e-6 determinism tolerance.", max_absolute_delta=max_delta)

    native_scale = "GOPT normalized training target units (nominal 0-2; regression outputs are not clamped)"
    window_predictions: list[dict[str, Any]] = []
    window_phone_scores: list[list[float]] = []
    window_word_scores: list[list[list[float]]] = []
    for batch_index, window in enumerate(windows):
        window_index = window.get("window_index", batch_index)
        phone_count = window["phone_end_index"] - window["phone_start_index"]
        window_utterance = {
            "utterance_accuracy": float(arrays[0][batch_index, 0]),
            "completeness": float(arrays[1][batch_index, 0]),
            "fluency": float(arrays[2][batch_index, 0]),
            "prosody": float(arrays[3][batch_index, 0]),
            "overall": float(arrays[4][batch_index, 0]),
            "native_scale": native_scale,
        }
        window_phone_scores.append(arrays[5][batch_index, :, 0][:phone_count].astype(float).tolist())
        window_word_scores.append(
            np.concatenate(
                [arrays[6][batch_index], arrays[7][batch_index], arrays[8][batch_index]], axis=1,
            )[:phone_count].astype(float).tolist()
        )
        window_predictions.append({
            "window_index": window_index,
            **{key: value for key, value in window.items() if key != "window_index"},
            "phone_count": phone_count,
            "utterance": window_utterance,
        })
    phone_scores = stitch_gopt_phone_scores(
        window_phone_scores, windows, len(phone_ids), allow_partial=allow_partial,
    )
    word_scores = stitch_gopt_phone_scores(
        window_word_scores, windows, len(phone_ids), allow_partial=allow_partial,
    )
    if len(windows) == 1 and not allow_partial:
        scores = {**window_predictions[0]["utterance"], "aggregation_status": "SINGLE_WINDOW_DIRECT"}
    else:
        scores = {
            "utterance_accuracy": None,
            "completeness": None,
            "fluency": None,
            "prosody": None,
            "overall": None,
            "native_scale": native_scale,
            "aggregation_status": "NOT_VALIDATED",
        }
    return {
        "utterance": scores,
        "phoneme_scores": phone_scores,
        "word_phone_scores": word_scores,
        "window_predictions": window_predictions,
    }, {
        "model_initialization_seconds": load_seconds,
        "first_forward_seconds": first_seconds,
        "warm_forward_seconds": warm_seconds,
        "max_absolute_repeat_delta": max_delta,
        "window_count": len(windows),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def kaldi_paths() -> dict[str, Path]:
    return {
        "mfcc_config": GOP_RECIPE / "conf" / "mfcc_hires.conf",
        "remove_phone_markers": GOP_RECIPE / "local" / "remove_phone_markers.pl",
        "model": ACOUSTIC_MODEL,
        "ivector": IVECTOR_EXTRACTOR,
        "lang": LANG,
    }


def _write_kaldi_text(work_dir: Path, audio: Path, words: list[str], expected: list[dict[str, Any]]) -> str:
    key = "utt0001"
    data = work_dir / "data" / "test"
    local_data = work_dir / "data" / "local"
    _write(data / "wav.scp", f"{key} {audio}\n")
    _write(data / "text", f"{key} {' '.join(words)}\n")
    _write(data / "utt2spk", f"{key} speaker0001\n")
    _write(data / "spk2utt", f"speaker0001 {key}\n")
    lines = []
    for word_index, word in enumerate(words):
        phones = [phone["marked_phone"] for phone in expected if phone["word_index"] == word_index]
        lines.append(f"{key}.{word_index} {' '.join(phones)}")
    _write(local_data / "text-phone", "\n".join(lines) + "\n")
    return key


def _save_feature_evidence(
    artifact_dir: Path,
    work_dir: Path,
    logs: Path,
    normalized_audio: Path,
    key: str,
    expected: list[dict[str, Any]],
    extracted_pure_ids: list[int],
    raw_features: Any,
    support_features: Any,
    result: dict[str, Any],
) -> None:
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise ScoreError("INVALID_INPUT", "evidence_output", "Evidence directory must be new or empty.", path=str(artifact_dir))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "kaldi").mkdir()
    shutil.copytree(work_dir / "data", artifact_dir / "kaldi" / "data", dirs_exist_ok=True)
    shutil.copytree(work_dir / "exp", artifact_dir / "kaldi" / "exp", dirs_exist_ok=True)
    shutil.copytree(logs, artifact_dir / "kaldi" / "logs", dirs_exist_ok=True)
    shutil.copy2(normalized_audio, artifact_dir / "audio_16khz_mono.wav")
    _write(artifact_dir / "kaldi" / "data" / "test" / "wav.scp", f"{key} {artifact_dir / 'audio_16khz_mono.wav'}\n")
    import numpy as np
    np.savez_compressed(
        artifact_dir / "gop_features.npz",
        raw_features=raw_features[:, 1:].astype(np.float32),
        output_xent_support_features=support_features[:, 1:].astype(np.float32),
        gopt_phone_ids=np.asarray([phone["gopt_phone_id"] for phone in expected], dtype=np.int64),
        pure_phone_ids=np.asarray(extracted_pure_ids, dtype=np.int64),
        normalized_features=((raw_features[:, 1:] - GOPT_MEAN) / GOPT_STD).astype(np.float32),
    )
    _write(artifact_dir / "result.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def score(
    audio: Path,
    reference: str,
    threads: int = 4,
    artifact_dir: Path | None = None,
    source_phone_words: list[list[str]] | None = None,
    feature_reconstruction_only: bool = False,
    recover_supported_words: bool = False,
    validate_reproducibility: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = deadline_from_env() or (time.monotonic() + 360.0)
    require_assets()
    if feature_reconstruction_only and source_phone_words is None:
        raise ScoreError("INVALID_INPUT", "source_phone_transcript", "Feature-reconstruction-only mode requires a source-native canonical phone sequence.")
    paths = kaldi_paths()
    model_phone_symbols, model_phone_ids = read_symbol_table(ACOUSTIC_MODEL / "phones.txt")
    lang_phone_symbols, lang_phone_ids = read_symbol_table(LANG / "phones.txt")
    if model_phone_symbols != lang_phone_symbols or model_phone_ids != lang_phone_ids:
        raise ScoreError("MODEL_INCOMPATIBLE", "phone_inventory", "LibriSpeech model and language phone tables differ.")
    _, pure_id_to_symbol = read_symbol_table(LANG / "phones.txt")

    # This is the same phone-marker removal and ID assignment used by the
    # Kaldi GOP recipe. Its output is generated in the isolated temp workdir.
    words = normalize_reference(reference)
    lexicon = read_alignment_lexicon(LANG / "phones" / "align_lexicon.txt") if source_phone_words is None else {}
    lexicon_prons = {word: values for word, values in lexicon.items()}

    # Build the pure-phone ID table once with the official recipe utility.
    temp_for_phone_map = Path(tempfile.mkdtemp(prefix="pte-phone-map-"))
    try:
        pure_table = temp_for_phone_map / "phones-pure.txt"
        phone_map_path = temp_for_phone_map / "phone-to-pure-phone.int"
        map_proc = run_scoring_command(
            "phone_mapping",
            ["perl", str(paths["remove_phone_markers"]), str(LANG / "phones.txt"), str(pure_table), str(phone_map_path)],
            timeout=30, deadline=deadline,
        )
        if map_proc.returncode != 0:
            raise ScoreError("MODEL_INCOMPATIBLE", "phone_mapping", "Kaldi phone-marker mapping utility failed.", stderr=map_proc.stderr[-1000:])
        pure_symbol_to_id, pure_id_to_symbol = read_symbol_table(pure_table)
        gopt_pure_to_class = load_training_phone_order(GOPT_DATA, pure_id_to_symbol)
        if source_phone_words is None:
            expected_raw, variant_warnings = expected_phones(words, lexicon_prons, pure_symbol_to_id)
        else:
            expected_raw = expected_phones_from_source(words, source_phone_words, pure_symbol_to_id)
            variant_warnings = []
        for phone in expected_raw:
            pure_id = phone["pure_phone_id"]
            if pure_id not in gopt_pure_to_class:
                raise ScoreError("UNSUPPORTED_PHONEME", "phone_mapping", "Reference phone is outside the pretrained GOPT inventory.", phone=phone["phone"], pure_phone_id=pure_id)
            phone["gopt_phone_id"] = gopt_pure_to_class[pure_id]
    finally:
        shutil.rmtree(temp_for_phone_map, ignore_errors=True)

    # Retain marked phone spellings for the exact forced alignment input.
    # Build the dictionary again from its selected first variant, preserving
    # the same deterministic rule as expected_phones().
    expected: list[dict[str, Any]] = []
    for word_index, word in enumerate(words):
        marked_sequence = lexicon_prons[word][0] if source_phone_words is None else source_phone_words[word_index]
        for marked_phone in marked_sequence:
            base = strip_phone_markers(marked_phone)
            pure_id = pure_symbol_to_id[base]
            expected.append({
                "word": word, "word_index": word_index, "phone": base,
                "pure_phone_id": pure_id, "gopt_phone_id": gopt_pure_to_class[pure_id],
                "marked_phone": marked_phone,
            })
    windows = plan_gopt_windows(expected)
    gopt_phone_ids = [phone["gopt_phone_id"] for phone in expected]

    with tempfile.TemporaryDirectory(prefix="pte-gopt-kaldi-") as temp_name:
        work_dir = Path(temp_name)
        work = build_work_area(work_dir, threads)
        env: dict[str, str] = work["_env"]
        env[DEADLINE_ENV] = str(deadline)
        logs = work_dir / "logs"
        preprocessing_start = time.perf_counter()
        normalized_audio, duration, audio_info = validate_audio(audio, work_dir, deadline)
        audio_preprocessing_seconds = time.perf_counter() - preprocessing_start
        key = _write_kaldi_text(work_dir, normalized_audio, words, expected)

        start = time.perf_counter()
        run_stage(
            "make_mfcc", [str(GOP_RECIPE / "steps" / "make_mfcc.sh"), "--nj", "1", "--mfcc-config", "conf/mfcc_hires.conf", "--cmd", "run.pl", "data/test", "exp/mfcc"],
            work_dir, env, logs,
        )
        run_stage("cmvn", [str(GOP_RECIPE / "steps" / "compute_cmvn_stats.sh"), "data/test"], work_dir, env, logs)
        run_stage("fix_data", [str(GOP_RECIPE / "utils" / "fix_data_dir.sh"), "data/test"], work_dir, env, logs)
        run_stage(
            "ivectors", [str(GOP_RECIPE / "steps" / "online" / "nnet2" / "extract_ivectors_online.sh"), "--cmd", "run.pl", "--nj", "1", "data/test", str(IVECTOR_EXTRACTOR), "data/test/ivectors"],
            work_dir, env, logs,
        )
        feature_seconds = time.perf_counter() - start

        start = time.perf_counter()
        run_stage(
            "posteriors", [str(GOP_RECIPE / "steps" / "nnet3" / "compute_output.sh"), "--nj", "1", "--cmd", "run.pl", "--online-ivector-dir", "data/test/ivectors", "--frame-subsampling-factor", "3", "--output-name", "output-xent", "data/test", str(ACOUSTIC_MODEL), "exp/probs"],
            work_dir, env, logs,
        )
        run_stage(
            "raw_acoustic_output", [str(GOP_RECIPE / "steps" / "nnet3" / "compute_output.sh"), "--nj", "1", "--cmd", "run.pl", "--online-ivector-dir", "data/test/ivectors", "--frame-subsampling-factor", "3", "data/test", str(ACOUSTIC_MODEL), "exp/raw-output"],
            work_dir, env, logs,
        )
        posterior_seconds = time.perf_counter() - start

        try:
            import kaldiio
            import numpy as np
        except ImportError as exc:
            raise ScoreError("MISSING_DEPENDENCY", "gop_extraction", "Install/use the existing .local/gopt-venv runtime containing kaldiio and NumPy.", error=str(exc)) from exc
        posterior_scp = work_dir / "exp" / "probs" / "output.scp"
        with contextlib.chdir(work_dir):
            posterior_matrices = dict(kaldiio.load_scp("exp/probs/output.scp"))
        if set(posterior_matrices) != {key}:
            raise ScoreError("ALIGNMENT_FAILED", "posterior_validation", "Posterior extraction did not return exactly the expected utterance.", keys=sorted(posterior_matrices))
        posteriors = np.asarray(posterior_matrices[key], dtype=np.float64)
        if posteriors.ndim != 2 or posteriors.shape[1] != 6024 or not np.isfinite(posteriors).all():
            raise ScoreError("MODEL_INCOMPATIBLE", "posterior_validation", "Acoustic output is not a finite [frames, 6024] log-posterior matrix.", shape=list(posteriors.shape))
        max_logsum_residual = float(np.max(np.abs(np.logaddexp.reduce(posteriors, axis=1))))
        if max_logsum_residual > 1e-3:
            raise ScoreError("MODEL_INCOMPATIBLE", "posterior_validation", "Selected acoustic output is not normalized log-softmax posteriors.", max_logsumexp_residual=max_logsum_residual)
        with contextlib.chdir(work_dir):
            acoustic_output_matrices = dict(kaldiio.load_scp("exp/raw-output/output.scp"))
        if set(acoustic_output_matrices) != {key}:
            raise ScoreError("GOP_EXTRACTION_FAILED", "acoustic_output_validation", "Raw acoustic extraction did not return exactly the expected utterance.", keys=sorted(acoustic_output_matrices))
        acoustic_output = np.asarray(acoustic_output_matrices[key], dtype=np.float64)
        if acoustic_output.ndim != 2 or acoustic_output.shape[1] != 6024 or not np.isfinite(acoustic_output).all():
            raise ScoreError("MODEL_INCOMPATIBLE", "acoustic_output_validation", "Raw acoustic output is not a finite [frames, 6024] matrix.", shape=list(acoustic_output.shape))
        validate_alignment_frame_count(int(posteriors.shape[0]), int(acoustic_output.shape[0]))
        output_logsumexp = np.logaddexp.reduce(acoustic_output, axis=1)

        text_int = work_dir / "data" / "test" / "text.int"
        text_phone_int = work_dir / "data" / "local" / "text-phone.int"
        # sym2int output is captured to a file through the normal shell because
        # its stdout must become Kaldi's text archive input.
        word_convert = run_scoring_command(
            "word_mapping",
            ["bash", "-c", f"{shlex_quote(str(GOP_RECIPE / 'utils' / 'sym2int.pl'))} -f 2- {shlex_quote(str(LANG / 'words.txt'))} data/test/text > {shlex_quote(str(text_int))}"],
            cwd=work_dir, env=env, timeout=30,
        )
        if word_convert.returncode != 0:
            raise ScoreError("UNSUPPORTED_VOCABULARY", "word_mapping", "Reference words cannot be mapped to the LibriSpeech language table.", stderr=word_convert.stderr[-1000:])
        phone_convert = run_scoring_command(
            "phone_mapping",
            ["bash", "-c", f"{shlex_quote(str(GOP_RECIPE / 'utils' / 'sym2int.pl'))} -f 2- {shlex_quote(str(LANG / 'phones.txt'))} data/local/text-phone > {shlex_quote(str(text_phone_int))}"],
            cwd=work_dir, env=env, timeout=30,
        )
        if phone_convert.returncode != 0:
            raise ScoreError("UNSUPPORTED_PHONEME", "phone_mapping", "Reference phones cannot be mapped to Kaldi phone IDs.", stderr=phone_convert.stderr[-1000:])

        graphs_dir = work_dir / "exp" / "align"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        graph_command = [
            "compile-train-graphs-without-lexicon",
            f"--read-disambig-syms={LANG / 'phones' / 'disambig.int'}",
            str(ACOUSTIC_MODEL / "tree"), str(ACOUSTIC_MODEL / "final.mdl"),
            f"ark,t:{text_int}", f"ark,t:{text_phone_int}",
            f"ark:|gzip -c >{graphs_dir / 'fsts.1.gz'}",
        ]
        run_stage("compile_graph", graph_command, work_dir, env, logs)
        _write(graphs_dir / "num_jobs", "1\n")
        align_start = time.perf_counter()
        run_stage(
            "align", [str(GOP_RECIPE / "steps" / "align_mapped.sh"), "--cmd", "run.pl", "--nj", "1", "--graphs", "exp/align", "data/test", "exp/probs", str(LANG), str(ACOUSTIC_MODEL), "exp/align"],
            work_dir, env, logs,
        )
        run_stage(
            "phone_alignment", ["ali-to-phones", "--per-frame=true", str(ACOUSTIC_MODEL / "final.mdl"), "ark:gunzip -c exp/align/ali.1.gz|", "ark,t:exp/align/ali-phone.1.ark"],
            work_dir, env, logs,
        )
        ctm_path = work_dir / "exp" / "align" / "phones.ctm"
        run_stage(
            "phone_boundaries", ["ali-to-phones", "--ctm-output=true", f"--frame-shift={FRAME_SHIFT_SECONDS}", str(ACOUSTIC_MODEL / "final.mdl"), "ark:gunzip -c exp/align/ali.1.gz|", str(ctm_path)],
            work_dir, env, logs,
        )
        frame_phone_ids = parse_text_vector_archive(work_dir / "exp" / "align" / "ali-phone.1.ark", key)
        validate_alignment_frame_count(len(frame_phone_ids), int(posteriors.shape[0]))
        ctm_segments = read_ctm(ctm_path, key)
        if any(
            ctm_segments[index]["start_seconds"] > ctm_segments[index + 1]["start_seconds"]
            or ctm_segments[index]["end_seconds"] > ctm_segments[index + 1]["start_seconds"] + 0.01
            for index in range(len(ctm_segments) - 1)
        ):
            raise ScoreError("ALIGNMENT_INVALID", "alignment_validation", "Phone boundaries are not ordered without overlap.")
        align_seconds = time.perf_counter() - align_start

        # Recreate the recipe's phone map in this temporary work area.
        pure_table = work_dir / "exp" / "phones-pure.txt"
        phone_map = work_dir / "exp" / "phone-to-pure-phone.int"
        map_proc = run_scoring_command(
            "phone_mapping",
            ["perl", str(paths["remove_phone_markers"]), str(LANG / "phones.txt"), str(pure_table), str(phone_map)],
            timeout=30, deadline=deadline,
        )
        if map_proc.returncode != 0:
            raise ScoreError("MODEL_INCOMPATIBLE", "phone_mapping", "Kaldi pure-phone map generation failed.", stderr=map_proc.stderr[-1000:])
        gop_dir = work_dir / "exp" / "gop-output"
        support_gop_dir = work_dir / "exp" / "gop-support"
        gop_dir.mkdir(parents=True, exist_ok=True)
        support_gop_dir.mkdir(parents=True, exist_ok=True)
        gop_start = time.perf_counter()
        gop_command = [
            "compute-gop", f"--phone-map={phone_map}", "--skip-phones-string=0:1:2",
            str(ACOUSTIC_MODEL / "final.mdl"),
            "ark,t:gunzip -c exp/align/ali.1.gz|", "ark:exp/align/ali-phone.1.ark",
            "ark:exp/raw-output/output.1.ark",
            f"ark,scp:{gop_dir / 'gop.1.ark'},{gop_dir / 'gop.1.scp'}",
            f"ark,scp:{gop_dir / 'feat.1.ark'},{gop_dir / 'feat.1.scp'}",
        ]
        run_stage("compute_gop_raw_output", gop_command, work_dir, env, logs)
        support_gop_command = [
            "compute-gop", f"--phone-map={phone_map}", "--skip-phones-string=0:1:2",
            str(ACOUSTIC_MODEL / "final.mdl"),
            "ark,t:gunzip -c exp/align/ali.1.gz|", "ark:exp/align/ali-phone.1.ark",
            "ark:exp/probs/output.1.ark",
            f"ark,scp:{support_gop_dir / 'gop.1.ark'},{support_gop_dir / 'gop.1.scp'}",
            f"ark,scp:{support_gop_dir / 'feat.1.ark'},{support_gop_dir / 'feat.1.scp'}",
        ]
        run_stage("compute_gop_output_xent_support", support_gop_command, work_dir, env, logs)
        gop_seconds = time.perf_counter() - gop_start
        if not (gop_dir / "feat.1.scp").is_file():
            raise ScoreError("GOP_EXTRACTION_FAILED", "gop_extraction", "Kaldi did not write a GOP feature index.")
        with contextlib.chdir(work_dir):
            feature_rows = dict(kaldiio.load_scp("exp/gop-output/feat.1.scp"))
            support_feature_rows = dict(kaldiio.load_scp("exp/gop-support/feat.1.scp"))
        if not feature_rows:
            raise ScoreError("GOP_EXTRACTION_FAILED", "gop_extraction", "Kaldi produced no non-silence GOP feature rows.")
        if set(feature_rows) != set(support_feature_rows):
            raise ScoreError("ALIGNMENT_INVALID", "alignment_support", "Raw and output-xent GOP branches returned different phone keys.", raw_keys=sorted(feature_rows), support_keys=sorted(support_feature_rows))
        ordered_keys = sorted(feature_rows, key=lambda value: int(value.rsplit(".", 1)[1]))
        raw_features = np.stack([np.asarray(feature_rows[row_key], dtype=np.float64) for row_key in ordered_keys])
        support_features = np.stack([np.asarray(support_feature_rows[row_key], dtype=np.float64) for row_key in ordered_keys])
        extracted_pure_ids = [int(round(float(row[0]))) for row in raw_features]
        if any(abs(float(row[0]) - pure_id) > 1e-5 for row, pure_id in zip(raw_features, extracted_pure_ids)):
            raise ScoreError("GOP_EXTRACTION_FAILED", "gop_validation", "GOP feature rows contain a non-integer phone ID.")
        validate_gop_phone_order(support_features, extracted_pure_ids, expected)
        verify_gopt_sequence_inputs(
            raw_features[:, 1:], [gopt_pure_to_class.get(pure_id, -1) for pure_id in extracted_pure_ids],
        )
        phone_log_posteriors = None if feature_reconstruction_only else phone_acoustic_support_values(
            support_features, extracted_pure_ids, expected,
        )

        phone_map_ids: dict[int, int] = {}
        for line in phone_map.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 2:
                phone_map_ids[int(fields[0])] = int(fields[1])
        visible_segments = []
        for segment in ctm_segments:
            old_id = segment["phone_id"]
            if old_id not in phone_map_ids:
                raise ScoreError("ALIGNMENT_INVALID", "alignment_validation", "Aligned phone ID is absent from the pure-phone map.", phone_id=old_id)
            pure_id = phone_map_ids[old_id]
            if pure_id in (0, 1, 2):
                continue
            visible_segments.append({**segment, "pure_phone_id": pure_id})
        if [segment["pure_phone_id"] for segment in visible_segments] != [phone["pure_phone_id"] for phone in expected]:
            raise ScoreError(
                "ALIGNMENT_INVALID", "alignment_validation", "Aligned lexical phone sequence differs from the explicit reference pronunciation sequence.",
                aligned_pure_phone_ids=[segment["pure_phone_id"] for segment in visible_segments],
                expected_pure_phone_ids=[phone["pure_phone_id"] for phone in expected],
            )
        if len(visible_segments) != len(raw_features):
            raise ScoreError(
                "ALIGNMENT_INVALID", "alignment_validation", "Non-silence phone boundaries do not match GOP feature rows.",
                aligned_phones=len(visible_segments), gop_rows=len(raw_features),
            )
        if any(segment["end_seconds"] > duration + FRAME_SHIFT_SECONDS for segment in visible_segments):
            raise ScoreError("ALIGNMENT_INVALID", "alignment_validation", "A phone boundary extends beyond the decoded audio duration.", audio_duration_seconds=duration)
        if extracted_pure_ids != [phone["pure_phone_id"] for phone in expected]:
            raise ScoreError("GOP_EXTRACTION_FAILED", "gop_validation", "GOP rows are not in the aligned canonical phone order.")

        if feature_reconstruction_only:
            result = {
                "status": "FEATURE_RECONSTRUCTION_COMPLETED",
                "engine": "kaldi-gop-feature-reconstruction",
                "input_audio": str(audio.resolve()),
                "audio_duration_seconds": round(duration, 3),
                "transcript": " ".join(words),
                "pearson_pte_score": None,
                "validation": {
                    "alignment_matches_source_canonical_phones": True,
                    "feature_dimensions_valid": True,
                    "checkpoint_phone_ids_valid": True,
                    "pronunciation_score_computed": False,
                    "acoustic_support_score_floor_applied": False,
                },
                "evidence": {
                    "phone_count": len(extracted_pure_ids),
                    "pure_phone_ids": extracted_pure_ids,
                    "gop_feature_shape": list(raw_features[:, 1:].shape),
                    "gop_feature_order": "42 log summed acoustic-output phone masses followed by 42 canonical-minus-alternative log mass ratios",
                    "gop_normalization": {"training_mean": GOPT_MEAN, "training_std": GOPT_STD, "padding": "all-zero features and phone ID -1; not normalized"},
                    "canonical_phone_sequence_source": "SpeechOcean762 resource/text-phone",
                    "posterior_frames": int(posteriors.shape[0]),
                    "alignment_frames": len(frame_phone_ids),
                    "frame_shift_seconds": FRAME_SHIFT_SECONDS,
                    "feature_acoustic_output_node": FEATURE_ACOUSTIC_OUTPUT_NODE,
                    "feature_output_frame_logsumexp_min_mean_max": [float(output_logsumexp.min()), float(output_logsumexp.mean()), float(output_logsumexp.max())],
                    "alignment_and_support_output_node": ALIGNMENT_SUPPORT_OUTPUT_NODE,
                    "output_xent_logsumexp_abs_max": max_logsum_residual,
                },
            }
            if artifact_dir is not None:
                _save_feature_evidence(artifact_dir, work_dir, logs, normalized_audio, key, expected, extracted_pure_ids, raw_features, support_features, result)
            return result

        supported_windows, window_support = partition_gopt_windows_by_support(
            windows, phone_log_posteriors,
            expected=expected if recover_supported_words else None,
        )
        if not supported_windows:
            unsupported_indices = [
                index for index, value in enumerate(phone_log_posteriors)
                if value < MIN_PHONE_LOG_POSTERIOR
            ]
            # Keep per-phone evidence in the failed result. Without it, a single
            # weak phone hides the boundary and support measurements for every
            # other phone in a rejected window, which prevents an independent
            # audit of the rejection policy.
            phone_support_diagnostics = [
                {
                    "index": index,
                    "word": expected[index]["word"],
                    "phone": expected[index]["phone"],
                    "start_seconds": round(float(visible_segments[index]["start_seconds"]), 3),
                    "end_seconds": round(float(visible_segments[index]["end_seconds"]), 3),
                    "canonical_phone_mean_log_posterior": round(float(phone_log_posteriors[index]), 6),
                    "support_floor": MIN_PHONE_LOG_POSTERIOR,
                    "support_passed": phone_log_posteriors[index] >= MIN_PHONE_LOG_POSTERIOR,
                }
                for index in range(len(expected))
            ]
            raise ScoreError(
                "ALIGNMENT_INVALID", "alignment_support",
                (
                    "No whole-word GOPT window passed the acoustic alignment support floor."
                    if not recover_supported_words else
                    "No supported whole-word region passed the acoustic alignment support floor."
                ),
                unsupported_phone_indices=unsupported_indices,
                withheld_window_count=len(window_support),
                phone_support_diagnostics=phone_support_diagnostics,
            )
        partial_windows = len(supported_windows) != len(window_support)
        gopt_result, gopt_timing = load_and_run_gopt(
            raw_features[:, 1:], gopt_phone_ids, threads, windows=supported_windows,
            allow_partial=partial_windows, validate_reproducibility=validate_reproducibility,
        )
        phone_results: list[dict[str, Any]] = []
        phone_window_indices: list[int | None] = [None] * len(expected)
        for diagnostic in window_support:
            for index in range(diagnostic["phone_start_index"], diagnostic["phone_end_index"]):
                if phone_window_indices[index] is not None:
                    raise ScoreError("ALIGNMENT_INVALID", "alignment_support", "A phone belongs to more than one support window.", phone_index=index)
                phone_window_indices[index] = diagnostic["window_index"]
        if any(index is None for index in phone_window_indices):
            raise ScoreError("ALIGNMENT_INVALID", "alignment_support", "One or more canonical phones have no support window.")
        for index, (phone, segment) in enumerate(zip(expected, visible_segments)):
            phone_prediction = gopt_result["phoneme_scores"][index]
            word_prediction = gopt_result["word_phone_scores"][index]
            gop_value = None
            if phone_prediction is not None:
                lpp = raw_features[index, 1:1 + PURE_PHONE_COUNT]
                valid_lpp = lpp[3:]
                gop_value = float(lpp[phone["pure_phone_id"]] - np.max(valid_lpp))
                if not math.isfinite(gop_value):
                    raise ScoreError("GOP_EXTRACTION_FAILED", "gop_validation", "GOP score is not finite.", phone=phone["phone"])
            window_status = window_support[phone_window_indices[index]]["status"]
            phone_results.append({
                "index": index,
                "window_index": phone_window_indices[index],
                "word_index": phone["word_index"],
                "word": phone["word"],
                "phone": phone["phone"],
                "start_seconds": round(float(segment["start_seconds"]), 3),
                "end_seconds": round(float(segment["end_seconds"]), 3),
                "canonical_phone_mean_log_posterior": round(phone_log_posteriors[index], 6),
                "pronunciation_evidence_status": "AVAILABLE_UNCALIBRATED" if phone_prediction is not None else window_status,
                "gop_raw_output_margin": None if gop_value is None else round(gop_value, 6),
                "gop_raw_output_margin_semantics": "target phone mass minus highest alternative mass from the unnormalized raw chain output; uncalibrated descriptor",
                "gopt_phone_score": None if phone_prediction is None else round(float(phone_prediction), 6),
                "gopt_word_scores": None if word_prediction is None else {
                    "accuracy": round(float(word_prediction[0]), 6),
                    "stress": round(float(word_prediction[1]), 6),
                    "overall": round(float(word_prediction[2]), 6),
                },
            })
        word_results = []
        for word_index, word in enumerate(words):
            indices = [i for i, phone in enumerate(expected) if phone["word_index"] == word_index]
            values = [gopt_result["word_phone_scores"][i] for i in indices]
            available = all(value is not None for value in values)
            word_results.append({
                "word_index": word_index,
                "word": word,
                "window_index": phone_results[indices[0]]["window_index"],
                "start_seconds": round(phone_results[indices[0]]["start_seconds"], 3),
                "end_seconds": round(phone_results[indices[-1]]["end_seconds"], 3),
                "pronunciation_evidence_status": "AVAILABLE_UNCALIBRATED" if available else "WINDOW_WITHHELD_ALIGNMENT_SUPPORT",
                "gopt_scores": None if not available else {
                    name: round(float(np.asarray([value[offset] for value in values], dtype=np.float64).mean()), 6)
                    for offset, name in enumerate(("accuracy", "stress", "overall"))
                },
            })

        window_diagnostics = []
        predicted_windows = {
            item["window_index"]: item for item in gopt_result["window_predictions"]
        }
        for window in window_support:
            window_index = window["window_index"]
            owned_phones = phone_results[window["phone_start_index"]:window["phone_end_index"]]
            prediction = predicted_windows.get(window_index)
            window_diagnostics.append({
                **window_support[window_index],
                "status": "SCORING_COMPLETED" if prediction is not None else window_support[window_index]["status"],
                "start_seconds": owned_phones[0]["start_seconds"],
                "end_seconds": owned_phones[-1]["end_seconds"],
                "processed_phone_count": sum(phone["gopt_phone_score"] is not None for phone in owned_phones),
                "utterance": None if prediction is None else prediction["utterance"],
            })
        processed_phone_count = sum(phone["gopt_phone_score"] is not None for phone in phone_results)
        aggregation_status = "SINGLE_WINDOW_DIRECT" if len(windows) == 1 and not partial_windows else "NOT_VALIDATED"
        long_form = {
            "reference_phone_count": len(expected),
            "processed_phone_count": processed_phone_count,
            "supported_phone_count": processed_phone_count,
            "alignment_supported_phone_count": sum(value >= MIN_PHONE_LOG_POSTERIOR for value in phone_log_posteriors),
            "planned_window_count": len(windows),
            "chunk_count": len(window_support),
            "processed_chunk_count": len(supported_windows),
            "withheld_chunk_count": len(window_support) - len(supported_windows),
            "coverage": round(processed_phone_count / len(expected), 6),
            "aggregation_status": aggregation_status,
            "windows": window_diagnostics,
        }

        warnings = list(variant_warnings)
        warnings.extend([
            "Scores are uncalibrated GOPT research signals and are not Pearson PTE scores.",
            "The raw chain output used for GOPT features is unnormalized and is not a calibrated posterior distribution.",
            "Forced alignment is conditioned on the supplied reference; successful alignment does not independently prove word completeness.",
            "A conservative -6 mean log-posterior floor rejects grossly unsupported forced phones; this is an alignment validity gate, not a calibrated pronunciation threshold.",
            "Kaldi's generic GOP recipe reports weaker GOP quality with chain models; the LibriSpeech chain model is retained because it is the acoustic model documented for this GOPT checkpoint.",
        ])
        if recover_supported_words and partial_windows:
            warnings.append(
                "Partial whole-word recovery was enabled after an independent ASR/reference exact match; this does not establish that the reference is complete or that the acoustic alignment is correct."
            )
        if len(window_support) > 1:
            warnings.append(
                "GOPT utterance heads are reported per window only; whole-utterance predictions are unavailable because window aggregation is not validated."
            )
        if partial_windows:
            warnings.append(
                "Whole-word regions containing a forced phone below the acoustic support floor were withheld; neighboring supported whole-word regions were scored independently and remain uncalibrated."
            )
        try:
            kaldi_commit = run_bounded(["git", "-C", str(KALDI_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
        except (OSError, subprocess.TimeoutExpired):
            kaldi_commit = "unknown"
        try:
            ffmpeg_version = run_bounded(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
        except (OSError, subprocess.TimeoutExpired, IndexError):
            ffmpeg_version = "unknown"
        try:
            kaldi_version = (KALDI_ROOT / "src" / ".version").read_text(encoding="utf-8").strip()
        except OSError:
            kaldi_version = "unknown"
        chain_archive = ROOT / ".local" / "models" / "0013_librispeech_v1_chain.tar.gz"
        peak_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        byte_factor = 1 if sys.platform == "darwin" else 1024
        peak_bytes = int(max(peak_self, peak_children) * byte_factor)
        end_to_end = time.perf_counter() - started
        result = {
            "status": "SCORING_COMPLETED_PARTIAL" if partial_windows else "SCORING_COMPLETED",
            "engine": "kaldi-gop-gopt-librispeech",
            "input_audio": str(audio.resolve()),
            "audio_duration_seconds": round(duration, 3),
            "processing_time_seconds": round(end_to_end, 3),
            "real_time_factor": round(end_to_end / duration, 4),
            "transcript": " ".join(words),
            "pronunciation": gopt_result["utterance"],
            "long_form": long_form,
            "phonemes": phone_results,
            "words": word_results,
            "warnings": warnings,
            "validation": {
                "alignment_valid": True,
                "gop_features_valid": True,
                "checkpoint_compatible": True,
                "transcript_independently_verified": False,
                "all_reference_phones_processed_once": processed_phone_count == len(expected),
                "processed_phone_count": processed_phone_count,
                "unsupported_phone_prediction_count": len(expected) - processed_phone_count,
            },
            "evidence": {
                "phone_count": len(phone_results),
                "gopt_window_count": len(windows),
                "max_phones_per_window": MAX_PHONES,
                "window_partition": window_diagnostics,
                "gop_feature_shape": list(raw_features[:, 1:].shape),
                "gop_feature_order": "42 log summed acoustic-output phone masses followed by 42 canonical-minus-alternative log mass ratios",
                "gop_feature_sample_mean": round(float(raw_features[:, 1:].mean()), 6),
                "gop_feature_sample_std": round(float(raw_features[:, 1:].std()), 6),
                "gop_normalization": {"training_mean": GOPT_MEAN, "training_std": GOPT_STD, "padding": "all-zero features and phone ID -1; not normalized"},
                "phone_order_source": "first-seen pure-phone IDs in official GOPT LibriSpeech training labels, checked against current model phone symbols",
                "canonical_phone_sequence_source": "SpeechOcean762 resource/text-phone" if source_phone_words is not None else "local align_lexicon first pronunciation",
                "posterior_frames": int(posteriors.shape[0]),
                "posterior_pdf_count": int(posteriors.shape[1]),
                "posterior_max_logsumexp_residual": round(max_logsum_residual, 8),
                "feature_acoustic_output_node": FEATURE_ACOUSTIC_OUTPUT_NODE,
                "feature_output_frame_logsumexp_min_mean_max": [round(float(output_logsumexp.min()), 6), round(float(output_logsumexp.mean()), 6), round(float(output_logsumexp.max()), 6)],
                "alignment_and_support_output_node": ALIGNMENT_SUPPORT_OUTPUT_NODE,
                "alignment_frames": len(frame_phone_ids),
                "frame_shift_seconds": FRAME_SHIFT_SECONDS,
                "canonical_phone_mean_log_posterior_min": round(min(phone_log_posteriors), 6),
                "canonical_phone_mean_log_posterior_floor": MIN_PHONE_LOG_POSTERIOR,
                "model": {
                    "acoustic": "Kaldi M13 LibriSpeech chain tdnn_1d_sp; raw output node supplies GOPT GOP features, output-xent supplies forced alignment and acoustic support validation",
                    "kaldi_version": kaldi_version,
                    "kaldi_commit": kaldi_commit,
                    "chain_archive_sha256": sha256_file(chain_archive) if chain_archive.is_file() else None,
                    "gopt_checkpoint_sha256": gopt_timing["checkpoint_sha256"],
                    "pytorch_version": gopt_timing["pytorch_version"],
                    "numpy_version": gopt_timing["numpy_version"],
                    "ffmpeg_version": ffmpeg_version,
                    "python_version": sys.version.split()[0],
                },
            },
            "performance": {
                "audio_preprocessing_seconds": round(audio_preprocessing_seconds, 4),
                "kaldi_feature_and_ivector_seconds": round(feature_seconds, 4),
                "acoustic_posterior_seconds": round(posterior_seconds, 4),
                "alignment_seconds": round(align_seconds, 4),
                "gop_extraction_seconds": round(gop_seconds, 4),
                **{key: round(float(value), 6) if isinstance(value, float) else value for key, value in gopt_timing.items() if key not in ("checkpoint_sha256", "pytorch_version", "numpy_version")},
                "peak_resident_memory_bytes_max_process": peak_bytes,
                "audio_preprocessing": audio_info,
            },
            "pearson_pte_score": None,
        }
        if artifact_dir is not None:
            _save_feature_evidence(artifact_dir, work_dir, logs, normalized_audio, key, expected, extracted_pure_ids, raw_features, result)
        return result


def shlex_quote(value: str) -> str:
    """Small POSIX shell quote helper for fixed local Kaldi text-conversion calls."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score reference-conditioned pronunciation with local Kaldi GOP + GOPT.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", required=True, help="English reference transcript; spell out numbers.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--keep-intermediates", type=Path, help="Copy Kaldi alignments, posteriors, GOP matrices, logs and JSON evidence to a new local directory.")
    parser.add_argument("--source-phone-text", type=Path, help="Optional source-native Kaldi text-phone file for feature reconstruction.")
    parser.add_argument("--source-utterance-id", help="Utterance key used with --source-phone-text.")
    parser.add_argument("--feature-reconstruction-only", action="store_true", help="Emit source-phone-aligned GOP features without applying the scoring support floor or producing scores.")
    parser.add_argument("--recover-supported-words", action="store_true", help="Experimental: score contiguous whole-word regions with per-phone support after an independent transcript check.")
    parser.add_argument("--validate-reproducibility", action="store_true", help="Run a second GOPT forward pass and enforce the 1e-6 determinism check.")
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 8:
        result = {"status": "INVALID_INPUT", "engine": "kaldi-gop-gopt-librispeech", "error": "--threads must be between 1 and 8.", "pearson_pte_score": None}
        print(json.dumps(result, indent=2))
        return 2
    try:
        if bool(args.source_phone_text) != bool(args.source_utterance_id):
            raise ScoreError("INVALID_INPUT", "source_phone_transcript", "--source-phone-text and --source-utterance-id must be supplied together.")
        source_phone_words = read_source_phone_transcript(args.source_phone_text, args.source_utterance_id) if args.source_phone_text else None
        result = score(
            args.audio.expanduser(), args.reference, args.threads, args.keep_intermediates,
            source_phone_words, args.feature_reconstruction_only, args.recover_supported_words,
            args.validate_reproducibility,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except ScoreError as exc:
        result = {
            "status": exc.status,
            "engine": "kaldi-gop-gopt-librispeech",
            "stage": exc.stage,
            "error": exc.message,
            "details": exc.details,
            "pronunciation": None,
            "validation": {"alignment_valid": False, "gop_features_valid": False, "checkpoint_compatible": False},
            "pearson_pte_score": None,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2
    except Exception as exc:  # Fail closed while preserving a machine-readable result.
        result = {
            "status": "PIPELINE_FAILED",
            "engine": "kaldi-gop-gopt-librispeech",
            "stage": "unexpected",
            "error": str(exc),
            "pronunciation": None,
            "validation": {"alignment_valid": False, "gop_features_valid": False, "checkpoint_compatible": False},
            "pearson_pte_score": None,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
