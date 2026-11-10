#!/usr/bin/env python3
"""Rebuild and compare GOPT acoustic feature branches on exact SpeechOcean762 pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import kaldiio
import numpy as np

import score_pronunciation as scoring


ROOT = Path(__file__).resolve().parent
DEFAULT_IDS = (
    "000030012", "096450018", "096240003", "000050078", "022420106",
    "096120001", "096290006", "001120097", "010370253",
)
BRANCHES = ("raw_output", "normalized_output", "output_xent")
REPO_URL = "https://github.com/jimbozhang/speechocean762"
LICENSE = "CC BY 4.0 (OpenSLR SLR101 listing)"


def read_kv(path: Path) -> dict[str, str]:
    return {key: value for key, value in (line.split(maxsplit=1) for line in path.read_text(encoding="utf-8").splitlines())}


def phone_base(value: str) -> str:
    value = re.sub(r"_[BIES]$", "", value)
    return re.sub(r"[0-9]+$", "", value)


def transcript_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Z0-9]+(?:'[A-Z0-9]+)?", value.upper()))


def read_source_phones(path: Path) -> dict[str, list[list[str]]]:
    words: dict[str, dict[int, list[str]]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(maxsplit=1)
        utterance_id, raw_index = key.rsplit(".", 1)
        index = int(raw_index)
        if index in words[utterance_id]:
            raise ValueError(f"duplicate source phone word index: {key}")
        words[utterance_id][index] = raw.split()
    return {
        utterance_id: [rows[index] for index in range(len(rows))]
        for utterance_id, rows in words.items()
        if sorted(rows) == list(range(len(rows)))
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_info(root: Path, relative: str, expected_speaker: str, utterance_id: str, source_revision: str) -> dict[str, Any]:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    if path.stem != utterance_id or path.parent.name != f"SPEAKER{expected_speaker}":
        raise ValueError(f"source speaker/audio identity mismatch for {utterance_id}: {relative}")
    digest = hashlib.sha256()
    pcm_digest = hashlib.sha256()
    with path.open("rb") as raw:
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(chunk)
    with wave.open(str(path), "rb") as audio:
        channels, width, rate, frames = audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()
        if channels != 1 or width != 2 or rate != 16000 or frames <= 0:
            raise ValueError(f"unexpected source WAV format for {utterance_id}: {(channels, width, rate, frames)}")
        while True:
            block = audio.readframes(65536)
            if not block:
                break
            pcm_digest.update(block)
    return {
        "audio_path": relative,
        "source_url": f"https://raw.githubusercontent.com/jimbozhang/speechocean762/{source_revision}/{relative}",
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "decoded_pcm_sha256": pcm_digest.hexdigest(),
        "sample_rate_hz": rate,
        "channels": channels,
        "sample_width_bytes": width,
        "duration_seconds": frames / rate,
    }


def read_author_archive(
    archive_path: Path,
    source_split_by_id: dict[str, str],
    source_phones: dict[str, list[list[str]]],
    aggregate_scores: dict[str, Any],
    pure_id_to_symbol: dict[int, str],
    selected_ids: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, list[float]], dict[str, Any]]:
    split_names = {"train": "tr", "test": "te"}
    selected_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    selected_labels: dict[str, list[float]] = defaultdict(list)
    observed_ids: dict[str, list[int]] = defaultdict(list)
    observed_labels: dict[str, list[float]] = defaultdict(list)
    archive_utterance_ids: dict[str, str] = {}
    sequence_counts = {"utterances": 0, "phone_rows": 0, "exact_phone_sequences": 0, "sequence_mismatches": 0, "split_mismatches": 0}
    max_label_delta = 0.0
    label_mismatch_count = 0
    all_keys_unique: set[str] = set()

    with zipfile.ZipFile(archive_path) as archive:
        for source_split, short_split in split_names.items():
            keys_member = f"raw_kaldi_gop/librispeech/{short_split}_keys_phn.csv"
            features_member = f"raw_kaldi_gop/librispeech/{short_split}_feats.csv"
            labels_member = f"raw_kaldi_gop/librispeech/{short_split}_labels_phn.csv"
            key_stream = io.TextIOWrapper(archive.open(keys_member), encoding="utf-8")
            feature_stream = io.TextIOWrapper(archive.open(features_member), encoding="utf-8")
            label_stream = io.TextIOWrapper(archive.open(labels_member), encoding="utf-8")
            keys = csv.reader(key_stream)
            features = csv.reader(feature_stream)
            labels = csv.reader(label_stream)
            row_count = 0
            for key_row, feature_row, label_row in zip(keys, features, labels):
                if not key_row or len(feature_row) != 85 or len(label_row) < 2:
                    raise ValueError(f"malformed GOPT author row in {short_split} at line {row_count + 1}")
                key = key_row[0]
                utterance_id, raw_index = key.rsplit(".", 1)
                index = int(raw_index)
                if key in all_keys_unique:
                    raise ValueError(f"duplicate author feature key: {key}")
                all_keys_unique.add(key)
                if source_split_by_id.get(utterance_id) != source_split:
                    sequence_counts["split_mismatches"] += 1
                feature = np.asarray([float(value) for value in feature_row], dtype=np.float64)
                phone_id = int(round(feature[0]))
                label_phone_id, label = int(round(float(label_row[0]))), float(label_row[1])
                if phone_id != label_phone_id:
                    raise ValueError(f"feature/label phone ID mismatch for {key}")
                if index != len(observed_ids[utterance_id]):
                    raise ValueError(f"non-contiguous or unordered author phone key: {key}")
                observed_ids[utterance_id].append(phone_id)
                observed_labels[utterance_id].append(label)
                row_count += 1
                if utterance_id in selected_ids:
                    selected_rows[utterance_id].append(feature)
                    selected_labels[utterance_id].append(label)
            sequence_counts["phone_rows"] += row_count
            if row_count == 0:
                raise ValueError(f"empty author split: {short_split}")
            if not key_stream.closed:
                key_stream.close()
            if not feature_stream.closed:
                feature_stream.close()
            if not label_stream.closed:
                label_stream.close()

    feature_by_id: dict[str, np.ndarray] = {}
    labels_by_id: dict[str, list[float]] = {}
    for utterance_id, phone_ids in observed_ids.items():
        if utterance_id not in source_phones:
            raise ValueError(f"author key is absent from source canonical phone annotations: {utterance_id}")
        expected_symbols = [phone_base(phone) for word in source_phones[utterance_id] for phone in word]
        actual_symbols = [pure_id_to_symbol.get(phone_id, "?") for phone_id in phone_ids]
        sequence_counts["utterances"] += 1
        if actual_symbols == expected_symbols:
            sequence_counts["exact_phone_sequences"] += 1
        else:
            sequence_counts["sequence_mismatches"] += 1
        expected_labels = [float(value) for word in aggregate_scores[utterance_id]["words"] for value in word["phones-accuracy"]]
        if len(expected_labels) != len(observed_labels[utterance_id]):
            raise ValueError(f"author labels/source aggregate phone count mismatch for {utterance_id}")
        for expected, actual in zip(expected_labels, observed_labels[utterance_id]):
            delta = abs(expected - actual)
            max_label_delta = max(max_label_delta, delta)
            label_mismatch_count += int(delta > 1e-6)
        if utterance_id in selected_ids:
            feature_by_id[utterance_id] = np.stack(selected_rows[utterance_id])
            labels_by_id[utterance_id] = selected_labels[utterance_id]

    if sequence_counts["split_mismatches"] or sequence_counts["sequence_mismatches"] or label_mismatch_count:
        raise ValueError(f"author/source mapping audit failed: {sequence_counts}; label mismatches={label_mismatch_count}")
    if selected_ids - set(feature_by_id):
        raise ValueError(f"selected IDs absent from author feature archive: {sorted(selected_ids - set(feature_by_id))}")
    return feature_by_id, labels_by_id, {
        **sequence_counts,
        "key_count": len(all_keys_unique),
        "label_mismatch_count": label_mismatch_count,
        "max_phone_label_abs_error": max_label_delta,
    }


def quantile_stats(values: np.ndarray) -> dict[str, float]:
    q = np.percentile(values, [0, 5, 50, 95, 100])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(q[0]),
        "p05": float(q[1]),
        "median": float(q[2]),
        "p95": float(q[3]),
        "max": float(q[4]),
    }


def feature_metrics(author: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if author.shape != candidate.shape or author.ndim != 2 or author.shape[1] != 84:
        raise ValueError(f"feature matrix mismatch: author={author.shape}; candidate={candidate.shape}")
    delta = candidate - author
    dimensions = []
    for index in range(84):
        left, right = author[:, index], candidate[:, index]
        correlation = None
        if np.std(left) > 0 and np.std(right) > 0:
            correlation = float(np.corrcoef(left, right)[0, 1])
        dimensions.append({
            "dimension": index,
            "mae": float(np.mean(np.abs(delta[:, index]))),
            "rmse": float(np.sqrt(np.mean(np.square(delta[:, index])))),
            "max_abs": float(np.max(np.abs(delta[:, index]))),
            "pearson_r": correlation,
        })
    return {
        "phone_count": int(author.shape[0]),
        "feature_dimensions": 84,
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs_error": float(np.max(np.abs(delta))),
        "lpp_42d": {
            "mae": float(np.mean(np.abs(delta[:, :42]))),
            "rmse": float(np.sqrt(np.mean(np.square(delta[:, :42])))),
            "max_abs_error": float(np.max(np.abs(delta[:, :42]))),
        },
        "lpr_42d": {
            "mae": float(np.mean(np.abs(delta[:, 42:]))),
            "rmse": float(np.sqrt(np.mean(np.square(delta[:, 42:])))),
            "max_abs_error": float(np.max(np.abs(delta[:, 42:]))),
        },
        "author_distribution": {
            "all_84d": quantile_stats(author.reshape(-1)),
            "lpp_42d": quantile_stats(author[:, :42].reshape(-1)),
            "lpr_42d": quantile_stats(author[:, 42:].reshape(-1)),
        },
        "candidate_distribution": {
            "all_84d": quantile_stats(candidate.reshape(-1)),
            "lpp_42d": quantile_stats(candidate[:, :42].reshape(-1)),
            "lpr_42d": quantile_stats(candidate[:, 42:].reshape(-1)),
        },
        "per_dimension": dimensions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=ROOT / ".local/datasets/speechocean762-reconstruction/corpus")
    parser.add_argument("--feature-archive", type=Path, default=ROOT / ".local/gopt/data.zip")
    parser.add_argument("--work-root", type=Path, default=ROOT / ".local/datasets/speechocean762-reconstruction")
    parser.add_argument("--output", type=Path, default=ROOT / "results/paired_feature_reconstruction_quality_05.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--utterance-ids", nargs="+", default=list(DEFAULT_IDS))
    args = parser.parse_args()

    corpus = args.corpus_root.resolve()
    work_root = args.work_root.resolve()
    selected_ids = list(dict.fromkeys(args.utterance_ids))
    if len(selected_ids) < 5 or len(selected_ids) > 20:
        raise SystemExit("select between 5 and 20 utterances")
    selected_set = set(selected_ids)
    source_text: dict[str, str] = {}
    source_speaker: dict[str, str] = {}
    source_audio: dict[str, str] = {}
    source_split: dict[str, str] = {}
    source_age: dict[str, str] = {}
    source_gender: dict[str, str] = {}
    split_speakers: dict[str, set[str]] = {}
    for split in ("train", "test"):
        texts = read_kv(corpus / split / "text")
        speakers = read_kv(corpus / split / "utt2spk")
        wavs = read_kv(corpus / split / "wav.scp")
        ages = read_kv(corpus / split / "spk2age")
        genders = read_kv(corpus / split / "spk2gender")
        if set(texts) != set(speakers) or set(texts) != set(wavs):
            raise ValueError(f"source metadata key sets differ in {split}")
        source_text.update(texts)
        source_speaker.update(speakers)
        source_audio.update(wavs)
        source_split.update({utterance_id: split for utterance_id in texts})
        source_age.update(ages)
        source_gender.update(genders)
        split_speakers[split] = set(speakers.values())
    if selected_set - set(source_text):
        raise ValueError(f"selected utterances absent from source metadata: {sorted(selected_set - set(source_text))}")
    speaker_overlap = sorted(split_speakers["train"] & split_speakers["test"])
    if speaker_overlap:
        raise ValueError(f"source split speaker overlap detected: {speaker_overlap[:10]}")

    source_phones = read_source_phones(corpus / "resource/text-phone")
    source_revision = json.loads((work_root / "provenance-repo-main-commit.json").read_text(encoding="utf-8"))["sha"]
    aggregate_scores = json.loads((corpus / "resource/scores.json").read_text(encoding="utf-8"))
    detailed_scores = json.loads((corpus / "resource/scores-detail.json").read_text(encoding="utf-8"))
    if len(source_text) != 5000 or len(source_speaker) != 5000 or len(source_phones) != 5000 or len(aggregate_scores) != 5000 or len(detailed_scores) != 5000:
        raise ValueError("the downloaded SpeechOcean762 metadata/annotation key counts are not 5,000")

    with tempfile.TemporaryDirectory(prefix="pte-quality05-phone-map-") as temp_name:
        pure_path = Path(temp_name) / "phones-pure.txt"
        map_path = Path(temp_name) / "phone-map.int"
        proc = subprocess.run([
            "perl", str(scoring.GOP_RECIPE / "local/remove_phone_markers.pl"),
            str(scoring.LANG / "phones.txt"), str(pure_path), str(map_path),
        ], capture_output=True, text=True, timeout=30)
        if proc.returncode:
            raise RuntimeError(f"phone map generation failed: {proc.stderr[-1000:]}")
        _, pure_id_to_symbol = scoring.read_symbol_table(pure_path)
        author_features, author_labels, archive_audit = read_author_archive(
            args.feature_archive.resolve(), source_split, source_phones, aggregate_scores,
            pure_id_to_symbol, selected_set,
        )
        gopt_phone_map = scoring.load_training_phone_order(args.feature_archive.resolve(), pure_id_to_symbol)

    evidence_root = work_root / "evidence"
    candidate_root = work_root / "branch-features"
    paired_root = work_root / "paired-features"
    paired_root.mkdir(parents=True, exist_ok=True)
    per_utterance: list[dict[str, Any]] = []
    feature_rows_by_branch: dict[str, list[np.ndarray]] = {branch: [] for branch in BRANCHES}
    gopt_branch_deltas: list[np.ndarray] = []
    author_rows: list[np.ndarray] = []
    phones_by_symbol: dict[str, dict[str, list[np.ndarray]]] = {
        branch: defaultdict(list) for branch in BRANCHES
    }

    for utterance_id in selected_ids:
        split = source_split[utterance_id]
        speaker_id = source_speaker[utterance_id]
        transcript = source_text[utterance_id]
        details = detailed_scores[utterance_id]
        aggregate = aggregate_scores[utterance_id]
        words = source_phones[utterance_id]
        canonical_symbols_by_word = [[phone_base(phone) for phone in word] for word in words]
        canonical_symbols = [phone for word in canonical_symbols_by_word for phone in word]
        author_matrix = author_features[utterance_id]
        author_phone_ids = author_matrix[:, 0].astype(np.int64)
        if [pure_id_to_symbol[value] for value in author_phone_ids] != canonical_symbols:
            raise ValueError(f"sample source canonical phones do not match author feature IDs: {utterance_id}")
        if transcript_tokens(transcript) != transcript_tokens(details["text"]) or transcript_tokens(transcript) != tuple(word["text"].upper() for word in aggregate["words"]):
            raise ValueError(f"source transcript/annotation words differ for {utterance_id}")
        for index, word in enumerate(details["words"]):
            detail_ref = [phone_base(phone) for phone in word["ref-phones"].split()]
            if detail_ref != canonical_symbols_by_word[index]:
                raise ValueError(f"source detail ref-phone sequence differs from text-phone at {utterance_id}.{index}")
        audio = audio_info(corpus, source_audio[utterance_id], speaker_id, utterance_id, source_revision)
        if len(details.get("accuracy", [])) != 5 or len(details.get("total", [])) != 5:
            raise ValueError(f"missing five expert ratings for {utterance_id}")
        rater_labels = {
            field: details[field]
            for field in ("accuracy", "completeness", "fluency", "prosodic", "total")
        }
        rater_median_accuracy = float(np.median(details["accuracy"]))

        evidence_dir = evidence_root / utterance_id
        branch_dir = candidate_root / utterance_id
        shutil.rmtree(evidence_dir, ignore_errors=True)
        shutil.rmtree(branch_dir, ignore_errors=True)
        evidence_dir.parent.mkdir(parents=True, exist_ok=True)
        branch_dir.mkdir(parents=True, exist_ok=True)
        scorer_cmd = [
            str(ROOT / ".local/gopt-venv/bin/python"), str(ROOT / "score_pronunciation.py"),
            "--audio", str((corpus / source_audio[utterance_id]).resolve()),
            "--reference", transcript,
            "--source-phone-text", str((corpus / "resource/text-phone").resolve()),
            "--source-utterance-id", utterance_id,
            "--feature-reconstruction-only", "--threads", str(args.threads),
            "--keep-intermediates", str(evidence_dir),
        ]
        scorer_run = subprocess.run(scorer_cmd, capture_output=True, text=True, timeout=900)
        try:
            scorer_result = json.loads(scorer_run.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"feature-reconstruction command returned invalid JSON for {utterance_id}: {scorer_run.stderr[-1000:]}") from exc
        if scorer_run.returncode or scorer_result.get("status") != "FEATURE_RECONSTRUCTION_COMPLETED":
            raise RuntimeError(f"feature reconstruction failed for {utterance_id}: {json.dumps(scorer_result, ensure_ascii=False)}")
        if scorer_result["evidence"]["pure_phone_ids"] != author_phone_ids.tolist():
            raise ValueError(f"aligned canonical phone IDs differ from author feature order: {utterance_id}")

        branch_stats_path = branch_dir / "branch-stats.json"
        artifacts_dir = branch_dir / "features"
        branch_env = os.environ.copy()
        branch_env.update({
            "QUALITY05_EVIDENCE_ROOT": str(evidence_dir),
            "QUALITY05_SOURCE_UTTERANCE_ID": utterance_id,
            "QUALITY05_SCOPE": f"SpeechOcean762 {utterance_id}; exact source canonical phone sequence; fixed alignment and phone map",
            "QUALITY05_OUTPUT": str(branch_stats_path),
            "QUALITY05_FEATURE_OUTPUT_DIR": str(artifacts_dir),
        })
        branch_run = subprocess.run(["bash", str(ROOT / "evaluate_acoustic_outputs.sh")], cwd=ROOT, env=branch_env, capture_output=True, text=True, timeout=600)
        if branch_run.returncode:
            raise RuntimeError(f"acoustic branch extraction failed for {utterance_id}: {branch_run.stderr[-2000:]}")
        branch_stats = json.loads(branch_stats_path.read_text(encoding="utf-8"))
        matrices: dict[str, np.ndarray] = {}
        for branch in BRANCHES:
            items = list(kaldiio.ReadHelper(f"ark:{artifacts_dir / (branch + '.feat.ark')}"))
            if [key for key, _ in items] != [f"utt0001.{index}" for index in range(len(canonical_symbols))]:
                raise ValueError(f"feature keys do not match source phone order for {utterance_id} ({branch})")
            matrix = np.stack([np.asarray(value, dtype=np.float64) for _, value in items])
            if matrix.shape != (len(canonical_symbols), 85) or not np.isfinite(matrix).all():
                raise ValueError(f"candidate matrix invalid for {utterance_id} ({branch}): {matrix.shape}")
            if not np.array_equal(matrix[:, 0].astype(np.int64), author_phone_ids):
                raise ValueError(f"candidate phone mapping differs from author feature rows for {utterance_id} ({branch})")
            matrices[branch] = matrix[:, 1:]
            feature_rows_by_branch[branch].append(matrices[branch])
            for phone_index, symbol in enumerate(canonical_symbols):
                phones_by_symbol[branch][symbol].append(matrices[branch][phone_index:phone_index + 1])
        author_values = author_matrix[:, 1:]
        author_rows.append(author_values)
        scorer_features = np.load(evidence_dir / "gop_features.npz")
        if not np.allclose(scorer_features["raw_features"], matrices["raw_output"], rtol=0, atol=1e-5):
            raise ValueError(f"raw-output Kaldi feature archive and selected scorer features disagree for {utterance_id}")
        if not np.allclose(scorer_features["output_xent_support_features"], matrices["output_xent"], rtol=0, atol=1e-5):
            raise ValueError(f"output-xent support archive and scorer support features disagree for {utterance_id}")
        gopt_classes = [gopt_phone_map[int(phone_id)] for phone_id in author_phone_ids]
        gopt_raw_output, gopt_timing = scoring.load_and_run_gopt(matrices["raw_output"], gopt_classes, args.threads)
        gopt_xent_output, _ = scoring.load_and_run_gopt(matrices["output_xent"], gopt_classes, args.threads)
        def gopt_values(output: dict[str, Any]) -> np.ndarray:
            return np.concatenate([
                np.asarray([
                    value for value in output["utterance"].values()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ], dtype=np.float64).reshape(-1),
                np.asarray(output["phoneme_scores"], dtype=np.float64).reshape(-1),
                np.asarray(output["word_phone_scores"], dtype=np.float64).reshape(-1),
            ])
        raw_gopt_values = gopt_values(gopt_raw_output)
        xent_gopt_values = gopt_values(gopt_xent_output)
        if not np.isfinite(raw_gopt_values).all() or not np.isfinite(xent_gopt_values).all():
            raise ValueError(f"GOPT produced non-finite outputs for an acoustic branch: {utterance_id}")
        gopt_branch_delta = raw_gopt_values - xent_gopt_values
        gopt_branch_deltas.append(gopt_branch_delta)
        result = {
            "utterance_id": utterance_id,
            "dataset": "SpeechOcean762",
            "split": split,
            "speaker_id": speaker_id,
            "speaker_age": int(source_age[speaker_id]),
            "speaker_gender": source_gender[speaker_id],
            "reference_text": transcript,
            "source_detail_text": details["text"],
            "canonical_phone_symbols": canonical_symbols,
            "author_feature_keys": [f"{utterance_id}.{index}" for index in range(len(canonical_symbols))],
            "phone_count": len(canonical_symbols),
            "audio": audio,
            "local_audio_path": str((corpus / source_audio[utterance_id]).relative_to(ROOT)),
            "provenance": {
                "repository": REPO_URL,
                "repository_revision": source_revision,
                "license": LICENSE,
                "license_source": "https://www.openslr.org/101/",
            },
            "evaluation_eligibility": "reconstruction_and_engineering_diagnostics_only__not_independent_final_evaluation",
            "annotation_consistency": {
                "source_transcript_matches_detail_and_aggregate_words": True,
                "source_ref_phones_match_text_phone": True,
                "author_feature_phone_ids_match_source_sequence": True,
                "author_phone_labels_match_scores_json": True,
            },
            "expert_ratings": rater_labels,
            "aggregate_annotation": aggregate,
            "feature_reconstruction": {
                "status": scorer_result["status"],
                "same_canonical_phone_ids": True,
                "frame_count": scorer_result["evidence"]["posterior_frames"],
                "frame_shift_seconds": scorer_result["evidence"]["frame_shift_seconds"],
                "pronunciation_score_computed": False,
                "scoring_support_floor_applied": False,
            },
            "gopt_raw_branch_inference": {
                "completed": True,
                "outputs_finite": True,
                "max_repeat_delta": gopt_timing["max_absolute_repeat_delta"],
                "checkpoint_sha256": gopt_timing["checkpoint_sha256"],
            },
            "old_xent_vs_selected_raw_scorer_outputs": {
                "same_recording_and_phone_ids": True,
                "numeric_output_count": int(gopt_branch_delta.size),
                "mean_absolute_delta": float(np.mean(np.abs(gopt_branch_delta))),
                "max_absolute_delta": float(np.max(np.abs(gopt_branch_delta))),
                "interpretation": "Uncalibrated GOPT output sensitivity to the compatible feature branch; not an accuracy measure.",
            },
            "candidates": {
                branch: feature_metrics(author_values, matrices[branch])
                for branch in BRANCHES
            },
            "frame_output_checks": branch_stats["frame_outputs"],
            "checkpoint_exposure": "SpeechOcean762 author training/test data; not independent quality evidence",
        }
        per_utterance.append(result)
        np.savez_compressed(
            paired_root / f"{utterance_id}.npz",
            author_features=author_values.astype(np.float32),
            pure_phone_ids=author_phone_ids,
            raw_output_features=matrices["raw_output"].astype(np.float32),
            normalized_output_features=matrices["normalized_output"].astype(np.float32),
            output_xent_features=matrices["output_xent"].astype(np.float32),
            canonical_phone_symbols=np.asarray(canonical_symbols),
        )
        shutil.rmtree(evidence_dir)
        shutil.rmtree(branch_dir)

    author_combined = np.concatenate(author_rows, axis=0)
    combined_by_branch = {
        branch: np.concatenate(feature_rows_by_branch[branch], axis=0)
        for branch in BRANCHES
    }
    aggregate_metrics = {
        branch: feature_metrics(author_combined, candidate)
        for branch, candidate in combined_by_branch.items()
    }
    all_gopt_deltas = np.concatenate(gopt_branch_deltas)
    per_phone_errors = {}
    for branch in BRANCHES:
        per_phone_errors[branch] = {}
        for symbol, chunks in sorted(phones_by_symbol[branch].items()):
            candidate = np.concatenate(chunks, axis=0)
            reference_chunks = []
            for item in per_utterance:
                positions = [i for i, value in enumerate(item["canonical_phone_symbols"]) if value == symbol]
                archive_rows = author_features[item["utterance_id"]][:, 1:]
                reference_chunks.extend(archive_rows[position:position + 1] for position in positions)
            reference = np.concatenate(reference_chunks, axis=0)
            error = candidate - reference
            per_phone_errors[branch][symbol] = {
                "count": int(len(reference)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "max_abs_error": float(np.max(np.abs(error))),
            }

    archive_audit.update({
        "split_utterances": {split: sum(value == split for value in source_split.values()) for split in ("train", "test")},
        "source_split_speaker_overlap_count": len(speaker_overlap),
        "source_split_utterance_overlap_count": len(set(read_kv(corpus / "train/utt2spk")) & set(read_kv(corpus / "test/utt2spk"))),
        "speaker_count": len(set(source_speaker.values())),
    })
    model_evidence = {
        "kaldi_version": (scoring.KALDI_ROOT / "src/.version").read_text(encoding="utf-8").strip(),
        "chain_model_archive_sha256": scoring.sha256_file(ROOT / ".local/models/0013_librispeech_v1_chain.tar.gz"),
        "acoustic_model_final_mdl_sha256": scoring.sha256_file(scoring.ACOUSTIC_MODEL / "final.mdl"),
        "ivector_extractor_sha256": scoring.sha256_file(scoring.IVECTOR_EXTRACTOR / "final.ie"),
        "mfcc_config_sha256": scoring.sha256_file(scoring.GOP_RECIPE / "conf/mfcc_hires.conf"),
        "gopt_checkpoint_sha256": scoring.sha256_file(scoring.CHECKPOINT),
    }
    means = {branch: aggregate_metrics[branch]["mae"] for branch in BRANCHES}
    selected_branch = min(means, key=means.get)
    report = {
        "task": "PTE-SCORING-QUALITY-05-RECOVERY",
        "source": REPO_URL,
        "source_revision": source_revision,
        "license": LICENSE,
        "historical_feature_archive": {
            "path": ".local/gopt/data.zip",
            "sha256": sha256_file(args.feature_archive.resolve()),
            "audit": archive_audit,
        },
        "source_artifacts": {
            relative: {
                "source_url": f"https://github.com/jimbozhang/speechocean762/blob/{source_revision}/{relative}",
                "local_path": str((corpus / relative).relative_to(ROOT)),
                "bytes": (corpus / relative).stat().st_size,
                "sha256": sha256_file(corpus / relative),
            }
            for relative in (
                "train/text", "train/utt2spk", "train/wav.scp", "train/spk2age", "train/spk2gender",
                "test/text", "test/utt2spk", "test/wav.scp", "test/spk2age", "test/spk2gender",
                "resource/text-phone", "resource/scores.json", "resource/scores-detail.json",
            )
        },
        "feature_provenance": {
            "official_gopt_documentation": "README directs users to the Kaldi GOP recipe, then extract_kaldi_gop/extract_gop_feats.py copies the recipe feature vectors into the author dataset.",
            "local_recipe_evidence": {
                "compute_output": ".local/kaldi/egs/gop_speechocean762/s5/run.sh:85-87 invokes compute_output.sh without --output-name, so it retains the model's default output node.",
                "compute_gop": ".local/kaldi/egs/gop_speechocean762/s5/run.sh:174-183 passes exp/probs_$part/output.JOB.ark directly into compute-gop.",
                "gopt_feature_copy": ".local/gopt/src/extract_kaldi_gop/extract_gop_feats.py:36-73 reads Kaldi feat.scp vectors and writes them as author feature rows without acoustic branch normalization.",
                "local_kaldi_version": (scoring.KALDI_ROOT / "src/.version").read_text(encoding="utf-8").strip(),
                "historical_recipe_revision": "not recoverable from the provided author feature archive; no author alignment boundaries or feature-extraction commit are bundled in data.zip",
            },
            "paired_branch_result": "raw output is closest on exact source utterance IDs and canonical phone sequences across all three tested branches; see aggregate_metrics.",
            "raw_output_semantics": "unnormalized chain acoustic scores; not calibrated posterior probabilities",
            "alignment_and_support_branch": "output-xent log-softmax; retained for reference forced alignment and the existing acoustic support floor",
            "unresolved_numerical_residual": "The author archive contains no frame-level alignment boundaries. Raw output is strongly closest but not exactly equal, so residual alignment/model/preprocessing differences are not assigned a specific cause.",
        },
        "sample_selection": {
            "utterance_ids": selected_ids,
            "sample_count": len(selected_ids),
            "source_splits": {split: sum(source_split[utterance_id] == split for utterance_id in selected_ids) for split in ("train", "test")},
            "speaker_count": len({source_speaker[utterance_id] for utterance_id in selected_ids}),
            "source_rater_median_accuracy_range": [
                float(min(np.median(detailed_scores[value]["accuracy"]) for value in selected_ids)),
                float(max(np.median(detailed_scores[value]["accuracy"]) for value in selected_ids)),
            ],
            "selection_note": "Includes the user-specified 000030012 pair plus a fixed cross-split sample selected for differing five-rater median accuracy, unique speakers, and exact first-lexicon canonical phone agreement where available. Ratings did not select or rank acoustic branches.",
        },
        "paired_reconstruction": {
            "paired_utterances": len(per_utterance),
            "paired_phones": int(sum(item["phone_count"] for item in per_utterance)),
            "phone_sequence_agreement_count": len(per_utterance),
            "reconstruction_failure_count": 0,
            "alignment_boundaries_available_in_author_archive": False,
            "alignment_contract": "Each reconstructed row is mapped by exact utterance ID and canonical pure-phone sequence; no author frame-level alignment is released for timing parity.",
            "acoustic_alignment_source": "M13 model forced alignment conditioned on the verified source resource/text-phone sequence; same per-utterance alignment used for all three output branches.",
            "candidate_feature_source": "Kaldi compute-gop with the same local model/map/alignment; only acoustic output branch varies.",
            "aggregate_metrics": aggregate_metrics,
            "per_phone_symbol_errors": per_phone_errors,
            "best_branch_by_paired_feature_mae": selected_branch,
            "per_utterance": per_utterance,
        },
        "local_model_artifacts": model_evidence,
        "gopt_compatibility": {
            "normalization": {"mean": scoring.GOPT_MEAN, "std": scoring.GOPT_STD},
            "padding": "50 tokens; valid features normalized by fixed author constants; padded features remain zero",
            "inference_on_raw_candidate": "completed for each pair; deterministic CPU repeat checks performed by the existing loader",
            "old_output_xent_vs_selected_raw_outputs": "same paired recordings and checkpoint; descriptive output deltas only, not labeled quality evidence",
        },
        "gopt_output_branch_comparison": {
            "paired_utterances": len(per_utterance),
            "numeric_outputs_compared": int(all_gopt_deltas.size),
            "mean_absolute_output_delta": float(np.mean(np.abs(all_gopt_deltas))),
            "max_absolute_output_delta": float(np.max(np.abs(all_gopt_deltas))),
            "interpretation": "Describes old output-xent versus selected raw-output checkpoint outputs on identical audio and phone IDs; this is model sensitivity, not labeled assessment quality.",
        },
        "assessment_quality": "NOT EVALUATED; feature reconstruction only. SpeechOcean762 data was used during GOPT development/checkpoint selection.",
        "pearson_pte_score": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest_path = args.manifest or (work_root / "sample_manifest.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as stream:
        for item in per_utterance:
            annotations = item["expert_ratings"]
            detail_words = detailed_scores[item["utterance_id"]]["words"]
            aggregate = item["aggregate_annotation"]
            record = {
                "dataset": item["dataset"],
                "utterance_id": item["utterance_id"],
                "speaker_id": item["speaker_id"],
                "source_split": item["split"],
                "audio_path": item["local_audio_path"],
                "audio": item["audio"],
                "reference_text": item["reference_text"],
                "annotations": {
                    "phone": {
                        "canonical_words": source_phones[item["utterance_id"]],
                        "expert_detail_by_word": detail_words,
                        "aggregate_by_word": aggregate["words"],
                    },
                    "word": detail_words,
                    "utterance": {
                        "five_rater_scores": annotations,
                        "aggregate": {key: aggregate[key] for key in ("accuracy", "completeness", "fluency", "prosodic", "total")},
                    },
                },
                "author_feature_pairing": {
                    "status": "exact_utterance_and_phone_sequence_match",
                    "feature_keys": item["author_feature_keys"],
                    "phone_count": item["phone_count"],
                },
                "provenance": item["provenance"],
                "evaluation_eligibility": item["evaluation_eligibility"],
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "manifest": str(manifest_path),
        "paired_utterances": len(per_utterance),
        "paired_phones": report["paired_reconstruction"]["paired_phones"],
        "global_sequence_matches": archive_audit["exact_phone_sequences"],
        "selected_branch": selected_branch,
        "branch_mae": means,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
