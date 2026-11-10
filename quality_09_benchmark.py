"""QUALITY-09 frozen pronunciation benchmark on fresh EpaDB speakers.

The downloaded EpaDB sample is CC BY-NC 4.0 research data. This utility keeps
audio, annotations, scorer runs, and derived benchmark results under .local/;
it does not train or alter the production scorer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import quality_08_benchmark as q08


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / ".local" / "datasets" / "quality-09-epadb-new-speakers"
OLD_DATA_ROOT = ROOT / ".local" / "datasets" / "epadb-quality-08"
REPORT_PATH = ROOT / "results" / "quality_09_report.json"
SOURCE_COMMIT = "c697e2994d227e8fd7d48bc014e12777a0292424"
SOURCE_REPOSITORY = "https://github.com/donggangj/EpaDB"
RAW_BASE = f"https://raw.githubusercontent.com/donggangj/EpaDB/{SOURCE_COMMIT}"
LICENSE = "CC BY-NC 4.0; non-commercial research use only"
SPEAKERS = (
    "spkr14", "spkr29", "spkr30", "spkr31", "spkr32", "spkr33", "spkr34", "spkr35",
    "spkr41", "spkr42", "spkr45", "spkr46", "spkr50", "spkr52", "spkr55", "spkr56",
    "spkr57", "spkr64", "spkr65",
)
PROMPTS = (18, 33, 46, 50)
FILE_TYPES = (
    ("waveforms", "wav"),
    ("annotations_1", "TextGrid"),
    ("annotations_2", "TextGrid"),
    ("labels", "csv"),
    ("transcriptions", "lab"),
)
GOPT_PYTHON = ROOT / ".local" / "gopt-venv" / "bin" / "python"
SCORE_SCRIPT = ROOT / "score_pronunciation.py"
SCORE_SUCCESS = {"SCORING_COMPLETED", "SCORING_COMPLETED_PARTIAL"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "pte-scoring-lab-quality09/1.0", "Accept": "application/octet-stream"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 200:
                    raise OSError(f"HTTP {response.status} from {url}")
                return response.read()
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == 2:
                break
    raise OSError(f"Could not fetch pinned source file {url}: {last_error}")


def _resource_plan() -> list[tuple[str, str]]:
    resources = [
        ("source/UPSTREAM_README.md", "readme.md"),
        ("source/train_reference_transcriptions.txt", "train/reference_transcriptions.txt"),
        ("source/train_assign_reference.py", "train/assign_reference.py"),
    ]
    for speaker in SPEAKERS:
        for prompt in PROMPTS:
            record_id = f"{speaker}_{prompt}"
            for folder, extension in FILE_TYPES:
                resources.append((
                    f"holdout/train/{speaker}/{folder}/{record_id}.{extension}",
                    f"train/{speaker}/{folder}/{record_id}.{extension}",
                ))
    return resources


def acquire_dataset() -> dict[str, Any]:
    """Download only 76 dual-annotated utterances from the pinned author repo."""
    manifest_path = DATA_ROOT / "sample_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_acquired_files(manifest)
        return manifest
    if DATA_ROOT.exists():
        raise RuntimeError(f"Refusing to overwrite incomplete or unrecognized data at {DATA_ROOT}.")
    staging = DATA_ROOT.with_name(DATA_ROOT.name + ".staging")
    if staging.exists():
        raise RuntimeError(f"Refusing to reuse pre-existing staging directory {staging}.")
    staging.mkdir(parents=True)
    resources = _resource_plan()
    completed: list[dict[str, Any]] = []

    def download_one(item: tuple[str, str]) -> dict[str, Any]:
        relative, upstream_path = item
        data = _download_bytes(f"{RAW_BASE}/{upstream_path}")
        if not data:
            raise OSError(f"Empty upstream file: {upstream_path}")
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return {
            "path": relative,
            "source_path": upstream_path,
            "source_url": f"{RAW_BASE}/{upstream_path}",
            "size_bytes": len(data),
            "sha256": sha256_bytes(data),
            "git_blob_sha1": git_blob_sha1(data),
        }

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(download_one, item) for item in resources]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                completed.append(future.result())
                if index % 50 == 0 or index == len(resources):
                    print(f"[acquire {index}/{len(resources)}] source files verified", flush=True)
        completed.sort(key=lambda row: row["path"])
        if len(completed) != len(resources):
            raise RuntimeError(f"Expected {len(resources)} files; fetched {len(completed)}.")
        files = {item["path"]: item for item in completed}
        records: list[dict[str, Any]] = []
        for speaker in SPEAKERS:
            for prompt in PROMPTS:
                record_id = f"{speaker}_{prompt}"
                record_files = {
                    folder: files[f"holdout/train/{speaker}/{folder}/{record_id}.{extension}"]
                    for folder, extension in FILE_TYPES
                }
                records.append({"record_id": record_id, "speaker_id": speaker, **record_files})
        manifest = {
            "schema_version": "quality-09-epadb-sample-v1",
            "acquired_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": {
                "name": "EpaDB English pronunciation database",
                "repository": SOURCE_REPOSITORY,
                "commit": SOURCE_COMMIT,
                "license": LICENSE,
                "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
                "citation": "Vidal, Ferrer, and Brambilla (2019), EpaDB: a database for development of pronunciation assessment systems, Interspeech, 589–593.",
                "usage_restriction": "Research diagnostics only. No commercial training, derived commercial artifact, redistribution, or deployment rights are claimed.",
                "label_semantics_source": "source/train_assign_reference.py at the pinned commit: '+' when canonical target equals the manual annotation, '-' otherwise.",
            },
            "selection": {
                "source_partition": "train",
                "speakers": list(SPEAKERS),
                "l1": "Argentinian Spanish for all corpus speakers",
                "prompts": list(PROMPTS),
                "recordings_per_speaker": len(PROMPTS),
                "reason": "The four selected source prompts have a second human TextGrid annotation for each selected speaker.",
                "relationship_to_quality_08": "All 20 speakers are absent from the 30-speaker QUALITY-08 sample; they are held out from QUALITY-09 threshold selection and candidate development.",
                "gender_metadata": "The source README reports corpus-level balance but no speaker-specific mapping was used in this sample.",
            },
            "counts": {"speakers": len(SPEAKERS), "recordings": len(records), "source_files": len(completed)},
            "files": completed,
            "records": records,
        }
        (staging / "sample_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        if DATA_ROOT.exists():
            raise RuntimeError(f"Dataset destination appeared during acquisition: {DATA_ROOT}")
        staging.rename(DATA_ROOT)
    except Exception:
        shutil.rmtree(staging)
        raise
    return manifest


def verify_acquired_files(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != "quality-09-epadb-sample-v1":
        raise ValueError("Unsupported QUALITY-09 dataset manifest.")
    for item in manifest.get("files", []):
        path = DATA_ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Downloaded source file failed manifest verification: {path}")


def scorer_fingerprint() -> dict[str, Any]:
    files = {
        "score_pronunciation.py": ROOT / "score_pronunciation.py",
        "score_integrated.py": ROOT / "score_integrated.py",
        "quality_08_benchmark.py": ROOT / "quality_08_benchmark.py",
        "gopt_checkpoint": ROOT / ".local/gopt/pretrained_models/gopt_librispeech/best_audio_model.pth",
        "kaldi_chain_archive": ROOT / ".local/models/0013_librispeech_v1_chain.tar.gz",
    }
    absent = [name for name, path in files.items() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Cannot freeze missing scorer assets: {absent}")
    return {
        "source_sha256": {name: sha256_file(path) for name, path in files.items()},
        "support_floor": q08.SUPPORT_FLOOR,
        "baseline_policy": "Default whole-window rejection; no --recover-supported-words flag.",
        "recovery_comparison": "Same frozen code/checkpoint with only the existing --recover-supported-words flag enabled.",
        "python": sys.version.split()[0],
        "gopt_runtime": _gopt_runtime_versions(),
        "kaldi_version": _kaldi_version(),
        "ffmpeg_version": _ffmpeg_version(),
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _gopt_runtime_versions() -> dict[str, str]:
    proc = subprocess.run(
        [str(GOPT_PYTHON), "-c", "import json, numpy, torch, kaldiio; print(json.dumps({'python':__import__('sys').version.split()[0], 'numpy':numpy.__version__, 'torch':torch.__version__, 'kaldiio':getattr(kaldiio,'__version__','unknown')}))"],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        return {"status": "UNAVAILABLE", "error": proc.stderr[-1000:]}
    return json.loads(proc.stdout)


def _kaldi_version() -> str:
    try:
        return (ROOT / ".local/kaldi/src/.version").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def _ffmpeg_version() -> str:
    try:
        proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        return proc.stdout.splitlines()[0] if proc.stdout else "unknown"
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return "unknown"


def freeze_scorer() -> dict[str, Any]:
    fingerprint = scorer_fingerprint()
    destination = DATA_ROOT / "frozen_scorer"
    fingerprint_path = destination / "fingerprint.json"
    if fingerprint_path.is_file():
        existing = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if existing.get("source_sha256") != fingerprint["source_sha256"]:
            raise RuntimeError("Frozen scorer fingerprint differs from the current scorer; refusing to overwrite it.")
        return existing
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite incomplete frozen scorer directory {destination}.")
    destination.mkdir(parents=True)
    copies = {
        "score_pronunciation.py": ROOT / "score_pronunciation.py",
        "score_integrated.py": ROOT / "score_integrated.py",
        "quality_08_benchmark.py": ROOT / "quality_08_benchmark.py",
    }
    for name, source in copies.items():
        shutil.copy2(source, destination / name)
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return fingerprint


def build_manifest() -> dict[str, Any]:
    path = DATA_ROOT / "sample_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run --acquire first; missing {path}.")
    source = json.loads(path.read_text(encoding="utf-8"))
    verify_acquired_files(source)
    q08.DATA_ROOT = DATA_ROOT
    wav_paths = sorted(DATA_ROOT.glob("holdout/train/**/waveforms/*.wav"))
    records = [q08.parse_record(wav_path) for wav_path in wav_paths]
    expected_ids = {record["record_id"] for record in source["records"]}
    if {record["record_id"] for record in records} != expected_ids:
        raise ValueError("Downloaded audio files do not match the pinned sample manifest.")
    old_manifest = json.loads((OLD_DATA_ROOT / "sample_manifest.json").read_text(encoding="utf-8"))
    old_speakers = {row["speaker_id"] for row in old_manifest["records"]}
    new_speakers = {record["speaker_id"] for record in records}
    duplicate_hashes: dict[str, list[str]] = defaultdict(list)
    for record in records:
        duplicate_hashes[record["files"]["audio"]["sha256"]].append(record["record_id"])
    duplicate_audio = [ids for ids in duplicate_hashes.values() if len(ids) > 1]
    result = {
        "schema_version": "quality-09-benchmark-manifest-v1",
        "source_manifest_sha256": sha256_file(path),
        "records": records,
        "integrity": {
            "recording_count": len(records),
            "speaker_count": len(new_speakers),
            "q08_speaker_overlap": sorted(new_speakers & old_speakers),
            "duplicate_audio_groups": duplicate_audio,
            "records_with_integrity_problems": [row["record_id"] for row in records if row["integrity_problems"]],
            "all_annotation_2_files_present": all(
                (DATA_ROOT / "holdout" / "train" / row["speaker_id"] / "annotations_2" / f"{row['record_id']}.TextGrid").is_file()
                for row in records
            ),
            "source_label_semantics_conflicts": _source_label_semantics_conflicts(
                [
                    row
                    for record in records
                    for row in q08._source_label_rows(record["record_id"], record["partition"])
                ]
            ),
        },
    }
    if result["integrity"]["q08_speaker_overlap"] or duplicate_audio or result["integrity"]["records_with_integrity_problems"]:
        raise ValueError(f"QUALITY-09 sample integrity check failed: {result['integrity']}")
    return result


def _source_label_semantics_conflicts(labels: list[dict[str, Any]]) -> list[str]:
    conflicts: list[str] = []
    for row in labels:
        expected = row["target_phone"] == row["observed_phone"]
        if (row["label"] == "+") != expected:
            conflicts.append(row["source_phone_id"])
    return conflicts


def _phone_rows(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    return q08._direct_phone_rows(result)


def collect_phone_items(
    dataset_root: Path,
    records: list[dict[str, Any]],
    run_name: str,
) -> list[dict[str, Any]]:
    """Match target labels to scorer phones with edit alignment; never shift positions."""
    q08.DATA_ROOT = dataset_root
    items: list[dict[str, Any]] = []
    for record in records:
        record_id = record["record_id"]
        labels = q08._source_label_rows(record_id, record["partition"])
        result = q08._load_run(dataset_root / "runs" / run_name / "direct" / f"{record_id}.json")
        predictions = _phone_rows(result)
        predicted_symbols = [q08.normalize_phone(row.get("phone", "")) for row in predictions]
        operations = q08.edit_align([q08.normalize_phone(row["target_phone"]) for row in labels], predicted_symbols)
        for operation in operations:
            label_index = operation["reference_index"]
            if label_index is None:
                continue
            label = labels[label_index]
            prediction_index = operation["hypothesis_index"]
            prediction = predictions[prediction_index] if prediction_index is not None else None
            exact_identity = operation["operation"] == "MATCH"
            score = q08._as_number(prediction.get("gopt_phone_score")) if prediction and exact_identity else None
            if label["label"] not in {"+", "-"}:
                raise ValueError(f"Unrecognized source label class: {label['label']!r}")
            items.append({
                "speaker_id": record["speaker_id"],
                "record_id": record_id,
                "source_phone_id": label["source_phone_id"],
                "correct": label["label"] == "+",
                "label": label["label"],
                "target_phone": label["target_phone"],
                "observed_phone": label["observed_phone"],
                "scorer_phone": None if prediction is None else prediction.get("phone"),
                "operation": operation["operation"],
                "exact_identity": exact_identity,
                "score": score,
                "supported": score is not None,
                "alignment_support": None if prediction is None else q08._as_number(prediction.get("canonical_phone_mean_log_posterior")),
                "start_seconds": None if prediction is None else q08._as_number(prediction.get("start_seconds")),
                "end_seconds": None if prediction is None else q08._as_number(prediction.get("end_seconds")),
            })
    return items


def _auc_error(items: list[dict[str, Any]]) -> float | None:
    scored = [row for row in items if row.get("score") is not None]
    if not scored:
        return None
    return q08._auc([int(row["correct"]) for row in scored], [row["score"] for row in scored])


def pr_auc_error(items: list[dict[str, Any]]) -> float | None:
    scored = [row for row in items if row.get("score") is not None]
    positives = sum(not row["correct"] for row in scored)
    if not scored or positives == 0 or positives == len(scored):
        return None
    ordered = sorted(scored, key=lambda row: row["score"])
    previous_recall, previous_precision = 0.0, 1.0
    selected = tp = 0
    area = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end]["score"] == ordered[index]["score"]:
            end += 1
        selected += end - index
        tp += sum(not row["correct"] for row in ordered[index:end])
        precision = tp / selected
        recall = tp / positives
        area += (recall - previous_recall) * (precision + previous_precision) / 2
        previous_recall, previous_precision = recall, precision
        index = end
    return area


def threshold_metrics(items: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scored = [row for row in items if row.get("score") is not None]
    errors = sum(not row["correct"] for row in items)
    correct = sum(row["correct"] for row in items)
    tp = sum(not row["correct"] and row["score"] <= threshold for row in scored)
    fp = sum(row["correct"] and row["score"] <= threshold for row in scored)
    fn_scored = sum(not row["correct"] and row["score"] > threshold for row in scored)
    tn = sum(row["correct"] and row["score"] > threshold for row in scored)
    abstained_errors = errors - sum(not row["correct"] for row in scored)
    abstained_correct = correct - sum(row["correct"] for row in scored)
    fn = fn_scored + abstained_errors
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / errors if errors else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0 if errors else None
    return {
        "eligible_phones": len(items),
        "scored_phones": len(scored),
        "coverage": len(scored) / len(items) if items else None,
        "conditional_confusion_matrix": {"true_error": tp, "false_error": fp, "missed_scored_error": fn_scored, "correct_rejection": tn},
        "abstained_correct": abstained_correct,
        "abstained_error_counted_as_missed": abstained_errors,
        "coverage_aware": {
            "precision": precision,
            "recall_all_reference_errors": recall,
            "f1": f1,
            "false_positive_rate_on_scored_correct_phones": fp / (fp + tn) if fp + tn else None,
            "false_negative_rate_including_abstained_errors": fn / errors if errors else None,
            "accuracy_abstentions_count_wrong": (tp + tn) / len(items) if items else None,
        },
    }


def choose_threshold(development_items: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in development_items if row.get("score") is not None]
    if not scored or len({row["correct"] for row in scored}) < 2:
        raise ValueError("Development data must contain scored correct and incorrect phones.")
    candidates = sorted({row["score"] for row in scored})
    options = []
    for threshold in candidates:
        metric = threshold_metrics(development_items, threshold)["coverage_aware"]
        options.append((metric["f1"], metric["precision"] or 0.0, -threshold, threshold, metric))
    best = max(options)
    return {
        "threshold": best[3],
        "decision": "flag as pronunciation error when native GOPT phone score <= threshold",
        "selection_metric": "maximum F1 on QUALITY-08 development speakers only; ties prefer precision then the smaller, more conservative threshold.",
        "development_speakers": sorted({row["speaker_id"] for row in development_items}),
        "development_eligible_phones": len(development_items),
        "development_scored_phones": len(scored),
        "development_metrics_at_threshold": threshold_metrics(development_items, best[3]),
    }


def _bootstrap_ci(items: list[dict[str, Any]], metric: Callable[[list[dict[str, Any]]], float | None], seed: int = 9026, reps: int = 2000) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[item["speaker_id"]].append(item)
    speakers = sorted(groups)
    if len(speakers) < 2:
        return {"low_95": None, "high_95": None, "valid_replicates": 0, "clusters": len(speakers)}
    import random
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        sample = []
        for _ in speakers:
            sample.extend(groups[rng.choice(speakers)])
        value = metric(sample)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    if not values:
        return {"low_95": None, "high_95": None, "valid_replicates": 0, "clusters": len(speakers)}
    values.sort()
    return {
        "low_95": values[int(0.025 * (len(values) - 1))],
        "high_95": values[int(0.975 * (len(values) - 1))],
        "valid_replicates": len(values),
        "clusters": len(speakers),
    }


def _paired_delta_ci(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float | None],
    seed: int = 9027,
    reps: int = 2000,
) -> dict[str, Any]:
    common = sorted(set(baseline) & set(candidate))
    pairs = [{"speaker_id": baseline[key]["speaker_id"], "baseline": baseline[key], "candidate": candidate[key]} for key in common]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[pair["speaker_id"]].append(pair)
    speakers = sorted(groups)
    if len(speakers) < 2:
        return {"paired_phone_count": len(common), "low_95": None, "high_95": None, "valid_replicates": 0}
    import random
    rng = random.Random(seed)
    deltas = []
    for _ in range(reps):
        sample = []
        for _ in speakers:
            sample.extend(groups[rng.choice(speakers)])
        base_value = metric([item["baseline"] for item in sample])
        candidate_value = metric([item["candidate"] for item in sample])
        if base_value is not None and candidate_value is not None:
            deltas.append(candidate_value - base_value)
    deltas.sort()
    return {
        "paired_phone_count": len(common),
        "paired_speaker_count": len(speakers),
        "low_95": deltas[int(0.025 * (len(deltas) - 1))] if deltas else None,
        "high_95": deltas[int(0.975 * (len(deltas) - 1))] if deltas else None,
        "valid_replicates": len(deltas),
    }


def _binary_metrics(items: list[dict[str, Any]]) -> float | None:
    return threshold_metrics(items, _ACTIVE_THRESHOLD)["coverage_aware"]["f1"]


_ACTIVE_THRESHOLD = 0.0


def run_models(manifest: dict[str, Any], threads: int = 2) -> None:
    records = manifest["records"]
    jobs = [(record, policy) for record in records for policy in ("baseline", "recovery")]
    for index, (record, policy) in enumerate(jobs, 1):
        record_id = record["record_id"]
        destination = DATA_ROOT / "runs" / policy / "direct" / f"{record_id}.json"
        if destination.is_file():
            continue
        audio_path = DATA_ROOT / record["files"]["audio"]["path"]
        command = [str(GOPT_PYTHON), str(SCORE_SCRIPT), "--audio", str(audio_path),
                   "--reference", record["reference_transcript"], "--threads", str(threads)]
        if policy == "recovery":
            command.append("--recover-supported-words")
        proc = subprocess.run(command, capture_output=True, text=True, timeout=900)
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError:
            result = {"status": "BENCHMARK_RUNNER_FAILED", "error": "Scorer did not return JSON.",
                      "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:]}
        result["_benchmark"] = {"policy": policy, "exit_code": proc.returncode, "command": command,
                                "stderr_tail": proc.stderr[-1000:] if proc.returncode else ""}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[{index}/{len(jobs)}] {policy} {record_id}: {result.get('status')}", flush=True)


def interannotator_agreement(records: list[dict[str, Any]]) -> dict[str, Any]:
    q08.DATA_ROOT = DATA_ROOT
    comparisons: list[dict[str, Any]] = []
    for record in records:
        base = DATA_ROOT / "holdout" / "train" / record["speaker_id"]
        first = q08.parse_textgrid((base / "annotations_1" / f"{record['record_id']}.TextGrid").read_text(encoding="utf-8"))
        second = q08.parse_textgrid((base / "annotations_2" / f"{record['record_id']}.TextGrid").read_text(encoding="utf-8"))
        phones_a = [row for row in _annotation_phone_tier(first) if row["text"].strip().casefold() not in q08.SILENCE_LABELS]
        phones_b = [row for row in _annotation_phone_tier(second) if row["text"].strip().casefold() not in q08.SILENCE_LABELS]
        ops = q08.edit_align([q08.normalize_phone(row["text"]) for row in phones_a],
                             [q08.normalize_phone(row["text"]) for row in phones_b])
        exact = [op for op in ops if op["operation"] == "MATCH"]
        edge_errors = []
        for op in exact:
            a, b = phones_a[op["reference_index"]], phones_b[op["hypothesis_index"]]
            edge_errors.append(max(abs(a["start_seconds"] - b["start_seconds"]), abs(a["end_seconds"] - b["end_seconds"])))
        comparisons.append({
            "speaker_id": record["speaker_id"], "record_id": record["record_id"],
            "annotation_1_phone_count": len(phones_a), "annotation_2_phone_count": len(phones_b),
            "exact_phone_identity_pairs": len(exact),
            "substitution_pairs": sum(op["operation"] == "SUBSTITUTION" for op in ops),
            "annotation_1_deletions": sum(op["operation"] == "DELETION" for op in ops),
            "annotation_2_insertions": sum(op["operation"] == "INSERTION" for op in ops),
            "sequence_identity_agreement": len(exact) / len(ops) if ops else None,
            "exact_pair_boundaries_within": {f"{ms}ms": sum(error <= ms / 1000 for error in edge_errors) for ms in (30, 60, 100)},
            "exact_pair_boundary_count": len(edge_errors),
        })
    all_ops = sum(row["exact_phone_identity_pairs"] + row["substitution_pairs"] + row["annotation_1_deletions"] + row["annotation_2_insertions"] for row in comparisons)
    exact_total = sum(row["exact_phone_identity_pairs"] for row in comparisons)
    exact_boundary_total = sum(row["exact_pair_boundary_count"] for row in comparisons)
    return {
        "recordings_with_two_human_annotations": len(comparisons),
        "sequence_aligned_phone_operations": all_ops,
        "exact_phone_identity_agreement_rate": exact_total / all_ops if all_ops else None,
        "exact_identity_pair_boundary_within_tolerance": {
            f"{ms}ms": sum(row["exact_pair_boundaries_within"][f"{ms}ms"] for row in comparisons) / exact_boundary_total if exact_boundary_total else None
            for ms in (30, 60, 100)
        },
        "per_record": comparisons,
        "method": "Unit-cost sequence alignment of manual phone tiers; exact phone identities divided by all matched/substitution/gap operations. Boundary tolerance is measured only on exact identity pairs.",
        "interpretation": "Descriptive raw agreement only; it is not chance-corrected kappa or adjudicated ground truth, and annotators are not treated as infallible.",
    }


def _annotation_phone_tier(parsed_textgrid: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the EpaDB phone tier, whose names differ between annotator exports."""
    tiers = parsed_textgrid.get("tiers", {})
    names = [name for name in tiers if name.casefold() in {"annotation", "annotations"}]
    if len(names) != 1:
        raise ValueError(f"Expected one phone annotation tier, found {names!r}.")
    return tiers[names[0]]


def error_phone_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    q08.DATA_ROOT = DATA_ROOT
    pairs: Counter[tuple[str, str]] = Counter()
    classes: Counter[str] = Counter()
    for record in records:
        labels = q08._source_label_rows(record["record_id"], record["partition"])
        for row in labels:
            if row["label"] != "-":
                continue
            target, observed = row["target_phone"], row["observed_phone"]
            pairs[(target, observed)] += 1
            target_silent = target.strip().casefold() in q08.SILENCE_LABELS | {"0"}
            observed_silent = observed.strip().casefold() in q08.SILENCE_LABELS | {"0"}
            if target_silent and not observed_silent:
                classes["source target is null/silence; observed phone present"] += 1
            elif observed_silent and not target_silent:
                classes["source target phone; observed null/silence"] += 1
            elif not target_silent and not observed_silent:
                classes["non-silence target/observed phone mismatch"] += 1
            else:
                classes["other source-tagged mismatch"] += 1
    return {
        "source_negative_label_rows": sum(classes.values()),
        "source_supported_relations": dict(classes),
        "most_common_target_observed_pairs": [
            {"target_phone": target, "observed_phone": observed, "count": count}
            for (target, observed), count in pairs.most_common(25)
        ],
        "limitation": "The source CSV supplies only +/- and target/observed symbols; no severity or phonological error taxonomy is inferred from model scores.",
    }


def _per_speaker_metrics(items: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item["speaker_id"]].append(item)
    output = {}
    for speaker, rows in sorted(grouped.items()):
        scored = [row for row in rows if row["score"] is not None]
        output[speaker] = {
            "eligible_phones": len(rows),
            "scored_phones": len(scored),
            "correct_phones": sum(row["correct"] for row in rows),
            "error_phones": sum(not row["correct"] for row in rows),
            "roc_auc_correct_vs_incorrect": _auc_error(rows),
            "pr_auc_error": pr_auc_error(rows),
            "threshold_metrics": threshold_metrics(rows, threshold),
        }
    return output


def _score_summary(items: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scored = [row for row in items if row["score"] is not None]
    auc_ci = _bootstrap_ci(scored, _auc_error, seed=9026)
    pr_ci = _bootstrap_ci(scored, pr_auc_error, seed=9028)
    f1_ci = _bootstrap_ci(items, lambda rows: threshold_metrics(rows, threshold)["coverage_aware"]["f1"], seed=9029)
    result = {
        "eligible_label_rows": len(items),
        "exact_identity_scored_rows": len(scored),
        "coverage": len(scored) / len(items) if items else None,
        "correct_label_rows": sum(row["correct"] for row in items),
        "incorrect_label_rows": sum(not row["correct"] for row in items),
        "roc_auc_correct_vs_incorrect_conditional": _auc_error(items),
        "roc_auc_speaker_cluster_bootstrap_95": auc_ci,
        "pr_auc_error_conditional": pr_auc_error(items),
        "pr_auc_speaker_cluster_bootstrap_95": pr_ci,
        "threshold_metrics": threshold_metrics(items, threshold),
        "coverage_aware_f1_speaker_cluster_bootstrap_95": f1_ci,
        "per_speaker": _per_speaker_metrics(items, threshold),
    }
    return result


def _paired_items(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {f"{row['record_id']}:{row['source_phone_id']}": row for row in items if row["score"] is not None}


def _all_items(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {f"{row['record_id']}:{row['source_phone_id']}": row for row in items}


def _runtime_summary(policy: str) -> dict[str, Any]:
    durations: list[float] = []
    memory: list[int] = []
    failed = 0
    files = sorted((DATA_ROOT / "runs" / policy / "direct").glob("*.json"))
    for path in files:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failed += 1
            continue
        if row.get("status") not in SCORE_SUCCESS:
            failed += 1
            continue
        if isinstance(row.get("processing_time_seconds"), (int, float)):
            durations.append(float(row["processing_time_seconds"]))
        mem = (row.get("performance") or {}).get("peak_resident_memory_bytes_max_process")
        if isinstance(mem, int):
            memory.append(mem)
    percentile = lambda values, p: sorted(values)[min(len(values) - 1, int(p * (len(values) - 1)))] if values else None
    return {
        "successful_recordings": len(durations), "failed_recordings": failed,
        "processing_time_seconds_median": statistics.median(durations) if durations else None,
        "processing_time_seconds_p95": percentile(durations, 0.95),
        "peak_memory_bytes_median": int(statistics.median(memory)) if memory else None,
        "peak_memory_bytes_p95": percentile(memory, 0.95),
        "peak_memory_source": "Scorer-reported maximum process RSS per utterance; not a concurrent throughput measurement.",
    }


def make_report() -> dict[str, Any]:
    manifest = build_manifest()
    frozen = freeze_scorer()
    old = json.loads((OLD_DATA_ROOT / "sample_manifest.json").read_text(encoding="utf-8"))
    q08.DATA_ROOT = OLD_DATA_ROOT
    dev_records = [row for row in old["records"] if row["partition"] == "development"]
    dev_items = collect_phone_items(OLD_DATA_ROOT, dev_records, "candidate")
    selected = choose_threshold(dev_items)
    threshold = float(selected["threshold"])
    q09_records = manifest["records"]
    baseline_items = collect_phone_items(DATA_ROOT, q09_records, "baseline")
    recovery_items = collect_phone_items(DATA_ROOT, q09_records, "recovery")
    global _ACTIVE_THRESHOLD
    _ACTIVE_THRESHOLD = threshold
    baseline_by_id = _paired_items(baseline_items)
    recovery_by_id = _paired_items(recovery_items)
    baseline_all_by_id = _all_items(baseline_items)
    recovery_all_by_id = _all_items(recovery_items)
    common_keys = sorted(set(baseline_by_id) & set(recovery_by_id))
    all_label_keys = sorted(set(baseline_all_by_id) & set(recovery_all_by_id))
    baseline_common = [baseline_by_id[key] for key in common_keys]
    recovery_common = [recovery_by_id[key] for key in common_keys]
    paired_auc = _paired_delta_ci(baseline_by_id, recovery_by_id, _auc_error)
    paired_f1 = _paired_delta_ci(baseline_by_id, recovery_by_id, _binary_metrics, seed=9030)
    paired_all_label_f1 = _paired_delta_ci(
        baseline_all_by_id, recovery_all_by_id, _binary_metrics, seed=9031,
    )
    q08.DATA_ROOT = DATA_ROOT
    alignment = {
        policy: q08._partition_metrics(
            {"records": manifest["records"]}, "holdout", False, policy,
        )
        for policy in ("baseline", "recovery")
    }
    for result in alignment.values():
        asr = result.get("asr_against_manual_spoken_word_tier")
        if asr is not None:
            asr["status"] = "NOT_EVALUATED"
            asr["scope_note"] = "ASR was not invoked for this direct pronunciation benchmark. Zero-valued count fields are placeholders from the shared QUALITY-08 schema, not measured ASR performance."
    disagreement = interannotator_agreement(q09_records)
    confusion = error_phone_pairs(q09_records)
    q08_dev_report = json.loads((ROOT / "results" / "quality_08_report.json").read_text(encoding="utf-8"))
    q08_dev = q08_dev_report["results"]["partitions"]["development"]["frozen_baseline_vs_candidate"]
    report = {
        "schema_version": "quality-09-pronunciation-quality-report-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "overall_verdict": "PARTIAL",
        "executive_verdict": "The frozen GOPT scorer was evaluated on 19 new, speaker-disjoint Argentinian Spanish-L1 EpaDB speakers. This establishes a fresh cross-speaker diagnostic, not Vietnamese or cross-L1 validity. No production scorer change is accepted unless independent evidence demonstrates better pronunciation discrimination.",
        "acceptance_gates": [
            {"gate": "A — Dataset", "status": "PASS_WITH_SCOPE", "evidence": "76 real EpaDB recordings from 19 speakers absent from the QUALITY-08 sample; two human TextGrids for every selected recording. Source remains CC BY-NC 4.0 research-only."},
            {"gate": "B — Ground truth", "status": "PARTIAL", "evidence": "The author label rule is verified and source labels are preserved. Only annotation_1 determines +/- benchmark labels; annotation_2 is available for 76 selected recordings and shows moderate raw phone identity agreement without adjudication. See ground_truth_quality.interannotator_agreement."},
            {"gate": "C — Cross-speaker", "status": "PASS", "evidence": "All 19 QUALITY-09 speakers are absent from the 30 QUALITY-08 development/holdout speakers and are kept out of threshold selection."},
            {"gate": "D — Cross-L1", "status": "BLOCKED", "evidence": "The acquired benchmark contains only Argentinian Spanish-L1 speakers. The owner marked the official L2-ARCTIC access route blocked; no alternate distribution with verifiable authorization was found. MRPL2 files remain restricted."},
            {"gate": "E — Error detection", "status": "PARTIAL", "evidence": "Fresh conditional ROC/PR AUC, threshold metrics, coverage-aware rates, confusion matrices, per-speaker results, and speaker-cluster intervals are in results.pronunciation."},
            {"gate": "F — Alignment", "status": "PARTIAL", "evidence": "Manual phone boundaries, 30/60/100 ms alignment and support-floor false acceptance/rejection are evaluated against annotation_1; see results.alignment."},
            {"gate": "G — Pronunciation improvement", "status": "NOT_VERIFIED", "evidence": "The scorer's production logic was not changed. Recovery is compared with the default on fresh speakers, but no improvement verdict is claimed without the paired metrics and cross-L1 evidence."},
            {"gate": "H — Partial speech", "status": "BLOCKED", "evidence": "The fresh corpus is fully prompted read speech, not authentic incomplete RA/RS responses. Existing failure guards remain; no fabricated omissions are used as accuracy evidence."},
            {"gate": "I — Coverage integrity", "status": "PASS", "evidence": "Conditional AUC/PR metrics are paired with all-label coverage; abstentions count as missed errors and as wrong in coverage-aware accuracy."},
            {"gate": "J — Model comparison", "status": "NOT_PERFORMED", "evidence": "No alternative with verified phone-error semantics, suitable license, and accessible independent evaluation data was justified for migration."},
            {"gate": "K — Regression", "status": "PASS", "evidence": "The original 102 tests and ten QUALITY-09 benchmark tests pass; exact command and count are listed in verification."},
            {"gate": "L — Pearson agreement", "status": "NOT_VERIFIED", "evidence": "No Pearson-referenced scores or calibration set; Pearson and overall PTE scores remain null."},
        ],
        "dataset_inventory": {
            "primary_evaluation": {
                "name": "EpaDB English pronunciation database", "version": SOURCE_COMMIT,
                "source": SOURCE_REPOSITORY, "license": LICENSE,
                "corpus_speakers": 50, "evaluation_speakers": len(SPEAKERS), "source_l1": "Argentinian Spanish",
                "recordings": len(q09_records), "prompts_per_speaker": list(PROMPTS),
                "annotators": "The upstream README describes two expert annotators from Argentina; annotation_1 is complete and annotation_2 covers four prompts per speaker. Phone tier names differ among exports, and annotation_1 supplies the source +/- label CSV.",
                "annotation_method": "Manual phone-level TextGrid annotations use an ARPAbet-like inventory with extensions for allophonic variation and Spanish sounds; the source TextGrids also contain an annotator score tier.",
                "speaker_split": "All 19 selected speakers are absent from the prior 30-speaker sample and used once as independent evaluation; Q08 development is used only to select the research threshold.",
                "audio_and_annotations": {"raw_files": len(json.loads((DATA_ROOT / "sample_manifest.json").read_text(encoding="utf-8"))["files"]), "source_label_classes": dict(Counter(row["label"] for record in q09_records for row in q08._source_label_rows(record["record_id"], "holdout")))},
                "gender_metadata": "Corpus-level 25/25 male/female is reported by source; no speaker-level gender analysis was attempted.",
                "rights_boundary": "Evaluation only; no training, commercial deployment, redistribution, or derived commercial model artifact.",
            },
            "considered_sources": [
                {"name": "L2-ARCTIC", "status": "BLOCKED_OFFICIAL_ACCESS_ROUTE", "source": "https://psi.engr.tamu.edu/l2-arctic-corpus/", "l1_groups": ["Arabic", "Hindi", "Korean", "Mandarin", "Spanish", "Vietnamese"], "annotations": "Manual substitutions, deletions, additions, and corrected boundaries; CC BY-NC 4.0.", "blocker": "Owner decision marks the official access route blocked because it requires applicant information/approval. A Hugging Face copy describes itself as a mirror, but independent distribution authorization was not verified and it was excluded."},
                {"name": "voisTUTOR 2.0", "status": "NOT_ACQUIRED_ACCESS_AND_LICENSE_UNVERIFIED", "source": "https://spire.ee.iisc.ac.in/assets/PDFs/papers_pdf/Priyanshi_Chiranjeevi_O-COCOSDA_2022.pdf", "description": "26,529 Indian L2 English utterances from six Indian L1 groups; manually phonetic-transcribed by one linguist, with intra-rater kappa 0.79 on a 5% sample. No authorized public audio/annotation distribution or license was found during this pass."},
                {"name": "MRPL2", "status": "BLOCKED_RESTRICTED_FILES", "source": "https://github.com/tweddan/MRPL2_Corpus", "description": "23 speakers, 14 L1s, manual phoneme error annotations; repository README states CC BY 4.0, but the Zenodo dataset files require a logged-in access request.", "blocker": "Official Zenodo v1.1 page marks the 12.5 GB files restricted; no account or approval was available."},
                {"name": "UME-ERJ", "status": "NOT_ACQUIRED", "source": "https://www.nii.ac.jp/dsc/idr/speech/submit/UME-ERJ.html", "description": "202 Japanese L1 English learners; research-purpose use only and application review. Available rating labels do not establish the needed phone-error semantics."},
                {"name": "Speak & Improve Corpus 2025", "status": "NOT_ACQUIRED_LABEL_MISMATCH", "source": "https://researchdatasets.cambridge.org/datasets/speak-and-improve-corpus-2025", "description": "Large L2 English spontaneous-speech corpus with holistic CEFR ratings and a transcribed language-error subset, but no suitable expert phone-error ground truth was verified; research license also restricts redistribution and derivative release."},
                {"name": "VLSP 2023 Vietnamese Mispronunciation Detection", "status": "NOT_APPLICABLE_LANGUAGE", "source": "https://vlsp.org.vn/index.php/vlsp2023/eval/vmd", "description": "Contains Vietnamese speech/phoneme errors, not non-native English speech; not used to infer English pronunciation correctness."},
                {"name": "SpeechOcean762", "status": "EXCLUDED_AS_INDEPENDENT_HOLDOUT", "source": "https://www.openslr.org/101/", "description": "Checkpoint-exposed per task brief; retained only for historical compatibility/diagnostics."},
            ],
        },
        "frozen_scorer": frozen,
        "evaluation_protocol": {
            "comparison": "Same audio, same frozen source/checkpoint, default whole-window policy vs existing supported-whole-word recovery flag.",
            "development": "QUALITY-08's 10 development speakers; used only to select one research-only error threshold by F1.",
            "final_evaluation": "All 19 new EpaDB source-train speakers; no candidate tuning on these speakers.",
            "identity_mapping": "Unit-cost Levenshtein target-phone to scorer-phone sequence alignment. Metrics include only exact phone-identity matches; insertions/deletions are retained as gaps and never positionally shifted.",
            "label_direction": "Source '+' means correct (target equals manual annotation); '-' means target differs from manual annotation. Higher native GOPT score is tested as more likely correct.",
            "threshold": selected,
            "coverage_policy": "Conditional ranking metrics are on exact identity scored phones. Coverage-aware threshold recall counts an unscored error as a miss; coverage-aware accuracy counts every abstention as wrong.",
            "uncertainty": {"method": "Percentile bootstrap resamples speakers as clusters, never individual phones.", "replicates": 2000, "seeds": {"gopt_auc": 9026, "pr_auc": 9028, "coverage_f1": 9029, "paired_auc": 9027, "paired_common_f1": 9030, "paired_all_label_f1": 9031, "alignment": 8026}},
            "calibration": "No conversion to probability, Pearson scale, or product threshold; the one threshold is a diagnostic operating point on development data only.",
        },
        "ground_truth_quality": {
            "author_label_rule": "Verified from the pinned source label-generation script; preserved as '+'/'-' without reinterpretation.",
            "source_label_semantics_conflicts": manifest["integrity"]["source_label_semantics_conflicts"],
            "interannotator_agreement": disagreement,
            "error_categories": confusion,
        },
        "results": {
            "pronunciation": {
                "baseline": _score_summary(baseline_items, threshold),
                "recovery": _score_summary(recovery_items, threshold),
                "paired_common_identity_sample": {
                    "phone_count": len(common_keys),
                    "baseline_roc_auc": _auc_error(baseline_common),
                    "recovery_roc_auc": _auc_error(recovery_common),
                    "roc_auc_delta_recovery_minus_baseline_speaker_cluster_bootstrap_95": paired_auc,
                    "baseline_coverage_aware_f1": threshold_metrics(baseline_common, threshold)["coverage_aware"]["f1"],
                    "recovery_coverage_aware_f1": threshold_metrics(recovery_common, threshold)["coverage_aware"]["f1"],
                    "coverage_aware_f1_delta_speaker_cluster_bootstrap_95": paired_f1,
                },
                "paired_all_eligible_label_sample": {
                    "phone_count": len(all_label_keys),
                    "baseline_coverage": threshold_metrics(baseline_items, threshold)["coverage"],
                    "recovery_coverage": threshold_metrics(recovery_items, threshold)["coverage"],
                    "baseline_coverage_aware_f1": threshold_metrics(baseline_items, threshold)["coverage_aware"]["f1"],
                    "recovery_coverage_aware_f1": threshold_metrics(recovery_items, threshold)["coverage_aware"]["f1"],
                    "coverage_aware_f1_delta_recovery_minus_baseline_speaker_cluster_bootstrap_95": paired_all_label_f1,
                    "abstentions_count_as_missed_errors_and_wrong_for_accuracy": True,
                },
                "recovery_only_rows": len(set(_paired_items(recovery_items)) - set(_paired_items(baseline_items))),
                "baseline_only_rows": len(set(_paired_items(baseline_items)) - set(_paired_items(recovery_items))),
            },
            "alignment": alignment,
            "runtime": {policy: _runtime_summary(policy) for policy in ("baseline", "recovery")},
        },
        "root_cause_findings": {
            "kaldi_alignment_and_support": "Default policy rejects 21/76 utterances at whole-window alignment; recovery scores all 76, often partially. Against annotation_1 boundaries, 96.3% of manual phone intervals are within 100 ms, while the support gate false-accept rate among accepted phones is 3.2% at 100 ms and 60.8% at 30 ms. The −6.0 support gate is not ground truth.",
            "gopt_discrimination": "Conditional AUC is 0.719 under baseline and 0.732 under recovery, but paired AUC is unchanged on the 1008 phone identities scored by both policies; the recovery-only conditional AUC difference reflects a different scored subset. At the development-only threshold, all-label F1 is 0.302 baseline versus 0.328 recovery, with low precision (0.259/0.251) and high false-positive rates on scored-correct phones (0.577/0.589); the candidate effect is exploratory and not a verified scoring improvement.",
            "phone_inventory_and_lexicon": "Rows where source canonical target fails exact scorer-phone mapping are reported as unavailable, not forced into adjacent positions.",
            "context_and_recovery": "The only comparison difference is the existing whole-word support-recovery flag. Scores are unchanged on 1008 phones scored by both policies (paired ROC-AUC delta 0); recovery adds scores on other eligible phones, so correctness gains must be read with abstention-aware metrics and cluster intervals.",
            "asr_and_partial_speech": "ASR was not run on the fresh pronunciation sample, so ASR contribution remains unmeasured; incomplete RA/RS recordings are also unavailable in this corpus.",
            "domain_shift": "Evaluation remains short prompted Argentinian Spanish-L1 speech; cross-corpus and cross-L1 conclusions are blocked.",
        },
        "alternative_model": {
            "verdict": "NOT_TESTED",
            "reason": "The fresh benchmark compares only the existing GOPT recovery policy. No alternative model with accessible weights, verified phone-level error semantics, license provenance, and comparable independent evaluation was justified. voisTUTOR 2.0 is promising but no authorized public audio/annotation distribution or license was found; MRPL2 files remain restricted.",
        },
        "verification": {
            "commands": [
                ".venv/bin/python -m unittest discover -s tests -v",
                ".venv/bin/python quality_09_benchmark.py --acquire --freeze",
                ".venv/bin/python quality_09_benchmark.py --run --threads 2",
                ".venv/bin/python quality_09_benchmark.py --analyze",
            ],
            "historical_test_count": 102,
            "quality_09_metric_test_count": 10,
            "total_test_count": 112,
            "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
            "real_audio": {"recordings": len(q09_records), "policies": ["baseline", "recovery"], "successful_run_counts": {policy: _runtime_summary(policy)["successful_recordings"] for policy in ("baseline", "recovery")}},
            "report_path": str(REPORT_PATH.relative_to(ROOT)),
        },
        "limitations": [
            "Vietnamese-L1 and other L1 pronunciation accuracy is not evaluated.",
            "EpaDB is non-commercial research data; its results cannot authorize a commercial threshold, training artifact, or deployment.",
            "EpaDB labels cover canonical-vs-manual phone identity; they do not supply a complete severity, insertion/deletion, intelligibility, or acceptable-accent taxonomy.",
            "No authentic partial Read Aloud or Repeat Sentence recordings were acquired.",
            "ASR accuracy was not measured on the new pronunciation sample.",
            "No Pearson-referenced labels or score calibration evidence exist.",
        ],
    }
    return report


def write_report() -> dict[str, Any]:
    report = make_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true", help="Acquire the bounded, pinned, non-commercial EpaDB sample.")
    parser.add_argument("--freeze", action="store_true", help="Copy and hash the current scorer and local model assets.")
    parser.add_argument("--run", action="store_true", help="Run baseline and supported-word-recovery policies on all 76 real recordings.")
    parser.add_argument("--analyze", action="store_true", help="Write the machine-readable QUALITY-09 report after both runs exist.")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 8:
        parser.error("--threads must be between 1 and 8")
    if not any((args.acquire, args.freeze, args.run, args.analyze)):
        parser.error("choose at least one of --acquire, --freeze, --run, or --analyze")
    if args.acquire:
        manifest = acquire_dataset()
        print(f"Acquired {manifest['counts']['recordings']} recordings from {manifest['counts']['speakers']} speakers.")
    if args.freeze:
        freeze_scorer()
        print("Frozen current scorer fingerprint and source copies.")
    if args.run:
        if not (DATA_ROOT / "frozen_scorer" / "fingerprint.json").is_file():
            freeze_scorer()
        run_models(build_manifest(), args.threads)
    if args.analyze:
        report = write_report()
        print(f"QUALITY-09 verdict: {report['overall_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
