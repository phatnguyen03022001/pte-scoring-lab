"""Frozen, local-only pronunciation evaluation for L2-ARCTIC v5.0.

The corpus is CC BY-NC 4.0. Audio, TextGrids, manifests containing selected
utterance IDs, scorer outputs, and derived results stay under ignored .local/.
This tool evaluates the frozen scorer; it never changes model behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import quality_08_benchmark as q08
import quality_09_benchmark as q09
from process_safety import run_bounded


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / ".local/datasets/l2-arctic-v5.0/extracted"
DATA_MANIFEST = ROOT / ".local/datasets/l2-arctic-v5.0/extraction.json"
SOURCE_MANIFEST = ROOT / ".local/datasets/l2-arctic-v5.0/acquisition.json"
PROTOCOL_PATH = ROOT / ".local/research/l2-arctic-task-03-1-protocol.json"
OUTPUT_ROOT = ROOT / ".local/research/l2-arctic-task-03-1"
GOPT_PYTHON = ROOT / ".local/gopt-venv/bin/python"
SCORE_SCRIPT = ROOT / "score_pronunciation.py"
SCORE_SUCCESS = {"SCORING_COMPLETED", "SCORING_COMPLETED_PARTIAL"}
THRESHOLD = 1.986756  # Frozen QUALITY-09 threshold selected on QUALITY-08 development speakers.
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AX", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
VIETNAMESE = ("HQTV", "PNV", "THV", "TLV")
L1 = {
    "ABA": "Arabic", "SKA": "Arabic", "YBAA": "Arabic", "ZHAA": "Arabic",
    "BWC": "Chinese", "LXC": "Chinese", "NCC": "Chinese", "TXHC": "Chinese",
    "ASI": "Hindi", "RRBI": "Hindi", "SVBI": "Hindi", "TNI": "Hindi",
    "HJK": "Korean", "HKK": "Korean", "YDCK": "Korean", "YKWK": "Korean",
    "EBVS": "Spanish", "ERMS": "Spanish", "MBMPS": "Spanish", "NJS": "Spanish",
    "HQTV": "Vietnamese", "PNV": "Vietnamese", "THV": "Vietnamese", "TLV": "Vietnamese",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_phone(text: str) -> str:
    """Drop ARPAbet stress and the documented foreign-accent deviation marker."""
    phone = re.sub(r"\s+", "", text).upper()
    phone = phone.replace("*", "")
    return re.sub(r"[012]$", "", phone)


def parse_manual_phones(textgrid: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Parse only official human manual-tier semantics; never use forced tiers."""
    grid = q08.parse_textgrid(textgrid)
    if not {"words", "phones"}.issubset(grid["tiers"]):
        raise ValueError("Manual annotation must contain both words and phones tiers.")
    eligible: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, interval in enumerate(grid["tiers"]["phones"]):
        raw = interval["text"].strip()
        if not raw or raw.casefold() in q08.SILENCE_LABELS:
            counts["unlabeled_or_silence_intervals"] += 1
            continue
        fields = [re.sub(r"\s+", "", value) for value in raw.split(",")]
        if len(fields) == 1:
            target = canonical_phone(fields[0])
            category, observed, is_error = "unchanged_correct", target, False
        elif len(fields) == 3 and fields[2].casefold() in {"s", "d", "a"}:
            target, observed, category = canonical_phone(fields[0]), canonical_phone(fields[1]), fields[2].casefold()
            if category == "a":
                if target != "SIL":
                    raise ValueError(f"Insertion tag must use a silence target: {raw!r}")
                counts["insertion_unscored"] += 1
                continue  # An inserted phone has no target-phone score in this reference-conditioned model.
            is_error = True
            if category == "s":
                category = "substitution"
                if observed == "ERR":
                    counts["ambiguous_perceived_phone"] += 1
            elif category == "d":
                if observed != "SIL":
                    raise ValueError(f"Deletion tag must use a silence observation: {raw!r}")
                category = "deletion"
        else:
            raise ValueError(f"Unrecognized manual phone annotation: {raw!r}")
        if target in {"", "SIL", "SP", "PAU"}:
            raise ValueError(f"Ground-truth target is not a non-silence phone: {raw!r}")
        eligible.append({
            "source_phone_id": str(index), "target_phone": target,
            "observed_phone": observed, "category": category,
            "correct": not is_error,
            "start_seconds": interval["start_seconds"],
            "end_seconds": interval["end_seconds"],
        })
        counts[category] += 1
    return eligible, counts


def _deterministic_sample(speaker: str, ids: list[str], count: int) -> list[str]:
    key = lambda record_id: hashlib.sha256(f"TASK-03.1|L2-ARCTIC-5.0|{speaker}|{record_id}".encode()).hexdigest()
    return sorted(sorted(ids, key=key)[:count])


def _speaker_inventory() -> dict[str, list[Path]]:
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Missing extracted L2-ARCTIC files at {DATA_ROOT}.")
    inventory: dict[str, list[Path]] = {}
    extraction = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    declared = {row["archive"][:-4]: row for row in extraction["records"] if row["archive"].endswith(".zip")}
    for speaker in sorted(L1):
        base = DATA_ROOT / speaker
        grids = sorted((base / "annotation").glob("*.TextGrid"))
        wavs = sorted((base / "wav").glob("*.wav"))
        if not grids:
            raise ValueError(f"No manual annotation TextGrids found for {speaker}.")
        if speaker not in declared or len(wavs) != declared[speaker]["wav_files"]:
            raise ValueError(f"Extracted audio count does not match the extraction manifest for {speaker}.")
        expected_grids = 149 if speaker == "YBAA" else 150
        if len(grids) != expected_grids:
            raise ValueError(f"Expected {expected_grids} documented manual TextGrids for {speaker}, found {len(grids)}.")
        ids = [path.stem for path in grids]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate manual annotation IDs for {speaker}.")
        for record_id in ids:
            for path in (base / "wav" / f"{record_id}.wav", base / "transcript" / f"{record_id}.txt"):
                if not path.is_file():
                    raise ValueError(f"Manual annotation has no matching source file: {path}")
            # Fail before freezing if any human label cannot be interpreted safely.
            grid_text = (base / "annotation" / f"{record_id}.TextGrid").read_text(encoding="utf-8")
            grid = q08.parse_textgrid(grid_text)
            parse_manual_phones(grid_text)
            with wave.open(str(base / "wav" / f"{record_id}.wav"), "rb") as audio:
                if audio.getframerate() != 44100 or audio.getnchannels() != 1:
                    raise ValueError(f"Unexpected source WAV format for {speaker}/{record_id}.")
                wav_duration = audio.getnframes() / audio.getframerate()
            if grid["duration_seconds"] > wav_duration + 0.02:
                raise ValueError(f"Manual annotation extends past audio for {speaker}/{record_id}.")
        inventory[speaker] = grids
    if len(inventory) != 24:
        raise ValueError(f"Expected 24 documented speakers, found {len(inventory)}.")
    return inventory


def prepare_protocol() -> dict[str, Any]:
    """Validate and freeze deterministic eligibility before any scorer call."""
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    extraction_manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    if source_manifest.get("dataset") != "L2-ARCTIC" or source_manifest.get("version") != "5.0":
        raise ValueError("Acquisition manifest does not identify L2-ARCTIC v5.0.")
    if sha256_file(ROOT / ".local/datasets/l2-arctic-v5.0/l2arctic_release_v5.0.zip") != source_manifest["archive_sha256"]:
        raise ValueError("Original L2-ARCTIC archive hash differs from acquisition manifest.")
    if len(extraction_manifest.get("records", [])) != 25 or extraction_manifest.get("speakers") != 24:
        raise ValueError("Extraction manifest does not match the documented 24-speaker release.")
    inventory = _speaker_inventory()
    selected: list[dict[str, Any]] = []
    all_annotation_counts: dict[str, dict[str, int]] = {}
    for speaker, grids in inventory.items():
        by_id = {path.stem: path for path in grids}
        chosen_ids = sorted(by_id) if speaker in VIETNAMESE else _deterministic_sample(speaker, sorted(by_id), 20)
        for record_id in chosen_ids:
            base = DATA_ROOT / speaker
            audio = base / "wav" / f"{record_id}.wav"
            transcript = base / "transcript" / f"{record_id}.txt"
            annotation = by_id[record_id]
            labels, counts = parse_manual_phones(annotation.read_text(encoding="utf-8"))
            if not labels:
                raise ValueError(f"No eligible manual target phones in {speaker}/{record_id}.")
            selected.append({
                "speaker_id": speaker, "l1": L1[speaker], "record_id": record_id,
                "audio_path": str(audio.relative_to(ROOT)),
                "transcript_path": str(transcript.relative_to(ROOT)),
                "annotation_path": str(annotation.relative_to(ROOT)),
                "audio_sha256": sha256_file(audio),
                "transcript_sha256": sha256_file(transcript),
                "annotation_sha256": sha256_file(annotation),
                "eligible_target_phones": len(labels),
                "label_counts": dict(sorted(counts.items())),
            })
        if speaker in VIETNAMESE:
            all_annotation_counts[speaker] = {
                "human_annotated_recordings": len(grids),
                "selected_recordings": len(chosen_ids),
            }
    protocol = {
        "schema_version": "l2-arctic-pronunciation-v1",
        "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
        "dataset": {"name": "L2-ARCTIC", "version": "5.0", "license": "CC BY-NC 4.0",
                    "archive_sha256": source_manifest["archive_sha256"],
                    "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
                    "extraction_manifest_sha256": sha256_file(DATA_MANIFEST)},
        "human_annotation_semantics": {
            "tier": "manual annotation/phones only; forced-alignment textgrid is not used as ground truth",
            "unchanged_label": "human-verified correct according to corpus README",
            "s": "human-annotated substitution; target phone is incorrect",
            "d": "human-annotated deletion; target phone is incorrect",
            "a": "human-annotated insertion; excluded from phone-score metrics because no target phone exists",
            "err": "retained as an error when it is the perceived side of a human-tagged substitution; ambiguous subtype retained separately",
            "unannotated_silence": "excluded; absence of a phone label is not treated as a correct pronunciation",
        },
        "selection": {
            "vietnamese": "all available manual annotation utterances from HQTV, PNV, THV, TLV",
            "cross_l1": "20 utterances per each of the other 20 speakers; lowest SHA-256 rank over the fixed task/dataset/speaker/utterance key; no score-dependent selection",
            "selected_recordings": len(selected),
            "selected_recording_ids_and_file_fingerprints": selected,
            "vietnamese_inventory": all_annotation_counts,
        },
        "model_exposure": {
            "checkpoint_source": "official GOPT LibriSpeech checkpoint; its documented training corpus is LibriSpeech",
            "acoustic_model_source": "Kaldi LibriSpeech chain model",
            "local_project_evidence": "TASK-02 and TASK-03 records state L2-ARCTIC was not acquired before 2026-11-11; no L2-ARCTIC training, checkpoint selection, experiment, or threshold tuning reference found in tracked source/history before acquisition",
            "status": "NO_LOCAL_EVIDENCE_OF_EXPOSURE; external exposure beyond recorded provenance cannot be disproved",
        },
        "frozen_scorer": {key: value for key, value in q09.scorer_fingerprint().items() if key != "frozen_at"},
        "decision": {"error_when_native_gopt_phone_score_le": THRESHOLD,
                     "source": "QUALITY-09 threshold selected on QUALITY-08 development speakers only; not refit on L2-ARCTIC",
                     "support_floor": -6.0,
                     "policy": "default scorer behavior; no supported-word recovery flag; one sequential process at a time; one scorer invocation per utterance"},
        "metrics": {
            "ranking": ["ROC-AUC", "PR-AUC"],
            "threshold": ["precision", "recall", "F1", "confusion_matrix", "false_positive_rate", "false_negative_rate"],
            "coverage": "conditional ranking and confusion metrics on exact target-phone matches; coverage-aware metrics count abstained errors as misses; abstention rate and unsupported target phones reported",
            "alignment": "unit-cost Levenshtein sequence alignment by canonical phone identity; mismatches/gaps are unsupported, never positionally shifted",
            "uncertainty": "speaker-cluster bootstrap on pooled all-L1 sample only; no confidence interval for Vietnamese 4-speaker subgroup",
        },
        "limits": ["read-speech only", "manual subset only", "four Vietnamese speakers", "insertion detection not evaluated", "source transcripts condition the pronunciation scorer", "not end-to-end PTE score validation"],
        "protocol_fingerprint": None,
    }
    serial = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    protocol["protocol_fingerprint"] = hashlib.sha256(serial.encode("utf-8")).hexdigest()
    PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_PATH.exists():
        previous = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if previous.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            raise RuntimeError(f"Existing frozen protocol differs; refusing to overwrite: {PROTOCOL_PATH}")
        return previous
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return protocol


def run_inference(protocol: dict[str, Any], max_seconds: int = 3600) -> dict[str, Any]:
    """Run bounded, sequential inference with cache reuse and per-child cleanup."""
    if protocol.get("protocol_fingerprint") != json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")).get("protocol_fingerprint"):
        raise RuntimeError("The supplied evaluation protocol is not the frozen local protocol.")
    if protocol.get("benchmark_script_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Benchmark implementation changed after protocol freeze.")
    now_fingerprint = q09.scorer_fingerprint()
    frozen = protocol["frozen_scorer"]
    if now_fingerprint["source_sha256"] != frozen["source_sha256"]:
        raise RuntimeError("Scorer code or model artifacts changed after protocol freeze.")
    if not GOPT_PYTHON.is_file():
        raise FileNotFoundError(f"Missing GOPT runtime: {GOPT_PYTHON}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    runs = OUTPUT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    start = time.monotonic()
    results: list[dict[str, Any]] = []
    records = protocol["selection"]["selected_recording_ids_and_file_fingerprints"]
    for index, record in enumerate(records, 1):
        destination = runs / record["speaker_id"] / f"{record['record_id']}.json"
        if destination.is_file():
            cached = q08._load_run(destination)
            stamp = (cached or {}).get("_benchmark", {})
            if stamp.get("protocol_fingerprint") == protocol["protocol_fingerprint"] and stamp.get("audio_sha256") == record["audio_sha256"] and stamp.get("annotation_sha256") == record["annotation_sha256"]:
                results.append(cached)
                continue
            raise RuntimeError(f"Existing output does not match frozen inputs; refusing to overwrite: {destination}")
        if time.monotonic() - start >= max_seconds:
            break
        audio = ROOT / record["audio_path"]
        transcript = (ROOT / record["transcript_path"]).read_text(encoding="utf-8").strip()
        command = [str(GOPT_PYTHON), str(SCORE_SCRIPT), "--audio", str(audio), "--reference", transcript, "--threads", "1"]
        try:
            proc = run_bounded(command, capture_output=True, text=True, timeout=420)
            result = json.loads(proc.stdout)
        except subprocess.TimeoutExpired as exc:
            result = {"status": "BENCHMARK_TIMEOUT", "error": str(exc), "phonemes": []}
            proc = None
        except (OSError, json.JSONDecodeError) as exc:
            result = {"status": "BENCHMARK_RUNNER_FAILED", "error": str(exc), "phonemes": []}
            proc = None
        result["_benchmark"] = {
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "audio_sha256": record["audio_sha256"],
            "annotation_sha256": record["annotation_sha256"],
            "exit_code": proc.returncode if proc else None,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        results.append(result)
        print(f"[{index}/{len(records)}] {record['speaker_id']} {record['record_id']} {result.get('status')}", flush=True)
        if time.monotonic() - start >= max_seconds:
            break
    return {"completed": len(results), "selected": len(records), "elapsed_seconds": round(time.monotonic() - start, 3),
            "remaining": len(records) - len(results), "output_root": str(OUTPUT_ROOT)}


def _eligible_for_scoring(record: dict[str, Any]) -> list[dict[str, Any]]:
    path = ROOT / record["annotation_path"]
    labels, _ = parse_manual_phones(path.read_text(encoding="utf-8"))
    return labels


def collect_items(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Any]]:
    items: list[dict[str, Any]] = []
    annotation_categories: Counter[str] = Counter()
    record_statuses: Counter[str] = Counter()
    unsupported_categories: Counter[str] = Counter()
    alignment_counts: Counter[str] = Counter()
    records = protocol["selection"]["selected_recording_ids_and_file_fingerprints"]
    for record in records:
        annotation = _eligible_for_scoring(record)
        counts = Counter(label["category"] for label in annotation)
        annotation_categories.update(counts)
        run = q08._load_run(OUTPUT_ROOT / "runs" / record["speaker_id"] / f"{record['record_id']}.json")
        if not run:
            record_statuses["NOT_RUN"] += 1
            predictions = []
        else:
            record_statuses[run.get("status", "UNKNOWN")] += 1
            predictions = q08._direct_phone_rows(run)
            if run.get("status") not in SCORE_SUCCESS:
                predictions = []
        predicted_symbols = [canonical_phone(row.get("phone", "")) for row in predictions]
        operations = q08.edit_align([row["target_phone"] for row in annotation], predicted_symbols)
        for operation in operations:
            label_index = operation["reference_index"]
            if label_index is None:
                continue
            label = annotation[label_index]
            prediction_index = operation["hypothesis_index"]
            prediction = predictions[prediction_index] if prediction_index is not None else None
            exact = operation["operation"] == "MATCH"
            score = q08._as_number(prediction.get("gopt_phone_score")) if prediction and exact and prediction.get("pronunciation_evidence_status") == "AVAILABLE_UNCALIBRATED" else None
            alignment_counts["exact_identity_match" if exact else "mismatch_or_gap"] += 1
            if prediction and prediction.get("pronunciation_evidence_status") != "AVAILABLE_UNCALIBRATED":
                alignment_counts[f"model_abstention:{prediction.get('pronunciation_evidence_status')}"] += 1
            if score is None:
                unsupported_categories[label["category"]] += 1
            items.append({
                "speaker_id": record["speaker_id"], "l1": record["l1"],
                "record_id": record["record_id"], "source_phone_id": label["source_phone_id"],
                "category": label["category"], "target_phone": label["target_phone"],
                "phone_class": "vowel" if label["target_phone"] in VOWELS else "consonant",
                "correct": label["correct"], "score": score,
                "prediction": ("error" if score is not None and score <= THRESHOLD else "correct" if score is not None else "abstain"),
            })
    return items, annotation_categories, {
        "record_status_counts": dict(record_statuses),
        "unsupported_target_phones_by_category": dict(unsupported_categories),
        "alignment_counts": dict(alignment_counts),
    }


def _metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [item for item in items if item["score"] is not None]
    metrics = {
        "eligible_target_phones": len(items), "scored_target_phones": len(scored),
        "coverage": len(scored) / len(items) if items else None,
        "abstention_rate": 1 - len(scored) / len(items) if items else None,
        "unscored_or_unsupported_phone_rate": 1 - len(scored) / len(items) if items else None,
        "ground_truth_errors": sum(not item["correct"] for item in items),
        "ground_truth_correct": sum(item["correct"] for item in items),
        "roc_auc_conditional": q09._auc_error(items),
        "pr_auc_conditional": q09.pr_auc_error(items),
        "threshold": THRESHOLD,
        "threshold_metrics": q09.threshold_metrics(items, THRESHOLD),
    }
    if len({row["speaker_id"] for row in items}) >= 10:
        metrics["speaker_cluster_bootstrap_95"] = {
            "roc_auc": q09._bootstrap_ci(items, q09._auc_error, seed=9036),
            "pr_auc": q09._bootstrap_ci(items, q09.pr_auc_error, seed=9038),
            "coverage_aware_f1": q09._bootstrap_ci(items, lambda rows: q09.threshold_metrics(rows, THRESHOLD)["coverage_aware"]["f1"], seed=9039),
            "method": "Percentile bootstrap over speakers as clusters, 2,000 replicates; estimates pooled cross-L1 sample only.",
        }
    else:
        metrics["speaker_cluster_bootstrap_95"] = {"status": "NOT_REPORTED", "reason": "Only four Vietnamese speaker clusters; too few for reliable population confidence intervals."}
    return metrics


def make_report(protocol: dict[str, Any]) -> dict[str, Any]:
    items, annotation_categories, diagnostics = collect_items(protocol)
    by_l1: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_l1[item["l1"]].append(item)
        by_speaker[item["speaker_id"]].append(item)
    by_phone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_phone_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_annotation_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_phone[item["target_phone"]].append(item)
        by_phone_class[item["phone_class"]].append(item)
        by_annotation_class[item["category"]].append(item)
    human_counts: Counter[str] = Counter()
    for row in protocol["selection"]["selected_recording_ids_and_file_fingerprints"]:
        human_counts.update(row["label_counts"])
    report = {
        "schema_version": "l2-arctic-pronunciation-report-v1",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "dataset_sha256": protocol["dataset"]["archive_sha256"],
        "all_l1": _metrics(items),
        "per_l1": {l1: _metrics(rows) for l1, rows in sorted(by_l1.items())},
        "per_speaker": {speaker: _metrics(rows) for speaker, rows in sorted(by_speaker.items())},
        "by_target_phoneme": {phone: _metrics(rows) for phone, rows in sorted(by_phone.items())},
        "by_target_phone_class": {kind: _metrics(rows) for kind, rows in sorted(by_phone_class.items())},
        "by_manual_annotation_category": {kind: _metrics(rows) for kind, rows in sorted(by_annotation_class.items())},
        "human_annotation_categories_including_unscored_insertions": dict(sorted(human_counts.items())),
        "diagnostics": diagnostics,
        "limitations": protocol["limits"],
        "usage": "CC BY-NC 4.0 research. Restricted audio, annotation identifiers, scorer outputs, and these derived results stay local; no Pearson agreement is implied.",
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen local-only L2-ARCTIC pronunciation evaluation.")
    parser.add_argument("--prepare-only", action="store_true", help="Validate data and freeze protocol without inference.")
    parser.add_argument("--run", action="store_true", help="Run bounded sequential scorer inference after protocol freeze.")
    parser.add_argument("--max-seconds", type=int, default=3600, help="Maximum sequential inference wall time for this invocation.")
    parser.add_argument("--report-only", action="store_true", help="Summarize completed cached outputs without inference.")
    args = parser.parse_args()
    try:
        protocol = prepare_protocol()
        print(f"Frozen protocol {protocol['protocol_fingerprint']}; {protocol['selection']['selected_recordings']} recordings.", flush=True)
        if args.prepare_only:
            return 0
        if args.run:
            print(json.dumps(run_inference(protocol, args.max_seconds), indent=2), flush=True)
        if args.run or args.report_only:
            report = make_report(protocol)
            print(json.dumps({"Vietnamese": report["per_l1"].get("Vietnamese"),
                              "all_l1": report["all_l1"],
                              "report_path": str(OUTPUT_ROOT / "report.json")}, indent=2, allow_nan=False), flush=True)
            return 0 if not any(status == "NOT_RUN" for status in report["diagnostics"]["record_status_counts"]) else 2
        parser.error("Choose --prepare-only, --run, or --report-only.")
    except Exception as exc:
        print(f"L2-ARCTIC benchmark failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
