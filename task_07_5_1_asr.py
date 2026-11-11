"""TASK-07.5.1: one-model-at-a-time, frozen-manifest ASR compatibility experiment.

No changes to historical TASK-07.5 evidence or production scorer. Outputs are
local research artifacts, never committed. Models must already be downloaded.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import resource
import time
from collections import Counter
from pathlib import Path

from faster_whisper import WhisperModel, decode_audio
from huggingface_hub import snapshot_download

import score_integrated as scoring
from task_07_5_asr import (
    ASR, MANIFEST, ROOT, _content, _gate_result, comparison, summarize,
    verify_manifest,
)

OUT = ROOT / ".local/research/task-07.5.1-tech"
BASE_REVISION = "3d3d5dee26484f91867d81cb899cfcf72b96be6c"
SMALL_REVISION = "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
SMALL_DIR = ROOT / ".local/models/asr-small-en-d1d751a5f827"


def evaluate(model_name: str) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    verify_manifest(manifest)
    if model_name == "base":
        model_path = Path(snapshot_download(
            repo_id="Systran/faster-whisper-base.en",
            revision=BASE_REVISION, local_files_only=True,
            allow_patterns=["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"],
        ))
        revision = BASE_REVISION
    else:
        model_path = SMALL_DIR
        revision = SMALL_REVISION
        for name in ("model.bin", "config.json", "tokenizer.json"):
            if not (model_path / name).is_file():
                raise ValueError(f"Missing pinned local candidate asset: {name}")
    output = OUT / f"{model_name}.json"
    if output.exists():
        raise FileExistsError(f"Existing experiment must not be overwritten: {output}")
    loaded_at = time.perf_counter()
    model = WhisperModel(
        str(model_path), device=ASR["device"], compute_type=ASR["compute_type"],
        cpu_threads=ASR["cpu_threads"], num_workers=ASR["num_workers"],
        local_files_only=True,
    )
    load_seconds = time.perf_counter() - loaded_at
    rows = []
    for index, row in enumerate(manifest["rows"]):
        started = time.perf_counter()
        stage = "decode"
        result = {
            "id": row["id"], "group": row["group"], "speaker": row["speaker"],
            "reference_type": row["reference_type"],
        }
        try:
            samples = decode_audio(str(ROOT / row["audio"]),
                                   sampling_rate=scoring.SAMPLE_RATE)
            stage = "vad"
            vad = scoring._vad_intervals(samples)
            stage = "asr"
            segments = []
            if vad:
                iterator, _ = model.transcribe(
                    samples, language=ASR["language"], beam_size=ASR["beam_size"],
                    vad_filter=ASR["vad_filter"],
                    condition_on_previous_text=ASR["condition_on_previous_text"],
                )
                segments = list(iterator)
            transcript = " ".join(s.text.strip() for s in segments).strip()
            stage = "content_and_gate"
            content = _content(row["reference"], transcript, segments, vad)
            result.update({
                "status": "ok", "transcript": transcript,
                "comparison": comparison(row["reference"], transcript),
                "gate_status": _gate_result(content),
                "content_status": content["status"],
                "content_alignment_types": dict(Counter(
                    a["classification"] for a in
                    (content.get("measurements") or {}).get("alignment", [])
                )),
                "duration_seconds": len(samples) / scoring.SAMPLE_RATE,
                "segment_count": len(segments), "vad_intervals": len(vad),
            })
        except Exception as exc:
            result.update({
                "status": "failure", "failure_stage": stage,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc)[:160],
            })
        result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        rows.append(result)
        print(f"{model_name} {index + 1}/{len(manifest['rows'])} {result['status']}", flush=True)
    report = {
        "manifest_fingerprint": manifest["fingerprint"],
        "model_version": importlib.metadata.version("faster-whisper"),
        "model_name": model_name, "model_revision": revision,
        "model_path": str(model_path),
        "asr_settings": ASR,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("faster-whisper", "ctranslate2", "av", "onnxruntime")},
        "model_load_seconds": round(load_seconds, 4),
        "maxrss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rows": rows,
    }
    report["summary"] = summarize(report)
    OUT.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("base", "small"), required=True)
    args = parser.parse_args()
    evaluate(args.model)
