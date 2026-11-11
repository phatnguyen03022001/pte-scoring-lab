"""TASK-07.5 bounded retrospective ASR/reference and gate diagnostic.

Private per-recording inputs and outputs are written exclusively under .local/.
No Pearson recordings, scores, GOPT calls, cloud access, or model downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import resource
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jiwer

import score_integrated as scoring

ROOT = Path(__file__).resolve().parent
OUT = ROOT / ".local/research/task-07.5-asr"
MANIFEST = OUT / "manifest.json"
DETAILS = OUT / "results.json"
SUMMARY = OUT / "summary.json"
L2 = ROOT / ".local/datasets/l2-arctic-v5.0/extracted"
CV = ROOT / ".local/datasets/commonvoice-en-v17-vietnamese-accent"
SPEAKERS = ("HQTV", "PNV", "THV", "TLV")
ASR = {"name": "base.en", "device": "cpu", "compute_type": "int8",
       "cpu_threads": 4, "num_workers": 1, "local_files_only": True,
       "language": "en", "beam_size": 5, "vad_filter": True,
       "condition_on_previous_text": False, "temperature": "library default"}
READ_PER_SPEAKER = 12


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def canonical(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def manual_words(path: Path) -> str:
    """L2-ARCTIC manually annotated words tier, NOT forced-aligned textgrid."""
    import re
    raw = path.read_text(encoding="utf-8")
    if raw.count('name = "words"') != 1:
        raise ValueError(f"Ambiguous human words tier: {path}")
    tier = raw.split('name = "words"', 1)[1].split('item [2]:', 1)[0]
    labels = re.findall(r'^\s*text = "(.*)"\s*$', tier, re.M)
    if not labels:
        raise ValueError(f"Missing human words: {path}")
    text = " ".join(word.strip() for word in labels if word.strip())
    if not scoring.normalize_tokens(text)[0]:
        raise ValueError(f"Empty human words: {path}")
    return text


def _row(group: str, speaker: str, unit: str, audio: Path, reference: str,
         reference_type: str, source: Path | None) -> dict[str, Any]:
    if not audio.is_file():
        raise FileNotFoundError(audio)
    if source is not None and not source.is_file():
        raise FileNotFoundError(source)
    return {"id": f"{group}:{speaker}:{unit}", "group": group, "speaker": speaker,
            "unit": unit, "audio": str(audio.relative_to(ROOT)),
            "audio_sha256": digest(audio), "reference": reference,
            "reference_type": reference_type,
            "reference_source": None if source is None else str(source.relative_to(ROOT)),
            "reference_source_sha256": None if source is None else digest(source)}


def freeze() -> None:
    if MANIFEST.exists():
        raise ValueError("Frozen manifest already exists; do not silently replace it.")
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {"manual_vs_orthographic": {}}
    narrative = L2 / "suitcase_corpus/suitcase_corpus"
    for speaker in SPEAKERS:
        name = speaker.lower()
        annotation = narrative / "annotation" / f"{name}.TextGrid"
        actual = manual_words(annotation)
        transcription = (narrative / "transcript" / f"{name}.txt").read_text(encoding="utf-8")
        a, b = scoring.normalize_tokens(actual)[0], scoring.normalize_tokens(transcription)[0]
        diff = jiwer.process_words(" ".join(a), " ".join(b))
        provenance["manual_vs_orthographic"][speaker] = {
            "human_word_tier_words": len(a), "orthographic_words": len(b),
            "differences": {"substitutions": diff.substitutions,
                            "deletions": diff.deletions, "insertions": diff.insertions},
            "human_word_tier_sha256": digest(annotation)}
        rows.append(_row("l2_suitcase_actual", speaker, name, narrative / "wav" / f"{name}.wav",
                         actual, "HUMAN_VERIFIED_ACTUAL_TRANSCRIPT", annotation))
        corpus = L2 / speaker
        annotated = list((corpus / "annotation").glob("*.TextGrid"))
        if len(annotated) != 150:
            raise ValueError(f"Unexpected annotated speaker inventory: {speaker} {len(annotated)}")
        ranked = sorted(annotated, key=lambda p: hashlib.sha256(
            f"TASK-07.5|{speaker}|{p.stem}".encode()).hexdigest())
        for textgrid in ranked[:READ_PER_SPEAKER]:
            script = corpus / "transcript" / f"{textgrid.stem}.txt"
            rows.append(_row("l2_read_intended", speaker, textgrid.stem,
                             corpus / "wav" / f"{textgrid.stem}.wav",
                             script.read_text(encoding="utf-8").strip(),
                             "INTENDED_REFERENCE_ONLY", script))
    cv = json.loads((CV / "manifest.json").read_text(encoding="utf-8"))
    for index, item in enumerate(cv["matches"]):
        rows.append(_row("cv_self_report_intended", item["speaker_hash_sha256"],
                         f"cv{index}", CV / item["relative_path"],
                         item["transcript"], "INTENDED_REFERENCE_ONLY",
                         CV / "manifest.json"))
        if rows[-1]["audio_sha256"] != item["sha256"]:
            raise ValueError("Common Voice audio checksum mismatch")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate unit IDs")
    manifest = {
        "schema": "task-07.5-asr-v1", "selection": "SHA256(TASK-07.5|speaker|utterance), lowest 12 from 150",
        "source_exposure": "L2 read-speech benchmark already exposed to TASK-07/07.1/07.2; "
                           "suitcase data co-reside in same downloaded corpus; NOT blind",
        "rights": {"l2": "CC BY-NC 4.0, non-commercial research",
                   "commonvoice": "CC0-1.0, self-reported Vietnamese accent, L1 unverified"},
        "config": ASR, "reference_provenance": provenance, "rows": rows}
    manifest["fingerprint"] = hashlib.sha256(canonical(manifest)).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST), "fingerprint": manifest["fingerprint"],
                      "counts": dict(Counter(r["group"] for r in rows))}))


def verify_manifest(manifest: dict[str, Any]) -> None:
    expected = manifest["fingerprint"]
    copy = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if hashlib.sha256(canonical(copy)).hexdigest() != expected:
        raise ValueError("Frozen manifest fingerprint changed")
    if manifest["config"] != ASR:
        raise ValueError("ASR configuration changed since freeze")
    seen: set[str] = set()
    for row in manifest["rows"]:
        if row["id"] in seen:
            raise ValueError("Duplicate unit")
        seen.add(row["id"])
        if digest(ROOT / row["audio"]) != row["audio_sha256"]:
            raise ValueError("Audio changed: " + row["id"])
        if row["reference_source"] and digest(ROOT / row["reference_source"]) != row["reference_source_sha256"]:
            raise ValueError("Reference source changed: " + row["id"])
        if row["reference_type"] == "HUMAN_VERIFIED_ACTUAL_TRANSCRIPT":
            if manual_words(ROOT / row["reference_source"]) != row["reference"]:
                raise ValueError("Human transcript changed: " + row["id"])


def comparison(reference: str, hypothesis: str) -> dict[str, Any]:
    ref, ref_flags = scoring.normalize_tokens(reference)
    hyp, hyp_flags = scoring.normalize_tokens(hypothesis)
    if not ref:
        raise ValueError("Reference without word tokens")
    if not scoring.content_alignment_within_budget(len(ref), len(hyp)):
        return {"status": "alignment_budget_exceeded", "reference_words": len(ref)}
    words = jiwer.process_words(" ".join(ref), " ".join(hyp))
    classifications = Counter(item["classification"] for item in scoring.classify_alignment(
        ref, hyp, reference_unsupported_numbers=ref_flags,
        hypothesis_unsupported_numbers=hyp_flags))
    return {"status": "ok", "reference_words": len(ref), "asr_words": len(hyp),
            "substitutions": words.substitutions, "deletions": words.deletions,
            "insertions": words.insertions, "hits": words.hits,
            "exact": ref == hyp, "wer": words.wer,
            "cer": jiwer.cer(" ".join(ref), " ".join(hyp)),
            "alignment_types": dict(classifications)}


def _content(reference: str, hypothesis: str, segments: list[Any], vad: list[dict]) -> dict[str, Any]:
    """Actual integrated Content logic on the ASR transcript; no GOPT inference."""
    by_index = {}
    cursor = 0
    for segment in segments:
        words, _ = scoring.normalize_tokens(segment.text)
        for index in range(cursor, cursor + len(words)):
            by_index[index] = segment
        cursor += len(words)
    transcription = {
        "raw_transcript": hypothesis, "normalized_tokens": scoring.normalize_tokens(hypothesis)[0],
        "unsupported_numeric_tokens": scoring.normalize_tokens(hypothesis)[1],
        "vad_intervals": vad, "recognized_token_segments": by_index,
        "diagnostic_thresholds": {"avg_logprob_threshold": -1.0, "no_speech_threshold": 0.6,
                                  "compression_ratio_threshold": 2.4},
        "segments": [{"start_seconds_original_timeline": float(s.start),
                      "end_seconds_original_timeline": float(s.end),
                      "diagnostic_flags": []} for s in segments],
        "model": {"name": ASR["name"]}, "decode_configuration": ASR,
        "vad_interval_duration_seconds": 0, "vad_interval_complement_duration_seconds": 0,
        "duration_after_vad_seconds": 0, "model_load_seconds": 0,
        "inference_seconds_excluding_model_load": 0, "elapsed_seconds_including_model_load": 0,
    }
    return scoring._build_content(reference, transcription, task="ra")


def _gate_result(content: dict[str, Any]) -> str:
    """Use production gate decision on an empty diagnostic shell; not GOPT scores."""
    shell = {"status": "AVAILABLE_UNCALIBRATED_MODEL_PREDICTIONS",
             "measurements": {"phonemes": [], "words": []}, "predictions": None}
    result = scoring.gate_pronunciation_attribution(shell, content)
    return result["speaker_attribution"]["status"]


def evaluate() -> None:
    from faster_whisper import WhisperModel, decode_audio

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    verify_manifest(manifest)
    if DETAILS.exists():
        raise ValueError("Existing results; do not overwrite private evidence")
    load_started = time.perf_counter()
    model = WhisperModel(ASR["name"], device=ASR["device"],
                         compute_type=ASR["compute_type"],
                         cpu_threads=ASR["cpu_threads"], num_workers=ASR["num_workers"],
                         local_files_only=True)
    load_seconds = time.perf_counter() - load_started
    result_rows: list[dict[str, Any]] = []
    for index, row in enumerate(manifest["rows"]):
        started = time.perf_counter()
        cpu_start = time.process_time()
        record: dict[str, Any] = {"id": row["id"], "group": row["group"],
                                  "speaker": row["speaker"], "reference_type": row["reference_type"]}
        stage = "audio_decode"
        try:
            samples = decode_audio(str(ROOT / row["audio"]), sampling_rate=scoring.SAMPLE_RATE)
            stage = "voice_activity_detection"
            vad = scoring._vad_intervals(samples)
            stage = "asr_inference"
            if vad:
                parts, _ = model.transcribe(
                    samples, language=ASR["language"], beam_size=ASR["beam_size"],
                    vad_filter=ASR["vad_filter"],
                    condition_on_previous_text=ASR["condition_on_previous_text"])
                segments = list(parts)
            else:
                segments = []
            transcript = " ".join(s.text.strip() for s in segments).strip()
            stage = "content_and_attribution_diagnostic"
            comparison_data = comparison(row["reference"], transcript)
            content = _content(row["reference"], transcript, segments, vad)
            record.update({"status": "ok", "transcript": transcript,
                           "comparison": comparison_data, "vad_intervals": len(vad),
                           "gate_status": _gate_result(content),
                           "content_status": content["status"],
                           "content_alignment_types": dict(Counter(
                               item["classification"] for item in
                               (content.get("measurements") or {}).get("alignment", []))),
                           "duration_seconds": len(samples) / scoring.SAMPLE_RATE,
                           "segment_count": len(segments)})
        except Exception as exc:
            record.update({"status": "failure", "failure_stage": stage,
                           "failure_type": type(exc).__name__, "failure_message": str(exc)[:160]})
        record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        record["process_cpu_seconds"] = round(time.process_time() - cpu_start, 4)
        result_rows.append(record)
        print(f"{index+1}/{len(manifest['rows'])} {row['group']} {row['speaker']} "
              f"{record['status']} {record['elapsed_seconds']}s", flush=True)
    private = {
        "manifest_fingerprint": manifest["fingerprint"], "model_load_seconds": load_seconds,
        "model_version": importlib.metadata.version("faster-whisper"),
        "maxrss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rows": result_rows}
    DETAILS.write_text(json.dumps(private, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = summarize(private)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in data["rows"]:
        groups[row["group"]].append(row)
    result: dict[str, Any] = {
        "manifest_fingerprint": data["manifest_fingerprint"],
        "model_load_seconds": round(data["model_load_seconds"], 3),
        "model_version": data["model_version"],
        "peak_rss_bytes_macos": data["maxrss_bytes_macos"],
        "groups": {}}
    for name, rows in groups.items():
        ok = [r for r in rows if r["status"] == "ok"]
        eligible = [r for r in ok if r["comparison"]["status"] == "ok"]
        errors = {key: sum(r["comparison"][key] for r in eligible)
                  for key in ("reference_words", "asr_words", "substitutions",
                              "deletions", "insertions", "hits")}
        error_count = sum(errors[k] for k in ("substitutions", "deletions", "insertions"))
        references = {x["id"]: x["reference"] for x in
                      json.loads(MANIFEST.read_text(encoding="utf-8"))["rows"]}
        n_chars = 0
        weighted_char_errors = 0.0
        for row in eligible:
            chars = len(" ".join(scoring.normalize_tokens(references[row["id"]])[0]))
            n_chars += chars
            weighted_char_errors += row["comparison"]["cer"] * chars
        # Group CER is reference-character weighted (not a mean of recording CER).
        result["groups"][name] = {
            "recordings": len(rows), "speakers": len({r["speaker"] for r in rows}),
            "successes": len(ok), "failures": len(rows) - len(ok),
            "empty_transcripts": sum(r.get("transcript") == "" for r in ok),
            "zero_vad_intervals": sum(r.get("vad_intervals") == 0 for r in ok),
            "counts": errors, "exact_recordings": sum(r["comparison"]["exact"] for r in eligible),
            "reference_disagreement_word_ratio": None if not errors["reference_words"]
                else error_count / errors["reference_words"],
            "reference_disagreement_character_ratio": None if not n_chars else weighted_char_errors / n_chars,
            "gate_status": dict(Counter(r["gate_status"] for r in ok)),
            "content_status": dict(Counter(r["content_status"] for r in ok)),
            "content_alignment_types": dict(sum((Counter(r["content_alignment_types"]) for r in ok), Counter())),
            "elapsed_total_seconds": round(sum(r["elapsed_seconds"] for r in rows), 3),
            "inference_wall_median_seconds": statistics.median(r["elapsed_seconds"] for r in rows) if rows else None,
            "per_speaker": {speaker: {
                "recordings": len(items),
                "reference_words": sum(x["comparison"]["reference_words"] for x in items if x["status"] == "ok"),
                "word_edit_units": sum(x["comparison"]["substitutions"] + x["comparison"]["deletions"]
                                      + x["comparison"]["insertions"] for x in items if x["status"] == "ok"),
                "gate_pass": sum(x.get("gate_status") == "ASR_MATCH_UNVERIFIED" for x in items),
            } for speaker, items in sorted(((s, [r for r in rows if r["speaker"] == s])
                                            for s in {r["speaker"] for r in rows}))}
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "verify", "evaluate", "summarize"))
    action = parser.parse_args().action
    if action == "freeze":
        freeze()
    elif action == "verify":
        verify_manifest(json.loads(MANIFEST.read_text(encoding="utf-8")))
        print("Frozen manifest verified")
    elif action == "evaluate":
        evaluate()
    else:
        print(json.dumps(summarize(json.loads(DETAILS.read_text(encoding="utf-8"))), indent=2))


if __name__ == "__main__":
    main()
