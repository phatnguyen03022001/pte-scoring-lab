"""TASK-07.5.2: frozen, offline Read Aloud provenance and attribution diagnostics.

This program never transcribes audio, creates human labels, or modifies scoring.
All recording-level inputs/outputs remain under ignored .local/.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jiwer

import score_integrated as scoring
from task_07_5_asr import MANIFEST, ROOT, comparison, digest, manual_words, verify_manifest

OUT = ROOT / ".local/research/task-07.5.2"
BASE = ROOT / ".local/research/task-07.5-asr/results.json"
SMALL = ROOT / ".local/research/task-07.5.1-tech/small.json"
ULTRA = ROOT / ".local/research/task-07.5.1.2/parakeet-ultra.json"
L2 = ROOT / ".local/datasets/l2-arctic-v5.0/extracted"
SPEAKERS = ("HQTV", "PNV", "THV", "TLV")
FINGERPRINT = "b52cafec4cfe23e49f602a36e5d2ae2c43ab89b9fd1e40f2a9ed9ba7d8ce497d"
HEX_SHA = re.compile(r"^[a-f0-9]{64}$")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(MANIFEST)
    verify_manifest(manifest)  # Frozen source and every recording hash.
    if manifest["fingerprint"] != FINGERPRINT:
        raise ValueError("Unexpected TASK-07.5 frozen manifest identity")
    read = [row for row in manifest["rows"] if row["group"] == "l2_read_intended"]
    if len(read) != 48 or Counter(row["speaker"] for row in read) != dict.fromkeys(SPEAKERS, 12):
        raise ValueError("Read Aloud frozen selection is not 12 x 4")
    for speaker in SPEAKERS:
        inventory = list((L2 / speaker / "annotation").glob("*.TextGrid"))
        if len(inventory) != 150:
            raise ValueError(f"Changed annotation inventory: {speaker}")
        ranked = sorted(inventory, key=lambda p: hashlib.sha256(
            f"TASK-07.5|{speaker}|{p.stem}".encode()).hexdigest())
        expected = {p.stem for p in ranked[:12]}
        actual = {row["unit"] for row in read if row["speaker"] == speaker}
        if actual != expected:
            raise ValueError(f"Changed frozen selection: {speaker}")
    return manifest, read


def word_interval_labels(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    if raw.count('name = "words"') != 1 or 'item [2]:' not in raw:
        raise ValueError("Missing or ambiguous TextGrid words tier")
    tier = raw.split('name = "words"', 1)[1].split('item [2]:', 1)[0]
    return re.findall(r'^\s*text = "(.*)"\s*$', tier, re.M)


def annotation_audit(row: dict[str, Any]) -> dict[str, Any]:
    speaker, unit = row["speaker"], row["unit"]
    directory = L2 / speaker
    hand = directory / "annotation" / f"{unit}.TextGrid"
    forced = directory / "textgrid" / f"{unit}.TextGrid"
    for path in (hand, forced):
        if not path.is_file():
            raise FileNotFoundError(path)
    reviewed = manual_words(hand)
    aligned = manual_words(forced)
    script = row["reference"]
    if scoring.normalize_tokens(reviewed)[0] != scoring.normalize_tokens(script)[0]:
        # Divergent annotation text does NOT automatically prove actual speech.
        lexical_status = "ANNOTATION_PROMPT_DISAGREEMENT_UNVERIFIED"
    else:
        lexical_status = "ANNOTATION_MATCHES_PROMPT"
    return {
        "id": row["id"], "speaker": speaker,
        "annotation_class": "HUMAN_REVIEWED_BUT_PROMPT_CONDITIONED",
        "ground_truth_eligible": False,
        "lexical_status": lexical_status,
        "annotation_vs_forced_interval_labels_equal": (
            word_interval_labels(hand) == word_interval_labels(forced)
        ),
        "annotation_vs_forced_lexical_equal": reviewed == aligned,
        "annotation_vs_forced_normalized_equal": (
            scoring.normalize_tokens(reviewed)[0] == scoring.normalize_tokens(aligned)[0]
        ),
        "annotation_sha256": digest(hand), "forced_sha256": digest(forced),
    }


def cached_models(manifest: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    expected = {row["id"] for row in manifest["rows"]}
    outputs = {}
    for name, path in (("base", BASE), ("small", SMALL), ("ultra", ULTRA)):
        obj = load_json(path)
        if obj["manifest_fingerprint"] != FINGERPRINT:
            raise ValueError(f"ASR output fingerprint mismatch: {name}")
        rows = obj["rows"]
        if len(rows) != len(expected) or {r["id"] for r in rows} != expected:
            raise ValueError(f"Missing/duplicated/misaligned ASR units: {name}")
        if len({r["id"] for r in rows}) != len(expected):
            raise ValueError(f"Duplicated ASR units: {name}")
        if any(row["status"] != "ok" for row in rows):
            raise ValueError(f"ASR failure in frozen results: {name}")
        outputs[name] = {row["id"]: row for row in rows}
    return outputs


def edits(reference: str, transcript: str) -> dict[str, Any]:
    result = comparison(reference, transcript)
    ref = " ".join(scoring.normalize_tokens(reference)[0])
    hyp = " ".join(scoring.normalize_tokens(transcript)[0])
    char = jiwer.process_characters(ref, hyp)
    return {
        "reference_words": result["reference_words"],
        "substitutions": result["substitutions"],
        "deletions": result["deletions"],
        "insertions": result["insertions"],
        "exact": result["exact"],
        "reference_chars": len(ref),
        "char_edits": char.substitutions + char.deletions + char.insertions,
    }


def aggregate(rows: list[dict[str, Any]], outputs: dict[str, dict[str, dict[str, Any]]]
              ) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for model, index in outputs.items():
        global_counts = Counter()
        speakers: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            record = index[row["id"]]
            metric = edits(row["reference"], record["transcript"])
            for bucket in (global_counts, speakers[row["speaker"]]):
                for field in ("reference_words", "substitutions", "deletions",
                              "insertions", "reference_chars", "char_edits"):
                    bucket[field] += metric[field]
                bucket["exact"] += int(metric["exact"])
                bucket["recordings"] += 1
        totals[model] = {"total": dict(global_counts),
                         "per_speaker": {sp: dict(speakers[sp]) for sp in SPEAKERS}}
    return totals


def decoder_support_compatibility(segments: list[dict[str, Any]]) -> str:
    """No uncalibrated native value can be treated as positive decoder support."""
    required = {"avg_logprob", "no_speech_prob", "compression_ratio"}
    if not segments or any(not required.issubset(segment) for segment in segments):
        return "MISSING_DECODER_SUPPORT_FAIL_CLOSED"
    return "UNCALIBRATED_DIAGNOSTIC_SEMANTICS_FAIL_CLOSED"


def paired_prompt_changes(read: list[dict[str, Any]],
                          models: dict[str, dict[str, dict[str, Any]]]
                          ) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for other in ("small", "ultra"):
        counts = Counter()
        for row in read:
            before = edits(row["reference"], models["base"][row["id"]]["transcript"])
            after = edits(row["reference"], models[other][row["id"]]["transcript"])
            b_edits = sum(before[key] for key in ("substitutions", "deletions", "insertions"))
            a_edits = sum(after[key] for key in ("substitutions", "deletions", "insertions"))
            counts["fewer_prompt_edits" if a_edits < b_edits
                   else "more_prompt_edits" if a_edits > b_edits
                   else "same_prompt_edits"] += 1
            if not before["exact"] and after["exact"]:
                counts["baseline_refusal_to_prompt_exact"] += 1
            if before["exact"] and not after["exact"]:
                counts["baseline_prompt_exact_to_mismatch"] += 1
            if before["exact"] and after["exact"]:
                counts["both_prompt_exact"] += 1
            if not before["exact"] and not after["exact"]:
                counts["both_prompt_mismatch"] += 1
        summary[f"base_vs_{other}"] = dict(counts)
    return summary


def three_way_diagnostic(prompt: str, actual: str, recognized: str, *,
                         independently_verified: bool = False,
                         uncertain_token_indices: tuple[int, ...] = ()) -> dict[str, Any]:
    """Conditional edit-level evidence; must never infer human truth from ASR/prompt.

    Ambiguous lexical spans exclude unqualified WER. Index linkage is only an
    edit-alignment hypothesis; repeated tokens and ties require human review.
    """
    if not independently_verified:
        raise ValueError("Actual speech must have independently verified provenance")
    prompt_tokens = scoring.normalize_tokens(prompt)[0]
    actual_tokens = scoring.normalize_tokens(actual)[0]
    asr_tokens = scoring.normalize_tokens(recognized)[0]
    if not prompt_tokens or not actual_tokens:
        raise ValueError("Empty prompt or actual speech is not unqualified truth")
    if any(not isinstance(i, int) or isinstance(i, bool)
           or i < 0 or i >= len(actual_tokens) for i in uncertain_token_indices):
        raise ValueError("Uncertain-word indices outside actual transcript")
    speaker = scoring.classify_alignment(prompt_tokens, actual_tokens)
    asr = scoring.classify_alignment(actual_tokens, asr_tokens)
    speaker_edits = [e for e in speaker if e["classification"] != "MATCH"]
    asr_edits = [e for e in asr if e["classification"] != "MATCH"]
    # Classifying both source edits on one recording does not mean independent
    # acoustic validation of the individual aligned events.
    repeated = any(count > 1 for seq in (prompt_tokens, actual_tokens, asr_tokens)
                   for count in Counter(seq).values())
    lexical_uncertain = bool(uncertain_token_indices)
    return {
        "status": "AMBIGUOUS_EXCLUDE_FROM_UNQUALIFIED_WER" if lexical_uncertain
                  else "CONDITIONAL_THREE_WAY_EDIT_ANALYSIS",
        "speaker_edit_events": speaker_edits,
        "asr_edit_events": asr_edits,
        "speaker_deviation_present": bool(speaker_edits),
        "asr_error_present": bool(asr_edits),
        "both_present": bool(speaker_edits and asr_edits),
        "possible_repeated_token_alignment_ambiguity": repeated,
        "human_uncertain_token_indices": list(uncertain_token_indices),
        "actual_vs_prompt_text_gate": actual_tokens == prompt_tokens,
        "asr_vs_prompt_text_gate": asr_tokens == prompt_tokens,
        "oracle_actual_vs_itself_text_gate": True,
        "decoder_confidence_eligible": None,
        "acoustic_support_eligible": None,
        "gopt_success": None,
        "pronunciation_attribution_verified": False,
        "actual_vs_asr": None if lexical_uncertain else edits(actual, recognized),
    }


def validate_handoff_submission(entry: dict[str, Any],
                                manifest_row: dict[str, Any]) -> str:
    """Structural validation ONLY. A form cannot establish a reviewer's identity."""
    if entry.get("recording_id") != manifest_row["id"] or (
        entry.get("audio_sha256") != manifest_row["audio_sha256"]
    ):
        raise ValueError("Recording identity or audio hash mismatch")
    if entry.get("status") != "HUMAN_REVIEW_SUBMITTED":
        raise ValueError("An absent human transcript cannot become ground truth")
    if not isinstance(entry.get("actual_speech_transcript"), str) or (
        not scoring.normalize_tokens(entry["actual_speech_transcript"])[0]
    ):
        raise ValueError("Human actual-speech transcript is missing")
    provenance = entry.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("kind") != "INDEPENDENT_HUMAN":
        raise ValueError("Model/prompt-derived transcript is not human evidence")
    annotator = provenance.get("annotator_id")
    reviewer = provenance.get("reviewer_id")
    if not isinstance(annotator, str) or not annotator.strip():
        raise ValueError("Missing traceable annotator_id")
    if reviewer is not None and (not isinstance(reviewer, str) or not reviewer.strip()):
        raise ValueError("Invalid reviewer_id")
    if reviewer is not None and annotator == reviewer:
        raise ValueError("Reviewer must be distinct from annotator")
    for key in ("audio_only_first_pass", "model_predictions_blinded",
                "prompt_aware_adjudication", "rights_confirmed"):
        if provenance.get(key) is not True:
            raise ValueError(f"Unverified review requirement: {key}")
    if not isinstance(provenance.get("source_evidence_sha256"), str) or not (
        HEX_SHA.fullmatch(provenance["source_evidence_sha256"])
    ):
        raise ValueError("Missing source annotation evidence hash")
    uncertain = entry.get("uncertain_token_indices")
    if not isinstance(uncertain, list) or any(
        not isinstance(i, int) or isinstance(i, bool) or i < 0
        or i >= len(scoring.normalize_tokens(entry["actual_speech_transcript"])[0])
        for i in uncertain
    ) or len(uncertain) != len(set(uncertain)):
        raise ValueError("Invalid ambiguous-word markers")
    # A single qualified human is possible, but inter-annotator agreement is
    # unavailable; a schema cannot verify an annotator's actual work.
    if reviewer is None:
        return "SCHEMA_VALID_SINGLE_ANNOTATOR_UNVERIFIED"
    return "SCHEMA_VALID_INDEPENDENCE_REQUIRES_EXTERNAL_VERIFICATION"


def _write_ignored(path: Path, data: dict[str, Any]) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Existing local evidence must not be overwritten: {path}")
        return
    with path.open("x", encoding="utf-8") as file:
        file.write(payload)


def run() -> dict[str, Any]:
    manifest, read = read_manifest()
    sources = {row["id"]: annotation_audit(row) for row in read}
    models = cached_models(manifest)
    actual_four = [row for row in manifest["rows"]
                   if row["group"] == "l2_suitcase_actual"]
    baseline_refusals = [row for row in read if models["base"][row["id"]][
        "gate_status"] == "UNSUPPORTED_ASR_MISMATCH_OR_UNCERTAINTY"]
    if len(baseline_refusals) != 37:
        raise ValueError("Frozen 37/48 refusals changed")
    for row in read:
        for name in ("base", "small"):
            cached = models[name][row["id"]]
            exact = edits(row["reference"], cached["transcript"])["exact"]
            if (cached["gate_status"] == "ASR_MATCH_UNVERIFIED") != exact:
                raise ValueError(f"Non-text diagnostic gate distinction: {name}")
    paired = {}
    decisions = {}
    for row in read:
        unit = row["id"]
        exact = {name: edits(row["reference"], model[unit]["transcript"])["exact"]
                 for name, model in models.items()}
        decisions[unit] = {"speaker": row["speaker"], "exact_prompt": exact,
                           "baseline_refused": row in baseline_refusals,
                           "causal_attribution": "INSUFFICIENT_EVIDENCE",
                           "human_actual_speech_verified": False}
    for speaker in SPEAKERS:
        subset = [row for row in read if row["speaker"] == speaker]
        paired[speaker] = {
            name: sum(decisions[row["id"]]["exact_prompt"][name] for row in subset)
            for name in models
        }
        paired[speaker]["baseline_refusals"] = sum(
            decisions[row["id"]]["baseline_refused"] for row in subset)
    ultra_segments = [seg for row in read for seg in models["ultra"][
        row["id"]]["segments"]]
    missing = ("avg_logprob", "no_speech_prob", "compression_ratio", "temperature")
    native_keys = sorted({key for seg in ultra_segments for key in seg})
    if any(key in native_keys for key in missing):
        raise ValueError("Parakeet native metadata changed; reassess compatibility")
    status = {
        "frozen_fingerprint": FINGERPRINT,
        "read_recordings": len(read), "speakers": len(SPEAKERS),
        "read_annotated_phone_tiers": len(sources),
        "human_verified_read_actual_speech": 0,
        "read_prompt_conditioned_word_tiers": sum(
            source["lexical_status"] == "ANNOTATION_MATCHES_PROMPT"
            for source in sources.values()),
        "read_interval_label_sequences_different_from_forced": sum(
            not source["annotation_vs_forced_interval_labels_equal"]
            for source in sources.values()),
        "read_normalized_word_tier_differences_from_forced": sum(
            not source["annotation_vs_forced_normalized_equal"]
            for source in sources.values()),
        "baseline_refusals": len(baseline_refusals),
        "baseline_refusals_with_independent_actual_speech": 0,
        "confirmed_asr_contribution_to_refusals": 0,
        "confirmed_speaker_contribution_to_refusals": 0,
        "confirmed_both_contribution_to_refusals": 0,
        "unattributable_refusals": len(baseline_refusals),
        "per_speaker_exact_prompt": paired,
        "paired_prompt_changes": paired_prompt_changes(read, models),
        "prompt_only_metrics": aggregate(read, models),
        "actual_speech_metrics_suitcase_only": aggregate(actual_four, models),
        "ultra_segments": len(ultra_segments),
        "ultra_native_cached_fields": native_keys,
        "ultra_decoder_equivalence_verified": False,
        "ultra_decoder_support_status": decoder_support_compatibility(ultra_segments),
        "verdict": "INDEPENDENT_TRANSCRIPTS_UNAVAILABLE",
    }
    handoff = {
        "schema": "task-07.5.2-local-human-handoff-v1",
        "disclosure": "LOCAL_ONLY_CC_BY_NC_4_0_NO_EXTERNAL_AUDIO_UPLOAD",
        "instructions": ("First transcribe audio without prompt or model outputs. "
                         "Then review with prompt, mark substitutions/omissions/"
                         "insertions/repetitions/false starts and uncertain words. "
                         "Retain independent source evidence, reviewer provenance "
                         "and license decision; no model consensus as truth."),
        "recordings": [
            {"recording_id": row["id"], "audio_sha256": row["audio_sha256"],
             "local_audio": row["audio"], "status": "UNANNOTATED",
             "actual_speech_transcript": None, "uncertain_token_indices": [],
             "provenance": None}
            for row in read
        ],
    }
    private = {"summary": status, "annotation_audit": sources,
               "read_unit_decisions": decisions,
               "baseline_refusal_ids": [row["id"] for row in baseline_refusals]}
    OUT.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write_ignored(OUT / "handoff.json", handoff)
    _write_ignored(OUT / "diagnostics.json", private)
    return status


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
