#!/usr/bin/env python3
"""Verify the locally released keyed GOPT feature archive and sequence build."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any


def _array(archive: zipfile.ZipFile, name: str, *, dtype: Any = float):
    import numpy as np

    with archive.open(name) as stream:
        value = np.loadtxt(stream, delimiter=",", dtype=dtype)
    return np.atleast_2d(value)


def _rebuild(features, keys, labels, phone_ids, utterance_count: int):
    import numpy as np

    if len(features) != len(keys) or len(keys) != len(labels):
        raise ValueError("raw feature, key, and label row counts differ")
    sequence = np.zeros((utterance_count, 50, features.shape[1] - 1), dtype=float)
    sequence_labels = np.zeros((utterance_count, 50, 2), dtype=float) - 1
    previous_utt = None
    row = -1
    seen_utterances = []
    seen_keys = set()
    invalid_keys = []
    for index, raw_key in enumerate(keys.reshape(-1)):
        key = str(raw_key)
        if key in seen_keys:
            invalid_keys.append({"key": key, "reason": "duplicate phone key"})
        seen_keys.add(key)
        try:
            utterance_id, token_text = key.split(".")
            token = int(token_text)
        except (ValueError, TypeError):
            invalid_keys.append({"key": key, "reason": "expected utterance_id.token_index"})
            continue
        if utterance_id != previous_utt:
            row += 1
            previous_utt = utterance_id
            seen_utterances.append(utterance_id)
        if not 0 <= row < utterance_count or not 0 <= token < 50:
            invalid_keys.append({"key": key, "reason": "sequence row or token index is outside fixed 50-phone shape"})
            continue
        label_key = str(labels[index, 0])
        if label_key not in phone_ids:
            invalid_keys.append({"key": key, "reason": f"phone label {label_key!r} is absent from training phone order"})
            continue
        sequence[row, token, :] = features[index, 1:]
        sequence_labels[row, token, 0] = phone_ids[label_key]
        try:
            sequence_labels[row, token, 1] = float(labels[index, 1])
        except (TypeError, ValueError):
            invalid_keys.append({"key": key, "reason": "phone rating is not numeric"})
    if len(seen_utterances) != utterance_count:
        invalid_keys.append({"key": None, "reason": f"key file has {len(seen_utterances)} grouped utterances; expected {utterance_count}"})
    return sequence, sequence_labels, seen_utterances, invalid_keys


def audit_archive(path: Path) -> dict[str, Any]:
    import numpy as np

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    with zipfile.ZipFile(path) as archive:
        crc_error = archive.testzip()
        expected_files = []
        for split in ("tr", "te"):
            expected_files.extend([
                f"raw_kaldi_gop/librispeech/{split}_feats.csv",
                f"raw_kaldi_gop/librispeech/{split}_keys_phn.csv",
                f"raw_kaldi_gop/librispeech/{split}_labels_phn.csv",
                f"seq_data_librispeech/{split}_feat.npy",
                f"seq_data_librispeech/{split}_label_phn.npy",
            ])
        missing = [name for name in expected_files if name not in archive.namelist()]
        if missing:
            return {"status": "FAIL", "archive": str(path), "sha256": digest.hexdigest(), "crc_error": crc_error, "missing_files": missing}

        train_labels = _array(archive, "raw_kaldi_gop/librispeech/tr_labels_phn.csv", dtype=str)
        phone_order = []
        for label in train_labels[:, 0]:
            key = str(label)
            if key not in phone_order:
                phone_order.append(key)
        phone_ids = {label: index for index, label in enumerate(phone_order)}
        report: dict[str, Any] = {
            "archive": str(path),
            "archive_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
            "zip_crc_error": crc_error,
            "phone_class_count": len(phone_order),
            "phone_order": phone_order,
            "splits": {},
        }
        split_failed = crc_error is not None or len(phone_order) != 39
        for split in ("tr", "te"):
            features = _array(archive, f"raw_kaldi_gop/librispeech/{split}_feats.csv")
            keys = _array(archive, f"raw_kaldi_gop/librispeech/{split}_keys_phn.csv", dtype=str).reshape(-1)
            labels = _array(archive, f"raw_kaldi_gop/librispeech/{split}_labels_phn.csv", dtype=str)
            with archive.open(f"seq_data_librispeech/{split}_feat.npy") as stream:
                released_features = np.load(io.BytesIO(stream.read()))
            with archive.open(f"seq_data_librispeech/{split}_label_phn.npy") as stream:
                released_labels = np.load(io.BytesIO(stream.read()))
            utterance_count = int(released_features.shape[0]) if released_features.ndim == 3 else 0
            rebuilt_features, rebuilt_labels, utterance_ids, key_issues = _rebuild(features, keys, labels, phone_ids, utterance_count)
            feature_error = float(np.max(np.abs(rebuilt_features - released_features))) if rebuilt_features.shape == released_features.shape and rebuilt_features.size else None
            labels_match = bool(rebuilt_labels.shape == released_labels.shape and np.array_equal(rebuilt_labels, released_labels))
            if released_features.ndim != 3 or released_features.shape[1:] != (50, 84):
                key_issues.append({"key": None, "reason": f"released feature shape {released_features.shape} is not [utterances, 50, 84]"})
            valid = released_features[:, :, 0] != 0 if released_features.ndim == 3 else np.zeros((0, 0), dtype=bool)
            valid_count = int(valid.sum())
            padding_count = int((~valid).sum())
            finite = bool(np.isfinite(released_features).all())
            padding_zero = bool(np.all(released_features[~valid] == 0)) if released_features.ndim == 3 else False
            labels_padding = bool(np.all(released_labels[~valid] == -1)) if released_labels.shape[:2] == valid.shape else False
            positive_lpp_fraction = None
            if released_features.ndim == 3 and valid_count:
                positive_lpp_fraction = float(np.mean(released_features[:, :, 3:42][valid] > 0))
            split_report = {
                "raw_feature_shape": list(features.shape),
                "raw_phone_rows": len(keys),
                "key_count": len(keys),
                "unique_key_count": len(set(keys.tolist())),
                "utterance_count": utterance_count,
                "utterance_ids": utterance_ids,
                "released_feature_shape": list(released_features.shape),
                "released_label_shape": list(released_labels.shape),
                "rebuilt_feature_max_abs_error": feature_error,
                "rebuilt_labels_exact": labels_match,
                "valid_phone_tokens": valid_count,
                "padding_slots": padding_count,
                "padding_features_zero": padding_zero,
                "padding_labels_minus_one": labels_padding,
                "features_finite": finite,
                "positive_non_skipped_lpp_fraction": positive_lpp_fraction,
                "key_issues": key_issues,
            }
            report["splits"][split] = split_report
            split_failed |= bool(key_issues or feature_error != 0 or not labels_match or not finite or not padding_zero or not labels_padding)
        report["status"] = "FAIL" if split_failed else "PASS"
        report["pairing_note"] = "Raw keys contain source utterance IDs and token indices; speaker/audio pairing still requires the source corpus inventory and recordings."
        report["speaker_ids_present_in_author_feature_archive"] = False
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(".local/gopt/data.zip"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_archive(args.archive)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
