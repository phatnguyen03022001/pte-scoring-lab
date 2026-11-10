"""Reproduce GOPT checkpoint quality on official SpeechOcean762 *held-out* feature arrays.

Important: authors' checkpoint selection used test metrics; this is a reproduction,
not an independent external benchmark or a Pearson PTE score.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / ".local" / "gopt" / "data"
GOPT_ROOT = ROOT / ".local" / "gopt"
PARAMS = {
    "librispeech": (84, 3.203, 4.045),
    "paiia": (86, -0.652, 9.737),
    "paiib": (88, -0.516, 9.247),
}
METRICS = ("accuracy", "completeness", "fluency", "prosody", "total")


def correlation(pred: np.ndarray, target: np.ndarray) -> float | None:
    if len(pred) < 2 or np.std(pred) == 0 or np.std(target) == 0:
        return None
    return round(float(np.corrcoef(pred, target)[0, 1]), 4)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including correct handling of ties."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype="float64")
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def spearman(pred: np.ndarray, target: np.ndarray) -> float | None:
    if len(pred) < 2:
        return None
    return correlation(average_ranks(pred), average_ranks(target))


def scores(pred: np.ndarray, actual: np.ndarray) -> dict:
    diff = np.abs(pred - actual)
    return {
        "pearson_r": correlation(pred, actual),
        "mae_0_to_10": round(float(diff.mean()), 4),
        "rmse_0_to_10": round(float(np.sqrt(np.mean(np.square(pred - actual)))), 4),
        "within_1_point_pct": round(float(np.mean(diff <= 1) * 100), 2),
        "within_2_points_pct": round(float(np.mean(diff <= 2) * 100), 2),
        "mean_pred": round(float(pred.mean()), 3),
        "mean_label": round(float(actual.mean()), 3),
    }


def bootstrap_total_ci(pred: np.ndarray, actual: np.ndarray, rng: np.random.Generator, repeats: int = 1000) -> dict:
    """Utterance-level percentile intervals; speaker IDs are unavailable for clustered resampling."""
    samples = {key: [] for key in ("pearson_r", "spearman_r", "mae", "rmse", "within_1_pct", "within_2_pct", "bias")}
    n = len(pred)
    for _ in range(repeats):
        indices = rng.integers(0, n, size=n)
        p, y = pred[indices], actual[indices]
        diff = p - y
        samples["pearson_r"].append(correlation(p, y))
        samples["spearman_r"].append(spearman(p, y))
        samples["mae"].append(float(np.mean(np.abs(diff))))
        samples["rmse"].append(float(np.sqrt(np.mean(np.square(diff)))))
        samples["within_1_pct"].append(float(np.mean(np.abs(diff) <= 1) * 100))
        samples["within_2_pct"].append(float(np.mean(np.abs(diff) <= 2) * 100))
        samples["bias"].append(float(np.mean(diff)))
    return {
        key: [round(float(x), 4) for x in np.percentile([v for v in values if v is not None], [2.5, 97.5])]
        for key, values in samples.items()
    }


def bootstrap_band_ci(pred: np.ndarray, actual: np.ndarray, seed: int, repeats: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    n = len(pred)
    values = {key: [] for key in ("mae", "within_1_pct", "within_2_pct", "bias")}
    for _ in range(repeats):
        indices = rng.integers(0, n, size=n)
        diff = pred[indices] - actual[indices]
        values["mae"].append(float(np.mean(np.abs(diff))))
        values["within_1_pct"].append(float(np.mean(np.abs(diff) <= 1) * 100))
        values["within_2_pct"].append(float(np.mean(np.abs(diff) <= 2) * 100))
        values["bias"].append(float(np.mean(diff)))
    return {
        key: [round(float(x), 4) for x in np.percentile(sample, [2.5, 97.5])]
        for key, sample in values.items()
    }


def evaluate(name: str, torch, batch_size: int, threads: int) -> dict:
    from models.gopt import GOPT

    dim, mean, std = PARAMS[name]
    data_path = DATA_ROOT / f"seq_data_{name}"
    inputs = np.load(data_path / "te_feat.npy")
    phn_labels = np.load(data_path / "te_label_phn.npy")
    labels = np.load(data_path / "te_label_utt.npy").astype("float32")
    word_labels = np.load(data_path / "te_label_word.npy").astype("float32")
    train_labels = np.load(data_path / "tr_label_utt.npy").astype("float32")
    if inputs.ndim != 3 or inputs.shape[1:] != (50, dim):
        raise ValueError(f"Unexpected input shape: {inputs.shape}")
    if phn_labels.shape != (len(inputs), 50, 2) or labels.shape != (len(inputs), 5):
        raise ValueError("Inconsistent feature or gold label shape")

    torch.set_num_threads(threads)
    model = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=dim)
    checkpoint = GOPT_ROOT / "pretrained_models" / f"gopt_{name}" / "best_audio_model.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."):value for key,value in state.items()}, strict=True)
    model.eval()

    all_utt = []
    all_phone = []
    all_word = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            end = min(len(inputs), start + batch_size)
            # Upstream GoPDataset: padded frames remain zero.
            batch = inputs[start:end].astype("float32")
            batch = np.where(batch[:, :, :1] != 0, (batch - mean) / std, 0).astype("float32")
            phns = phn_labels[start:end, :, 0].astype("int64")
            if phns.min() < -1 or phns.max() >= 39:
                raise ValueError("Unexpected phoneme index")
            outputs = model(torch.from_numpy(batch), torch.from_numpy(phns))
            all_utt.append(torch.cat(outputs[:5], dim=1).cpu().numpy() * 5)
            all_phone.append(outputs[5].squeeze(2).cpu().numpy())
            all_word.append(torch.cat(outputs[6:], dim=2).cpu().numpy() * 5)
    seconds = time.perf_counter() - t0
    predictions = np.concatenate(all_utt, axis=0)
    phone_predictions = np.concatenate(all_phone, axis=0)
    word_predictions = np.concatenate(all_word, axis=0)
    baseline_value = np.median(train_labels, axis=0)
    report = {
        "model": f"GOPT-{name}",
        "dataset": "SpeechOcean762, official GOPT precomputed GOP features",
        "split": "authors' te_* files (checkpoint selection used test metrics; not truly independent model selection)",
        "test_utterances": len(inputs),
        "inference_sec_cpu": round(seconds, 3),
        "inference_ms_per_utterance": round(1000 * seconds / len(inputs), 3),
        "uncertainty_method": "percentile bootstrap over utterances; speaker IDs are absent from supplied feature arrays, so intervals are not speaker-clustered",
        "utterance": {},
    }
    for j, metric in enumerate(METRICS):
        pred = predictions[:, j]
        gold = labels[:, j]
        result = scores(pred, gold)
        baseline_diff = baseline_value[j] - gold
        result["spearman_r"] = spearman(pred, gold)
        result["train_median_baseline"] = {
            "score": round(float(baseline_value[j]), 4),
            "mae_0_to_10": round(float(np.mean(np.abs(baseline_diff))), 4),
            "rmse_0_to_10": round(float(np.sqrt(np.mean(np.square(baseline_diff)))), 4),
            "within_1_point_pct": round(float(np.mean(np.abs(baseline_diff) <= 1) * 100), 2),
            "within_2_points_pct": round(float(np.mean(np.abs(baseline_diff) <= 2) * 100), 2),
            "bias_pred_minus_true": round(float(np.mean(baseline_diff)), 4),
        }
        # Retain the original field for consumers of the first result schema.
        result["train_median_baseline_mae"] = result["train_median_baseline"]["mae_0_to_10"]
        if metric == "total":
            result["utterance_bootstrap_95pct_ci"] = bootstrap_total_ci(pred, gold, np.random.default_rng(20261109))
            result["label_distribution"] = {str(int(k)): int(v) for k, v in zip(*np.unique(gold, return_counts=True))}
            result["out_of_range_prediction_pct"] = round(float(np.mean((pred < 0) | (pred > 10)) * 100), 2)
            groups = {"low_0_to_5": gold <= 5, "mid_6_to_7": (gold >= 6) & (gold <= 7), "high_8_to_10": gold >= 8}
            result["by_true_score_band"] = {
                key: {"n": int(mask.sum()),
                      "mae": round(float(np.mean(np.abs(pred[mask] - gold[mask]))), 3),
                      "within_1_pct": round(float(np.mean(np.abs(pred[mask] - gold[mask]) <= 1) * 100), 2),
                      "within_2_pct": round(float(np.mean(np.abs(pred[mask] - gold[mask]) <= 2) * 100), 2),
                      "bias_pred_minus_true": round(float(np.mean(pred[mask] - gold[mask])), 3),
                      "train_median_baseline_mae": round(float(np.mean(np.abs(baseline_value[j] - gold[mask]))), 3),
                      "train_median_baseline_within_1_pct": round(float(np.mean(np.abs(baseline_value[j] - gold[mask]) <= 1) * 100), 2),
                      "utterance_bootstrap_95pct_ci": bootstrap_band_ci(pred[mask], gold[mask], 20261110 + band_index)}
                for band_index, (key, mask) in enumerate(groups.items()) if mask.any()
            }
        report["utterance"][metric] = result

    valid_phn = phn_labels[:, :, 1] >= 0
    gold_phn = phn_labels[:, :, 1][valid_phn]
    predicted_phn = phone_predictions[valid_phn]
    report["phone"] = {
        "n": int(valid_phn.sum()),
        "pearson_r": correlation(predicted_phn, gold_phn),
        "spearman_r": spearman(predicted_phn, gold_phn),
        "mae_0_to_2": round(float(np.mean(np.abs(predicted_phn - gold_phn))), 4),
    }

    # Word scores: score one value per word, not per phone.
    valid_word_pred = []
    valid_word_gold = []
    for i in range(len(inputs)):
        ids = word_labels[i, :, 3]
        start = 0
        prev = 0
        for j, wid in enumerate(ids):
            if wid != prev:
                if j > start:
                    valid_word_pred.append(word_predictions[i, start:j, :].mean(axis=0))
                    valid_word_gold.append(word_labels[i, start:j, :3].mean(axis=0))
                if wid == -1:
                    break
                start = j
                prev = wid
    wp = np.array(valid_word_pred)
    wg = np.array(valid_word_gold)
    report["word"] = {
        "n": len(wp),
        **{m: {
            "pearson_r": correlation(wp[:, j], wg[:, j]),
            "spearman_r": spearman(wp[:, j], wg[:, j]),
            "mae_0_to_10": round(float(np.mean(np.abs(wp[:, j] - wg[:, j]))), 4)
        } for j, m in enumerate(("accuracy", "stress", "total"))},
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/gopt_speechocean762.json"))
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("batch-size and threads must be >= 1")
    import torch
    sys.path.insert(0, str(GOPT_ROOT / "src"))
    reports = [evaluate(name, torch, args.batch_size, args.threads) for name in PARAMS]
    output = {
        "status": "REPRODUCED_PRETRAINED_GOPT_EVALUATION",
        "evidence_scope": "SpeechOcean762 held-out files only; not PTE and not end-to-end new audio",
        "source_model_repo": "YuanGongND/gopt@bed909daf8eca035095871e51642525acc5b9b55",
        "limitations": [
            "The GOPT training script selects the best checkpoint with test-set phone MSE.",
            "No clean external validation set and no PTE score labels are available.",
            "Speechocean762 is non-native Mandarin L1 speech, not representative of all learners.",
            "Input features come from authors' precomputed Kaldi pipeline, not our own live audio pipeline.",
            "Utterance scores on SpeechOcean762 use a 0-10 rubric, not the Pearson PTE scale.",
            "Metrics are on author test data and may be optimistic due to selection on test.",
            "The installed Kaldi chain model conflicts with the TDNN model prescribed by the GOP recipe; no novel audio was scored.",
        ],
        "models": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "result_path": str(args.output),
        "summary": [
            {"model": m["model"],"n": m["test_utterances"],
             "total": m["utterance"]["total"],"phone":m["phone"],"word_total":m["word"]["total"],
             "inference_sec":m["inference_sec_cpu"]}
            for m in reports
        ]
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
