#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$PWD/.local/kaldi/src/nnet3bin:$PWD/.local/kaldi/src/featbin:$PWD/.local/kaldi/src/bin:$PATH"
P=$(mktemp -d "${TMPDIR:-/tmp}/pte-acoustic-05.XXXXXX")
export P
trap 'rm -rf "$P"' EXIT
EVIDENCE_ROOT="${QUALITY05_EVIDENCE_ROOT:-$PWD/.local/audio-scoring/evidence-validated}"
B="$EVIDENCE_ROOT/kaldi"
M="$PWD/.local/kaldi/egs/librispeech/s5/exp/chain_cleaned/tdnn_1d_sp/final.mdl"
printf 'utt0001 %s/data/test/data/raw_mfcc_test.1.ark:8\n' "$B" > "$P/feats.scp"
printf 'speaker0001 %s/data/test/data/cmvn_test.ark:12\n' "$B" > "$P/cmvn.scp"
printf 'utt0001 %s/data/test/ivectors/ivector_online.1.ark:8\n' "$B" > "$P/ivector.scp"
printf 'utt0001 speaker0001\n' > "$P/utt2spk"
nnet3-compute --use-gpu=no --online-ivectors=scp:"$P/ivector.scp" --online-ivector-period=10 --frame-subsampling-factor=3 --frames-per-chunk=50 "$M" "ark:apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:$P/utt2spk scp:$P/cmvn.scp scp:$P/feats.scp ark:- |" "ark,t:$P/output.txt"
copy-feats "ark:$B/exp/probs/output.1.ark" "ark,t:$P/xent.txt"

.local/gopt-venv/bin/python - <<'PY'
import io, json, os
from pathlib import Path
import numpy as np

p = Path(os.environ["P"])

def read_matrix(path):
    text = Path(path).read_text()
    body = text.split("[", 1)[1].rsplit("]", 1)[0]
    values = np.loadtxt(io.StringIO(body))
    return np.atleast_2d(values)

raw = read_matrix(p / "output.txt")
xent = read_matrix(p / "xent.txt")
if raw.shape != xent.shape or raw.shape[1] != 6024:
    raise SystemExit(f"branch matrix shape mismatch: output={raw.shape}, output-xent={xent.shape}")
if not np.isfinite(raw).all() or not np.isfinite(xent).all():
    raise SystemExit("non-finite acoustic output")
lse = np.logaddexp.reduce(raw, axis=1, keepdims=True)
normalized = raw - lse
with (p / "normalized.txt").open("w") as f:
    f.write("utt0001  [\n")
    np.savetxt(f, normalized, fmt="%.9g")
    f.write("]\n")
normalized_text = read_matrix(p / "normalized.txt")

frame_stats = {
    "frames": int(raw.shape[0]),
    "pdfs": int(raw.shape[1]),
    "output_logsumexp_min_mean_max": [float(lse.min()), float(lse.mean()), float(lse.max())],
    "normalized_output_text_logsumexp_abs_max": float(np.max(np.abs(np.logaddexp.reduce(normalized_text, axis=1)))),
    "output_xent_logsumexp_abs_max": float(np.max(np.abs(np.logaddexp.reduce(xent, axis=1)))),
    "output_xent_max_abs_delta_from_framewise_normalized_output": float(np.max(np.abs(xent - normalized_text))),
}
(p / "frame_stats.json").write_text(json.dumps(frame_stats))
PY

perl .local/kaldi/egs/gop_speechocean762/s5/local/remove_phone_markers.pl .local/kaldi/egs/librispeech/s5/data/lang_test_tgsmall/phones.txt "$P/pure.txt" "$P/map.int"

for branch in raw_output normalized_output output_xent; do
  case "$branch" in
    raw_output) input="$P/output.txt" ;;
    normalized_output) input="$P/normalized.txt" ;;
    output_xent) input="$P/xent.txt" ;;
  esac
  compute-gop --phone-map="$P/map.int" --skip-phones-string=0:1:2 "$M" \
    "ark:gunzip -c $B/exp/align/ali.1.gz|" "ark:$B/exp/align/ali-phone.1.ark" \
    "ark,t:$input" "ark:$P/$branch.gop.ark" "ark:$P/$branch.feat.ark"
done

.local/gopt-venv/bin/python - <<'PY'
import json, os
from pathlib import Path
import kaldiio
import numpy as np

p = Path(os.environ["P"])
frame_stats = json.loads((p / "frame_stats.json").read_text())
branches = {}

def quantiles(values):
    q = np.percentile(values, [0, 5, 25, 50, 75, 95, 100])
    return {"min": float(q[0]), "p05": float(q[1]), "p25": float(q[2]), "p50": float(q[3]), "p75": float(q[4]), "p95": float(q[5]), "max": float(q[6])}

def stats(values):
    return {"mean": float(values.mean()), "std": float(values.std()), **quantiles(values)}

for branch in ("raw_output", "normalized_output", "output_xent"):
    items = list(kaldiio.ReadHelper(f"ark:{p / (branch + '.feat.ark')}"))
    expected_keys = [f"utt0001.{index}" for index in range(len(items))]
    if not items or [key for key, _ in items] != expected_keys:
        raise SystemExit(f"unexpected GOP feature keys for {branch}: {[key for key, _ in items]}")
    matrix = np.stack([np.asarray(value, dtype=np.float64) for _, value in items])
    if matrix.ndim != 2 or matrix.shape[1] != 85 or not np.isfinite(matrix).all():
        raise SystemExit(f"invalid {branch} feature matrix: {matrix.shape}")
    phone_ids = matrix[:, 0].astype(np.int64)
    if not np.array_equal(phone_ids, matrix[:, 0]):
        raise SystemExit(f"non-integral phone IDs in {branch}")
    features = matrix[:, 1:]
    lpp, lpr = features[:, :42], features[:, 42:]
    canonical_lpr = np.array([lpr[row, phone] for row, phone in enumerate(phone_ids)])
    scalar_gop = np.array([lpp[row, phone] - lpp[row].max() for row, phone in enumerate(phone_ids)])
    feature_stats = []
    for dim, values in enumerate(features.T):
        feature_stats.append({"dimension": dim, **stats(values)})
    per_phone = {}
    for phone in range(42):
        mask = phone_ids == phone
        if mask.any():
            per_phone[str(phone)] = {
                "count": int(mask.sum()),
                "canonical_lpp": stats(lpp[mask, phone]),
                "canonical_lpr_mean": float(lpr[mask, phone].mean()),
                "scalar_gop": stats(scalar_gop[mask]),
            }
    branches[branch] = {
        "shape_including_phone_id": list(matrix.shape),
        "phone_count": int(len(matrix)),
        "feature_84d": stats(features.reshape(-1)),
        "lpp_42d": stats(lpp.reshape(-1)),
        "lpr_42d": stats(lpr.reshape(-1)),
        "feature_dimensions": feature_stats,
        "per_phone": per_phone,
        "phone_id_order": phone_ids.tolist(),
        "skipped_lpp_max_abs_error_from_minus_10": float(np.max(np.abs(lpp[:, :3] + 10))),
        "canonical_lpr_max_abs": float(np.max(np.abs(canonical_lpr))),
        "scalar_gop_zero_count": int(np.count_nonzero(np.abs(scalar_gop) < 1e-6)),
        "scalar_gop_max_abs": float(np.max(np.abs(scalar_gop))),
    }

raw = np.stack([np.asarray(value, dtype=np.float64) for _, value in kaldiio.ReadHelper(f"ark:{p / 'raw_output.feat.ark'}")])
normalized = np.stack([np.asarray(value, dtype=np.float64) for _, value in kaldiio.ReadHelper(f"ark:{p / 'normalized_output.feat.ark'}")])
if not np.array_equal(raw[:, 0], normalized[:, 0]):
    raise SystemExit("phone order changed between raw and normalized output")
raw_lpr, normalized_lpr = raw[:, 43:85], normalized[:, 43:85]
gop_delta = []
for row, phone in enumerate(raw[:, 0].astype(int)):
    raw_gop = raw[row, 1 + phone] - raw[row, 1:43].max()
    norm_gop = normalized[row, 1 + phone] - normalized[row, 1:43].max()
    gop_delta.append(abs(raw_gop - norm_gop))
comparison = {
    "same_phone_order": True,
    "non_skipped_lpr_max_abs_delta": float(np.max(np.abs(raw_lpr[:, 3:] - normalized_lpr[:, 3:]))),
    "skipped_lpr_max_abs_delta": float(np.max(np.abs(raw_lpr[:, :3] - normalized_lpr[:, :3]))),
    "scalar_gop_max_abs_delta": float(max(gop_delta)),
    "interpretation": "Framewise normalization preserves active phone ratios and scalar GOP; skipped LPP dimensions are overwritten to -10 and their derived LPRs therefore are not invariant.",
}

summary = {
    "scope": os.environ.get("QUALITY05_SCOPE", "one retained 4.4-second provider recording; same stored alignment, phone map, model and 35 aligned non-skipped phones"),
    "source_utterance_id": os.environ.get("QUALITY05_SOURCE_UTTERANCE_ID"),
    "not_a_paired_learner_or_author_feature_reconstruction": True,
    "frame_outputs": frame_stats,
    "branches": branches,
    "raw_vs_normalized_output": comparison,
}
destination = os.environ.get("QUALITY05_OUTPUT")
rendered = json.dumps(summary, indent=2, allow_nan=False)
if destination:
    Path(destination).write_text(rendered + "\n")
feature_destination = os.environ.get("QUALITY05_FEATURE_OUTPUT_DIR")
if feature_destination:
    from shutil import copy2
    feature_path = Path(feature_destination)
    feature_path.mkdir(parents=True, exist_ok=True)
    for branch in ("raw_output", "normalized_output", "output_xent"):
        copy2(p / f"{branch}.feat.ark", feature_path / f"{branch}.feat.ark")
    copy2(p / "frame_stats.json", feature_path / "frame_stats.json")
print(rendered)
PY
