"""Compare one sampled complex update with its backprop update."""

from pathlib import Path

import numpy as np


RESULTS = Path(__file__).resolve().parents[1] / "results_4" / "inner_loop" / "70"
EPISODE = 1
LAYER = 0
SAMPLED_TIME = 5


def compute_cosine_similarity(delta_w_cs: np.ndarray, delta_w_bp: np.ndarray) -> float:
    """Return cosine similarity, or raise when either update is zero."""
    u = np.asarray(delta_w_cs, dtype=np.float64).ravel()
    v = np.asarray(delta_w_bp, dtype=np.float64).ravel()
    if u.shape != v.shape:
        raise ValueError(f"Update shapes differ: complex={u.shape}, BP={v.shape}")

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        raise ValueError(
            f"Cosine is undefined for a zero update (complex norm={norm_u:g}, BP norm={norm_v:g}). "
            "Regenerate the saved arrays after fixing compute_bp_delta_w."
        )

    return float(np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0))


def main() -> None:
    prefix = f"episode_{EPISODE:04d}"
    cs_path = RESULTS / f"weight_changes_{prefix}.npy"
    bp_path = RESULTS / f"bp_weight_changes_{prefix}.npy"
    cs = np.load(cs_path)
    bp = np.load(bp_path)
    cs_shapes = np.load(RESULTS / f"weight_changes_{prefix}_layer_shapes.npy")
    bp_shapes = np.load(RESULTS / f"bp_weight_changes_{prefix}_layer_shapes.npy")

    if cs.ndim != 4 or bp.ndim != 4:
        raise ValueError(f"Expected padded 4D arrays; got complex={cs.shape}, BP={bp.shape}")
    if cs_shapes[LAYER].tolist() != bp_shapes[LAYER].tolist():
        raise ValueError(
            f"Layer {LAYER} shapes differ: complex={cs_shapes[LAYER]}, BP={bp_shapes[LAYER]}"
        )
    if not 0 <= SAMPLED_TIME < min(cs.shape[1], bp.shape[1]):
        raise IndexError(f"Sampled time {SAMPLED_TIME} is outside saved arrays")
    
    rows, cols = map(int, cs_shapes[LAYER])
    cs_update = cs[LAYER, SAMPLED_TIME, :rows, :cols]
    bp_update = bp[LAYER, SAMPLED_TIME, :rows, :cols]
    cosine = compute_cosine_similarity(cs_update, bp_update)
    print(
        f"Episode {EPISODE}, layer {LAYER}, sampled time {SAMPLED_TIME}: "
        f"cosine={cosine:.8f} (complex norm={np.linalg.norm(cs_update):.6g}, "
        f"BP norm={np.linalg.norm(bp_update):.6g})"
    )


if __name__ == "__main__":
    main()
