"""A real tiny pointer-head gradient update on synthetic vectors, not a language model."""
import numpy as np


def run(seed=7):
    rng = np.random.default_rng(seed)
    queries = rng.normal(size=(12, 8))
    candidates = rng.normal(size=(3, 8))
    target = np.arange(12) % 3
    weights = rng.normal(scale=0.01, size=(8, 8))

    def forward(w):
        logits = queries @ w @ candidates.T / np.sqrt(8)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(12), target]).mean()
        return probs, float(loss)

    p, before = forward(weights)
    residual = p.copy()
    residual[np.arange(12), target] -= 1
    gradient = queries.T @ residual @ candidates / (12 * np.sqrt(8))
    weights -= 0.05 * gradient
    _, after = forward(weights)
    if not np.isfinite(gradient).all() or after >= before:
        raise RuntimeError("Synthetic pointer gradient smoke failed")
    return {"kind": "synthetic_numpy_pointer_head", "loss_before": before,
            "loss_after": after, "gradient_norm": float(np.linalg.norm(gradient)),
            "qwen_loaded": False, "clinical_validation": False}
