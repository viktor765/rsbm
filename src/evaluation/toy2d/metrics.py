import numpy as np
import ot

from ..common import TOY2D_DEFAULTS


def deterministic_subsample(x: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x
    rng = np.random.default_rng(seed)
    indices = rng.choice(x.shape[0], size=max_samples, replace=False)
    indices.sort()
    return x[indices]


def empirical_w2(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int = TOY2D_DEFAULTS["w2_samples"],
    seed: int = TOY2D_DEFAULTS["w2_seed"],
) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = deterministic_subsample(x, max_samples=max_samples, seed=seed)
    y = deterministic_subsample(y, max_samples=max_samples, seed=seed + 1)
    out = {
        "w2_n_source": int(x.shape[0]),
        "w2_n_target": int(y.shape[0]),
    }
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        out["w2"] = float(np.inf)
        out["w2_squared"] = float(np.inf)
    else:
        w2_squared = ot.solve_sample(x, y).value
        w2_squared = max(w2_squared, 0.0)
        out["w2"] = float(np.sqrt(w2_squared))
        out["w2_squared"] = w2_squared
    return out


def support_metrics(
    x: np.ndarray, boundary_eps: float = TOY2D_DEFAULTS["boundary_eps"]
) -> dict[str, float]:
    x = np.asarray(x)
    flat = x.reshape(-1, x.shape[-1])
    violation = np.maximum(np.maximum(-flat, flat - 1.0), 0.0).max(axis=1)
    return {
        "out_of_bounds_fraction": float(np.mean(violation > boundary_eps)),
        "max_violation": float(np.max(violation)) if violation.size else 0.0,
    }
