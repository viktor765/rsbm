import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch as th

DEFAULT_OOB_THRESHOLDS = (0.0, 0.001, 0.01, 0.05, 0.1)


def get_oob_thresholds(eval_conf) -> tuple[float, ...]:
    thresholds = eval_conf.get("oob_thresholds", DEFAULT_OOB_THRESHOLDS)
    return tuple(float(threshold) for threshold in thresholds)


def compute_oob_metrics(
    generated_pixels: th.Tensor,
    thresholds: list[float] | tuple[float, ...] = DEFAULT_OOB_THRESHOLDS,
) -> dict[str, Any]:
    x = generated_pixels.detach().cpu().to(dtype=th.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected generated pixels with shape (N,C,H,W), got {x.shape}")
    n_images = int(x.shape[0])
    pixels_per_image = int(np.prod(x.shape[1:]))
    total_values = int(n_images * pixels_per_image)

    below = x < 0
    above = x > 1
    oob = below | above
    magnitudes = th.where(below, -x, th.where(above, x - 1, th.zeros_like(x)))

    oob_per_image = oob.flatten(1).sum(dim=1).to(dtype=th.float64)
    oob_fraction_per_image = oob_per_image / pixels_per_image
    oob_per_image_np = oob_per_image.numpy()
    oob_values = magnitudes[oob].numpy()
    per_image_max_magnitude = magnitudes.flatten(1).max(dim=1).values.numpy()

    def percentile(values: np.ndarray, q: float) -> float:
        if values.size == 0:
            return 0.0
        return float(np.percentile(values, q))

    metrics: dict[str, Any] = {
        "n_images": n_images,
        "pixels_per_image": pixels_per_image,
        "total_pixel_values": total_values,
        "total_oob_pixels": int(oob.sum().item()),
        "total_oob_pixel_fraction": float(oob.sum().item() / total_values),
        "below_zero_pixels": int(below.sum().item()),
        "below_zero_pixel_fraction": float(below.sum().item() / total_values),
        "above_one_pixels": int(above.sum().item()),
        "above_one_pixel_fraction": float(above.sum().item() / total_values),
        "oob_pixels_per_image_mean": float(oob_per_image_np.mean()),
        "oob_pixels_per_image_median": float(np.median(oob_per_image_np)),
        "oob_pixels_per_image_max": int(oob_per_image_np.max()),
        "oob_magnitude_mean": float(oob_values.mean()) if oob_values.size else 0.0,
        "oob_magnitude_median": percentile(oob_values, 50),
        "oob_magnitude_p95": percentile(oob_values, 95),
        "oob_magnitude_p99": percentile(oob_values, 99),
        "oob_magnitude_max": float(oob_values.max()) if oob_values.size else 0.0,
        "per_image_max_oob_magnitude_mean": float(per_image_max_magnitude.mean()),
        "per_image_max_oob_magnitude_p95": percentile(per_image_max_magnitude, 95),
        "per_image_max_oob_magnitude_max": float(per_image_max_magnitude.max()),
    }

    threshold_metrics = {}
    for threshold in thresholds:
        threshold = float(threshold)
        key = f"{threshold:g}"
        if threshold <= 0:
            value = (oob_fraction_per_image > 0).to(dtype=th.float32).mean()
        else:
            value = (oob_fraction_per_image >= threshold).to(dtype=th.float32).mean()
        threshold_metrics[key] = float(value.item())
    metrics["fraction_images_oob_fraction_ge_threshold"] = threshold_metrics
    metrics["fraction_images_with_any_oob"] = threshold_metrics.get(
        "0", float((oob_fraction_per_image > 0).to(dtype=th.float32).mean().item())
    )
    return metrics


def write_oob_metrics(
    *,
    output_dir: Path,
    metrics_by_item: dict[str, dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    nested: dict[str, dict[str, Any]] = {}
    for item_key, metrics in metrics_by_item.items():
        split, direction = item_key.split("_", maxsplit=1)
        nested.setdefault(split, {})[direction] = metrics

    with (output_dir / "image_oob_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(nested, f, indent=2, ensure_ascii=False)

    rows = []
    for item_key, metrics in metrics_by_item.items():
        split, direction = item_key.split("_", maxsplit=1)
        row = {
            "split": split,
            "direction": direction,
            "n_images": metrics["n_images"],
            "total_oob_pixel_fraction": metrics["total_oob_pixel_fraction"],
            "fraction_images_with_any_oob": metrics["fraction_images_with_any_oob"],
            "oob_pixels_per_image_mean": metrics["oob_pixels_per_image_mean"],
            "oob_pixels_per_image_max": metrics["oob_pixels_per_image_max"],
            "oob_magnitude_mean": metrics["oob_magnitude_mean"],
            "oob_magnitude_p95": metrics["oob_magnitude_p95"],
            "oob_magnitude_max": metrics["oob_magnitude_max"],
            "cache_path": metrics["cache_path"],
        }
        for threshold, value in metrics[
            "fraction_images_oob_fraction_ge_threshold"
        ].items():
            row[f"fraction_images_oob_fraction_ge_{threshold}"] = value
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "image_oob_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_cached_sample_oob_metrics(
    *, context, cached_by_item: dict[str, dict[str, Any]], output_dir: str | Path
) -> None:
    if not cached_by_item:
        return

    output_dir = Path(output_dir)
    thresholds = get_oob_thresholds(context.eval_conf)
    metrics_by_item = {}
    for item_key, result in cached_by_item.items():
        cache = result["cache"]
        metrics = compute_oob_metrics(cache["generated_pixels"], thresholds=thresholds)
        metrics.update(
            {
                "cache_path": str(result["cache_path"]),
                "selected_start_count": int(cache["metadata"]["n_selected"]),
                "dataset_size": int(cache["metadata"]["dataset_size"]),
            }
        )
        metrics_by_item[item_key] = metrics
    write_oob_metrics(output_dir=output_dir, metrics_by_item=metrics_by_item)
