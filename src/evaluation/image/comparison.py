# ruff: noqa: E402
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from .cached_samples import SPLITS, derive_seed

COMPARISON_LABELS = [
    "Source",
    r"$\alpha$-DSBM",
    r"Clipped",
    r"$\alpha$-RSBM",
]


def expected_cache_filenames() -> tuple[str, ...]:
    return tuple(
        f"{split}_{direction}.pt"
        for split in SPLITS
        for direction in ("0_to_1", "1_to_0")
    )


def is_cache_dir(path: Path) -> bool:
    return path.is_dir() and all(
        (path / filename).is_file() for filename in expected_cache_filenames()
    )


def latest_eval_cache_dir(run_dir: Path) -> Path | None:
    eval_dir = run_dir / "eval"
    if not eval_dir.is_dir():
        return None
    candidates = [
        child / "cache"
        for child in eval_dir.iterdir()
        if child.is_dir() and is_cache_dir(child / "cache")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.parent.name)


def resolve_cache_dir(run_or_cache_dir: str | Path) -> Path:
    path = Path(run_or_cache_dir).expanduser().resolve()
    if is_cache_dir(path):
        return path
    cache_dir = path / "cache"
    if is_cache_dir(cache_dir):
        return cache_dir
    latest_cache_dir = latest_eval_cache_dir(path)
    if latest_cache_dir is not None:
        return latest_cache_dir
    raise FileNotFoundError(f"Could not find cached samples under {path}")


def default_comparison_dir(reflected_cache_dir: Path) -> Path:
    return reflected_cache_dir.parent / "comparison"


def load_cache(path: Path) -> dict[str, Any]:
    return th.load(path, map_location="cpu", weights_only=False)


def tensor_to_image_array(x: th.Tensor) -> np.ndarray:
    x = x.detach().cpu().to(dtype=th.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C,H,W), got {x.shape}.")
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    elif x.shape[0] != 3:
        raise ValueError(f"Expected image tensor with C==1 or C==3, got {x.shape}.")
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def projected_image_array(x: th.Tensor) -> np.ndarray:
    return tensor_to_image_array(x.clamp(0, 1))


def oob_overlay_array(x: th.Tensor) -> np.ndarray:
    image = tensor_to_image_array(x)
    oob_mask = ((x < 0) | (x > 1)).detach().cpu()
    if oob_mask.shape[0] > 1:
        oob_mask_2d = oob_mask.any(dim=0).numpy()
    else:
        oob_mask_2d = oob_mask[0].numpy()
    image[oob_mask_2d] = np.array([1.0, 0.0, 1.0])
    return image


def select_random_generated_indices(
    n_generated: int,
    plot_count: int,
    seed: int,
    split: str,
    direction_name: str,
) -> th.Tensor:
    n_plot = min(int(plot_count), int(n_generated))
    rng = np.random.default_rng(
        derive_seed(seed, "random_pairs_plot", split, direction_name, bits=32)
    )
    chosen = rng.permutation(n_generated)[:n_plot]
    return th.as_tensor(chosen, dtype=th.long)


def select_top_oob_indices(generated: th.Tensor, top_k: int) -> th.Tensor:
    oob_counts = ((generated < 0) | (generated > 1)).flatten(1).sum(dim=1)
    n_plot = min(int(top_k), generated.shape[0])
    return th.argsort(oob_counts, descending=True)[:n_plot]


def validate_comparison_caches(
    nonref_cache: dict[str, Any], reflected_cache: dict[str, Any]
) -> None:
    nonref_meta = nonref_cache["metadata"]
    reflected_meta = reflected_cache["metadata"]
    if str(nonref_meta.get("sde_type", "")).lower() == "reflected":
        raise ValueError("First cache must be from a non-reflected run.")
    if str(reflected_meta.get("sde_type", "")).lower() != "reflected":
        raise ValueError("Second cache must be from a reflected run.")
    for key in ("split", "direction", "source_index", "target_index", "seed"):
        if nonref_meta.get(key) != reflected_meta.get(key):
            raise ValueError(f"Cache metadata field {key!r} differs.")
    if not th.equal(nonref_cache["selected_indices"], reflected_cache["selected_indices"]):
        raise ValueError("Cache selected_indices differ.")
    for key in ("source_pixels", "generated_pixels"):
        if nonref_cache[key].shape != reflected_cache[key].shape:
            raise ValueError(f"Cache tensor {key!r} shapes differ.")
    if not th.equal(nonref_cache["source_pixels"], reflected_cache["source_pixels"]):
        raise ValueError("Cache source_pixels differ.")


def comparison_rows(
    *,
    nonref_cache: dict[str, Any],
    reflected_cache: dict[str, Any],
    generated_indices: th.Tensor,
) -> list[list[np.ndarray]]:
    sources = nonref_cache["source_pixels"].detach().cpu()
    nonref_generated = nonref_cache["generated_pixels"].detach().cpu()
    reflected_generated = reflected_cache["generated_pixels"].detach().cpu()

    rows = []
    for generated_index in generated_indices.tolist():
        rows.append(
            [
                tensor_to_image_array(sources[generated_index]),
                oob_overlay_array(nonref_generated[generated_index]),
                projected_image_array(nonref_generated[generated_index]),
                tensor_to_image_array(reflected_generated[generated_index]),
            ]
        )
    return rows


def write_comparison_grid(
    *,
    rows: list[list[np.ndarray]],
    path: Path,
    comparison_images_per_row: int,
    labels: list[str] | None = COMPARISON_LABELS,
    cell_size: float = 1.45,
    label_fontsize: int = 18,
) -> None:
    if not rows:
        return
    images_per_row = int(comparison_images_per_row)
    if images_per_row < 1:
        raise ValueError("comparison_images_per_row must be at least 1.")
    images_per_group = images_per_row**2
    if len(rows) % images_per_group != 0:
        raise ValueError(
            "Comparison row count must be divisible by "
            "comparison_images_per_row ** 2."
        )
    n_categories = len(rows[0])
    if n_categories == 0:
        raise ValueError("Comparison rows must contain at least one image.")
    if any(len(row) != n_categories for row in rows):
        raise ValueError("All comparison rows must have the same number of columns.")

    first_image = np.asarray(rows[0][0])
    image_shape = first_image.shape
    if any(np.asarray(image).shape != image_shape for row in rows for image in row):
        raise ValueError("All comparison images must have the same shape.")

    image_h, image_w = image_shape[:2]
    min_image_dim = min(image_h, image_w)
    inner_margin = max(1, int(round(min_image_dim * 0.04)))
    main_col_margin = max(inner_margin + 1, int(round(min_image_dim * 0.10)))
    content_h = images_per_row * image_h + (images_per_row - 1) * inner_margin
    content_w = images_per_row * image_w + (images_per_row - 1) * inner_margin
    sample_groups = len(rows) // images_per_group
    canvas_h = sample_groups * content_h + (sample_groups - 1) * inner_margin
    canvas_w = n_categories * content_w + (n_categories - 1) * main_col_margin
    canvas = np.ones((canvas_h, canvas_w, *image_shape[2:]), dtype=np.float32)

    for sample_index, row_images in enumerate(rows):
        sample_group = sample_index // images_per_group
        group_offset = sample_index % images_per_group
        subrow = group_offset // images_per_row
        subcol = group_offset % images_per_row
        for category, image in enumerate(row_images):
            y0 = sample_group * (content_h + inner_margin) + subrow * (
                image_h + inner_margin
            )
            x0 = category * (content_w + main_col_margin) + subcol * (
                image_w + inner_margin
            )
            canvas[y0 : y0 + image_h, x0 : x0 + image_w] = np.asarray(
                image, dtype=np.float32
            )

    label_top_inches = 0.5 if labels is not None else 0.0
    width = cell_size * n_categories
    height = cell_size * sample_groups + label_top_inches
    top = 1 - label_top_inches / height if label_top_inches > 0 else 1.0

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, top])
    ax.imshow(canvas, vmin=0, vmax=1, interpolation="nearest")
    ax.set_axis_off()

    if labels is not None:
        for col, label in enumerate(labels):
            center_px = col * (content_w + main_col_margin) + 0.5 * content_w
            x = center_px / canvas_w
            fig.text(
                x,
                1 - 0.22 / height,
                label,
                ha="center",
                va="top",
                fontsize=label_fontsize,
                fontfamily="serif",
                math_fontfamily="cm",
            )

    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


def write_comparison_artifacts(
    *,
    nonref_cache: dict[str, Any],
    reflected_cache: dict[str, Any],
    output_dir: Path,
    plot_count: int,
    top_k: int,
    comparison_images_per_row: int,
) -> dict[str, Any]:
    validate_comparison_caches(nonref_cache, reflected_cache)
    metadata = nonref_cache["metadata"]
    generated = nonref_cache["generated_pixels"].detach().cpu()
    random_indices = select_random_generated_indices(
        generated.shape[0],
        plot_count,
        int(metadata["seed"]),
        str(metadata["split"]),
        str(metadata["direction"]),
    )
    top_indices = select_top_oob_indices(generated, top_k)

    random_path = output_dir / f"{plot_count}_random_comparison_rows.png"
    top_path = output_dir / f"top{top_k}_oob_comparison_rows.png"
    write_comparison_grid(
        rows=comparison_rows(
            nonref_cache=nonref_cache,
            reflected_cache=reflected_cache,
            generated_indices=random_indices,
        ),
        path=random_path,
        comparison_images_per_row=comparison_images_per_row,
    )
    write_comparison_grid(
        rows=comparison_rows(
            nonref_cache=nonref_cache,
            reflected_cache=reflected_cache,
            generated_indices=top_indices,
        ),
        path=top_path,
        comparison_images_per_row=comparison_images_per_row,
    )
    return {
        "split": metadata["split"],
        "direction": metadata["direction"],
        "random_generated_indices": [int(x) for x in random_indices.tolist()],
        "top_oob_generated_indices": [int(x) for x in top_indices.tolist()],
        "paths": {"random_rows": str(random_path), "top_oob_rows": str(top_path)},
    }


def compare_runs(
    *,
    nonref_run_dir: str | Path,
    reflected_run_dir: str | Path,
    plot_count: int,
    top_k: int,
    comparison_images_per_row: int,
) -> Path:
    nonref_cache_dir = resolve_cache_dir(nonref_run_dir)
    reflected_cache_dir = resolve_cache_dir(reflected_run_dir)
    output_dir = default_comparison_dir(reflected_cache_dir)
    manifest = {
        "nonref_run_dir": str(Path(nonref_run_dir).expanduser().resolve()),
        "reflected_run_dir": str(Path(reflected_run_dir).expanduser().resolve()),
        "nonref_cache_dir": str(nonref_cache_dir),
        "reflected_cache_dir": str(reflected_cache_dir),
        "plots": [],
    }
    for split in SPLITS:
        for direction in ("0_to_1", "1_to_0"):
            filename = f"{split}_{direction}.pt"
            artifact = write_comparison_artifacts(
                nonref_cache=load_cache(nonref_cache_dir / filename),
                reflected_cache=load_cache(reflected_cache_dir / filename),
                output_dir=output_dir / f"{split}_{direction}",
                plot_count=plot_count,
                top_k=top_k,
                comparison_images_per_row=comparison_images_per_row,
            )
            manifest["plots"].append(artifact)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return output_dir
