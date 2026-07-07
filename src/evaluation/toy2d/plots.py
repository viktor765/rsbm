# ruff: noqa: E402
import os
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.collections import LineCollection
from PIL import Image, ImageOps

from ... import data_utils
from ..common import COLORS, DPI, next_filename, savefig, select_evenly_spaced


def set_unit_square_axes(
    ax, margin: float = 0.1, width_height_aspect: float = 1.0
) -> None:
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "-", color=COLORS["black"], linewidth=1)
    ax.set_xlim(-margin, 1 + margin)
    ax.set_ylim(-margin, 1 + margin)
    ax.set_box_aspect(1.0 / width_height_aspect)
    ax.set_axis_off()


def plot_2d(
    x: np.ndarray,
    dir: str,
    file_name: str,
    ref_x: np.ndarray | None = None,
    extension: str = "png",
    enumerate_existing: bool = True,
    transparent: bool = False,
    width_height_aspect: float = 1.0,
) -> None:
    os.makedirs(dir, exist_ok=True)
    fig, ax = plt.subplots(1, figsize=(4.4 * width_height_aspect, 4.4))
    set_unit_square_axes(ax, margin=0.15, width_height_aspect=width_height_aspect)
    if ref_x is not None:
        ax.scatter(
            *ref_x.T,
            s=5,
            alpha=0.25,
            color=COLORS["vermillion"],
            label="target",
            edgecolor="none",
        )
    out_of_bounds = (x[:, 0] < 0) | (x[:, 0] > 1) | (x[:, 1] < 0) | (x[:, 1] > 1)
    if np.any(~out_of_bounds):
        ax.scatter(
            *x[~out_of_bounds].T,
            s=5,
            alpha=0.55,
            color=COLORS["grey"],
            label="sampled",
            edgecolor="none",
        )
    if np.any(out_of_bounds):
        ax.scatter(
            *x[out_of_bounds].T,
            s=7,
            alpha=0.55,
            color=COLORS["vermillion"],
            label="outside",
            edgecolor="none",
        )
    path = os.path.join(dir, f"{file_name}.{extension}")
    if enumerate_existing:
        savefig(fig, path, dpi=DPI, transparent=transparent)
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=DPI, transparent=transparent)
    plt.close(fig)


def add_time_colored_path(ax, path: np.ndarray, linewidth: float, alpha: float) -> None:
    points = path[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    collection = LineCollection(
        segments,
        cmap="viridis",
        linewidth=linewidth,
        alpha=alpha,
    )
    collection.set_array(np.linspace(0, 1, max(len(path) - 1, 1)))
    ax.add_collection(collection)


def plot_trajectories(
    paths: np.ndarray, dir: str, file_name: str, title: str | None = None
) -> None:
    os.makedirs(dir, exist_ok=True)
    fig, ax = plt.subplots(1, figsize=(4.8, 4.8))
    set_unit_square_axes(ax, margin=0.15)
    for path in paths:
        add_time_colored_path(
            ax,
            path,
            linewidth=2.0 if len(paths) == 1 else 1.25,
            alpha=0.95 if len(paths) == 1 else 0.78,
        )
    ax.scatter(
        paths[:, 0, 0],
        paths[:, 0, 1],
        s=28,
        marker="o",
        color=COLORS["black"],
        label="start",
        zorder=3,
    )
    ax.scatter(
        paths[:, -1, 0],
        paths[:, -1, 1],
        s=36,
        marker="X",
        color=COLORS["orange"],
        edgecolor=COLORS["black"],
        linewidth=0.4,
        label="end",
        zorder=3,
    )
    if title is not None:
        ax.set_title(title)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    savefig(fig, os.path.join(dir, f"{file_name}.png"), dpi=DPI)
    plt.close(fig)


def plot_trajectory_gif(
    paths: np.ndarray, dir: str, file_name: str, n_frames: int
) -> None:
    os.makedirs(dir, exist_ok=True)
    if paths.shape[1] > n_frames:
        frame_indices = np.linspace(0, paths.shape[1] - 1, n_frames).round().astype(int)
    else:
        frame_indices = np.arange(paths.shape[1])
    frames = []
    for frame_index in frame_indices:
        fig, ax = plt.subplots(1, figsize=(4.8, 4.8))
        set_unit_square_axes(ax, margin=0.15)
        for path in paths:
            ax.plot(
                path[: frame_index + 1, 0],
                path[: frame_index + 1, 1],
                color=COLORS["grey"],
                linewidth=0.8,
                alpha=0.20,
            )
        current = paths[:, frame_index]
        ax.scatter(
            current[:, 0],
            current[:, 1],
            s=12,
            color=COLORS["orange"],
            alpha=0.78,
            edgecolor="none",
        )
        ax.set_title(f"t = {frame_index / (paths.shape[1] - 1):.2f}")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    frames[0].save(
        os.path.join(dir, f"{file_name}.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
    )


def plot_trajectory_frame(
    paths: np.ndarray,
    dir: str,
    file_name: str,
    frame_index: int | None = None,
    show_title: bool = True,
    enumerate_existing: bool = True,
    extension: str = "png",
) -> None:
    os.makedirs(dir, exist_ok=True)
    if frame_index is None:
        frame_index = paths.shape[1] - 1

    fig, ax = plt.subplots(1, figsize=(4.8, 4.8))
    set_unit_square_axes(ax, margin=0.15)
    for path in paths:
        ax.plot(
            path[: frame_index + 1, 0],
            path[: frame_index + 1, 1],
            color=COLORS["black"],
            linewidth=0.8,
            alpha=0.20,
        )
    current = paths[:, frame_index]
    ax.scatter(
        current[:, 0],
        current[:, 1],
        s=12,
        color=COLORS["black"],
        alpha=0.78,
        edgecolor="none",
    )
    if show_title:
        ax.set_title(f"t = {frame_index / max(paths.shape[1] - 1, 1):.2f}")
    path = os.path.join(dir, f"{file_name}.{extension}")
    if enumerate_existing:
        savefig(fig, path, dpi=DPI)
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _center_crop_preserve_resolution(
    image: Image.Image, centering: tuple[float, float]
) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = int(round((width - side) * centering[0]))
    top = int(round((height - side) * centering[1]))
    return image.crop((left, top, left + side, top + side))


def load_density_display_image(conf, *, full_resolution: bool = False) -> np.ndarray:
    if conf.dataset != "image_density":
        raise ValueError(f"Expected image_density dataset, got {conf.dataset}")
    image_path = data_utils._resolve_repo_path(conf.image_path)
    image_size = int(conf.get("image_size", 256))
    is_segmented_density = conf.get("density", "darkness") == "bridge_segmentation"
    image = Image.open(image_path).convert("RGB" if is_segmented_density else "L")
    preprocessing = (
        "resize_aspect" if is_segmented_density else conf.get("preprocessing", "resize")
    )
    centering = tuple(conf.get("centering", [0.5, 0.5]))
    if preprocessing not in {"resize", "resize_aspect", "center_crop"}:
        raise ValueError(f"Unknown image preprocessing: {preprocessing}")
    if full_resolution:
        if preprocessing == "center_crop":
            image = _center_crop_preserve_resolution(image, centering)
    else:
        if preprocessing == "resize":
            image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        elif preprocessing == "resize_aspect":
            image_height = int(conf.get("image_height", image_size))
            image = data_utils.resize_aspect(image, image_height)
        elif preprocessing == "center_crop":
            image = ImageOps.fit(
                image,
                (image_size, image_size),
                method=Image.Resampling.LANCZOS,
                centering=centering,
            )
    if (not is_segmented_density) and conf.get("autocontrast", True):
        image = ImageOps.autocontrast(image, cutoff=float(conf.get("cutoff", 1.0)))
    return np.asarray(image, dtype=np.float32) / 255.0


def image_aspect(image: np.ndarray) -> float:
    return float(image.shape[1] / image.shape[0])


def _set_generation_axes(ax, width_height_aspect: float = 1.0) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1.0 / width_height_aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(COLORS["black"])


def _plot_generation_points(
    ax,
    points: np.ndarray,
    *,
    color: str = COLORS["grey"],
    point_size: float = 1.0,
    point_alpha: float = 0.5,
    width_height_aspect: float = 1.0,
) -> None:
    _set_generation_axes(ax, width_height_aspect=width_height_aspect)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=point_size,
        alpha=point_alpha,
        color=color,
        edgecolor="none",
        rasterized=True,
    )


def _plot_generation_paths(
    ax, paths: np.ndarray, *, path_linewidth: float = 0.75
) -> None:
    _set_generation_axes(ax, width_height_aspect=1.0)
    for path in paths:
        segments = np.stack([path[:-1], path[1:]], axis=1)
        collection = LineCollection(
            segments, linewidth=path_linewidth, alpha=0.8, color=COLORS["grey"]
        )
        collection.set_array(np.linspace(0, 1, max(path.shape[0] - 1, 1)))
        ax.add_collection(collection)
    ax.scatter(
        paths[:, -1, 0],
        paths[:, -1, 1],
        s=10,
        marker="o",
        color=COLORS["black"],
        linewidth=0.25,
        zorder=3,
    )


def _image_axes(ax, image: np.ndarray) -> None:
    if image.ndim == 2:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(np.clip(image, 0.0, 1.0))
    ax.set_box_aspect(image.shape[0] / image.shape[1])
    ax.set_axis_off()


def _arrow_axes(ax, direction: str) -> None:
    ax.set_axis_off()
    if direction == "right":
        xy, xytext = (0.92, 0.5), (0.08, 0.5)
    elif direction == "left":
        xy, xytext = (0.08, 0.5), (0.92, 0.5)
    else:
        raise ValueError(direction)
    ax.annotate(
        "",
        xy=xy,
        xytext=xytext,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.05, "color": COLORS["black"]},
    )


def plot_generation_procedure(
    *,
    image0: np.ndarray,
    image1: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    fake_01: np.ndarray,
    fake_10: np.ndarray,
    paths_01: np.ndarray,
    paths_10: np.ndarray,
    dir: str,
    file_name: str = "generation_procedure",
    n_points: int = 5000,
    n_paths: int = 7,
) -> None:
    os.makedirs(dir, exist_ok=True)
    x0_plot = x0[:n_points]
    x1_plot = x1[:n_points]
    fake_01_plot = fake_01[:n_points]
    fake_10_plot = fake_10[:n_points]
    paths_01_plot = paths_01[20 : 20 + n_paths]
    paths_10_plot = paths_10[20 : 20 + n_paths]

    aspect0 = image_aspect(image0)
    aspect1 = image_aspect(image1)
    width_ratios = [
        2.0 * aspect0,
        0.04,
        0.24,
        aspect0,
        0.24,
        1.0,
        0.24,
        aspect1,
        0.24,
        0.04,
        2.0 * aspect1,
    ]
    figure_height = 3.6
    figure_width = figure_height * (sum(width_ratios) / 2.0)

    fig = plt.figure(figsize=(figure_width, figure_height), dpi=DPI)
    gs = fig.add_gridspec(
        2,
        11,
        width_ratios=width_ratios,
        wspace=0.0,
        hspace=0.05,
    )

    _image_axes(fig.add_subplot(gs[:, 0:2]), image0)
    _image_axes(fig.add_subplot(gs[:, 9:11]), image1)

    _plot_generation_points(
        fig.add_subplot(gs[0, 3]),
        x0_plot,
        color=COLORS["grey"],
        width_height_aspect=aspect0,
    )
    _plot_generation_paths(fig.add_subplot(gs[0, 5]), paths_01_plot)
    _plot_generation_points(
        fig.add_subplot(gs[0, 7]),
        fake_01_plot,
        color=COLORS["grey"],
        width_height_aspect=aspect1,
    )
    for col in (2, 4, 6):
        _arrow_axes(fig.add_subplot(gs[0, col]), "right")

    _plot_generation_points(
        fig.add_subplot(gs[1, 7]),
        x1_plot,
        color=COLORS["grey"],
        width_height_aspect=aspect1,
    )
    _plot_generation_paths(fig.add_subplot(gs[1, 5]), paths_10_plot)
    _plot_generation_points(
        fig.add_subplot(gs[1, 3]),
        fake_10_plot,
        color=COLORS["grey"],
        width_height_aspect=aspect0,
    )
    for col in (4, 6, 8):
        _arrow_axes(fig.add_subplot(gs[1, col]), "left")

    png_path = next_filename(os.path.join(dir, f"{file_name}.png"))
    pdf_path = next_filename(os.path.join(dir, f"{file_name}.pdf"))
    fig.savefig(
        png_path, dpi=400, bbox_inches="tight", pad_inches=0.05, facecolor="white"
    )
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def plot_snapshots_and_vector_fields(
    paths: np.ndarray,
    model: th.nn.Module,
    direction: int,
    device: th.device,
    autocast,
    dir: str,
    file_name: str,
    n_times: int,
    n_particles: int,
    vector_grid_size: int,
    extension: str = "png",
    enumerate_existing: bool = True,
) -> None:
    os.makedirs(dir, exist_ok=True)
    time_indices = np.linspace(0, paths.shape[1] - 1, n_times).round().astype(int)
    particles = select_evenly_spaced(paths, n_particles)
    grid_axis = np.linspace(0, 1, vector_grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(grid_axis, grid_axis)
    grid_np = np.column_stack([xx.ravel(), yy.ravel()])
    grid = th.from_numpy(grid_np).to(device=device)

    fig, axes = plt.subplots(2, n_times, figsize=(2.2 * n_times, 4.8))
    if n_times == 1:
        axes = axes.reshape(2, 1)

    for col, time_index in enumerate(time_indices):
        t_value = time_index / (paths.shape[1] - 1)
        ax = axes[0, col]
        set_unit_square_axes(ax, margin=0.05)
        ax.scatter(
            particles[:, time_index, 0],
            particles[:, time_index, 1],
            s=5,
            alpha=0.55,
            color=COLORS["grey"],
            edgecolor="none",
        )
        ax.set_title(f"t = {t_value:.2f}")

        t = th.full((grid.shape[0],), t_value, device=device, dtype=grid.dtype)
        with th.no_grad(), autocast():
            vectors = model(grid, t, direction=direction).float().cpu().numpy()
        magnitudes = np.linalg.norm(vectors, axis=1)
        robust_max = max(float(np.percentile(magnitudes, 95)), 1e-8)
        scale = robust_max / 0.075

        ax = axes[1, col]
        set_unit_square_axes(ax, margin=0.05)
        ax.quiver(
            grid_np[:, 0],
            grid_np[:, 1],
            vectors[:, 0],
            vectors[:, 1],
            magnitudes,
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=0.004,
            alpha=0.90,
        )

    path = os.path.join(dir, f"{file_name}.{extension}")
    if enumerate_existing:
        savefig(fig, path, dpi=DPI)
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=DPI)
    plt.close(fig)
