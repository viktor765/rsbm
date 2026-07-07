# ruff: noqa: E402
import logging
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "black": "#111111",
    "grey": "#5F6368",
}

DPI = 300

TOY2D_DEFAULTS = {
    "w2_samples": 4096,
    "w2_seed": 0,
    "final_artifacts": True,
    "n_snapshot_times": 6,
    "n_snapshot_particles": 1000,
    "n_gif_paths": 128,
    "n_gif_frames": 31,
    "n_generation_points": 5000,
    "n_generation_paths": 7,
    "overleaf_dir_name": "overleaf_upload",
    "vector_grid_size": 21,
    "boundary_eps": 1e-3,
}

logger = logging.getLogger(__name__)


def file_enum_format(path: str, index: int) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}_({index:03d}){ext}"


def next_filename(path: str) -> str:
    if not os.path.exists(path):
        return path

    i = 0
    new_path = file_enum_format(path, i)
    while os.path.exists(new_path):
        i += 1
        new_path = file_enum_format(path, i)
    return new_path


def savefig(fig, path: str, **kwargs):
    """Saves figure, saves copy if `path` already is a file."""
    fig.tight_layout()
    if not os.path.isfile(path):
        fig.savefig(path, **kwargs)
    else:
        zeroth_path = file_enum_format(path, 0)
        if not os.path.isfile(zeroth_path):
            os.rename(path, zeroth_path)
        fig.savefig(path, **kwargs)
        fig.savefig(next_filename(path), **kwargs)


def imsave(path: str, img: np.ndarray, **kwargs) -> None:
    """Saves image, saves copy if `path` already is a file."""
    if not os.path.isfile(path):
        plt.imsave(path, img, **kwargs)
    else:
        zeroth_path = file_enum_format(path, 0)
        if not os.path.isfile(zeroth_path):
            os.rename(path, zeroth_path)
        plt.imsave(path, img, **kwargs)
        plt.imsave(next_filename(path), img, **kwargs)


def get_toy2d_eval_conf(eval_conf) -> dict:
    conf = dict(TOY2D_DEFAULTS)
    conf.update(dict(eval_conf.get("toy2d", {})))
    return conf


def select_evenly_spaced(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[0] <= n:
        return x
    indices = np.linspace(0, x.shape[0] - 1, n).round().astype(int)
    return x[indices]


def plot_loss(losses, dir: str) -> None:
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    savefig(fig, os.path.join(dir, "training_loss.png"))
    plt.close(fig)


def display_path(path: str) -> str:
    path = os.path.abspath(path)
    parts = os.path.normpath(path).split(os.sep)
    if "outputs" in parts:
        return os.path.join(*parts[parts.index("outputs") :])
    return path


def copy_if_exists(src: str, dst: str, warn_missing: bool = True) -> bool:
    if not os.path.isfile(src):
        if warn_missing:
            logger.warning("Missing Overleaf artifact source: %s", src)
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True
