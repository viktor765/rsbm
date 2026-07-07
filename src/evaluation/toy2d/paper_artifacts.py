import json
import logging
import os

import numpy as np
import torch as th

from ..common import copy_if_exists, display_path, select_evenly_spaced
from .plots import (
    load_density_display_image,
    plot_2d,
    plot_generation_procedure,
    plot_snapshots_and_vector_fields,
    plot_trajectories,
    plot_trajectory_frame,
    plot_trajectory_gif,
)

logger = logging.getLogger(__name__)


def write_final_direction_artifacts(
    *,
    context,
    model: th.nn.Module,
    direction_dir: str,
    direction_name: str,
    sampling_direction: int,
    trajectory: np.ndarray,
) -> None:
    if not context.toy2d_conf["final_artifacts"]:
        return
    plot_trajectories(
        trajectory[:1],
        direction_dir,
        "single_trajectory",
        title=direction_name,
    )
    plot_trajectories(
        select_evenly_spaced(trajectory, 5),
        direction_dir,
        "five_trajectories",
        title=direction_name,
    )
    gif_paths = select_evenly_spaced(
        trajectory,
        context.toy2d_conf["n_gif_paths"],
    )
    plot_trajectory_gif(
        gif_paths,
        direction_dir,
        "paths",
        n_frames=context.toy2d_conf["n_gif_frames"],
    )
    plot_trajectory_frame(gif_paths, direction_dir, "paths_final_frame")
    plot_snapshots_and_vector_fields(
        trajectory,
        model=model,
        direction=sampling_direction,
        device=context.device,
        autocast=context.autocast,
        dir=direction_dir,
        file_name="snapshots_and_vector_field",
        n_times=context.toy2d_conf["n_snapshot_times"],
        n_particles=context.toy2d_conf["n_snapshot_particles"],
        vector_grid_size=context.toy2d_conf["vector_grid_size"],
    )


def write_generation_procedure_artifact(
    *,
    context,
    base_dir: str,
    true_samples: dict[int, th.Tensor],
    direction_artifacts: dict[str, dict[str, np.ndarray]],
) -> None:
    if not context.toy2d_conf["final_artifacts"]:
        return
    if not (
        context.data_conf.data0.dataset == "image_density"
        and context.data_conf.data1.dataset == "image_density"
    ):
        return
    if "0_to_1" not in direction_artifacts or "1_to_0" not in direction_artifacts:
        logger.warning("Skipping generation procedure figure; missing direction data.")
        return

    image0 = load_density_display_image(context.data_conf.data0, full_resolution=True)
    image1 = load_density_display_image(context.data_conf.data1, full_resolution=True)
    plot_generation_procedure(
        image0=image0,
        image1=image1,
        x0=context.to_plot_numpy(0, true_samples[0]),
        x1=context.to_plot_numpy(1, true_samples[1]),
        fake_01=direction_artifacts["0_to_1"]["sampled"],
        fake_10=direction_artifacts["1_to_0"]["sampled"],
        paths_01=direction_artifacts["0_to_1"]["trajectory"],
        paths_10=direction_artifacts["1_to_0"]["trajectory"],
        dir=base_dir,
        n_points=context.toy2d_conf["n_generation_points"],
        n_paths=context.toy2d_conf["n_generation_paths"],
    )


def write_overleaf_upload_artifacts(
    *,
    context,
    model: th.nn.Module,
    base_dir: str,
    metrics: dict,
    direction_artifacts: dict[str, dict[str, np.ndarray]],
) -> None:
    upload_dir = os.path.join(base_dir, context.toy2d_conf["overleaf_dir_name"])
    os.makedirs(upload_dir, exist_ok=True)

    run_dir = os.path.dirname(context.out_dir)
    hydra_config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    metadata_path = os.path.join(upload_dir, "metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(f"output_subdir: {display_path(base_dir)}\n")
        f.write(f"output_subdir_absolute: {os.path.abspath(base_dir)}\n")
        f.write(f"hydra_config_path: {display_path(hydra_config_path)}\n")
        hydra_config_abs = os.path.abspath(hydra_config_path)
        f.write(f"hydra_config_path_absolute: {hydra_config_abs}\n")
        f.write("\nhydra_config:\n")
        if os.path.isfile(hydra_config_path):
            with open(hydra_config_path, encoding="utf-8") as config_file:
                f.write(config_file.read())
        else:
            f.write("<missing>\n")

    metrics_path = os.path.join(upload_dir, "eval_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, indent=2, ensure_ascii=False))
        f.write("\n")

    copy_if_exists(
        os.path.join(base_dir, "generation_procedure.pdf"),
        os.path.join(upload_dir, "generation_procedure.pdf"),
    )

    direction_map = {
        "0_to_1": ("fwd", 1),
        "1_to_0": ("bwd", -1),
    }
    for direction_name, (upload_name, sampling_direction) in direction_map.items():
        target_index = 1 if direction_name == "0_to_1" else 0
        target_aspect = context.plot_aspect(target_index)
        direction_dir = os.path.join(base_dir, direction_name)
        upload_direction_dir = os.path.join(upload_dir, upload_name)
        os.makedirs(upload_direction_dir, exist_ok=True)

        if direction_name in direction_artifacts:
            sampled_np = direction_artifacts[direction_name]["sampled"]
            plot_2d(
                sampled_np,
                upload_direction_dir,
                "sampled_raw",
                extension="pdf",
                enumerate_existing=False,
                transparent=True,
                width_height_aspect=target_aspect,
            )
            if os.path.isfile(os.path.join(direction_dir, "sampled_projected.png")):
                plot_2d(
                    np.clip(sampled_np, 0.0, 1.0),
                    upload_direction_dir,
                    "sampled_projected",
                    extension="pdf",
                    enumerate_existing=False,
                    transparent=True,
                    width_height_aspect=target_aspect,
                )

            trajectory = direction_artifacts[direction_name]["trajectory"]
            gif_paths = select_evenly_spaced(
                trajectory, context.toy2d_conf["n_gif_paths"]
            )
            plot_trajectory_frame(
                gif_paths,
                upload_direction_dir,
                "paths_final_frame",
                show_title=False,
                enumerate_existing=False,
                extension="pdf",
            )
            plot_snapshots_and_vector_fields(
                trajectory,
                model=model,
                direction=sampling_direction,
                device=context.device,
                autocast=context.autocast,
                dir=upload_direction_dir,
                file_name="snapshots_and_vector_field",
                n_times=context.toy2d_conf["n_snapshot_times"],
                n_particles=context.toy2d_conf["n_snapshot_particles"],
                vector_grid_size=context.toy2d_conf["vector_grid_size"],
                extension="pdf",
                enumerate_existing=False,
            )
        else:
            logger.warning(
                "Missing trajectory data for Overleaf artifact: %s", direction_name
            )
