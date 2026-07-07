import json
import logging
import os

import numpy as np
import torch as th

from ...rsbm.sdes import ReflectedSDE
from ..sampling import sample_direction
from .metrics import empirical_w2, support_metrics
from .paper_artifacts import (
    write_final_direction_artifacts,
    write_generation_procedure_artifact,
    write_overleaf_upload_artifacts,
)
from .plots import plot_2d

logger = logging.getLogger(__name__)


class Toy2DEvaluator:
    def __init__(self, context):
        self.context = context
        self.is_reflected = isinstance(context.sde, ReflectedSDE)

    def prepare(self) -> None:
        if self.context.eval_conf.get("compute_fid", True):
            logger.warning("FID requested for 2D toy data; skipping FID.")
        if self.context.eval_conf.get("plot_initial", True):
            self.write_initial_plots()

    def write_initial_plots(self) -> None:
        for dataset_index in (0, 1):
            data = self.context.postprocessing[dataset_index](
                th.stack(
                    [
                        self.context.test_data[dataset_index].dataset[j][0]
                        for j in range(self.context.n_images)
                    ]
                )
            )
            plot_2d(
                data.numpy(),
                self.context.out_dir,
                f"ds{dataset_index}_true",
                width_height_aspect=self.context.plot_aspect(dataset_index),
            )
            plot_2d(
                data.numpy(),
                self.context.out_dir,
                f"ds{dataset_index}_true",
                extension="pdf",
                transparent=True,
                width_height_aspect=self.context.plot_aspect(dataset_index),
            )

    def evaluate(
        self, model: th.nn.Module, step: int, final_artifacts: bool = False
    ) -> None:
        generator = self.context.new_generator()
        model.eval()

        true_samples = self.context.true_eval_samples()
        base_dir = self.context.step_dir(step)
        os.makedirs(base_dir, exist_ok=True)

        metrics = {}
        direction_artifacts = {}
        for direction in self.context.directions:
            direction_dir = os.path.join(base_dir, direction.name)
            os.makedirs(direction_dir, exist_ok=True)

            source = true_samples[direction.source_index]
            target = true_samples[direction.target_index]
            sampled, trajectory = sample_direction(
                sde=self.context.sde,
                model=model,
                source=source,
                direction=direction,
                device=self.context.device,
                generator=generator,
                autocast=self.context.autocast,
                return_trajectory=True,
            )

            target_np = self.context.to_plot_numpy(direction.target_index, target)
            sampled_np = self.context.to_plot_numpy(direction.target_index, sampled)
            trajectory_np = trajectory.numpy()
            direction_artifacts[direction.name] = {
                "sampled": sampled_np,
                "trajectory": trajectory_np,
            }

            target_aspect = self.context.plot_aspect(direction.target_index)
            plot_2d(
                sampled_np,
                direction_dir,
                "sampled_raw_w_ref",
                ref_x=target_np,
                width_height_aspect=target_aspect,
            )
            plot_2d(
                sampled_np,
                direction_dir,
                "sampled_raw",
                width_height_aspect=target_aspect,
            )

            direction_metrics = {
                "raw": {
                    **empirical_w2(
                        sampled_np,
                        target_np,
                        max_samples=self.context.toy2d_conf["w2_samples"],
                        seed=self.context.toy2d_conf["w2_seed"],
                    ),
                    "final_support": support_metrics(
                        sampled_np,
                        boundary_eps=self.context.toy2d_conf["boundary_eps"],
                    ),
                }
            }

            if not self.is_reflected:
                projected_np = np.clip(sampled_np, 0.0, 1.0)
                plot_2d(
                    projected_np,
                    direction_dir,
                    "sampled_projected",
                    ref_x=target_np,
                    width_height_aspect=target_aspect,
                )
                direction_metrics["projected"] = {
                    **empirical_w2(
                        projected_np,
                        target_np,
                        max_samples=self.context.toy2d_conf["w2_samples"],
                        seed=self.context.toy2d_conf["w2_seed"],
                    ),
                    "final_support": support_metrics(
                        projected_np,
                        boundary_eps=self.context.toy2d_conf["boundary_eps"],
                    ),
                }

            if final_artifacts:
                write_final_direction_artifacts(
                    context=self.context,
                    model=model,
                    direction_dir=direction_dir,
                    direction_name=direction.name,
                    sampling_direction=direction.sampling_direction,
                    trajectory=trajectory_np,
                )

            metrics_path = os.path.join(direction_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(direction_metrics, f, indent=2, ensure_ascii=False)
            metrics[direction.name] = direction_metrics
            logger.info(
                "Step %s: %s W2 %.6f",
                step,
                direction.name,
                direction_metrics["raw"]["w2"],
            )

        if final_artifacts:
            write_generation_procedure_artifact(
                context=self.context,
                base_dir=base_dir,
                true_samples=true_samples,
                direction_artifacts=direction_artifacts,
            )

        metrics_path = os.path.join(base_dir, "eval_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        if final_artifacts:
            write_overleaf_upload_artifacts(
                context=self.context,
                model=model,
                base_dir=base_dir,
                metrics=metrics,
                direction_artifacts=direction_artifacts,
            )
