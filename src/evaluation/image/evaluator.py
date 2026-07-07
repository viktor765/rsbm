import os

import torch as th

from ..directions import DIRECTION_BY_NAME
from ..sampling import sample_direction
from .cached_samples import write_cached_samples
from .metrics import ImageMetricEvaluator, write_image_metrics
from .oob_metrics import write_cached_sample_oob_metrics
from .plots import plot_images


class ImageEvaluator:
    def __init__(self, context):
        self.context = context
        self.metrics = ImageMetricEvaluator(context)

    def prepare(self) -> None:
        if self.context.eval_conf.get("plot_initial", True):
            self.write_initial_plots()
        self.metrics.prepare()

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
            plot_images(data, self.context.out_dir, f"ds{dataset_index}_true", True)
            plot_images(data, self.context.out_dir, f"ds{dataset_index}_true", False)

    def evaluate(
        self, model: th.nn.Module, step: int, final_artifacts: bool = False
    ) -> None:
        generator = self.context.new_generator()
        model.eval()

        true_samples = self.context.true_eval_samples()
        x0_true = true_samples[0]
        x1_true = true_samples[1]
        x1_sampled = sample_direction(
            sde=self.context.sde,
            model=model,
            source=x0_true,
            direction=DIRECTION_BY_NAME["0_to_1"],
            device=self.context.device,
            generator=generator,
            autocast=self.context.autocast,
        )
        x0_sampled = sample_direction(
            sde=self.context.sde,
            model=model,
            source=x1_true,
            direction=DIRECTION_BY_NAME["1_to_0"],
            device=self.context.device,
            generator=generator,
            autocast=self.context.autocast,
        )
        samples = {0: x0_sampled, 1: x1_sampled}

        eval_dir = self.context.step_dir(step)
        os.makedirs(eval_dir, exist_ok=True)
        for dataset_index in (0, 1):
            plot_images(
                self.context.postprocessing[dataset_index](samples[dataset_index]),
                eval_dir,
                f"ds{dataset_index}_sampled",
                highlight_out_of_bounds=True,
            )
            plot_images(
                self.context.postprocessing[dataset_index](samples[dataset_index]),
                eval_dir,
                f"ds{dataset_index}_sampled",
                highlight_out_of_bounds=False,
            )

        if self.metrics.enabled:
            metrics = self.metrics.compute(model, step)
            write_image_metrics(os.path.join(eval_dir, "eval_metrics.json"), metrics)

        if final_artifacts:
            cached_by_item = write_cached_samples(self.context, model, eval_dir)
            write_cached_sample_oob_metrics(
                context=self.context,
                cached_by_item=cached_by_item,
                output_dir=eval_dir,
            )
