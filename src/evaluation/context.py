import os
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch as th

from .. import data_utils
from .common import get_toy2d_eval_conf
from .directions import DIRECTIONS, DirectionSpec
from .toy2d.plots import image_aspect, load_density_display_image


@dataclass
class EvaluationContext:
    data_conf: object
    eval_conf: object
    synth_data_base_seed: int
    sde: object
    generator_state: th.Tensor
    out_dir: str
    device: th.device
    postprocessing0: object
    postprocessing1: object
    clip_fid: bool = True
    autocast: object | None = None
    n_images: int = field(init=False)
    train_data: dict = field(init=False)
    test_data: dict = field(init=False)
    postprocessing: dict[int, object] = field(init=False)
    toy2d_conf: dict = field(init=False)
    _plot_aspect_cache: dict[int, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.n_images = self.eval_conf.get("n_samples", 64)
        self.train_data = data_utils.create_dataloaders(
            self.data_conf,
            1024,
            True,
            False,
            self.synth_data_base_seed,
            dl_generator=None,
        )
        self.test_data = data_utils.create_dataloaders(
            self.data_conf,
            max(256, self.n_images),
            False,
            False,
            self.synth_data_base_seed,
            dl_generator=None,
            num_workers=0,
        )
        self.postprocessing = {0: self.postprocessing0, 1: self.postprocessing1}
        self.toy2d_conf = get_toy2d_eval_conf(self.eval_conf)
        self.autocast = nullcontext if self.autocast is None else self.autocast

    @property
    def is_2d(self) -> bool:
        sample = self.test_data[0].dataset[0][0]
        return sample.ndim == 1 and sample.shape[0] == 2

    @property
    def directions(self) -> tuple[DirectionSpec, ...]:
        return DIRECTIONS

    def lpips_supported(self) -> bool:
        return all(
            self.test_data[i].dataset[0][0].ndim == 3
            and self.test_data[i].dataset[0][0].shape[0] == 3
            for i in (0, 1)
        )

    def new_generator(self) -> th.Generator:
        generator = th.Generator(device=self.device)
        generator.set_state(self.generator_state.clone())
        return generator

    def true_eval_samples(self) -> dict[int, th.Tensor]:
        return {
            0: next(iter(self.test_data[0]))[0][: self.n_images],
            1: next(iter(self.test_data[1]))[0][: self.n_images],
        }

    def to_plot_numpy(self, dataset_index: int, x: th.Tensor):
        return self.postprocessing[dataset_index](x).numpy()

    def plot_aspect(self, dataset_index: int) -> float:
        if dataset_index not in self._plot_aspect_cache:
            conf = self.data_conf[f"data{dataset_index}"]
            if conf.dataset == "image_density":
                image = load_density_display_image(conf)
                self._plot_aspect_cache[dataset_index] = image_aspect(image)
            else:
                self._plot_aspect_cache[dataset_index] = 1.0
        return self._plot_aspect_cache[dataset_index]

    def step_dir(self, step: int) -> str:
        return os.path.join(self.out_dir, str(step).zfill(6))
