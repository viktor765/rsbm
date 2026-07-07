import json
import logging
import os

import numpy as np
import torch as th
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

logger = logging.getLogger(__name__)

INCEPTION_WEIGHTS_PATH = (
    r"data/checkpoints/weights-inception-2015-12-05-6726825d.pth"
)


class ImageMetricEvaluator:
    def __init__(self, context):
        self.context = context
        self.compute_fid = context.eval_conf.get("compute_fid", True)
        self.compute_lpips = context.eval_conf.get("compute_lpips", False)
        self.fids = None
        self.lpips = None

    def prepare(self) -> None:
        if self.compute_fid:
            self.fids = self._init_fids()
        self.lpips = self._init_lpips() if self.compute_lpips else None

    @property
    def enabled(self) -> bool:
        return self.compute_fid or self.lpips is not None

    def _init_fids(self):
        fids = {
            i: FrechetInceptionDistance(
                normalize=True,
                sync_on_compute=False,
                reset_real_features=False,
                feature_extractor_weights_path=INCEPTION_WEIGHTS_PATH,
            ).to(self.context.device)
            for i in (0, 1)
        }
        for i, fid in fids.items():
            for x, _ in self.context.train_data[i]:
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                fid.update(
                    self.context.postprocessing[i](x.to(self.context.device)),
                    real=True,
                )

        return fids

    def _init_lpips(self):
        if not self.context.lpips_supported():
            logger.warning(
                "LPIPS requested but requires RGB image datasets; skipping LPIPS."
            )
            return None

        try:
            return LearnedPerceptualImagePatchSimilarity().to(self.context.device)
        except Exception as exc:
            logger.warning("Failed to initialize LPIPS; skipping LPIPS. %s", exc)
            return None

    def compute(self, model: th.nn.Module, step: int) -> dict[str, dict[str, float]]:
        generator = self.context.new_generator()
        eval_metrics = {}
        for i, sampling_direction in zip((0, 1), (-1, 1), strict=True):
            eval_metrics[ds_key := f"ds{i}"] = {}
            fid = self.fids[i] if self.compute_fid else None
            if fid is not None:
                fid.reset()
            if self.lpips is not None:
                self.lpips.reset()

            msd = 0.0
            n_samples = 0
            for x, _ in self.context.test_data[1 - i]:
                x = x.to(self.context.device)
                x_sampled = self.context.sde.forward_euler(
                    model,
                    x,
                    direction=sampling_direction,
                    generator=generator,
                    autocast=self.context.autocast,
                )
                msd += th.sum((x_sampled - x) ** 2) / int(np.prod(x.shape[1:]))
                n_samples += x.shape[0]
                if fid is not None:
                    x_sampled_fid = x_sampled
                    if x_sampled_fid.shape[1] == 1:
                        x_sampled_fid = x_sampled_fid.repeat(1, 3, 1, 1)
                    x_sampled_fid = self.context.postprocessing[i](x_sampled_fid)
                    if self.context.clip_fid:
                        x_sampled_fid = x_sampled_fid.clamp(0, 1)
                    fid.update(x_sampled_fid, real=False)
                if self.lpips is not None:
                    self.lpips.update(x_sampled.clamp(-1, 1), x.clamp(-1, 1))

            msd = msd.item() / n_samples
            fid_score = fid.compute().item() if fid is not None else None
            lpips_score = (
                self.lpips.compute().item() if self.lpips is not None else None
            )

            metrics_log = [f"Step {step}: dataset_{i}"]
            if fid_score is not None:
                metrics_log.append(f"FID: {fid_score:.4f}")
                eval_metrics[ds_key]["fid"] = fid_score
            metrics_log.append(f"MSD: {msd:.6f}")
            if lpips_score is not None:
                metrics_log.append(f"LPIPS: {lpips_score:.6f}")
                eval_metrics[ds_key]["lpips"] = lpips_score
            logger.info(", ".join(metrics_log))
            eval_metrics[ds_key]["msd"] = msd
        return eval_metrics


def write_image_metrics(path: str, metrics: dict[str, dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
