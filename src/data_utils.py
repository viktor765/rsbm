import os
import random
from pathlib import Path

import numpy as np
import torch as th
from PIL import Image, ImageFilter, ImageOps
from sklearn.datasets import make_moons, make_swiss_roll
from torchvision import datasets, transforms

from .rsbm import normalizations


def resize_aspect(image: Image.Image, image_height: int) -> Image.Image:
    aspect = image.width / image.height
    width = int(round(image_height * aspect))
    return image.resize((width, image_height), Image.Resampling.LANCZOS)


def _seed_worker(worker_id):
    worker_seed = th.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def postprocessing(dataset: str):
    # To offset preprocessing normalization to e.g. [-1, 1]
    def _id(x):
        return x

    def _pm1(x):
        return (x + 1) / 2

    if dataset in {"mnist", "emnist", "afhq64"}:
        return _pm1
    else:
        return _id


def create_dataloaders(
    data_conf,
    batch_size: int,
    train_data: bool,
    for_training: bool,
    synth_data_base_seed: int,
    dl_generator: th.Generator | None = None,
    num_workers: int | None = None,
) -> dict[int, th.utils.data.DataLoader]:
    data_loaders = {}
    for i in [0, 1]:
        dataset = get_dataset(
            data_conf[f"data{i}"], train_data, for_training, synth_data_base_seed + i
        )
        if len(dataset) == 0:
            raise ValueError(f"Dataset {data_conf[f'data{i}'].dataset} is empty.")
        loader_batch_size = batch_size
        if len(dataset) < loader_batch_size and not for_training:
            loader_batch_size = len(dataset)
        if len(dataset) < loader_batch_size:
            raise ValueError(
                f"Dataset {data_conf[f'data{i}'].dataset} size {len(dataset)} "
                f"is smaller than batch size {loader_batch_size}."
            )
        if num_workers is None:
            num_workers_int: int = data_conf.num_workers
        else:
            num_workers_int = num_workers
        data_loaders[i] = th.utils.data.DataLoader(
            dataset,
            batch_size=loader_batch_size,
            shuffle=for_training,
            num_workers=num_workers_int,
            pin_memory=data_conf.pin_memory,
            drop_last=for_training,
            worker_init_fn=_seed_worker,
            generator=dl_generator,
        )
    return data_loaders


def _filter_dataset_by_labels(
    dataset, labels: list[str] | None
) -> th.utils.data.Dataset:
    if labels is None:
        return dataset
    label_indices = set(dataset.class_to_idx[label] for label in labels)
    indices = [
        i for i, target in enumerate(dataset.targets) if int(target) in label_indices
    ]
    return th.utils.data.Subset(dataset, indices)


def get_dataset(
    data_conf, train_data: bool, for_training: bool, synth_data_base_seed: int
) -> th.utils.data.Dataset:
    dataset_name = data_conf.dataset
    root = data_conf.get("root", "data")
    if dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,), (0.5,)
                ),  # scale to ([0, 1] - .5) / .5 = [-1, 1]
            ]
        )
        dataset = datasets.MNIST(
            root=root, train=train_data, download=True, transform=transform
        )
        return _filter_dataset_by_labels(dataset, data_conf.labels)
    elif dataset_name == "emnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.transpose(1, 2)),
                transforms.Normalize((0.5,), (0.5,)),  # scale to [-1, 1]
            ]
        )
        dataset = datasets.EMNIST(
            root=root,
            split="letters",
            train=train_data,
            download=True,
            transform=transform,
        )
        return _filter_dataset_by_labels(dataset, data_conf.labels)
    elif dataset_name == "afhq64":
        if for_training:
            transf_comp = transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            transf_comp = transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        dataset = datasets.ImageFolder(
            root=os.path.join("data", "AFHQ", "train" if train_data else "test"),
            transform=transf_comp,
        )
        return _filter_dataset_by_labels(dataset, data_conf.labels)
    elif dataset_name in {
        "two_moons",
        "checkerboard",
        "swiss_roll",
        "image_density",
    }:
        return get_toy_dataset(data_conf, train_data, synth_data_base_seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _normalize(x: np.ndarray):  # to unit cube
    return (x - x.min(0)) / (x.max(0) - x.min(0))


def make_checkerboard(n_samples: int, rng: np.random.Generator):
    x = rng.random((n_samples, 2))
    x[x[:, 0] % 0.5 >= 0.25, 1] += 1
    x[:, 1] += rng.integers(2, size=n_samples) * 2
    x[:, 1] /= 4
    return x


def _resolve_repo_path(path: str | os.PathLike) -> Path:
    path = Path(path)
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Could not resolve path: {path}")

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    repo_path = Path(__file__).resolve().parents[1] / path
    if repo_path.exists():
        return repo_path

    raise FileNotFoundError(f"Could not resolve path: {path}")


def _load_density_image(conf) -> np.ndarray:
    density = conf.get("density", "darkness")
    if density == "bridge_segmentation":
        return _load_bridge_segmentation_density(conf)
    if density != "darkness":
        raise ValueError(f"Unknown image density type: {density}")

    image_path = _resolve_repo_path(conf.image_path)
    image_size = int(conf.get("image_size", 256))
    image = Image.open(image_path).convert("L")

    preprocessing = conf.get("preprocessing", "resize")
    if preprocessing == "resize":
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    elif preprocessing == "resize_aspect":
        image_height = int(conf.get("image_height", image_size))
        image = resize_aspect(image, image_height)
    elif preprocessing == "center_crop":
        centering = tuple(conf.get("centering", [0.5, 0.5]))
        image = ImageOps.fit(
            image,
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
            centering=centering,
        )
    else:
        raise ValueError(f"Unknown image preprocessing: {preprocessing}")

    if conf.get("autocontrast", True):
        image = ImageOps.autocontrast(image, cutoff=float(conf.get("cutoff", 1.0)))

    darkness = 1.0 - np.asarray(image, dtype=np.float32) / 255.0
    blur = float(conf.get("blur", 0.0))
    if blur > 0:
        darkness_image = Image.fromarray(np.uint8(np.clip(darkness, 0.0, 1.0) * 255))
        darkness_image = darkness_image.filter(ImageFilter.GaussianBlur(radius=blur))
        darkness = np.asarray(darkness_image, dtype=np.float32) / 255.0

    eps = float(conf.get("eps", 0.0))
    gamma = float(conf.get("gamma", 1.0))
    weights = eps + np.power(np.clip(darkness, 0.0, 1.0), gamma)
    weights = np.asarray(weights, dtype=np.float64)
    weights_sum = weights.sum(dtype=np.float64)
    if not np.isfinite(weights_sum) or weights_sum <= 0:
        raise ValueError(
            f"Image density weights must have positive finite sum, got {weights_sum}"
        )
    return weights / weights_sum


def _config_dict(conf_section) -> dict:
    if conf_section is None:
        return {}
    return {str(key): value for key, value in conf_section.items()}


def _load_bridge_segmentation_density(conf) -> np.ndarray:
    from .image_density_segmentation import (
        BridgeClassWeights,
        BridgeGradientConfig,
        BridgeSegmentationConfig,
        BridgeThresholdMaskConfig,
        apply_class_gradients,
        class_weight_map,
        load_aspect_image,
        probability_from_weights,
        segment_bridge_image,
        segment_bridge_image_threshold_masks,
    )

    image_path = _resolve_repo_path(conf.image_path)
    image_height = int(conf.get("image_height", conf.get("image_size", 288)))
    image = load_aspect_image(image_path, image_height=image_height)

    threshold_config = BridgeThresholdMaskConfig(
        **_config_dict(conf.get("threshold_mask", None))
    )
    result = segment_bridge_image_threshold_masks(image, threshold_config)

    class_weights = BridgeClassWeights(**_config_dict(conf.get("class_weights", None)))
    weights = class_weight_map(result.classes, class_weights)
    gradient_config = BridgeGradientConfig(
        **_config_dict(conf.get("gradient", None))
    )
    weights = apply_class_gradients(weights, result.classes, gradient_config)

    return probability_from_weights(weights)


def make_image_density(conf, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    density = _load_density_image(conf)
    height, width = density.shape
    probabilities = density.ravel().astype(np.float64)
    probabilities /= probabilities.sum(dtype=np.float64)

    indices = rng.choice(height * width, size=n_samples, p=probabilities)
    pixel_y, pixel_x = np.divmod(indices, width)

    x = (pixel_x + rng.random(n_samples)) / width
    y = 1.0 - (pixel_y + rng.random(n_samples)) / height
    return np.column_stack([x, y])


def get_toy_dataset(conf, train_data: bool, synth_data_base_seed: int):
    dataset_name = conf.dataset
    n_samples = conf.n_samples

    seed = (
        synth_data_base_seed + {True: 1299782495, False: 1031696584}[train_data]
    ) % 2**32

    if dataset_name == "two_moons":
        x, _ = make_moons(n_samples=n_samples, noise=conf.noise, random_state=seed)
        x = _normalize(x)
    elif dataset_name == "swiss_roll":
        x, _ = make_swiss_roll(n_samples=n_samples, noise=conf.noise, random_state=seed)
        x = x[:, [0, 2]]
        x = _normalize(x)
    elif dataset_name == "checkerboard":
        x = make_checkerboard(
            n_samples, np.random.default_rng(seed)
        )  # already normalized
    elif dataset_name == "image_density":
        x = make_image_density(conf, n_samples, np.random.default_rng(seed))
    else:
        raise NotImplementedError(f"Toy dataset {dataset_name} not implemented")

    return th.utils.data.TensorDataset(th.from_numpy(x).to(th.float32))


def make_infinite(dataloader: th.utils.data.DataLoader):
    while True:
        yield from dataloader


def sample_timesteps(
    batch_size: int, device: th.device, generator: th.Generator, eps=1e-4
):
    return (
        th.rand(batch_size, device=device, generator=generator) * (1.0 - 2 * eps) + eps
    )


def get_data_normalization(data_conf) -> normalizations.Normalization:
    # Heuristics; can be extended later
    if (
        "mnist" in data_conf.data0.dataset.lower()
        or "afhq" in data_conf.data0.dataset.lower()
    ):
        return normalizations.PM1
    else:
        return normalizations.IDENTITY
