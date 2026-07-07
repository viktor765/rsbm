from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
    sobel,
)

CLASS_SKY = 0
CLASS_WATER = 1
CLASS_LIGHT_BRIDGE = 2
CLASS_DARK_BRIDGE = 3

CLASS_NAMES = ("sky", "water", "light bridge", "dark bridge")
CLASS_COLORS = ("#b8dcf4", "#1f5c93", "#ded8c9", "#1b1e23")


@dataclass(frozen=True)
class BridgeSegmentationConfig:
    image_height: int = 288
    autocontrast_cutoff: float = 1.0
    blue_excess_offset: float = 0.015
    blue_excess_scale: float = 0.23
    saturation_offset: float = 0.08
    saturation_scale: float = 0.50
    blue_blur: float = 1.2
    blue_threshold: float = 0.22
    horizon_search_low: float = 0.48
    horizon_search_high: float = 0.90
    horizon_transition_width: float = 0.035
    structural_dark_percentile: float = 73.0
    structural_dark_edge_threshold: float = 0.12
    bridge_dark_quantile_low: float = 35.0
    bridge_dark_quantile_high: float = 75.0


@dataclass(frozen=True)
class BridgeThresholdMaskConfig:
    reference_height: int = 288
    reference_width: int = 384
    light_bridge_red_threshold: float = 0.70
    light_bridge_col_start: int = 175
    sky_row_stop: int = 238
    water_row_start: int = 235
    dark_bridge_exclude_lower_left_row_start: int = 200
    dark_bridge_exclude_lower_left_col_stop: int = 175
    water_blue_threshold: float = 0.50
    dark_bridge_blue_threshold: float = 0.52
    autocontrast_cutoff: float = 1.0


@dataclass(frozen=True)
class BridgeGradientConfig:
    sky_top_multiplier: float = 1.8
    sky_bottom_multiplier: float = 0.0
    sky_curve: float = 0.5
    preserve_class_mean: bool = False


@dataclass(frozen=True)
class BridgeClassWeights:
    sky: float = 0.05
    water: float = 0.75
    light_bridge: float = 1.50
    dark_bridge: float = 5.00

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.sky, self.water, self.light_bridge, self.dark_bridge],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class SegmentationResult:
    image: np.ndarray
    luminance: np.ndarray
    darkness: np.ndarray
    blue_score: np.ndarray
    structure: np.ndarray
    classes: np.ndarray
    horizon_row: int
    bridge_dark_threshold: float
    aspect: float


def load_aspect_image(path: str | Path, image_height: int = 288) -> Image.Image:
    image = Image.open(path).convert("RGB")
    aspect = image.width / image.height
    width = int(round(image_height * aspect))
    return image.resize((width, image_height), Image.Resampling.LANCZOS)


def robust01(values: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lower, upper = np.percentile(values, [lo, hi])
    if upper <= lower + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def _structure_score(luminance: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter(luminance, sigma=0.8, mode="nearest")
    gradient = np.hypot(
        sobel(smooth, axis=0, mode="nearest"),
        sobel(smooth, axis=1, mode="nearest"),
    )
    gradient = robust01(gradient, 2.0, 99.7)

    dog_small = np.abs(
        gaussian_filter(luminance, 0.7, mode="nearest")
        - gaussian_filter(luminance, 3.5, mode="nearest")
    )
    dog_big = np.abs(
        gaussian_filter(luminance, 1.5, mode="nearest")
        - gaussian_filter(luminance, 8.0, mode="nearest")
    )
    dog = robust01(
        0.65 * robust01(dog_small, 2.0, 99.5) + 0.35 * robust01(dog_big, 2.0, 99.5),
        1.0,
        99.5,
    )

    mean = gaussian_filter(luminance, sigma=7.0, mode="nearest")
    variance = gaussian_filter((luminance - mean) ** 2, sigma=4.0, mode="nearest")
    local_contrast = robust01(np.sqrt(np.maximum(variance, 0.0)), 2.0, 99.5)
    structure = robust01(
        0.45 * gradient + 0.35 * dog + 0.20 * local_contrast,
        1.0,
        99.5,
    )
    return robust01(
        0.55 * gaussian_filter(structure, 1.0, mode="nearest")
        + 0.45 * gaussian_filter(structure, 4.0, mode="nearest"),
        1.0,
        99.5,
    )


def _estimate_horizon_row(
    luminance: np.ndarray, config: BridgeSegmentationConfig
) -> int:
    height = luminance.shape[0]
    row_luminance = gaussian_filter(luminance.mean(axis=1), sigma=3.0)
    row_gradient = np.abs(np.gradient(row_luminance))
    start = int(config.horizon_search_low * height)
    stop = int(config.horizon_search_high * height)
    search_rows = np.arange(max(0, start), min(height, stop))
    if search_rows.size == 0:
        return height // 2
    return int(search_rows[np.argmax(row_gradient[search_rows])])


def segment_bridge_image(
    image: Image.Image,
    config: BridgeSegmentationConfig | None = None,
) -> SegmentationResult:
    if config is None:
        config = BridgeSegmentationConfig(image_height=image.height)

    if image.mode != "RGB":
        image = image.convert("RGB")

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1e-6)

    gray = ImageOps.autocontrast(
        ImageOps.grayscale(image),
        cutoff=float(config.autocontrast_cutoff),
    )
    luminance = np.asarray(gray, dtype=np.float32) / 255.0
    darkness = 1.0 - luminance
    structure = _structure_score(luminance)

    blue_excess = blue - 0.5 * (red + green)
    blue_score = np.clip(
        (blue_excess - config.blue_excess_offset) / config.blue_excess_scale,
        0.0,
        1.0,
    ) * np.clip(
        (saturation - config.saturation_offset) / config.saturation_scale,
        0.0,
        1.0,
    )
    blue_score = gaussian_filter(blue_score, sigma=config.blue_blur, mode="nearest")

    height, width = luminance.shape
    horizon_row = _estimate_horizon_row(luminance, config)
    rows = np.arange(height)[:, None]
    transition = config.horizon_transition_width * height
    water_prior = 1.0 / (1.0 + np.exp(-(rows - horizon_row) / transition))
    sky_prior = 1.0 - water_prior

    blue_mask = blue_score > config.blue_threshold
    blue_mask = binary_opening(blue_mask, structure=np.ones((2, 2)))
    blue_mask = binary_closing(blue_mask, structure=np.ones((3, 3)))

    edge = np.hypot(
        sobel(luminance, axis=0, mode="nearest"),
        sobel(luminance, axis=1, mode="nearest"),
    )
    edge = np.clip(edge / (np.percentile(edge, 99.5) + 1e-8), 0.0, 1.0)
    structural_dark = (
        darkness > np.percentile(darkness, config.structural_dark_percentile)
    ) & (edge > config.structural_dark_edge_threshold)

    bridge_mask = (~blue_mask) | structural_dark
    sky_mask = blue_mask & (sky_prior >= 0.35) & (~bridge_mask)
    water_mask = blue_mask & (water_prior > 0.35) & (~bridge_mask)
    unassigned = ~(sky_mask | water_mask | bridge_mask)
    sky_mask = sky_mask | (unassigned & (sky_prior >= water_prior))
    water_mask = water_mask | (unassigned & (water_prior > sky_prior))
    bridge_mask = ~(sky_mask | water_mask)

    bridge_values = darkness[bridge_mask]
    if bridge_values.size:
        c0, c1 = np.percentile(
            bridge_values,
            [config.bridge_dark_quantile_low, config.bridge_dark_quantile_high],
        )
        for _ in range(25):
            high = np.abs(bridge_values - c1) < np.abs(bridge_values - c0)
            if high.any():
                c1 = float(bridge_values[high].mean())
            if (~high).any():
                c0 = float(bridge_values[~high].mean())
        bridge_dark_threshold = float(0.5 * (c0 + c1))
    else:
        bridge_dark_threshold = float(np.median(darkness))

    dark_bridge = bridge_mask & (darkness >= bridge_dark_threshold)
    light_bridge = bridge_mask & (~dark_bridge)

    classes = np.zeros((height, width), dtype=np.int16)
    classes[sky_mask] = CLASS_SKY
    classes[water_mask] = CLASS_WATER
    classes[light_bridge] = CLASS_LIGHT_BRIDGE
    classes[dark_bridge] = CLASS_DARK_BRIDGE

    return SegmentationResult(
        image=rgb.astype(np.float32),
        luminance=luminance.astype(np.float32),
        darkness=darkness.astype(np.float32),
        blue_score=blue_score.astype(np.float32),
        structure=structure.astype(np.float32),
        classes=classes,
        horizon_row=horizon_row,
        bridge_dark_threshold=bridge_dark_threshold,
        aspect=width / height,
    )


def _scaled_row(row: int, height: int, reference_height: int) -> int:
    return int(np.clip(round(row * height / reference_height), 0, height))


def _scaled_col(col: int, width: int, reference_width: int) -> int:
    return int(np.clip(round(col * width / reference_width), 0, width))


def nearest_valid_classes(
    mask_stack: np.ndarray,
    class_ids: tuple[int, ...],
    *,
    fallback_class: int = CLASS_SKY,
) -> np.ndarray:
    masks = np.asarray(mask_stack, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(
            f"Expected mask stack with shape (n_classes, H, W), got {masks.shape}"
        )
    if masks.shape[0] != len(class_ids):
        raise ValueError(f"Expected {masks.shape[0]} class ids, got {len(class_ids)}")

    coverage = masks.sum(axis=0)
    classes = np.full(masks.shape[1:], -1, dtype=np.int16)
    valid = coverage == 1
    for mask, class_id in zip(masks, class_ids):
        classes[valid & mask] = class_id

    if valid.any():
        _, indices = distance_transform_edt(~valid, return_indices=True)
        invalid = ~valid
        classes[invalid] = classes[tuple(indices[:, invalid])]
    else:
        classes[:, :] = fallback_class

    return classes


def segment_bridge_image_threshold_masks(
    image: Image.Image,
    config: BridgeThresholdMaskConfig | None = None,
) -> SegmentationResult:
    if config is None:
        config = BridgeThresholdMaskConfig()
    if image.mode != "RGB":
        image = image.convert("RGB")

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    height, width, _ = rgb.shape

    gray = ImageOps.autocontrast(
        ImageOps.grayscale(image),
        cutoff=float(config.autocontrast_cutoff),
    )
    luminance = np.asarray(gray, dtype=np.float32) / 255.0
    darkness = 1.0 - luminance
    structure = _structure_score(luminance)

    sky_row_stop = _scaled_row(config.sky_row_stop, height, config.reference_height)
    water_row_start = _scaled_row(
        config.water_row_start, height, config.reference_height
    )
    light_bridge_col_start = _scaled_col(
        config.light_bridge_col_start, width, config.reference_width
    )
    dark_bridge_exclude_row_start = _scaled_row(
        config.dark_bridge_exclude_lower_left_row_start,
        height,
        config.reference_height,
    )
    dark_bridge_exclude_col_stop = _scaled_col(
        config.dark_bridge_exclude_lower_left_col_stop,
        width,
        config.reference_width,
    )

    light_bridge_mask0 = rgb[..., 0] > config.light_bridge_red_threshold
    light_bridge_mask1 = np.zeros((height, width), dtype=bool)
    light_bridge_mask1[:, light_bridge_col_start:] = True
    light_bridge_mask = light_bridge_mask0 & light_bridge_mask1

    sky_mask0 = np.zeros((height, width), dtype=bool)
    sky_mask0[:sky_row_stop, :] = True

    water_mask0 = np.zeros((height, width), dtype=bool)
    water_mask0[water_row_start:, :] = True
    water_mask1 = rgb[..., 2] < config.water_blue_threshold
    water_mask = water_mask0 & water_mask1

    dark_bridge_mask1 = rgb[..., 2] < config.dark_bridge_blue_threshold
    dark_bridge_mask = sky_mask0 & dark_bridge_mask1
    dark_bridge_exclude = np.zeros((height, width), dtype=bool)
    dark_bridge_exclude[
        dark_bridge_exclude_row_start:, :dark_bridge_exclude_col_stop
    ] = True
    dark_bridge_mask = dark_bridge_mask & (~dark_bridge_exclude)

    sky_mask = sky_mask0 & ~(dark_bridge_mask | light_bridge_mask)

    mask_stack = np.stack(
        [sky_mask, water_mask, light_bridge_mask, dark_bridge_mask],
        axis=0,
    )
    classes = nearest_valid_classes(
        mask_stack,
        (CLASS_SKY, CLASS_WATER, CLASS_LIGHT_BRIDGE, CLASS_DARK_BRIDGE),
    )

    return SegmentationResult(
        image=rgb.astype(np.float32),
        luminance=luminance.astype(np.float32),
        darkness=darkness.astype(np.float32),
        blue_score=rgb[..., 2].astype(np.float32),
        structure=structure.astype(np.float32),
        classes=classes,
        horizon_row=water_row_start,
        bridge_dark_threshold=float(config.dark_bridge_blue_threshold),
        aspect=width / height,
    )


def class_weight_map(
    classes: np.ndarray,
    weights: BridgeClassWeights | Mapping[str, float] | Mapping[int, float],
) -> np.ndarray:
    if isinstance(weights, BridgeClassWeights):
        weight_values = weights.as_array()
    else:
        default = BridgeClassWeights()
        weight_values = default.as_array()
        for key, value in weights.items():
            if isinstance(key, str):
                class_index = {
                    "sky": CLASS_SKY,
                    "water": CLASS_WATER,
                    "light_bridge": CLASS_LIGHT_BRIDGE,
                    "light bridge": CLASS_LIGHT_BRIDGE,
                    "dark_bridge": CLASS_DARK_BRIDGE,
                    "dark bridge": CLASS_DARK_BRIDGE,
                }[key]
            else:
                class_index = int(key)
            weight_values[class_index] = float(value)
    result = weight_values[np.asarray(classes, dtype=np.int16)]
    return np.asarray(result, dtype=np.float32)


def apply_class_gradients(
    weights: np.ndarray,
    classes: np.ndarray,
    config: BridgeGradientConfig | None = None,
) -> np.ndarray:
    if config is None:
        config = BridgeGradientConfig()

    weights = np.asarray(weights, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int16)
    if weights.shape != classes.shape:
        raise ValueError(
            f"Weight map and class map must have same shape, got {weights.shape} and {classes.shape}"
        )

    height = weights.shape[0]
    row_fraction = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    sky_gradient = config.sky_top_multiplier + (
        config.sky_bottom_multiplier - config.sky_top_multiplier
    ) * np.power(row_fraction, config.sky_curve)
    sky_gradient = np.broadcast_to(sky_gradient, weights.shape).astype(np.float32)

    multipliers = np.ones_like(weights, dtype=np.float32)
    sky_mask = classes == CLASS_SKY
    if sky_mask.any():
        if config.preserve_class_mean:
            sky_mean = float(sky_gradient[sky_mask].mean())
            if sky_mean > 0:
                sky_gradient = sky_gradient / sky_mean
        multipliers[sky_mask] = sky_gradient[sky_mask]

    return np.asarray(weights * multipliers, dtype=np.float32)


def probability_from_weights(weights: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum(dtype=np.float64)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Weight map must have positive finite sum, got {total}")
    return probabilities / total


def sample_probability_grid(
    probabilities: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = probabilities.shape
    flat = probabilities.ravel().astype(np.float64)
    flat /= flat.sum(dtype=np.float64)
    indices = rng.choice(height * width, size=n_samples, replace=True, p=flat)
    pixel_y, pixel_x = np.divmod(indices, width)
    x = (pixel_x + rng.random(n_samples)) / width
    y = 1.0 - (pixel_y + rng.random(n_samples)) / height
    samples = np.column_stack([x, y]).astype(np.float32)
    if not np.isfinite(samples).all():
        raise ValueError("Samples contain non-finite values")
    if not ((samples >= 0.0) & (samples <= 1.0)).all():
        raise ValueError("Samples are outside [0, 1]^2")
    return samples


def effective_support(probabilities: np.ndarray) -> float:
    flat = probabilities.ravel().astype(np.float64)
    entropy = -np.sum(flat * np.log(flat + 1e-300))
    return float(np.exp(entropy) / flat.size)


def boundary_mass(probabilities: np.ndarray, band_fraction: float = 0.05) -> float:
    height, width = probabilities.shape
    band_y = max(1, int(round(height * band_fraction)))
    band_x = max(1, int(round(width * band_fraction)))
    return float(
        probabilities[:band_y, :].sum()
        + probabilities[-band_y:, :].sum()
        + probabilities[:, :band_x].sum()
        + probabilities[:, -band_x:].sum()
    )


def class_masses(probabilities: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    return {
        name: float(probabilities[np.asarray(classes) == class_index].sum())
        for class_index, name in enumerate(CLASS_NAMES)
    }
