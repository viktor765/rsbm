import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch as th

from ... import data_utils
from ...rsbm import sdes
from ..directions import DIRECTIONS, DirectionSpec
from ..sampling import sample_direction

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHED_SAMPLES_CONF = {
    "enabled": True,
    "n_samples": 1024,
    "seed": 0,
}
CACHED_SAMPLE_BATCH_SIZE = 32
SPLITS = ("test", "train")


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def derive_seed(base_seed: int, *parts: str, bits: int = 32) -> int:
    payload = stable_json({"base_seed": int(base_seed), "parts": parts})
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return value % (2**bits)


def cached_samples_conf(eval_conf) -> dict[str, Any]:
    conf = dict(DEFAULT_CACHED_SAMPLES_CONF)
    conf.update(dict(eval_conf.get("cached_samples", {}) or {}))
    return conf


def cached_samples_enabled(eval_conf) -> bool:
    return bool(cached_samples_conf(eval_conf)["enabled"])


def select_start_indices(
    dataset_size: int,
    n_samples: int,
    seed: int,
    split: str,
    direction_name: str,
) -> np.ndarray:
    if dataset_size <= 0:
        raise ValueError("Cannot create cached samples from an empty dataset.")
    if int(n_samples) <= 0:
        raise ValueError("eval.cached_samples.n_samples must be positive.")
    n_selected = min(int(n_samples), int(dataset_size))
    rng = np.random.default_rng(
        derive_seed(seed, "start_indices", split, direction_name, bits=32)
    )
    return rng.choice(dataset_size, size=n_selected, replace=False).astype(np.int64)


def _sde_type_name(sde) -> str:
    if isinstance(sde, sdes.ReflectedSDE):
        return "reflected"
    if isinstance(sde, sdes.SDE):
        return "brownian"
    return sde.__class__.__name__


def _source_dataset(context, split: str, direction: DirectionSpec):
    train_data = split == "train"
    return data_utils.get_dataset(
        context.data_conf[f"data{direction.source_index}"],
        train_data,
        False,
        int(context.synth_data_base_seed) + direction.source_index,
    )


def _subset_loader(dataset, selected_indices: np.ndarray):
    subset = th.utils.data.Subset(dataset, selected_indices.tolist())
    return th.utils.data.DataLoader(
        subset,
        batch_size=CACHED_SAMPLE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def _cache_metadata_matches(
    *,
    cache: dict[str, Any],
    split: str,
    direction: DirectionSpec,
    n_samples: int,
    seed: int,
    dataset_size: int,
) -> bool:
    metadata = cache.get("metadata", {})
    return (
        metadata.get("schema_version") == CACHE_SCHEMA_VERSION
        and metadata.get("split") == split
        and metadata.get("direction") == direction.name
        and metadata.get("source_index") == direction.source_index
        and metadata.get("target_index") == direction.target_index
        and int(metadata.get("n_requested", -1)) == int(n_samples)
        and int(metadata.get("seed", -1)) == int(seed)
        and int(metadata.get("dataset_size", -1)) == int(dataset_size)
        and int(metadata.get("sampling_batch_size", -1)) == CACHED_SAMPLE_BATCH_SIZE
    )


def _cache_payload_matches(
    *,
    cache: dict[str, Any],
    selected_indices: np.ndarray,
) -> bool:
    expected_indices = th.from_numpy(selected_indices.copy())
    selected = cache.get("selected_indices")
    source = cache.get("source_pixels")
    generated = cache.get("generated_pixels")
    return (
        isinstance(selected, th.Tensor)
        and isinstance(source, th.Tensor)
        and isinstance(generated, th.Tensor)
        and th.equal(selected.cpu(), expected_indices)
        and source.shape[0] == len(selected_indices)
        and generated.shape[0] == len(selected_indices)
    )


def _load_matching_cache(
    *,
    cache_path: Path,
    split: str,
    direction: DirectionSpec,
    n_samples: int,
    seed: int,
    dataset_size: int,
    selected_indices: np.ndarray,
) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    try:
        cache = th.load(cache_path, map_location="cpu", weights_only=False)
    except Exception:
        logger.warning("Ignoring unreadable cached samples file: %s", cache_path)
        return None
    if not isinstance(cache, dict):
        logger.warning("Ignoring invalid cached samples file: %s", cache_path)
        return None
    if not _cache_metadata_matches(
        cache=cache,
        split=split,
        direction=direction,
        n_samples=n_samples,
        seed=seed,
        dataset_size=dataset_size,
    ):
        return None
    if not _cache_payload_matches(cache=cache, selected_indices=selected_indices):
        return None
    logger.info("Using cached samples from %s", cache_path)
    return cache


def write_cached_samples_for_direction(
    *,
    context,
    model: th.nn.Module,
    split: str,
    direction: DirectionSpec,
    output_dir: Path,
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    dataset = _source_dataset(context, split, direction)
    selected_indices = select_start_indices(
        len(dataset), n_samples, seed, split, direction.name
    )
    cache_path = output_dir / "cache" / f"{split}_{direction.name}.pt"
    cached = _load_matching_cache(
        cache_path=cache_path,
        split=split,
        direction=direction,
        n_samples=n_samples,
        seed=seed,
        dataset_size=len(dataset),
        selected_indices=selected_indices,
    )
    if cached is not None:
        return {"cache": cached, "cache_path": cache_path}

    loader = _subset_loader(dataset, selected_indices)
    generator = th.Generator(device=context.device)
    generator.manual_seed(derive_seed(seed, "sde_noise", split, direction.name, bits=63))

    source_batches = []
    generated_batches = []
    model.eval()
    with th.inference_mode():
        for batch in loader:
            source_model_space = batch[0] if isinstance(batch, (tuple, list)) else batch
            source_batches.append(
                context.postprocessing[direction.source_index](
                    source_model_space.detach().cpu().to(dtype=th.float32)
                )
            )
            sampled = sample_direction(
                sde=context.sde,
                model=model,
                source=source_model_space,
                direction=direction,
                device=context.device,
                generator=generator,
                autocast=context.autocast,
            )
            generated_batches.append(
                context.postprocessing[direction.target_index](
                    sampled.detach().cpu().to(dtype=th.float32)
                )
            )

    source_pixels = th.cat(source_batches, dim=0)
    generated_pixels = th.cat(generated_batches, dim=0)
    cache = {
        "metadata": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "split": split,
            "direction": direction.name,
            "source_index": direction.source_index,
            "target_index": direction.target_index,
            "n_requested": int(n_samples),
            "n_selected": int(len(selected_indices)),
            "seed": int(seed),
            "sde_type": _sde_type_name(context.sde),
            "dataset_size": int(len(dataset)),
            "sampling_batch_size": CACHED_SAMPLE_BATCH_SIZE,
        },
        "selected_indices": th.from_numpy(selected_indices.copy()),
        "source_pixels": source_pixels,
        "generated_pixels": generated_pixels,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    th.save(cache, cache_path)
    return {"cache": cache, "cache_path": cache_path}


def write_cached_samples(
    context, model: th.nn.Module, output_dir: str | Path
) -> dict[str, dict[str, Any]]:
    conf = cached_samples_conf(context.eval_conf)
    if not conf["enabled"]:
        return {}

    output_dir = Path(output_dir)
    cached_by_item = {}
    for split in SPLITS:
        for direction in DIRECTIONS:
            item_key = f"{split}_{direction.name}"
            logger.info("Writing cached samples for %s", item_key)
            cached_by_item[item_key] = write_cached_samples_for_direction(
                context=context,
                model=model,
                split=split,
                direction=direction,
                output_dir=output_dir,
                n_samples=int(conf["n_samples"]),
                seed=int(conf["seed"]),
            )
    return cached_by_item
