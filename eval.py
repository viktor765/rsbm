import argparse
import logging
import os

from omegaconf import DictConfig, OmegaConf

from src import data_utils, models, utils
from src.rsbm import create_sde_and_bridge, trainers

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation of model at checkpoint."
    )
    parser.add_argument("dir", type=str, help="Path to config file.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Explicit choice of config file, otherwise inferred",
    )
    args = parser.parse_args()
    checkpoint_dir = args.dir
    if os.path.isfile(checkpoint_dir):
        checkpoint_dir = os.path.dirname(checkpoint_dir)

    out_dir = os.path.split(os.path.split(checkpoint_dir)[0])[0]

    if args.config is None:
        config_path = os.path.join(out_dir, ".hydra", "config.yaml")
    else:
        config_path = args.config

    print(checkpoint_dir)
    print(config_path)

    config = OmegaConf.load(config_path)
    assert isinstance(config, DictConfig), f"Expected DictConfig, got {type(config)}"
    config.out_dir = out_dir
    config.training.resume.dir = checkpoint_dir
    config.eval.compute_fid = True
    return config


def main(cfg: DictConfig):
    logger.info("Config:")
    logger.info(OmegaConf.to_yaml(cfg))

    utils.set_seed(cfg.seed)

    model = models.create_model(cfg.model)
    sde, bridge = create_sde_and_bridge(
        cfg.sde, data_utils.get_data_normalization(cfg.data)
    )

    logger.info(
        f"Model size (nbr of params): {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    if cfg.method == "alpha_imf":
        trainer_class = trainers.get_trainer(model.__class__)
        trainer = trainer_class(
            model=model,
            sde=sde,
            bridge=bridge,
            training_conf=cfg.training,
            eval_conf=cfg.eval,
            data_conf=cfg.data,
            seed=cfg.seed,
            out_dir=cfg.out_dir,
        )
    else:
        raise NotImplementedError(f"Method {cfg.method} not implemented.")

    trainer.evaluate(final_artifacts=True)


if __name__ == "__main__":
    main(parse_args())
