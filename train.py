import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.contrib.logging import logging_redirect_tqdm
from accelerate import PartialState

from src import data_utils, models, utils
from src.rsbm import create_sde_and_bridge, trainers

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    is_main_process = PartialState().is_main_process

    with logging_redirect_tqdm():
        if is_main_process:
            logger.info(f"Output dir: {cfg.out_dir}")
            logger.info("Config:")
            logger.info(OmegaConf.to_yaml(cfg))

        utils.set_seed(cfg.seed)

        model = models.create_model(cfg.model)
        sde, bridge = create_sde_and_bridge(
            cfg.sde, data_utils.get_data_normalization(cfg.data)
        )

        if is_main_process:
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

        trainer.train()

        if is_main_process:
            logger.info("Training completed")


if __name__ == "__main__":
    main()
