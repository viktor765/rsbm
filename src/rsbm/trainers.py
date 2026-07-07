import logging
import os

import accelerate
import torch as th
from accelerate import Accelerator
from torch.nn.functional import mse_loss
from torch.optim import Adam
from tqdm import trange

from .. import data_utils, models, utils
from ..evaluation import create_evaluator, plot_loss
from . import bridges, sdes

logger = logging.getLogger(__name__)


def get_pretrain_lr(
    base_lr: float, step: int, n_pretrain_steps: int, lr_warmup_steps: int
) -> float:
    effective_warmup_steps = min(max(lr_warmup_steps, 0), n_pretrain_steps)
    if effective_warmup_steps == 0 or step >= effective_warmup_steps:
        return base_lr
    return base_lr * (step + 1) / effective_warmup_steps


def set_optimizer_lr(optimizer: Adam, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def create_optimizer(model, training_conf):
    lr = training_conf.lr
    betas = tuple(training_conf.adam_betas)
    if isinstance(model, models.MultiNet):
        return {
            key: Adam(net.parameters(), lr=lr, betas=betas)
            for key, net in model.nets.items()
        }
    else:
        return Adam(model.parameters(), lr=lr, betas=betas)


class AlphaIMFTrainer:
    def __init__(
        self,
        model,
        sde: sdes.SDE,
        bridge: bridges.BrownianBridge,
        training_conf,
        eval_conf,
        data_conf,
        seed: int,
        out_dir: str,
    ):
        dl_config = accelerate.DataLoaderConfiguration(split_batches=True)
        self.mixed_precision = training_conf.get("mixed_precision", "no")
        self.accelerator = Accelerator(
            dataloader_config=dl_config, mixed_precision=self.mixed_precision
        )

        # set attributes
        self.step = 0
        self.start_step = 0
        self.losses = []

        self.training_conf = training_conf
        self.n_pretrain_steps = training_conf.n_pretrain_steps
        self.n_finetune_steps = training_conf.n_finetune_steps
        self.base_lr = training_conf.lr
        self.lr_warmup_steps = training_conf.lr_warmup_steps
        self.use_ema = training_conf.use_ema
        self.ema_rate = training_conf.ema_rate
        self.compile_ema = training_conf.get("compile_ema", False)
        self.log_freq = training_conf.log_freq
        self.checkpoint_freq = training_conf.checkpoint.frequency
        self.max_grad_norm = training_conf.max_grad_norm

        self.eval_freq = eval_conf.freq
        self.eval_dir = os.path.join(out_dir, "eval")
        self.checkpoint_dir = os.path.join(out_dir, "checkpoints")
        if self.accelerator.is_main_process:
            os.makedirs(self.eval_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model = model
        self.ema_model = models.create_eval_copy(self.model)
        self._ema_inference_model = None
        self.sde = sde
        self.bridge = bridge

        # build stuff
        self.optimizer_pretrain = create_optimizer(self.model, training_conf)
        self.optimizer_finetune = create_optimizer(self.model, training_conf)
        self.data_gen = utils.create_generators(seed + 1, "cpu")
        self.train_gen = utils.create_generators(
            seed + 2 + self.accelerator.process_index * 10, self.accelerator.device
        )
        self.eval_gen = utils.create_generators(seed + 3, self.accelerator.device)

        # build more stuff
        self.train_dataloaders = data_utils.create_dataloaders(
            data_conf,
            training_conf.batch_size,
            True,
            True,
            seed,
            dl_generator=self.data_gen,
        )

        (
            self.model,
            self.ema_model,
            self.optimizer_pretrain,
            self.optimizer_finetune,
            self.train_dataloaders[0],
            self.train_dataloaders[1],
        ) = self.accelerator.prepare(
            self.model,
            self.ema_model,
            self.optimizer_pretrain,
            self.optimizer_finetune,
            self.train_dataloaders[0],
            self.train_dataloaders[1],
        )
        self.optimizer = self.optimizer_pretrain
        self.device = self.accelerator.device

        # load checkpoint if specified
        self.load_checkpoint(training_conf.resume.dir)

        self.evaluator = None
        if self.accelerator.is_main_process:
            self.evaluator = create_evaluator(
                data_conf=data_conf,
                eval_conf=eval_conf,
                synth_data_base_seed=seed,
                sde=self.sde,
                generator_state=self.eval_gen.get_state(),  # detaches state to create new generator
                out_dir=self.eval_dir,
                device=self.accelerator.device,
                postprocessing0=data_utils.postprocessing(data_conf.data0.dataset),
                postprocessing1=data_utils.postprocessing(data_conf.data1.dataset),
                clip_fid=not isinstance(self.sde, sdes.ReflectedSDE),
                autocast=self.autocast,
            )
            self.evaluator.prepare()

    def autocast(self):
        return self.accelerator.autocast()

    def get_ema_inference_model(self):
        ema_model = self.accelerator.unwrap_model(self.ema_model)
        ema_model.eval()
        if not self.compile_ema:
            return ema_model
        if self._ema_inference_model is None:
            if not hasattr(th, "compile"):
                raise RuntimeError(
                    "training.compile_ema=true requires torch.compile to be available."
                )
            try:
                self._ema_inference_model = th.compile(ema_model)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to compile the EMA model. Disable training.compile_ema "
                    "to fall back to the uncompiled EMA model."
                ) from exc
        self._ema_inference_model.eval()
        return self._ema_inference_model

    def evaluate(self, final_artifacts: bool = False):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.use_ema:
                eval_model = self.get_ema_inference_model()
            else:
                eval_model = self.accelerator.unwrap_model(self.model)
            eval_model.eval()
            with th.inference_mode():
                self.evaluator.evaluate(
                    eval_model, step=self.step, final_artifacts=final_artifacts
                )
        self.accelerator.wait_for_everyone()
        if not self.use_ema:
            self.model.train()

    def train(self):
        try:
            self._train()
        except KeyboardInterrupt:
            if self.accelerator.is_main_process:
                logger.warning("Training interrupted, plotting losses")
        if self.accelerator.is_main_process:
            plot_loss(self.losses, self.eval_dir)

    def _train(self):
        train_dataloaders = {
            key: data_utils.make_infinite(loader)
            for key, loader in self.train_dataloaders.items()
        }

        total_steps = self.n_pretrain_steps + self.n_finetune_steps
        for i in trange(
            self.start_step,
            total_steps,
            disable=not self.accelerator.is_main_process,
        ):
            if i < self.n_pretrain_steps:
                self.optimizer = self.optimizer_pretrain
                lr = get_pretrain_lr(
                    self.base_lr, i, self.n_pretrain_steps, self.lr_warmup_steps
                )
                set_optimizer_lr(self.optimizer_pretrain, lr)
            else:
                self.optimizer = self.optimizer_finetune
                lr = self.base_lr

            x0 = next(train_dataloaders[0])[0]
            x1 = next(train_dataloaders[1])[0]

            batch_size = x0.shape[0]

            times_fwd = data_utils.sample_timesteps(
                batch_size, self.device, self.train_gen
            )
            times_bwd = data_utils.sample_timesteps(
                batch_size, self.device, self.train_gen
            )

            if i < self.n_pretrain_steps:
                loss = self.pretrain_step(x0, x1, times_fwd)
            else:
                loss = self.finetune_step(x0, x1, times_fwd, times_bwd)

            accelerated_loss = self.accelerator.gather(loss)
            if self.accelerator.is_main_process:
                self.losses.append(accelerated_loss.mean().item())

            self.step += 1
            assert self.step == i + 1

            self.accelerator.wait_for_everyone()
            models.update_ema(self.model, self.ema_model, self.ema_rate)

            if self.accelerator.is_main_process:
                if self.step % self.log_freq == 0:
                    logger.info(f"Step: {i + 1} \t\tLoss: {loss:.6f} \t\tLR: {lr:.8f}")
            if self.step % self.checkpoint_freq == 0:
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, f"c_{self.step}")
                )
            is_final_step = self.step == total_steps
            if (
                self.eval_freq is not None
                and self.step % self.eval_freq == 0
                and not is_final_step
            ):
                self.evaluate()

        self.save_checkpoint(os.path.join(self.checkpoint_dir, "c_final"))
        is_final_step = self.step == total_steps
        if self.eval_freq is None or self.step % self.eval_freq != 0 or is_final_step:
            self.evaluate(final_artifacts=True)

    def fwd_loss_fn(self, x1, xt, t) -> th.Tensor:
        target = self.bridge.sigma**2 * self.bridge.score_1_given_t(x1, xt, t)
        with self.autocast():
            pred = self.model(xt, t, direction=1)
        return mse_loss(pred.float(), target.float())

    def bwd_loss_fn(self, x0, xt, t) -> th.Tensor:
        target = self.bridge.sigma**2 * self.bridge.score_t_given_0(xt, x0, t)
        with self.autocast():
            pred = self.model(xt, 1 - t, direction=-1)
        return mse_loss(pred.float(), target.float())

    def grad_step(self, fwd_kwargs, bwd_kwargs) -> th.Tensor:
        self.optimizer.zero_grad()
        loss = 0.5 * (self.fwd_loss_fn(**fwd_kwargs) + self.bwd_loss_fn(**bwd_kwargs))
        self.accelerator.backward(loss)
        if self.max_grad_norm is not None:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
        self.optimizer.step()
        return loss

    def pretrain_step(self, x0, x1, times):
        B = x0.shape[0]
        b = B // 2

        xt = self.bridge.sample_given_01(x0, x1, times, generator=self.train_gen)

        loss = self.grad_step(
            fwd_kwargs={"x1": x1[:b], "xt": xt[:b], "t": times[:b]},
            bwd_kwargs={"x0": x0[b:], "xt": xt[b:], "t": times[b:]},
        )
        return loss

    def finetune_step(self, x0, x1, times_fwd, times_bwd):
        if self.use_ema:
            sampling_model = self.get_ema_inference_model()
        else:
            sampling_model = self.model
        sampling_model.eval()
        x1_hat = self.sde.forward_euler(
            sampling_model,
            x0,
            direction=1,
            generator=self.train_gen,
            autocast=self.autocast,
        )
        x0_hat = self.sde.forward_euler(
            sampling_model,
            x1,
            direction=-1,
            generator=self.train_gen,
            autocast=self.autocast,
        )
        self.model.train()

        xt_fwd = self.bridge.sample_given_01(
            x0_hat, x1, times_fwd, generator=self.train_gen
        )
        xt_bwd = self.bridge.sample_given_01(
            x0, x1_hat, times_bwd, generator=self.train_gen
        )

        loss = self.grad_step(
            fwd_kwargs={"x1": x1, "xt": xt_fwd, "t": times_fwd},
            bwd_kwargs={"x0": x0, "xt": xt_bwd, "t": times_bwd},
        )
        return loss

    def _state_dict(self):
        return {
            "step": self.step,
            "losses": self.losses,
            "data_gen": self.data_gen.get_state(),
            "train_gen": self.train_gen.get_state(),
            "eval_gen": self.eval_gen.get_state(),
        }

    def _load_state_dict(self, state) -> None:
        self.step = state["step"]
        self.start_step = state["step"]
        self.losses = state["losses"]
        self.data_gen.set_state(
            state["data_gen"]
        )  # note: training data sequence is not perfectly restored after loading a checkpoint
        self.train_gen.set_state(state["train_gen"])
        self.eval_gen.set_state(state["eval_gen"])

    @property
    def _state_file_name(self):
        return f"misc-{self.accelerator.process_index}.pt"

    def save_checkpoint(self, out_dir: str) -> None:
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(out_dir, safe_serialization=True)

        th.save(self._state_dict(), os.path.join(out_dir, self._state_file_name))

    def load_checkpoint(self, dir: str) -> None:
        if dir is None:
            return
        self.accelerator.load_state(dir, {"weights_only": True})

        self._load_state_dict(
            th.load(os.path.join(dir, self._state_file_name), weights_only=True)
        )


class AlphaIMFTrainerDoubleNet(AlphaIMFTrainer):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Doublenet trainer is not properly implemented with accelerator,\n"
            "need to deal with preparation of the two optimizers."
        )
        super().__init__(*args, **kwargs)

    def grad_step(self, fwd_kwargs, bwd_kwargs) -> float:
        losses = []

        for direction, loss_fn, kwargs in [
            (1, self.fwd_loss_fn, fwd_kwargs),
            (-1, self.bwd_loss_fn, bwd_kwargs),
        ]:
            self.optimizer[direction].zero_grad()
            loss = loss_fn(**kwargs)
            self.accelerator.backward(loss)
            if self.max_grad_norm is not None:
                # nn.utils.clip_grad_norm_(self.model.nets[direction].parameters(), self.max_grad_norm)
                self.accelerator.clip_grad_norm_(
                    self.model.nets[direction].parameters(), self.max_grad_norm
                )
            self.optimizer[direction].step()
            losses.append(loss.item())

        return sum(losses) / len(losses)


def get_trainer(model_class):
    if model_class is models.MultiNet:
        return AlphaIMFTrainerDoubleNet
    else:
        return AlphaIMFTrainer
