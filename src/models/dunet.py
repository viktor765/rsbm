# (D)irectional U-Net
import torch as th
import torch.nn as nn

from ..guided_diffusion.nn import linear, timestep_embedding
from ..guided_diffusion.unet import ResBlock
from . import utils
from .unet import UNet


class DirectionalUNet(UNet):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        emb_rescale=1,
    ):
        super().__init__(
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            num_classes,
            use_checkpoint,
            use_fp16,
            num_heads,
            num_head_channels,
            num_heads_upsample,
            use_scale_shift_norm,
            resblock_updown,
            use_new_attention_order,
            emb_rescale,
        )

        time_embed_dim = model_channels * 4  # like in UNet
        direction_embed_dim = time_embed_dim
        total_embed_dim = time_embed_dim + direction_embed_dim

        # Add module for direction embedding
        self.direction_embed = nn.Sequential(
            linear(model_channels, direction_embed_dim),
            nn.SiLU(),
            linear(direction_embed_dim, direction_embed_dim),
        )

        # Extend class embedding to total embedding dim
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, total_embed_dim)

        # Update embed_dim of ResBlocks
        for module in self.modules():
            if isinstance(module, ResBlock):
                module.emb_channels = total_embed_dim
                module.emb_layers = nn.Sequential(
                    nn.SiLU(),
                    linear(
                        total_embed_dim,
                        2 * module.out_channels
                        if module.use_scale_shift_norm
                        else module.out_channels,
                    ),
                )

    def forward(
        self,
        x: th.Tensor,
        timesteps: th.Tensor,
        y=None,
        *,
        direction: int | None = None,
    ):
        assert (y is not None) == (self.num_classes is not None), (
            "must specify y if and only if the model is class-conditional"
        )

        direction_01 = utils.direction_to_01(direction, x.shape[0], x.device)

        emb_t = self.time_embed(
            timestep_embedding(timesteps * self.emb_rescale, self.model_channels)
        )
        emb_d = self.direction_embed(
            timestep_embedding(direction_01 * self.emb_rescale, self.model_channels)
        )
        emb = th.cat([emb_t, emb_d], dim=1)

        if y is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
