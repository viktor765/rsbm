import torch.nn as nn
from ._guided_diffusion_checkpoint import install as install_guided_diffusion_patch
from .unet import UNet
from .dunet import DirectionalUNet
from .litenet import LiteNet, DirectionalLiteNet
from .multinet import MultiNet
from .utils import create_eval_copy, update_ema


install_guided_diffusion_patch()


def create_model(conf) -> nn.Module:
    if conf.type.lower() == 'unet':
        if conf.directional:
            return DirectionalUNet(**conf.args)
        else:
            return MultiNet({
                +1: UNet(**conf.args),
                -1: UNet(**conf.args),
            })
    elif conf.type.lower() == 'litenet':
        if conf.directional:
            return DirectionalLiteNet(conf.args.input_dim,
                                              conf.args.time_embed_dim,
                                              conf.args.dir_embed_dim)
        else:
            return MultiNet({
                +1: LiteNet(conf.args.input_dim, conf.args.time_embed_dim),
                -1: LiteNet(conf.args.input_dim, conf.args.time_embed_dim),
            })

    else:
        raise ValueError(f'Unknown model type: {conf.type}')
