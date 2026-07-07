import numpy as np
import torch
import torch.nn as nn

from . import utils


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Random weights that stay fixed (not trained)
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class GaussianFourierProjection2d(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, scale=30.0):
        super().__init__()
        # Random weights that stay fixed (not trained)
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2, input_dim) * scale, requires_grad=False
        )

    def forward(self, x):
        x_proj = (x[:, None, :] * self.W[None, :, :]).sum(dim=-1)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class LiteNet(nn.Module):
    def __init__(self, input_dim=2, time_emb_dim=32):
        super().__init__()
        # We embed the sigma level (noise level) so the net knows how coarse/fine to look
        self.embed = nn.Sequential(
            GaussianFourierProjection(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Embedding layer for the data
        self.data_embed = nn.Sequential(
            GaussianFourierProjection2d(input_dim=input_dim, embed_dim=64, scale=30.0),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

        # Input: Data (2) + Sigma Embedding (32)
        self.input_layer = nn.Linear(input_dim + time_emb_dim + 64, 512)
        self.mid_layer1 = nn.Linear(512, 512)
        self.mid_layer2 = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, input_dim)

        self.act = nn.SiLU()

    def forward(self, x, timesteps, y=None):
        if y is not None:
            raise NotImplementedError("Conditional LiteNet not implemented.")
        # Embed the noise level index
        # We normalize index 0..N to 0..1 for the embedding layer
        embed = self.embed(timesteps)
        data_embed = self.data_embed(x)

        h = torch.cat([x, embed, data_embed], dim=1)
        h = self.act(self.input_layer(h))
        h = self.act(self.mid_layer1(h))
        h = self.act(self.mid_layer2(h))

        # Output is the unnormalized score
        # Typically we divide result by sigma later for stability
        return self.output_layer(h)


class DirectionalLiteNet(LiteNet):
    def __init__(self, input_dim=2, time_emb_dim=32, dir_emb_dim=32):
        super().__init__(input_dim, time_emb_dim)
        self.direction_embed = nn.Embedding(2, dir_emb_dim)
        self.input_layer = nn.Linear(input_dim + time_emb_dim + dir_emb_dim + 64, 512)

    def forward(self, x, timesteps, y=None, *, direction: int | None = None):
        if y is not None:
            raise NotImplementedError("Conditional DirectionalLiteNet not implemented.")
        time_embed = self.embed(timesteps)
        dir_embed = self.direction_embed(
            utils.direction_to_01(direction, x.shape[0], x.device)
        )
        data_embed = self.data_embed(x)

        h = torch.cat([x, time_embed, dir_embed, data_embed], dim=1)
        h = self.act(self.input_layer(h))
        h = self.act(self.mid_layer1(h))
        h = self.act(self.mid_layer2(h))

        return self.output_layer(h)
