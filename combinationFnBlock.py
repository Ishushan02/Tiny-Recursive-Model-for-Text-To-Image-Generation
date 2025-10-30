import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from diffusers import AutoencoderDC
import numpy as np
import math



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" The Device is : {device}")


class TimeEmbedding(nn.Module):
    def __init__(self, embedDimension):
        super().__init__()
        self.embedDimension = embedDimension
        self.linear1 = nn.Linear(embedDimension, 4 * embedDimension)
        self.silu = nn.SiLU()
        self.outlayer = nn.Linear(4 * embedDimension, embedDimension)

        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.outlayer.weight, std=0.02)

    def forward(self, t):

        half = self.embedDimension// 2
        exponent = -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
        freq = torch.exp(exponent.to(device))

        # timedimMap = t.float().unsqueeze(0) * freq[None, :]
        timedimMap = t[:, None].float() * freq[None, :]
        sinusoidal = torch.cat([torch.cos(timedimMap), torch.sin(timedimMap)], dim = -1)

        sinusoidal = self.linear1(sinusoidal)
        sinusoidal = self.silu(sinusoidal)
        out = self.outlayer(sinusoidal)

        return out

# tEmbed = TimeEmbedding(768)
# tEmbed.to(device)
# time = torch.tensor([5]).to(device)
# out = tEmbed(time)
# print(out.shape)