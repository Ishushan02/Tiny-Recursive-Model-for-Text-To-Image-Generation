import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from diffusers import AutoencoderDC
import numpy as np
import math
import torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from torchvision import transforms
from diffusers import AutoencoderDC
from diffusers import DDPMScheduler, DDIMScheduler
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os
from torch.optim.lr_scheduler import StepLR
import random
# from CombinationFunctions import ImageInputToDiT, NDiTModule, Decoder, TimeEmbedding, TextEmbedding
from tqdm import tqdm



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

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, embedDimension):
        super().__init__()
        self.embedDimension = embedDimension
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedDimension, 6 * embedDimension)
        )
        self.scaleShiftParameters = nn.Parameter(torch.zeros(6, embedDimension))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)      
    
    def forward(self, t):
        batchSize, _ = t.shape
        t = self.adaLN(t)
        t = t.reshape(batchSize, 6, -1)

        # gamma_msa, beta_msa, alpha_msa, gamma_mlp, beta_mlp, alpha_mlp = (
        #     (self.scaleShiftParameters[None] + t).chunk(6, dim = 1)
        # )

        scale_shift = self.scaleShiftParameters.unsqueeze(0)  
        t = t + scale_shift                               

        gamma_msa, beta_msa, alpha_msa, gamma_mlp, beta_mlp, alpha_mlp = t.chunk(6, dim=1)

        gamma_msa = gamma_msa.squeeze(1)
        beta_msa = beta_msa.squeeze(1)
        alpha_msa = alpha_msa.squeeze(1)
        gamma_mlp = gamma_mlp.squeeze(1)
        beta_mlp = beta_mlp.squeeze(1)
        alpha_mlp = alpha_mlp.squeeze(1)
        return gamma_msa, beta_msa, alpha_msa, gamma_mlp, beta_mlp, alpha_mlp
    
# embedDimension = 768
# tEmbed = TimeEmbedding(embedDimension=embedDimension)
# tEmbed.to(device)
# time = torch.tensor([1000]).to(device)
# tout = tEmbed(time)
# adaNorm = AdaptiveLayerNorm(768)
# g1, b1, a1, g2, b2, a2 = adaNorm(tout)
# print(g1.shape, b1.shape, a1.shape, g2.shape, b2.shape, a2.shape)


class Rotary2DPositionalEncoding(nn.Module):
    def __init__(self, height, width, embedDimension):
        super().__init__()
        self.height = height
        self.width = width
        self.embedDimension = embedDimension

        self.dimHalf = embedDimension // 2
        self.dimQuarter = embedDimension // 4
        inverseFrequency = 1.0 / (10000 ** (torch.arange(0, self.dimQuarter, dtype=torch.float32) / self.dimQuarter))

        heightPositions = torch.arange(height, dtype=torch.float32)
        widthPositions = torch.arange(width, dtype=torch.float32)

        sinusoidHeight = torch.einsum("i,j->ij", heightPositions, inverseFrequency)
        sinusoidWidth = torch.einsum("i,j->ij", widthPositions, inverseFrequency)

        self.register_buffer("sinHeight", sinusoidHeight.sin(), persistent=False)
        self.register_buffer("cosHeight", sinusoidHeight.cos(), persistent=False)
        self.register_buffer("sinWidth", sinusoidWidth.sin(), persistent=False)
        self.register_buffer("cosWidth", sinusoidWidth.cos(), persistent=False)

    def rotateEveryTwo(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    
    def applyRope(self, x, sinHeight, cosHeight, sinWidth, cosWidth):

        xHeight = x[..., :self.dimHalf]
        xWidth = x[..., self.dimHalf:]

        sinHeight = sinHeight[None, :, None, :].to(x.device)
        cosHeight = cosHeight[None, :, None, :].to(x.device)
        sinWidth = sinWidth[None, None, :, :].to(x.device)
        cosWidth = cosWidth[None, None, :, :].to(x.device)

        xHeightRotional = (xHeight[..., :self.dimQuarter] * cosHeight) + (self.rotateEveryTwo(xHeight[..., :self.dimQuarter]) * sinHeight)
        xWidthRotional = (xWidth[..., :self.dimQuarter] * cosWidth) + (self.rotateEveryTwo(xWidth[..., :self.dimQuarter]) * sinWidth)

        xHeightRotional = torch.cat([xHeightRotional, xHeight[..., self.dimQuarter:]], dim=-1)
        xWidthRotional = torch.cat([xWidthRotional, xWidth[..., self.dimQuarter:]], dim=-1)
        rotated = torch.cat([xHeightRotional, xWidthRotional], dim=-1)
        return rotated


    def forward(self, x):
        B, L, D = x.shape
        assert D == self.embedDimension
        assert L == self.height * self.width, f"Expected seq_len {self.height*self.width}, got {L}"

        x = x.view(B, self.height, self.width, D)
        x = self.applyRope(x, self.sinHeight, self.cosHeight, self.sinWidth, self.cosWidth)
        return x.view(B, L, D)

# rope2D = Rotary2DPositionalEncoding(4, 4, 768)
# imagePatches = torch.randn(2, 16, 768)

# out = rope2D(imagePatches)
# print(out.shape)



class PatchEmbedding(nn.Module):
    def __init__(self, imageSize, patchSize, inChannels, embedDimension):
        super().__init__()
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.embedDimension = embedDimension
        self.imageSize = imageSize

        self.patches = imageSize//patchSize * imageSize//patchSize

        self.encode = nn.Conv2d(in_channels = inChannels, out_channels = inChannels, kernel_size = patchSize, stride = patchSize, bias = True)
        self.decode = nn.ConvTranspose2d(in_channels=inChannels, out_channels=inChannels, kernel_size=patchSize, stride=patchSize, bias=True)
    

    def unPatchify(self, x):
        batchSize, NPatches, EmbedDim = x.shape
        patchPerDim = self.imageSize // self.patchSize
        x = x.transpose(1, 2).reshape(batchSize, EmbedDim, patchPerDim, patchPerDim)
        out = self.decode(x)
        return out


    def forward(self, latentImage):

        allPatch = self.encode(latentImage)
        # print(allPatch.shape)
        flattened = allPatch.flatten(2).transpose(1, 2)
        # print(flattened.shape, self.positionalEmbedding.shape)
        out = flattened
        return out

# latent = torch.randn(128, 8, 8).unsqueeze(0)
# pEmbed = PatchEmbedding(imageSize = 8, patchSize = 2, inChannels = 128, embedDimension = 768)
# out = pEmbed(latent)
# unpatched = pEmbed.unPatchify(out)
# print(out.shape, unpatched.shape)

def shiftModulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class ScaleShiftBlock(nn.Module):
    def __init__(self, embedDimension):
        super().__init__()
        self.embedDimension = embedDimension
        self.norm = nn.LayerNorm(embedDimension, elementwise_affine=False, eps=1e-6)
        # nn.init.ones_(self.norm.weight)
        # nn.init.zeros_(self.norm.bias)

    def forward(self, x, beta, gamma):
        B, N, W = x.shape
        x_norm = self.norm(x)
        out = shiftModulate(x_norm, gamma, beta)
        return out


def scaleModulate(x, scale):
    return x * (1 + scale.unsqueeze(1))

class ScaleBlock(nn.Module):
    def __init__(self, embedDimension):
        super().__init__()
        self.embedDimension = embedDimension
        self.norm = nn.LayerNorm(embedDimension, elementwise_affine=False, eps=1e-6)
        # nn.init.ones_(self.norm.weight)
        # nn.init.zeros_(self.norm.bias)

    def forward(self, x, alpha):
        B, N, W = x.shape
        x_norm = self.norm(x)
        out = scaleModulate(x_norm, alpha)
        return out
    
# patchify_latents = torch.randn(1, 16, 768)
# scShft = ScaleBlock(embedDimension)
# out = scShft(patchify_latents, a1)
# out.shape


class FeedForwardBlock(nn.Module):
    def __init__(self, embedDimension):
        super().__init__()

        self.linear1 = nn.Linear(embedDimension, embedDimension * 4)
        self.linear2 = nn.Linear(embedDimension * 4, embedDimension)
        self.gelu = nn.GELU()

        nn.init.zeros_(self.linear1.weight)
        nn.init.zeros_(self.linear2.weight)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x
    
# latents = torch.randn(1, 16, 768)
# ff = FeedForwardBlock(768)
# out = ff(latents)
# out.shape