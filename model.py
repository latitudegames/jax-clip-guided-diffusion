import torch
import torch.utils.data
from torchvision.transforms import functional as TF
from torchvision import datasets, transforms, utils
from IPython import display
from lib.util import pil_from_tensor, pil_to_tensor
from lib import util
from lib.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import clip_jax
from tqdm.notebook import tqdm
from jaxtorch import PRNG, Context, Module, nn, init
import jaxtorch
import jax.numpy as jnp
import jax
import numpy as np
import requests
from PIL import Image
from dataclasses import dataclass
from functools import partial
import functools
import os
import time
import io
import math
import sys
sys.path.append('./CLIP_JAX')
sys.path.append('./jax-guided-diffusion')

devices = jax.devices()
n_devices = len(devices)
print('Using device:', devices)

# Define necessary functions


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def fetch_model(url_or_path):
    basename = os.path.basename(url_or_path)
    if os.path.exists(basename):
        return basename
    else:
        !curl - OL '{url_or_path}'
        return basename


def grey(image):
    [*_, c, h, w] = image.shape
    return jnp.broadcast_to(image.mean(axis=-3, keepdims=True), image.shape)


@jax.tree_util.register_pytree_node_class
class MakeCutouts(object):
    def __init__(self, cut_size, cutn, cut_pow=1., p_grey=0.2):
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.p_grey = p_grey

    def __call__(self, input, key):
        [b, c, h, w] = input.shape
        rng = PRNG(key)
        max_size = min(h, w)
        min_size = min(h, w, self.cut_size)
        cut_us = jax.random.uniform(
            rng.split(), shape=[self.cutn//2])**self.cut_pow
        sizes = (min_size + cut_us * (max_size - min_size + 1)
                 ).astype(jnp.int32).clamp(min_size, max_size)
        offsets_x = jax.random.uniform(
            rng.split(), [self.cutn//2], minval=0, maxval=w - sizes)
        offsets_y = jax.random.uniform(
            rng.split(), [self.cutn//2], minval=0, maxval=h - sizes)
        cutouts = util.cutouts_images(input, offsets_x, offsets_y, sizes)

        lcut_us = jax.random.uniform(rng.split(), shape=[self.cutn//2])
        lsizes = (max(h, w) + 10 + lcut_us * 10).astype(jnp.int32)
        loffsets_x = jax.random.uniform(
            rng.split(), [self.cutn//2], minval=-20, maxval=0)
        loffsets_y = jax.random.uniform(
            rng.split(), [self.cutn//2], minval=-20, maxval=0)
        lcutouts = util.cutouts_images(input, loffsets_x, loffsets_y, lsizes)

        cutouts = jnp.concatenate([cutouts, lcutouts], axis=1)

        grey_us = jax.random.uniform(
            rng.split(), shape=[b, self.cutn, 1, 1, 1])
        cutouts = jnp.where(grey_us < self.p_grey, grey(cutouts), cutouts)
        cutouts = cutouts.rearrange('b n c h w -> (n b) c h w')
        return cutouts

    def tree_flatten(self):
        return ([self.p_grey, self.cut_pow], (self.cut_size, self.cutn))

    @staticmethod
    def tree_unflatten(static, dynamic):
        (cut_size, cutn) = static
        (p_grey, cut_pow) = dynamic
        return MakeCutouts(cut_size, cutn, cut_pow, p_grey)


def Normalize(mean, std):
    mean = jnp.array(mean).reshape(3, 1, 1)
    std = jnp.array(std).reshape(3, 1, 1)

    def forward(image):
        return (image - mean) / std
    return forward


def norm1(x):
    """Normalize to the unit sphere."""
    return x / x.square().sum(axis=-1, keepdims=True).sqrt()


def spherical_dist_loss(x, y):
    x = norm1(x)
    y = norm1(y)
    return (x - y).square().sum(axis=-1).sqrt().div(2).arcsin().square().mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    # input = jnp.pad(input, ((0,0), (0,0), (0,1), (0,1)), mode='edge')
    # x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    # y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    # return (x_diff**2 + y_diff**2).mean([1, 2, 3])
    x_diff = input[..., :, 1:] - input[..., :, :-1]
    y_diff = input[..., 1:, :] - input[..., :-1, :]
    return x_diff.square().mean([1, 2, 3]) + y_diff.square().mean([1, 2, 3])


def downscale2d(image, f):
    [c, n, h, w] = image.shape
    return jax.image.resize(image, [c, n, h//f, w//f], method='cubic')


def upscale2d(image, f):
    [c, n, h, w] = image.shape
    return jax.image.resize(image, [c, n, h*f, w*f], method='cubic')


def rms(x):
    return x.square().mean().sqrt()


@dataclass
@jax.tree_util.register_pytree_node_class
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor

    def tree_flatten(self):
        return [self.v, self.pred, self.eps], []

    @classmethod
    def tree_unflatten(cls, static, dynamic):
        return cls(*dynamic)


def alpha_sigma_to_t(alpha, sigma):
    return jnp.arctan2(sigma, alpha) * 2 / math.pi


def get_ddpm_alphas_sigmas(t):
    log_snrs = -jnp.expm1(1e-4 + 10 * t**2).log()
    alphas_squared = jax.nn.sigmoid(log_snrs)
    sigmas_squared = jax.nn.sigmoid(-log_snrs)
    return alphas_squared.sqrt(), sigmas_squared.sqrt()


def get_cosine_alphas_sigmas(t):
    return jnp.cos(t * math.pi/2), jnp.sin(t * math.pi/2)

# Common nn modules.


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, cx, input):
        return jnp.concatenate([self.main(cx, input), self.skip(cx, input)], axis=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = init.normal(out_features // 2, in_features, stddev=std)

    def forward(self, cx, input):
        f = 2 * math.pi * input @ cx[self.weight].transpose()
        return jnp.concatenate([f.cos(), f.sin()], axis=-1)


class AvgPool2d(nn.Module):
    def forward(self, cx, x):
        [n, c, h, w] = x.shape
        x = x.reshape([n, c, h//2, 2, w//2, 2])
        x = x.mean((3, 5))
        return x


def expand_to_planes(input, shape):
    return input[..., None, None].broadcast_to(list(input.shape) + [shape[2], shape[3]])
