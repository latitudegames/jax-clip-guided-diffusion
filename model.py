import sys
sys.path.append('./CLIP_JAX')
sys.path.append('./jax-guided-diffusion')

import math
import io
import time
import os
import functools
from functools import partial
from dataclasses import dataclass

from PIL import Image
import requests

import numpy as np
import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch import PRNG, Context, Module, nn, init
from tqdm.notebook import tqdm

import clip_jax

from lib.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from lib import util
from lib.util import pil_from_tensor, pil_to_tensor

from IPython import display
from torchvision import datasets, transforms, utils
from torchvision.transforms import functional as TF
import torch.utils.data
import torch

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

# Secondary Model


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(),
        )


class SecondaryDiffusionImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        self.timestep_embed = FourierFeatures(1, 16)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, c),
            ConvBlock(c, c),
            SkipBlock([
                AvgPool2d(),
                # nn.image.Downsample2d('linear'),
                ConvBlock(c, c * 2),
                ConvBlock(c * 2, c * 2),
                SkipBlock([
                    AvgPool2d(),
                    # nn.image.Downsample2d('linear'),
                    ConvBlock(c * 2, c * 4),
                    ConvBlock(c * 4, c * 4),
                    SkipBlock([
                        AvgPool2d(),
                        # nn.image.Downsample2d('linear'),
                        ConvBlock(c * 4, c * 8),
                        ConvBlock(c * 8, c * 4),
                        nn.image.Upsample2d('linear'),
                    ]),
                    ConvBlock(c * 8, c * 4),
                    ConvBlock(c * 4, c * 2),
                    nn.image.Upsample2d('linear'),
                ]),
                ConvBlock(c * 4, c * 2),
                ConvBlock(c * 2, c),
                nn.image.Upsample2d('linear'),
            ]),
            ConvBlock(c * 2, c),
            nn.Conv2d(c, 3, 3, padding=1),
        )

    def forward(self, cx, input, t):
        timestep_embed = expand_to_planes(
            self.timestep_embed(cx, t[:, None]), input.shape)
        v = self.net(cx, jnp.concatenate([input, timestep_embed], axis=1))
        alphas, sigmas = get_cosine_alphas_sigmas(t)
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


class SecondaryDiffusionImageNet2(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = AvgPool2d()
        self.up = nn.image.Upsample2d('linear')

        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock([
                self.down,
                ConvBlock(cs[0], cs[1]),
                ConvBlock(cs[1], cs[1]),
                SkipBlock([
                    self.down,
                    ConvBlock(cs[1], cs[2]),
                    ConvBlock(cs[2], cs[2]),
                    SkipBlock([
                        self.down,
                        ConvBlock(cs[2], cs[3]),
                        ConvBlock(cs[3], cs[3]),
                        SkipBlock([
                            self.down,
                            ConvBlock(cs[3], cs[4]),
                            ConvBlock(cs[4], cs[4]),
                            SkipBlock([
                                self.down,
                                ConvBlock(cs[4], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[4]),
                                self.up,
                            ]),
                            ConvBlock(cs[4] * 2, cs[4]),
                            ConvBlock(cs[4], cs[3]),
                            self.up,
                        ]),
                        ConvBlock(cs[3] * 2, cs[3]),
                        ConvBlock(cs[3], cs[2]),
                        self.up,
                    ]),
                    ConvBlock(cs[2] * 2, cs[2]),
                    ConvBlock(cs[2], cs[1]),
                    self.up,
                ]),
                ConvBlock(cs[1] * 2, cs[1]),
                ConvBlock(cs[1], cs[0]),
                self.up,
            ]),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, cx, input, t):
        timestep_embed = expand_to_planes(
            self.timestep_embed(cx, t[:, None]), input.shape)
        v = self.net(cx, jnp.concatenate([input, timestep_embed], axis=1))
        alphas, sigmas = get_cosine_alphas_sigmas(t)
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


secondary1_model = SecondaryDiffusionImageNet()
secondary1_params = secondary1_model.init_weights(jax.random.PRNGKey(0))
secondary1_params = jaxtorch.pt.load(fetch_model(
    'https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet.pth'))

secondary2_model = SecondaryDiffusionImageNet2()
secondary2_params = secondary2_model.init_weights(jax.random.PRNGKey(0))
secondary2_params = jaxtorch.pt.load(fetch_model(
    'https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth'))

# Anti-JPEG model


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, cx, input):
        return self.main(cx, input) + self.skip(cx, input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.LeakyReLU(),
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
        ], skip)


CHANNELS = 64


class JPEGModel(nn.Module):
    def __init__(self, c=CHANNELS):
        super().__init__()

        self.timestep_embed = FourierFeatures(1, 16, std=1.0)
        self.class_embed = nn.Embedding(3, 16)

        self.arch = '11(22(22(2)22)22)11'

        self.net = nn.Sequential(
            nn.Conv2d(3 + 16 + 16, c, 1),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.image.Downsample2d(),
                ResConvBlock(c,     c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.image.Downsample2d(),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, 2 * 2, c * 2),
                    SkipBlock([
                        nn.image.Downsample2d(),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        nn.image.Upsample2d(),
                    ]),
                    ResConvBlock(c * 4, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    nn.image.Upsample2d(),
                ]),
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.image.Upsample2d(),
            ]),
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, dropout=False),
        )

    def forward(self, cx, input, ts, cond):
        [n, c, h, w] = input.shape
        timestep_embed = expand_to_planes(
            self.timestep_embed(cx, ts[:, None]), input.shape)
        class_embed = expand_to_planes(self.class_embed(cx, cond), input.shape)
        v = self.net(cx, jnp.concatenate(
            [input, timestep_embed, class_embed], axis=1))
        alphas, sigmas = get_cosine_alphas_sigmas(ts)
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


jpeg_model = JPEGModel()
jpeg_params = jpeg_model.init_weights(jax.random.PRNGKey(0))
jpeg_params = jaxtorch.pt.load(fetch_model(
    'https://set.zlkj.in/models/diffusion/jpeg-db-oi-614.pt'))['params_ema']

# Secondary Anti-JPEG Classifier

CHANNELS = 64


class Classifier(nn.Module):
    def __init__(self, c=CHANNELS):
        super().__init__()

        self.timestep_embed = FourierFeatures(1, 16, std=1.0)

        self.arch = '11-22-22-22'

        self.net = nn.Sequential(
            nn.Conv2d(3 + 16, c, 1),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
            nn.image.Downsample2d(),
            ResConvBlock(c,     c * 2, c * 2),
            ResConvBlock(c * 2, c * 2, c * 2),
            nn.image.Downsample2d(),
            ResConvBlock(c * 2, c * 2, c * 2),
            ResConvBlock(c * 2, 2 * 2, c * 2),
            nn.image.Downsample2d(),
            ResConvBlock(c * 2, c * 2, c * 2),
            ResConvBlock(c * 2, c * 2, c * 2),
            ResConvBlock(c * 2, c * 2, 1, dropout=False),
        )

    def forward(self, cx, input, ts):
        [n, c, h, w] = input.shape
        timestep_embed = expand_to_planes(
            self.timestep_embed(cx, ts[:, None]), input.shape)
        return self.net(cx, jnp.concatenate([input, timestep_embed], axis=1))

    def score(self, cx, reals, ts, cond, flood_level):
        cond = cond[:, None, None, None]
        logits = self.forward(cx, reals, ts)
        loss = -jax.nn.log_sigmoid(jnp.where(cond == 0, logits, -logits))
        loss = loss.clamp(minval=flood_level, maxval=None)
        return loss.mean()


@jax.jit
def classifier_probs(classifier_params, x, ts):
    n = x.shape[0]
    cx = Context(classifier_params, jax.random.PRNGKey(0)).eval_mode_()
    probs = jax.nn.sigmoid(classifier_model(cx, x, ts.broadcast_to([n])))
    return probs

# Model Settings


classifier_model = Classifier()
classifier_params = classifier_model.init_weights(jax.random.PRNGKey(0))
classifier_params = jaxtorch.pt.load(fetch_model(
    'https://set.zlkj.in/models/diffusion/jpeg-classifier-72.pt'))['params_ema']

model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '1000',
    'image_size': 512,     # Change to either 256 or 512 to select the openai model
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_scale_shift_norm': True,
    'use_checkpoint': False  # Set to True to save memory
})

# Load models

model, diffusion = create_model_and_diffusion(**model_config)
model_params = model.init_weights(jax.random.PRNGKey(0))

print('Loading state dict...')
model_urls = {
    512: 'https://set.zlkj.in/models/diffusion/512x512_diffusion_uncond_finetune_008100.pt',
    256: 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
}
with torch.no_grad():
    model_params = model.load_state_dict(model_params, jaxtorch.pt.load(
        fetch_model(model_urls[model_config['image_size']])))

# Define combinators.

# These (ab)use the jax pytree registration system to define parameterised
# objects for doing various things, which are compatible with jax.jit.

# For jit compatibility an object needs to act as a pytree, which means implementing two methods:
#  - tree_flatten(self): returns two lists of the object's fields:
#       1. 'dynamic' parameters: things which can be jax tensors, or other pytrees
#       2. 'static' parameters: arbitrary python objects, will trigger recompilation when changed
#  - tree_unflatten(static, dynamic): reconstitutes the object from its parts

# With these tricks, you can simply define your cond_fn as an object, as is done
# below, and pass it into the jitted sample step as a regular argument. JAX will
# handle recompiling the jitted code whenever a control-flow affecting parameter
# is changed (such as cut_batches).


@jax.tree_util.register_pytree_node_class
class CosineModel(object):
    def __init__(self, model, params, **kwargs):
        self.model = model
        self.params = params
        self.kwargs = kwargs

    @jax.jit
    def __call__(self, x, t, key):
        n = x.shape[0]
        alpha, sigma = get_ddpm_alphas_sigmas(t)
        cosine_t = alpha_sigma_to_t(alpha, sigma)
        cx = Context(self.params, key).eval_mode_()
        return self.model(cx, x, cosine_t.broadcast_to([n]), **self.kwargs)

    def tree_flatten(self):
        return [self.params, self.kwargs], [self.model]

    def tree_unflatten(static, dynamic):
        [params, kwargs] = dynamic
        [model] = static
        return CosineModel(model, params, **kwargs)


@jax.tree_util.register_pytree_node_class
class OpenaiModel(object):
    def __init__(self, model, params):
        self.model = model
        self.params = params

    @jax.jit
    def __call__(self, x, t, key):
        n = x.shape[0]
        alpha, sigma = get_ddpm_alphas_sigmas(t)
        cx = Context(self.params, key).eval_mode_()
        openai_t = (t * 1000).broadcast_to([n])
        eps = self.model(cx, x, openai_t)[:, :3, :, :]
        pred = (x - eps * sigma) / alpha
        v = (eps - x * sigma) / alpha
        return DiffusionOutput(v, pred, eps)

    def tree_flatten(self):
        return [self.params], [self.model]

    def tree_unflatten(static, dynamic):
        [params] = dynamic
        [model] = static
        return OpenaiModel(model, params)


@jax.tree_util.register_pytree_node_class
class Perceptor(object):
    # Wraps a CLIP instance and its parameters.
    def __init__(self, image_fn, text_fn, clip_params):
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.clip_params = clip_params

    @jax.jit
    def embed_cutouts(self, cutouts):
        return norm1(self.image_fn(self.clip_params, cutouts))

    def embed_text(self, text):
        tokens = clip_jax.tokenize([text])
        text_embed = self.text_fn(self.clip_params, tokens)
        return norm1(text_embed.reshape(512))

    def tree_flatten(self):
        return [self.clip_params], [self.image_fn, self.text_fn]

    def tree_unflatten(static, dynamic):
        [clip_params] = dynamic
        [image_fn, text_fn] = static
        return Perceptor(image_fn, text_fn, clip_params)


@jax.tree_util.register_pytree_node_class
class LerpModels(object):
    """Linear combination of diffusion models."""

    def __init__(self, models):
        self.models = models

    def __call__(self, x, t, key):
        outputs = [m(x, t, key) for (m, w) in self.models]
        v = sum(out.v * w for (out, (m, w)) in zip(outputs, self.models))
        pred = sum(out.pred * w for (out, (m, w)) in zip(outputs, self.models))
        eps = sum(out.eps * w for (out, (m, w)) in zip(outputs, self.models))
        return DiffusionOutput(v, pred, eps)

    def tree_flatten(self):
        return [self.models], []

    def tree_unflatten(static, dynamic):
        return LerpModels(*dynamic)
# Cond Fns


@jax.tree_util.register_pytree_node_class
class CondCLIP(object):
    # CLIP guidance loss. Pushes the image toward a text prompt.
    def __init__(self, text_embed, clip_guidance_scale, perceptor, make_cutouts, cut_batches):
        self.text_embed = text_embed
        self.clip_guidance_scale = clip_guidance_scale
        self.perceptor = perceptor
        self.make_cutouts = make_cutouts
        self.cut_batches = cut_batches

    def __call__(self, x_in, key):
        n = x_in.shape[0]

        def main_clip_loss(x_in, key):
            cutouts = normalize(self.make_cutouts(x_in.add(1).div(2), key))
            image_embeds = self.perceptor.embed_cutouts(
                cutouts).reshape([self.make_cutouts.cutn, n, 512])
            losses = spherical_dist_loss(image_embeds, self.text_embed).mean(0)
            return losses.sum() * self.clip_guidance_scale
        num_cuts = self.cut_batches
        keys = jnp.stack(jax.random.split(key, num_cuts))
        main_clip_grad = jax.lax.scan(lambda total, key: (total + jax.grad(main_clip_loss)(x_in, key), key),
                                      jnp.zeros_like(x_in),
                                      keys)[0] / num_cuts
        return main_clip_grad

    def tree_flatten(self):
        return [self.text_embed, self.clip_guidance_scale, self.perceptor, self.make_cutouts], [self.cut_batches]

    def tree_unflatten(static, dynamic):
        [text_embed, clip_guidance_scale, perceptor, make_cutouts] = dynamic
        [cut_batches] = static
        return CondCLIP(text_embed, clip_guidance_scale, perceptor, make_cutouts, cut_batches)


@jax.tree_util.register_pytree_node_class
class CondTV(object):
    # Multiscale Total Variation loss. Tries to smooth out the image.
    def __init__(self, tv_scale):
        self.tv_scale = tv_scale

    def __call__(self, x_in, key):
        def sum_tv_loss(x_in, f=None):
            if f is not None:
                x_in = downscale2d(x_in, f)
            return tv_loss(x_in).sum() * self.tv_scale
        tv_grad_512 = jax.grad(sum_tv_loss)(x_in)
        tv_grad_256 = jax.grad(partial(sum_tv_loss, f=2))(x_in)
        tv_grad_128 = jax.grad(partial(sum_tv_loss, f=4))(x_in)
        return tv_grad_512 + tv_grad_256 + tv_grad_128

    def tree_flatten(self):
        return [self.tv_scale], []

    def tree_unflatten(static, dynamic):
        return CondTV(*dynamic)


@jax.tree_util.register_pytree_node_class
class CondSat(object):
    # Saturation loss. Tries to prevent the image from going out of range.
    def __init__(self, sat_scale):
        self.sat_scale = sat_scale

    def __call__(self, x_in, key):
        def saturation_loss(x_in):
            return jnp.abs(x_in - x_in.clamp(minval=-1, maxval=1)).mean()
        return self.sat_scale * jax.grad(saturation_loss)(x_in)

    def tree_flatten(self):
        return [self.sat_scale], []

    def tree_unflatten(static, dynamic):
        return CondSat(*dynamic)


@jax.tree_util.register_pytree_node_class
class CondMSE(object):
    # MSE loss. Targets the output towards an image.
    def __init__(self, target, mse_scale):
        self.target = target
        self.mse_scale = mse_scale

    def __call__(self, x_in, key):
        def mse_loss(x_in):
            return (x_in - self.target).square().mean()
        return self.mse_scale * jax.grad(mse_loss)(x_in)

    def tree_flatten(self):
        return [self.target, self.mse_scale], []

    def tree_unflatten(static, dynamic):
        return CondMSE(*dynamic)


@jax.tree_util.register_pytree_node_class
class MainCondFn(object):
    # Used to construct the main cond_fn. Accepts a diffusion model which will
    # be used for denoising, plus a list of 'conditions' which will
    # generate gradient of a loss wrt the denoised, to be summed together.
    def __init__(self, diffusion, conditions, use='pred'):
        self.diffusion = diffusion
        self.conditions = conditions
        self.use = use

    @jax.jit
    def __call__(self, key, x, t):
        rng = PRNG(key)
        n = x.shape[0]

        alphas, sigmas = get_ddpm_alphas_sigmas(t)

        def denoise(key, x):
            pred = self.diffusion(x, t, key).pred
            if self.use == 'pred':
                return pred
            elif self.use == 'x_in':
                return pred * sigmas + x * alphas
        (x_in, backward) = jax.vjp(partial(denoise, rng.split()), x)

        total = jnp.zeros_like(x_in)
        for cond in self.conditions:
            total += cond(x_in, rng.split())
        final_grad = -backward(total)[0]

        # clamp gradients to a max of 0.2
        magnitude = final_grad.square().mean(axis=(1, 2, 3), keepdims=True).sqrt()
        final_grad = final_grad * \
            jnp.where(magnitude > 0.2, 0.2 / magnitude, 1.0)
        return final_grad

    def tree_flatten(self):
        return [self.diffusion, self.conditions], [self.use]

    def tree_unflatten(static, dynamic):
        return MainCondFn(*dynamic, *static)


@jax.tree_util.register_pytree_node_class
class ClassifierFn(object):
    def __init__(self, model, params, guidance_scale, **kwargs):
        self.model = model
        self.params = params
        self.guidance_scale = guidance_scale
        self.kwargs = kwargs

    @jax.jit
    def __call__(self, key, x, t):
        n = x.shape[0]
        alpha, sigma = get_ddpm_alphas_sigmas(t)
        cosine_t = alpha_sigma_to_t(alpha, sigma).broadcast_to([n])

        def fwd(x):
            cx = Context(self.params, key).eval_mode_()
            return self.guidance_scale * self.model.score(cx, x, cosine_t, **self.kwargs)
        return -jax.grad(fwd)(x)

    def tree_flatten(self):
        return [self.params, self.guidance_scale, self.kwargs], [self.model]

    def tree_unflatten(static, dynamic):
        [params, guidance_scale, kwargs] = dynamic
        [model] = static
        return ClassifierFn(model, params, guidance_scale, **kwargs)


@jax.tree_util.register_pytree_node_class
class CondFns(object):
    def __init__(self, *conditions):
        self.conditions = conditions

    def __call__(self, key, x, t):
        rng = PRNG(key)
        total = jnp.zeros_like(x)
        for cond in self.conditions:
            total += cond(rng.split(), x, t)
        return total

    def tree_flatten(self):
        return [self.conditions], []

    def tree_unflatten(static, dynamic):
        [conditions] = dynamic
        return MixCondFn(*conditions)


def sample_step(key, x, t1, t2, diffusion, cond_fn, eta):
    rng = PRNG(key)

    n = x.shape[0]
    alpha1, sigma1 = get_ddpm_alphas_sigmas(t1)
    alpha2, sigma2 = get_ddpm_alphas_sigmas(t2)

    # Run the model
    out = diffusion(x, t1, rng.split())
    eps = out.eps
    pred0 = out.pred

    # # Predict the denoised image
    # pred0 = (x - eps * sigma1) / alpha1

    # Adjust eps with conditioning gradient
    cond_score = cond_fn(rng.split(), x, t1)
    eps = eps - sigma1 * cond_score

    # Predict the denoised image with conditioning
    pred = (x - eps * sigma1) / alpha1

    # Negative eta allows more extreme levels of noise.
    ddpm_sigma = (sigma2**2 / sigma1**2).sqrt() * \
        (1 - alpha1**2 / alpha2**2).sqrt()
    ddim_sigma = jnp.where(eta >= 0.0,
                           eta * ddpm_sigma,  # Normal: eta interpolates between ddim and ddpm
                           -eta * sigma2)    # Extreme: eta interpolates between ddim and q_sample(pred)
    adjusted_sigma = (sigma2**2 - ddim_sigma**2).sqrt()

    # Recombine the predicted noise and predicted denoised image in the
    # correct proportions for the next step
    x = pred * alpha2 + eps * adjusted_sigma

    # Add the correct amount of fresh noise
    x += jax.random.normal(rng.split(), x.shape) * ddim_sigma
    return x, pred0


# Load CLIP Models

clip_size = 224
normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                      std=[0.26862954, 0.26130258, 0.27577711])

image_fn, text_fn, clip_params, _ = clip_jax.load('ViT-B/32')
vit32 = Perceptor(image_fn, text_fn, clip_params)
image_fn, text_fn, clip_params, _ = clip_jax.load('ViT-B/16')
vit16 = Perceptor(image_fn, text_fn, clip_params)

# Run Configuration

image_size = (640, 512)

batch_size = 2
title = "waters of the death tarot card by greg rutkowski"

# Note: with two perceptors, combined guidance scale is 2x because they are added together.
clip_guidance_scale = 2000
tv_scale = 150  # Smooths out the image.
sat_scale = 600  # Prevents image from over/under-saturating.
steps = 250    # Number of steps for sampling. Generally, more = better.
eta = 1.0  # 0.0: DDIM | 1.0: DDPM | -1.0: Extreme noise (q_sample)

cutn = 16  # effective cutn is cut_batches * this
cut_pow = 1.0
cut_batches = 4
make_cutouts = MakeCutouts(clip_size, cutn, cut_pow=cut_pow)

n_batches = 4
init_image = None
skip_timesteps = 0
seed = None  # if None, uses the current time in seconds.

# OpenAI used T=1000 to 0. We've just rescaled to between 1 and 0.
schedule = jnp.linspace(1, 0, steps)[skip_timesteps:]


openai = OpenaiModel(model, model_params)
secondary1 = CosineModel(secondary1_model, secondary1_params)
secondary2 = CosineModel(secondary2_model, secondary2_params)
jpeg_0 = CosineModel(jpeg_model, jpeg_params,
                     cond=jnp.array([0]*batch_size))  # Clean class
jpeg_1 = CosineModel(jpeg_model, jpeg_params,
                     cond=jnp.array([2]*batch_size))  # Noisy class

jpeg_classifier_fn = ClassifierFn(classifier_model, classifier_params,
                                  guidance_scale=10000.0,  # will generally depend on image size
                                  # Clean class
                                  cond=jnp.array([0]*batch_size),
                                  flood_level=0.7)  # Prevent over-optimization

diffusion = LerpModels([(openai, 1.0),
                        (jpeg_0, 1.0),
                        (jpeg_1, -1.0)])
cond_model = secondary2

# target = Image.open(fetch('https://pbs.twimg.com/media/FBbTU8hVEAM10dL?format=png&name=small')).convert('RGB')
# target = target.resize(image_size, Image.LANCZOS)
# target = jnp.array(TF.to_tensor(target)) * 2 - 1

cond_fn = CondFns(MainCondFn(cond_model, [
    CondCLIP(vit32.embed_text(title), clip_guidance_scale,
             vit32, make_cutouts, cut_batches),
    CondCLIP(vit16.embed_text(title), clip_guidance_scale,
             vit16, make_cutouts, cut_batches),
    # CondTV(tv_scale),
    # CondMSE(target, 256*256*3),
    CondSat(sat_scale),
], use='pred'),
    jpeg_classifier_fn)

# Run Execution


def proc_init_image(init_image):
    if init_image.endswith(':parts512'):
        url = init_image.rsplit(':', 1)[0]
        init = Image.open(fetch(url)).convert('RGB')
        init = pil_to_tensor(init).mul(2).sub(1)
        [c, h, w] = init.shape
        indices = [(x, y)
                   for y in range(0, h, 512)
                   for x in range(0, w, 512)]
        indices = (indices * batch_size)[:batch_size]
        parts = [init[:, y:y+512, x:x+512] for (x, y) in indices]
        init = jnp.stack(parts)
        init = jax.image.resize(
            init, [batch_size, c, image_size[1], image_size[0]], method='lanczos3')
        return init

    init = Image.open(fetch(init_image)).convert('RGB')
    init = init.resize(image_size, Image.LANCZOS)
    init = pil_to_tensor(init).unsqueeze(0).mul(2).sub(1)
    return init


@torch.no_grad()
def run():
    if seed is None:
        local_seed = int(time.time())
    else:
        local_seed = seed
    print(f'Starting run with seed {local_seed}...')
    rng = PRNG(jax.random.PRNGKey(local_seed))

    init = None
    if init_image is not None:
        if type(init_image) is list:
            init = jnp.concatenate([proc_init_image(url)
                                   for url in init_image], axis=0)
        else:
            init = proc_init_image(init_image)

    for i in range(n_batches):
        timestring = time.strftime('%Y%m%d%H%M%S')

        ts = schedule
        alphas, sigmas = get_ddpm_alphas_sigmas(ts)
        cosine_ts = alpha_sigma_to_t(alphas, sigmas)

        x = sigmas[0] * jax.random.normal(rng.split(),
                                          [batch_size, 3, image_size[1], image_size[0]])

        if init is not None:
            x = x + alphas[0] * init

        # Main loop
        local_steps = schedule.shape[0] - 1
        for j in tqdm(range(local_steps)):
            if ts[j] != ts[j+1]:
                # Skip steps where the ts are the same, to make it easier to
                # make complicated schedules out of cat'ing linspaces.
                x, pred = sample_step(
                    rng.split(), x, ts[j], ts[j+1], diffusion, cond_fn, eta)
            if j % 50 == 0 or j == local_steps - 1:
                images = pred.add(1).div(2).clamp(0, 1)
                # probs = classifier_probs(classifier_params, x, cosine_ts[j+1])
                # probs = jax.image.resize(probs * jnp.ones([batch_size, 3, 1, 1]), pred.shape, 'cubic')
                # images = jnp.concatenate([images, probs],axis=0)
                images = torch.tensor(np.array(images))
                display.display(TF.to_pil_image(
                    utils.make_grid(images, 4).cpu()))

        # Save samples
        os.makedirs('samples', exist_ok=True)
        os.makedirs(save_location, exist_ok=True)
        for k in range(batch_size):
            this_title = title[:100]
            dname = f'samples/{timestring}_{k}_{this_title}.png'
            pil_image = TF.to_pil_image(images[k])
            pil_image.save(dname)
            pil_image.save(
                f'{save_location}/{timestring}_{k}_{this_title}.png')


try:
    run()
    success = True
except:
    import traceback
    traceback.print_exc()
    success = False
assert success
