import os
import matplotlib.pyplot as plt
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch
import numpy as np
from utils import restore_checkpoint
from PIL import Image
import torchvision.transforms as transforms

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
from configs.ve import celebahq_256_ncsnpp_continuous as configs

from super_resolution.measure import SuperResolutionOperator, GaussianNoise
from super_resolution.abo import get_recon_solver

def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))

def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img

root = "./super_resolution"
img_dir = os.path.join(root, "img")
ckpt_filename = os.path.join(root, "checkpoint_48.pth")
config = configs.get_config()
sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=1000)
# sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
sampling_eps = 1e-5

batch_size = 1 #@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())
  
# PC sampling
img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = config.sampling.snr #@param {"type": "number"}
n_steps =  1 #@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device)

# samples_uncond, n = sampling_fn(score_model)
# plt.imsave(os.path.join(img_dir, "samples_uncond.png"), clear_color(samples_uncond))

# super resolution recon
ref_img = Image.open(os.path.join(img_dir, "ref_img.png"))
transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
ref_img = transform(ref_img)
ref_img = ref_img[None, ...].to(config.device)
operator = SuperResolutionOperator(([1, 3, 256, 256]), 4, config.device)
noiser = GaussianNoise(0.05)

y_0 = noiser(operator.forward(ref_img)) # y_0 = Ax_0 + n measurements
print("measurement size", y_0.shape)
plt.imsave(os.path.join(img_dir, "input.png"), clear_color(y_0)) # y = Ax+n
plt.imsave(os.path.join(img_dir, "project.png"), clear_color(operator.project(ref_img, y_0)))

sampling_shape = (config.eval.batch_size,
                    config.data.num_channels, 
                    config.data.image_size,
                    config.data.image_size)
recon_solver = get_recon_solver(config, sde, operator, sampling_shape, inverse_scaler, eps=sampling_eps)
samples_cond = recon_solver(score_model, y_0, config.sampling.snr)
plt.imsave(os.path.join(img_dir, "recon.png"), clear_color(samples_cond))

