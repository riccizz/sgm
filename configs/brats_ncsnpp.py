# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""NCSNv3 on bedroom, with continuous sigmas."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.batch_size = 64
  training.n_iters = 2400001
  training.snapshot_sampling = True
  training.snapshot_freq = 100000
  training.sde = 'vesde'
  training.continuous = True
  training.snapshot_freq_for_preemption = 100000
  # eval
  evaluate = config.eval
  evaluate.begin_ckpt = 1
  evaluate.end_ckpt = 9
  evaluate.batch_size = 128
  evaluate.num_samples = 500
  # evaluate.ckpt_id = 20
  evaluate.enable_sampling = True
  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.cs_solver = 'projection'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.n_projections = 10
  sampling.projection_sigma_rate = 1.586
  sampling.task = 'mri'
  sampling.snr = 0.517
  sampling.coeff = 1.0
  # data
  data = config.data
  data.dataset = 'brats'
  data.image_size = 240
  data.num_channels = 1
  data.centered = False
  data.random_flip = False
  data.uniform_dequantization = False
  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.sigma_max = 128
  model.num_scales = 1000
  model.ema_rate = 0.999
  model.sigma_min = 0.01
  model.beta_min = 0.1
  model.beta_max = 20.
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 32
  model.ch_mult = (1, 1, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (30,)
  model.dropout = 0.
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  # optim
  optim = config.optim
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.amsgrad = False
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42

  return config
