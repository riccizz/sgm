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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import torchvision.datasets as ds
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
# from GPUtil import showUtilization as gpu_usage
from torchvision import transforms
from torch.utils.data import DataLoader
from brats_dataset import Brats
import tqdm
# recon
import cs
from PIL import Image, ImageDraw
import piq

FLAGS = flags.FLAGS

def cycle(dl):
    while True:
        for data in dl:
            yield data

def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  # print(score_model)
  # print("GPU usage after loading model")
  # gpu_usage()
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  # train_ds, eval_ds, _ = datasets.get_dataset(config,
  #                                             uniform_dequantization=config.data.uniform_dequantization)

  # MRI
  dataset = Brats(transform=transforms.ToTensor())
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [90000, 2897], generator=torch.Generator().manual_seed(99))
  train_iter = cycle(DataLoader(dataset=train_dataset, batch_size = config.training.batch_size, shuffle = True, pin_memory = True))
  eval_iter = cycle(DataLoader(dataset=test_dataset, batch_size = config.eval.batch_size, shuffle = False, pin_memory = True))

  # CIFAR
  # dataset = ds.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
  # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)], generator=torch.Generator().manual_seed(99))
  # train_iter = cycle(DataLoader(dataset=train_dataset, batch_size = config.training.batch_size, shuffle = True, pin_memory = True))
  # eval_iter = cycle(DataLoader(dataset=test_dataset, batch_size = config.eval.batch_size, shuffle = False, pin_memory = True))

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  for step in tqdm.tqdm(range(initial_step, num_train_steps + 1)):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    # batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
    # batch = next(train_iter)[0].to(config.device).float()
    # batch = batch.permute(0, 3, 1, 2)
    batch = next(train_iter).to(config.device)
    batch = scaler(batch)
    # Execute one training step
    loss = train_step_fn(state, batch)
    # print(f"GPU usage after step {step}")
    # gpu_usage()
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      writer.add_scalar("training_loss", loss, step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      # eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
      # eval_batch = next(eval_iter)[0].to(config.device).float()
      # eval_batch = eval_batch.permute(0, 3, 1, 2)
      eval_batch = next(eval_iter).to(config.device)
      eval_batch = scaler(eval_batch)
      eval_loss = eval_step_fn(state, eval_batch)
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
      writer.add_scalar("eval_loss", eval_loss.item(), step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # Generate and save samples
      if config.training.snapshot_sampling:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample, n = sampling_fn(score_model)
        ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        tf.io.gfile.makedirs(this_sample_dir)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
          np.save(fout, sample)

        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
          save_image(image_grid, fout)


def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data pipeline
  # train_ds, eval_ds, _ = datasets.get_dataset(config,
  #                                             uniform_dequantization=config.data.uniform_dequantization,
  #                                             evaluation=True)

  # MRI
  # dataset = Brats(transform=transforms.ToTensor())
  # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [90000, 2897], generator=torch.Generator().manual_seed(99))

  # CIFAR
  dataset = ds.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)], generator=torch.Generator().manual_seed(99))


  train_dl = DataLoader(dataset=train_dataset, batch_size = config.training.batch_size, shuffle = True, pin_memory = True)
  eval_dl = DataLoader(dataset=test_dataset, batch_size = config.eval.batch_size, shuffle = False, pin_memory = True)
  eval_iter = cycle(eval_dl)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)


  # # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  # train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
  #                                                     uniform_dequantization=True, evaluation=True)
  # if config.eval.bpd_dataset.lower() == 'train':
  #   ds_bpd = train_ds_bpd
  #   bpd_num_repeats = 1
  # elif config.eval.bpd_dataset.lower() == 'test':
  #   # Go over the dataset 5 times when computing likelihood on the test dataset
  #   ds_bpd = eval_ds_bpd
  #   bpd_num_repeats = 5
  # else:
  #   raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    while not tf.io.gfile.exists(ckpt_filename):
      if not waiting_message_printed:
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
      time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(60)
      try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
      except:
        time.sleep(120)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss and False:
      all_losses = []
      # eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in tqdm.tqdm(enumerate(eval_dl)):
        # eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        # eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = batch[0].to(config.device)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          eval_batch = scaler(eval_batch)
          bpd = likelihood_fn(score_model, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      for r in tqdm.tqdm(range(num_sampling_rounds)):
        break
        logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        real_img = next(eval_iter)[0].to(config.device)
        # real_img = torch.cat([real_img, real_img, real_img], dim=1)
        real_img = np.clip(real_img.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)

        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
        tf.io.gfile.makedirs(this_sample_dir)
        samples, n = sampling_fn(score_model)
        # samples = torch.cat([samples, samples, samples], dim=1)
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, 3))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
          fout.write(io_buffer.getvalue())
        
        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(real_img, inception_model,
                                                       inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"real_statistics_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
          fout.write(io_buffer.getvalue())

      # Compute inception scores, FIDs and KIDs.
      # Load all statistics that have been previously computed and saved for each host
      all_logits = []
      all_pools = []
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
      for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
          stat = np.load(fin)
          if not inceptionv3:
            all_logits.append(stat["logits"])
          all_pools.append(stat["pool_3"])

      if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
      all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]
      
      # # Load pre-computed dataset statistics.
      # data_stats = evaluation.load_dataset_stats(config)
      # data_pools = data_stats["pool_3"]
      data_pools = []
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "real_statistics_*.npz"))
      for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
          stat = np.load(fin)
          data_pools.append(stat["pool_3"])
      data_pools = np.concatenate(data_pools, axis=0)[:config.eval.num_samples]

      # Compute FID/KID/IS on all samples together.
      if not inceptionv3:
        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
      else:
        inception_score = -1

      fid1 = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools)
      fid2 = evaluation.cal_fid(data_pools,all_pools)
      print(fid1)
      print(fid2)
      exit()

      # Hack to get tfgan KID work for eager execution.
      tf_data_pools = tf.convert_to_tensor(data_pools)
      tf_all_pools = tf.convert_to_tensor(all_pools)
      kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
      del tf_data_pools, tf_all_pools

      logging.info(
        "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
          ckpt, inception_score, fid, kid))

      with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                             "wb") as f:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
        f.write(io_buffer.getvalue())
        

def recon(config,workdir,recon_folder="recon"):
  """Evaluate trained models for reconstruction.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    recon_folder: The subfolder for storing evaluation results. Default to
      "recon".
  """
  # Create directory to recon_folder
  recon_dir = os.path.join(workdir, recon_folder)
  tf.io.gfile.makedirs(recon_dir)

  # Build data pipeline
  test_data_dir = {
    'ct2d_320': 'LIDC_320.npz',
    'ldct_512': 'LDCT.npz',
    'brats': 'BraTS.npz'
  }[config.data.dataset]
  test_data_dir = os.path.join('test_data', test_data_dir)
  test_imgs = np.load(test_data_dir)['all_imgs']


  # MRI
  dataset = Brats(transform=transforms.ToTensor())
  _, test_dataset = torch.utils.data.random_split(dataset, [90000, 2897], generator=torch.Generator().manual_seed(99))
  eval_iter = DataLoader(dataset=test_dataset, batch_size = 128, shuffle = False, pin_memory = True)
  for x in eval_iter:
    current_batch = x.numpy()
    break


  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)
  
  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{config.eval.end_ckpt}.pth')
  state = restore_checkpoint(ckpt_path, state, device=config.device)
  ema.copy_to(score_model.parameters())

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  sampling_shape = (config.eval.batch_size,
                    config.data.num_channels, 
                    config.data.image_size,
                    config.data.image_size)

  cs_solver = cs.get_cs_solver(config, sde, sampling_shape, inverse_scaler, eps=sampling_eps)

  hyper_params = {
    'projection': [config.sampling.coeff, config.sampling.snr],
    'langevin_projection': [config.sampling.coeff, config.sampling.snr],
    'langevin': [config.sampling.projection_sigma_rate, config.sampling.snr],
    'baseline': [config.sampling.projection_sigma_rate, config.sampling.snr]
  }[config.sampling.cs_solver]

  per_host_batch_size = config.eval.batch_size
  num_batches = int(np.ceil(len(test_imgs) / per_host_batch_size))

  # Create a circular mask
  img_size = config.data.image_size
  mask = Image.new('L', (img_size, img_size), 0)
  draw = ImageDraw.Draw(mask)
  draw.pieslice([0, 0, img_size, img_size], 0, 360, fill=255)
  toTensor = transforms.ToTensor()
  mask = toTensor(mask)[0]

  def get_metric(predictions, targets, mask_roi=False, hist_norm=False):
    with torch.no_grad():
      if hist_norm:
        pred_hist = torch.histc(predictions, bins=255)
        targ_hist = torch.histc(targets, bins=255)

        peak_pred1 = torch.argmax(pred_hist[:75]) / 255.
        peak_pred2 = (torch.argmax(pred_hist[75:]) + 75) / 255.
        peak_targ1 = torch.argmax(targ_hist[:75]) / 255.
        peak_targ2 = (torch.argmax(targ_hist[75:]) + 75) / 255.

        predictions = torch.clamp((predictions - peak_pred1) / (peak_pred2 - peak_pred1), min=0)
        targets = torch.clamp((targets - peak_targ1) / (peak_targ2 - peak_targ1), min=0)

        predictions = torch.clamp(predictions, max=torch.max(targets).item(), min=0)
        predictions /= torch.max(targets)
        targets /= torch.max(targets)

      # Mask Region of Interest
      if mask_roi:
        predictions = predictions * mask
        targets = targets * mask

      return (piq.psnr(predictions[None, None, ...], targets[None, None, ...], data_range=1.).item(),
              piq.ssim(predictions[None, None, ...], targets[None, None, ...], data_range=1.).item())

  all_samples = []
  all_ssims = []
  all_psnrs = []
  all_ssims_mask = []
  all_psnrs_mask = []
  all_ssims_mask_hist = []
  all_psnrs_mask_hist = []
  all_mar_ssims = []
  all_mar_psnrs = []
  all_mar_rmses = []

  # for batch in tqdm.tqdm(range(num_batches)):
  for batch in tqdm.tqdm(range(1)):
    # current_batch = np.asarray(test_imgs[batch * per_host_batch_size:
    #                                       min((batch + 1) * per_host_batch_size,
    #                                           len(test_imgs))], dtype=np.float32) / 255.

    n_effective_samples = len(current_batch)
    if n_effective_samples < per_host_batch_size:
      pad_len = per_host_batch_size - len(current_batch)
      current_batch = np.pad(current_batch, ((0, pad_len), (0, 0), (0, 0)),
                              mode='constant', constant_values=0.)
    
    current_batch = current_batch.reshape(*sampling_shape)
    img = scaler(current_batch)

    samples = cs_solver(score_model, torch.from_numpy(img), *hyper_params)

    samples = np.clip(np.asarray(samples.cpu().numpy()), 0., 1.)
    samples = samples.reshape((-1, config.data.image_size, config.data.image_size, 1))[:n_effective_samples]
    all_samples.extend(samples)

    ground_truth = np.asarray(inverse_scaler(img)).reshape((-1, config.data.image_size,
                                                              config.data.image_size, 1))
    ground_truth = np.clip(ground_truth, 0., 1.)
    ground_truth = torch.from_numpy(ground_truth).permute(0, 3, 1, 2)
    samples = torch.from_numpy(samples).permute(0, 3, 1, 2)

    for i in range(n_effective_samples):
      p, s = get_metric(samples[i].squeeze(), ground_truth[i].squeeze())
      all_psnrs.append(p)
      all_ssims.append(s)

      p, s = get_metric(samples[i].squeeze(), ground_truth[i].squeeze(), mask_roi=True)
      all_psnrs_mask.append(p)
      all_ssims_mask.append(s)

      p, s = get_metric(samples[i].squeeze(), ground_truth[i].squeeze(), mask_roi=True, hist_norm=True)
      all_psnrs_mask_hist.append(p)
      all_ssims_mask_hist.append(s)

    print(f'PSNR: {np.asarray(all_psnrs).mean():.2f}±{np.asarray(all_psnrs).std():.2f}, SSIM: {np.asarray(all_ssims).mean():.3f}±{np.asarray(all_ssims).std():.3f}')
    print(f'with mask: PSNR: {np.asarray(all_psnrs_mask).mean():.2f}±{np.asarray(all_psnrs_mask).std():.2f}, SSIM: {np.asarray(all_ssims_mask).mean():.3f}±{np.asarray(all_ssims_mask).std():.3f}')
    print(
      f'with mask & hist: PSNR: {np.asarray(all_psnrs_mask_hist).mean():.2f}±{np.asarray(all_psnrs_mask_hist).std():.2f}, SSIM: {np.asarray(all_ssims_mask_hist).mean():.3f}±{np.asarray(all_ssims_mask_hist).std():.3f}')

  all_samples = (np.stack(all_samples, axis=0) * 255.).astype(np.uint8)
  np.savez_compressed(os.path.join(recon_dir, "reconstructions.npz"), recon=all_samples)

  all_psnrs = np.asarray(all_psnrs)
  all_ssims = np.asarray(all_ssims)
  all_psnrs_mask = np.asarray(all_psnrs_mask)
  all_ssims_mask = np.asarray(all_ssims_mask)
  all_psnrs_mask_hist = np.asarray(all_psnrs_mask_hist)
  all_ssims_mask_hist = np.asarray(all_ssims_mask_hist)

  np.savez_compressed(os.path.join(recon_dir, "metrics.npz"),
                      psnrs=all_psnrs,
                      ssims=all_ssims,
                      psnrs_mask=all_psnrs_mask,
                      ssims_mask=all_ssims_mask,
                      psnrs_mask_hist=all_psnrs_mask_hist,
                      ssims_mask_hist=all_ssims_mask_hist)

  
