import numpy as np
from sampling import NoneCorrector, NonePredictor, ReverseDiffusionPredictor, get_predictor, get_corrector, \
  shared_predictor_update_fn, shared_corrector_update_fn
from models import utils as mutils
import functools
import torch
import tqdm


def get_cs_solver(config, sde, shape, inverse_scaler, eps=1e-5):
  cs_solver = config.sampling.cs_solver
  # Probability flow ODE sampling with black-box ODE solvers
  predictor = get_predictor(config.sampling.predictor.lower())
  corrector = get_corrector(config.sampling.corrector.lower())

  if cs_solver.lower() == 'projection':
    sampling_fn = get_projection_sampler(config, sde, shape, predictor, corrector,
                                         inverse_scaler,
                                         n_steps=config.sampling.n_steps_each,
                                         probability_flow=config.sampling.probability_flow,
                                         continuous=config.training.continuous,
                                         denoise=config.sampling.noise_removal,
                                         eps=eps)
  else:
    raise ValueError(f"CS solver name {cs_solver} unknown.")

  return sampling_fn


def get_cartesian_mask(shape, n_keep=30):
  # shape [Tuple]: (H, W)
  size = shape[0]                                                       # 240
  center_fraction = n_keep / 1000                                       # 0.03
  acceleration = size / n_keep                                          # acc=8

  num_rows, num_cols = shape[0], shape[1]                               # 240, 240
  num_low_freqs = int(round(num_cols * center_fraction))                # 7

  # create the mask
  mask = torch.zeros((num_rows, num_cols), dtype=torch.float32)         # 240x240
  pad = (num_cols - num_low_freqs + 1) // 2                             # 117
  mask[:, pad: pad + num_low_freqs] = True                              # 240x(117:124) set low freqs to True

  # determine acceleration rate by adjusting for the number of low frequencies
  adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
      num_low_freqs * acceleration - num_cols
  )                                                                     # -1864/-184=10.13

  offset = round(adjusted_accel) // 2                                   # 5

  accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
  accel_samples = np.around(accel_samples).astype(np.int32)
  mask[:, accel_samples] = True

  return mask


def get_masks(config, img):
  if config.sampling.task == 'mri':
    mask = get_cartesian_mask((config.data.image_size, config.data.image_size), n_keep=config.sampling.n_projections)
    mask = mask[None, None, :, :].float()
    return mask

  else:
    raise ValueError(f"task {config.sampling.mask} not recognized.")


def get_known(config, img):
  if config.sampling.task == 'mri':
    print(torch.max(img),torch.min(img))
    return get_kspace(img, axes=(2, 3))

  else:
    raise ValueError(f"task {config.sampling.mask} not recognized.")


def get_kspace(img, axes):
  shape = img.shape[axes[0]]
  return torch.fft.fftshift(
    torch.fft.fftn(torch.fft.ifftshift(
      img, dim=axes
    ), dim=axes),
    dim=axes
  ) / shape


def kspace_to_image(kspace, axes):
  shape = kspace.shape[axes[0]]
  return torch.fft.fftshift(
    torch.fft.ifftn(torch.fft.ifftshift(
      kspace, dim=axes
    ), dim=axes),
    dim=axes
  ) * shape


def merge_known_with_mask(config, x_space, known, mask, coeff=1.):
  # P(Lambda)^{-1}y_t * Lambda * lambda + Tx_t * (I - lambda * Lambda)
  return known * mask * coeff + x_space * (1. - mask * coeff)
    
    
def get_projection_sampler(config, sde, shape, predictor, corrector,
                           inverse_scaler, n_steps=1,
                           probability_flow=False, continuous=True,
                           denoise=True, eps=1e-5, device='cuda'):

  to_space = lambda x: get_kspace(x, (2, 3))
  from_space = lambda x: kspace_to_image(x, (2, 3)).real

  def get_inpaint_update_fn(update_fn):
    def inpaint_update_fn(i, x, t, mask, known, coeff):
      x_space = to_space(x) # Tx_t

      mean, std = sde.marginal_prob(known, t) # alpha(t)Tx_0, beta(t)
      noise = torch.randn(size=x.shape, device=t.device) # z
      noise_space = to_space(noise) # Tz
      noisy_known = mean + std[:,None,None,None] * noise_space 
      # P(Lambda)^{-1}y_t = alpha(t)Tx_0 + beta(t)*Tz

      x_space = merge_known_with_mask(config, x_space, noisy_known, mask, coeff)
      x = from_space(x_space) # T^{-1}
      x, x_mean = update_fn(x, t)

      x0 = from_space(known)
      xtx0 = torch.sum((x - x0)**2).cpu().numpy()
      Axy = torch.sum((from_space(mask * (to_space(x) - noisy_known)))**2).cpu().numpy()
      print(f"{i}, xt-x0:{xtx0:.3f}, Axt-yt:{Axy:.3f}")
      return x

    return inpaint_update_fn

  def projection_sampler(model, img, coeff, snr):
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                              sde=sde,
                                              model=model,
                                              predictor=predictor,
                                              probability_flow=probability_flow,
                                              continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            model=model,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    cs_predictor_update_fn = get_inpaint_update_fn(predictor_update_fn)
    cs_corrector_update_fn = get_inpaint_update_fn(corrector_update_fn)

    # Initial sample
    with torch.no_grad():
      x = sde.prior_sampling(shape).to(device)

      mask = get_masks(config, img).to(device)  # Lambda
      known = get_known(config, img).to(device) # Tx_0

      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

      for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x = cs_corrector_update_fn(i, x, vec_t, mask, known, coeff)
        x = cs_predictor_update_fn(i, x, vec_t, mask, known, coeff)

      if denoise:
        t_eps = torch.full((x.shape[0],), eps, device=device)
        k, std = sde.marginal_prob(torch.ones_like(x), t_eps)
        score_fn = mutils.get_score_fn(sde, model,
                                      train=False, continuous=continuous)
        score = score_fn(x, t_eps)
        x = x / k + (std[:,None,None,None] ** 2 * score / k)
        x_space = to_space(x)
        x_space = merge_known_with_mask(config, x_space, known, mask, 1.)
        x = from_space(x_space)

      return inverse_scaler(x)

  return projection_sampler
