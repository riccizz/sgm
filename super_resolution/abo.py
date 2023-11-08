from models import utils as mutils
import torch
import numpy as np
from sampling import get_predictor, get_corrector, shared_corrector_update_fn, shared_predictor_update_fn
from models.utils import get_score_fn
import functools
from tqdm import tqdm


def get_recon_solver(config, sde, operator, shape, inverse_scaler, eps=1e-5):
  predictor = get_predictor(config.sampling.predictor.lower())
  corrector = get_corrector(config.sampling.corrector.lower())

  sampling_fn = get_projection_sampler(config, sde, shape, predictor, corrector,
                                        inverse_scaler, operator,
                                        n_steps=config.sampling.n_steps_each,
                                        probability_flow=config.sampling.probability_flow,
                                        continuous=config.training.continuous,
                                        denoise=config.sampling.noise_removal,
                                        eps=eps)
  return sampling_fn

def get_projection_sampler(config, sde, shape, predictor, corrector,
                           inverse_scaler, operator, n_steps=1,
                           probability_flow=False, continuous=True,
                           denoise=True, eps=1e-5, device='cuda'):

  def get_recon_update_fn(update_fn):
    def recon_update_fn(i, model, x, v, t, y_0, cond=False):
      x, x_mean = update_fn(x, t)

      if cond:
        # hyper
        m_inv = 4
        eps = 1e-8
        gamma = 1
        tao = 0.5

        # perturb y_0 to y_t
        mean, std = sde.marginal_prob(y_0, t) # alpha(t)y_0, beta(t)
        z = torch.randn(size=y_0.shape, device=t.device)
        Az = operator.forward(z) # Az
        y_t = mean + std[:,None,None,None] * Az # y_t = alpha(t)y_0 + beta(t) * Az

        # calc \nabla_{x_t} \| Ax_t - y_t \|^2
        Ax_t = operator.forward(x)
        difference = Ax_t - y_t
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
        print(norm_grad.shape)

        score_fn = get_score_fn(sde, model, train=False, continuous=continuous)
        score = score_fn(x, t)
        print(score.shape)

        cond_score = norm_grad + score
        w = torch.randn_like(x)
        x = x + eps * m_inv * v
        v = v + eps * cond_score
        v = torch.exp(-gamma*eps) * v + torch.sqrt(tao * (1 - torch.exp(-2*gamma*eps)) / m_inv) * w

      # x0 = operator.transpose(known)
      # xtx0 = torch.sum((x - x0)**2).cpu().numpy()
      # Axy = torch.sum((operator.transpose(mask * (operator.forward(x) - noisy_known)))**2).cpu().numpy()
      # print(f"{i}, xt-x0:{xtx0:.3f}, Axt-yt:{Axy:.3f}")
      return x, v

    return recon_update_fn

  def projection_sampler(model, y_0, snr):
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

    cs_predictor_update_fn = get_recon_update_fn(predictor_update_fn)
    cs_corrector_update_fn = get_recon_update_fn(corrector_update_fn)

    # Initial sample
    x = sde.prior_sampling(shape).to(device)
    v = torch.randn_like(x)
    timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

    for i in tqdm(range(1000)):
    # for i in tqdm(range(sde.N)):
      t = timesteps[i]
      vec_t = torch.ones(shape[0], device=t.device) * t
      x, v = cs_corrector_update_fn(i, model, x, v, vec_t, y_0, False)
      x, v = cs_predictor_update_fn(i, model, x, v, vec_t, y_0, False)

      # if denoise:
      #   t_eps = torch.full((x.shape[0],), eps, device=device)
      #   k, std = sde.marginal_prob(torch.ones_like(x), t_eps)
      #   score_fn = get_score_fn(sde, model,
      #                                 train=False, continuous=continuous)
      #   score = score_fn(x, t_eps)
      #   x = x / k + (std[:,None,None,None] ** 2 * score / k)
      #   x_space = to_space(x)
      #   x_space = merge_known_with_mask(config, x_space, known, mask, 1.)
      #   x = from_space(x_space)

    return inverse_scaler(x)

  return projection_sampler





