"""
Adaptation from Elucidating Diffusion
https://github.com/NVlabs/edm/blob/main/example.py

NOTE: denoised(x, sigma) = x - sigma * model(.); for vanilla noise predictor model
NOTE: PVD predicts noise (eps ~ N(0, sigma**2))
TODO: double check if equation is actually denoised(x, sigma) = x - model(.)
TODO: ^ double check scalings and training details of PVD that may affect conversion

"""
import torch
import numpy as np


def sample(data, model, opt,
#            num_steps=18,            # NOTE: N in the paper
           sigma_min=0.002,
           sigma_max=80,
           rho=7,                   # NOTE: p in the paper
           # stochastic sampling parameters (worse performance v deterministic according to paper)
           S_churn=0,               # NOTE: s_churn = 0 means deterministic sampling
           S_min=0,
           S_max=float('inf'),
           S_noise=1,
           device=torch.device('cuda')
          ):
    torch.manual_seed(0)

    # NOTE: adapt if PVD affects sigma_min, sigma_max
    # # Adjust noise levels based on what's supported by the network.
    # sigma_min = max(sigma_min, net.sigma_min)
    # sigma_max = min(sigma_max, net.sigma_max)

    num_steps = opt.time_num
    # initial gaussian noise
    latents = torch.randn_like(data, device=device)

    # t_steps: list of noise levels from sigma_max to sigma_min
    step_indices = torch.arange(num_steps, dtype=torch.float, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])      # appends 0 at end

    x_next = latents.to(torch.float) * t_steps[0]     # scales noise by sigma_max
    for i, (t_cur, t_next) in list(enumerate(zip(t_steps[:-1], t_steps[1:]))):  # 0, ..., N-1
        x_cur = x_next

        # NOTE: stochasticity lines (does nothing if S_churn = 0;t_hat = t_cur, x_hat = x_cur)
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = torch.as_tensor(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        # denoised = net(x_hat, t_hat).to(torch.float64)        # original code for when model outputs denoised sample rather than noise
        denoised = x_hat - t_hat * model._denoise(x_hat, t_hat).to(torch.float)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            # denoised = net(x_next, t_next).to(torch.float64)  # original code
            denoised = x_next - t_next * model._denoise(x_next, t_next).to(torch.float)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    # NOTE: still need to rescale x_next based on mean and std of data (done in caller function)
    return x_next

# TODO: delete after debugging done
if __name__ == '__main__':
    data = torch.rand(100,3)
    opt = lambda: None
    opt.nc = 3
    model = lambda: None
    model._denoise = lambda x,t: x
    sample(data, model, opt)
