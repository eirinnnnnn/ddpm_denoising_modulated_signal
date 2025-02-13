import torch
# from denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import denoising_diffusion_pytorch_1d as ddm
import numpy as np
import scipy.io
import os
import sys 
from pathlib import Path 

# from datasets import load_dataset
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

np.set_printoptions(precision=None, threshold=10000000, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None, sign=None, floatmode=None)
torch.cuda.set_device(0)

device = "cuda"

savepath = sys.argv[1]
os.makedirs(savepath, exist_ok = True)

timesteps = 10000
epochs = 5
batch_size = 1000 

# betas = ddm.linear_beta_schedule(timesteps=timesteps)
# betas = ddm.exp_beta_schedule(timesteps=timesteps)
betas = ddm.const_beta_schedule(timesteps = timesteps, betaval=0.0001)
print(betas.shape)
print(betas)
print(np.sum(betas.numpy()))

# sample_n = pow(2,32)
sample_n = 100000
seq_length = 16

# alphas for forward process noise distribute 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis = 0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value = 1.0)
sqrt_recip_alphas = torch.sqrt(1.0/ alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# %% bpsk data config 
M_PSK = 4
data = ddm.N_PSK(M_PSK=M_PSK, seq_length=seq_length, sample_n=sample_n, sgn = True)
# data[:,1,:] = 0
print(data.shape)
print(data[:3, :, :5])
scipy.io.savemat(savepath + "data.mat", {'bpsk': data})

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.normal(mean=0.0, std=1, size = x_start.shape).to(device) 
    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    y = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return y.float()


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l2"):
    global forward_set
    tnp = t.cpu().numpy()
    if noise is None:
        noise = torch.normal(mean=0.0, std=1, size = x_start.shape).to(device) 
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    
    predicted_noise = denoise_model(x_noisy, t)
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss.float()

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean.float()
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # noise = torch.normal(mean=0.0, std=1.0, size = x.shape).to(device)
        # Algorithm 2 line 4:
        return (model_mean + torch.sqrt(posterior_variance_t) * noise).float()


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

print(data.shape)

data = torch.tensor(data, dtype=torch.float).to(device)
dataset = ddm.Dataset1D(data)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

# model = ddm.Unet1D(
#     dim = seq_length,
#     init_dim = None,
#     out_dim = None,
#     dim_mults = (1,2,4,8),
#     channels = 2,
#     self_condition = False,
#     resnet_block_groups = 1,
#     learned_variance = False,
#     learned_sinusoidal_cond = False,
#     random_fourier_features = False,
#     learned_sinusoidal_dim = 16,
#     sinusoidal_pos_emb_theta = 10000,
#     attn_dim_head = 32,
#     attn_heads = 4
# )

model = ddm.n_MLP(
    dim = seq_length,
    init_dim = None,
    out_dim = None,
    channels = 2,
    n=3,
    resnet_block_groups = 1,
    sinusoidal_pos_emb_theta = 10000,
)

optimizer = Adam(model.parameters(), lr = 1e-3)
model = model.to(device)

for epoch in range(epochs):
    print("epoch:", epoch, "/", epochs)
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()
      

      batch_size = batch.shape[0]
      batch = batch.to(device)

      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, timesteps, (batch_size,), device=device).long()
    #   print(t)
      loss = p_losses(model, batch, t, loss_type="l2")

      if step % 10 == 0:
        print("Loss:", loss.item())

      loss.backward()
      optimizer.step()
    # torch.save(model.state_dict(), savepath +f"model_{epoch}.pt")


# Save the model state dictionary
scipy.io.savemat(savepath+"fin_model.mat", 
                 {
                     "seq_len" : seq_length, 
                     "b" : betas.numpy(),
                     "t" : timesteps,
                     "ch" : 2,
                     "M_PSK" : M_PSK,
                     "sam_n" : sample_n
                }, long_field_names=True)
torch.save(model.state_dict(), savepath +"model.pt" )
print(betas)
# scipy.io.savemat(savepath + "forward.mat", {'diffusion': forward_set})
