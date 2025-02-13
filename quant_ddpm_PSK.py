import torch
# from denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import denoising_diffusion_pytorch_1d as ddm
import ddpm_utils as ddpm
import numpy as np
import scipy.io 
import sys

# from datasets import load_dataset
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import os
print(torch.cuda.is_available())
device = "cuda"
model_path = sys.argv[1]
save_path = sys.argv[2]
os.makedirs(save_path, exist_ok = True)



trained_package = scipy.io.loadmat(model_path+"fin_model.mat")
seq_length = trained_package["seq_len"][0][0]
sample_n = trained_package["sam_n"][0][0]
betas = trained_package["b"][0]
timesteps = trained_package["t"][0][0]
channels = trained_package["ch"][0][0]
M_PSK = trained_package["M_PSK"][0][0]
print(betas,seq_length, timesteps, channels, M_PSK)
testing_n = 10000
sample_n = 100000
div = 1
torch.cuda.set_device(div)

snr_beta = np.cumsum(betas)
snr_beta = 10*np.log10(snr_beta)
scipy.io.savemat(save_path + "beta_snr.mat", {'snr': snr_beta})


del_var = [0]
snr_id = np.arange(2000, 9000, 1000)

betas = torch.from_numpy(betas)

# model = ddm.Unet1D(
#     dim = seq_length,
#     init_dim = None,
#     out_dim = None,
#     dim_mults = (1,2,4,8),
#     channels = channels,
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
# if (sys.argv[2]=="U") else  
model = ddm.n_MLP(
    dim = seq_length,
    init_dim = None,
    out_dim = None,
    channels = 2,
    n=3,
    resnet_block_groups = 1,
    sinusoidal_pos_emb_theta = 10000,
)

model.load_state_dict(torch.load(model_path+"model.pt"))
# model.load_state_dict(md)

torch.cuda.set_device(div)
print(torch.cuda.current_device())
model.to(device)
print("ouo")
# model.eval()



alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis = 0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value = 1.0)
sqrt_recip_alphas = torch.sqrt(1.0/ alphas)



sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    # if (t > timesteps):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)


def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start).to(device)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return (sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise).float().to(device)

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l2"):
    if noise is None:
        noise = torch.normal(mean=0.0, std=1.0, size = x_start.shape).to(device)
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
    # device = next(model.parameters()).device
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
        # noise = torch.randn_like(x)
        noise = torch.normal(mean=0.0, std=1.0, size = x.shape).to(device)
        # Algorithm 2 line 4:
        return (model_mean + torch.sqrt(posterior_variance_t) * noise).float().to(device)


def p_sample_loop(model, shape, t, mult, data = None, vardif = 0):
    b = shape[0]

    if (data==None):
        data = ddm.N_PSK(M_PSK=M_PSK, sample_n=shape[0], seq_length=shape[2], sgn = True)
        if (M_PSK==2): data[:,1,:] = 0
        print(data[:3, :, :5])
        scipy.io.savemat(save_path + f"testdata_t{t}_dt{vardif}_{mult}.mat", {'bpsk': data})
        data = torch.tensor(data, device=device)
        data = data.float()
        print(data.shape)

    img = q_sample(data, t=torch.full((data.shape[0],), t+vardif, device= device, dtype= torch.long))
    imgs = []

    for i in tqdm(reversed(range(0, t)), desc='sampling loop time step', total=t):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        if (i==0): imgs.append(img.cpu().numpy())
    return imgs

# def sample(model, shape, t, data, mult):

    # return
 
# t = 1000
torch.cuda.set_device(div)
print(torch.cuda.current_device())
for vardif in del_var:
    for mult in range (1): 
        for t in snr_id:
            samp_test = p_sample_loop(model=model, shape=(sample_n, channels, seq_length, ), t = t, mult = mult, vardif = vardif)
            samp_test = np.array(samp_test)
            scipy.io.savemat(save_path + f"kana_t{t}_dt{vardif}_{mult}.mat", {'samp': samp_test})