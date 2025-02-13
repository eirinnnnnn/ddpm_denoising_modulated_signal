import dataclasses
import torch
# from denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import denoising_diffusion_pytorch_1d as ddm
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

torch.cuda.set_device(0)

device = "cuda"

savepath = sys.argv[1]

timesteps = 5000


betas = ddm.const_beta_schedule(timesteps = timesteps, betaval=0.0001)

# snr = betas.numpy()
# snr = np.square(snr)
# snr = np.sum(snr)
# snr = np.sqrt(snr)
# snr = 1.55
# snr = 10 
# M_PSK = 2

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
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)


def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start).to(device)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return (sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise).float().to(device)


def noise_img(x, snr, t=None):
    # x_noisy = q_sample(x_start=x, t=t)
    
    # amplitude = pow(10.0,(-snr/40))
    gaussian_noise = np.random.normal(loc=0.0, scale=(snr), size=x.shape)

    x_noisy = gaussian_noise
    return x_noisy

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    
    if noise is None:
        # noise = torch.normal(mean=0.0, std=1.0, size = x_start.shape).to(device)
        noise = torch.normal(mean=0.0, std=1.0, size = x_start.shape).to(device)
        # noise = torch.normal(mean=0.0, std=extract(betas, t=t, x_shape=x_start.shape), size = x_start.shape).to(device) 
        # noise = torch.zeros_like(x_start).to(device) 
        # noise = torch.normal(mean=0.0, std=1/1.414, size = x_start.shape).to(device) 
        # noise = torch.normal(size = x_start)
        # noise = torch.randn_like(x_start).to(device)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)



    # print(f"xnoisy type: {x_noisy.dtype}")

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





# Algorithm 2 (including returning all images)

@torch.no_grad()
def p_sample_loop_mid(model, shape, t, ii, snr, test, mult, M_PSK):
    # device = next(model.parameters()).device

    b = shape[0]
    data = []
    # start from pure noise (for each example in the batch)
    if(test==False):
        datamatf = scipy.io.loadmat(savepath+f"data.mat")
        data = datamatf['bpsk']
        print(data.shape)
        data=data[:ii]
    else:
        data = np.random.randint(0, M_PSK, (shape[0], 1, shape[2])) 
        data = np.tile(data, (1, 2, 1))
        data[data==0] = -1
        data[:,1,:] = 0
        print(data.shape)
        scipy.io.savemat(savepath + f"testdata_t{t}_{mult}.mat", {'bpsk': data})


    
    # data=data.reshape((1,data.shape[0], data.shape[1]))
        # print(data[0])

    # pass the signal through the AWGN with given beta schedule.
    # noise = noise_img(x=data, snr=snr)
    
    data = torch.tensor(data, device=device)
    data = data.float()
    print(data.shape)
    # snr = pow(10, -snr/20) 
    # img = data + np.random.normal(loc=0, scale=1, size = data.shape)*snr
    # scipy.io.savemat(savepath + "noisysig.mat", {'bpsk': img})
    # plt.scatter(img[data[0]==-1][0], img[data[0]==-1][1], color='blue', s=1)
    # plt.scatter(img[data[0]==1][0], img[data[0]==1][1], color='red', s=1)
    img = q_sample(data, t=torch.full((data.shape[0],), t, device= device, dtype= torch.long))
     

    # print(img[0])
    # img = img.to(device)
    
    # img = torch.tensor(img, device=device)
    # img = img.float().to(device)
    imgs = []

    # img = latent 
    # imgs = []

    for i in tqdm(reversed(range(0, t)), desc='sampling loop time step', total=t):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        if (i % (t/5) == 0): imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def looping_p_sample(model, shape, t, data):
    # device = next(model.parameters()).device

    b = shape[0]
    data = torch.tensor(data, device=device)
    data = data.float()
    print(data.shape)
    img = q_sample(data, t=torch.full((data.shape[0],), t, device= device, dtype= torch.long))
     
    imgs = []

    for i in tqdm(reversed(range(0, t)), desc='sampling loop time step', total=t):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        if (i % (t/5) == 0): imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def p_sample_loop(model, shape):
    # device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def p_sample_loop_val(model, shape, test, snr, ii=-1):
    # device = next(model.parameters()).device
    # print(snr)
    b = shape[0]
    # print(b)
    sample_n, channel, seq_length = shape 
    # add the noise onto the generated signals
    if(test==True):    
        # for rep same sequence
        # data = np.random.randint(0, M_PSK, (1, 1, seq_length)) 
        # data = np.tile(data, (1,2,1))
        # data = np.tile(data, (sample_n,1,1))

        # random sequences
        data = np.random.randint(0, M_PSK, (sample_n, 1, seq_length)) 
        data = np.tile(data, (1, channel, 1))

        # data = np.random.randint(0, M_PSK, (sample_n, 1, 1)) 
        # data = np.tile(data, (1,2,seq_length))

        data[data==0] = -1
        data[:,1,:] = 0
        print(data.shape)
        print(data[0])
        scipy.io.savemat(savepath + f"valtest_{snr}.mat", {'bpsk': data})

        
    elif (test==False):
        datamatf = scipy.io.loadmat(savepath+"data.mat")
        data = datamatf['bpsk']
        data = data[:shape[0], :, :]
        print(data[0])

    else:
        datamatf = scipy.io.loadmat(savepath+f"valtest_onesamp_{snr}.mat")
        data = datamatf['bpsk']
        data=data[ii]
        data=data.reshape((1,data.shape[0], data.shape[1]))
        # print(data[0])

    # pass the signal through the AWGN with given beta schedule.
    # noise = noise_img(x=data, snr=snr)
    
    data = torch.tensor(data)
    data = data.float()
    print(data.shape)

    snr = pow(10, -snr/20) 
    img = data + np.random.normal(loc=0, scale=1, size = data.shape)*snr
    # scipy.io.savemat(savepath + "noisysig.mat", {'bpsk': img})
    # plt.scatter(img[data[0]==-1][0], img[data[0]==-1][1], color='blue', s=1)
    # plt.scatter(img[data[0]==1][0], img[data[0]==1][1], color='red', s=1)
    

    # print(img[0])
    # img = img.to(device)
    
    # img = torch.tensor(img, device=device)
    img = img.float().to(device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        # print(torch.full((b,), i, device=device, dtype=torch.long))
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


    

@torch.no_grad()
def sample(model, image_size=64, batch_size=16, channels=1, test=None, snr=0, i=-1, start_t = None, mult=None, M_PSK=2):
    if (start_t != None): return p_sample_loop_mid(model, shape=(batch_size, channels, image_size, ), t = start_t, ii = i, snr=snr, test=test, mult=mult)
    else:
        data = ddm.N_PSK(M_PSK=M_PSK)
        scipy.io.savemat(savepath+f"valtest_t{start_t}_{mult}.mat", {"bpsk":data})
        # return p_sample_loop_val(model, shape=(batch_size, channels, image_size), test=test, snr=snr, ii=i)
        return looping_p_sample(model, shape=(batch_size, channels, image_size,), t = start_t,data=data)
from pathlib import Path

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


