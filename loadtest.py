import scipy.io
import numpy as np
import matplotlib.pyplot as plt 
import denoising_diffusion_pytorch as ddm
import torch 
import sys
# from denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D


from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

np.set_printoptions(precision=None, threshold=10000000, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None, sign=None, floatmode=None)

# torch.cuda.set_device(0)
# device = "cuda"

dat_loadpath = sys.argv[1]
kana_loadpath = sys.argv[2]
# loadpath = "./one_batch"
 
# datamatf = scipy.io.loadmat(loadpath+'/data.mat')

# data = datamatf['bpsk']
# data = data[:,0,:]
# # data = data[:4]
# print(data)
# print(data.shape)
# print(data.size)
# print(data[0].size)
# # data = np.asarray(data).astype(int)

# sample_n = (data.size/data[0].size)
# data = np.real(data)

# kanapath = f"kana_t2400.mat"
# kanapath = "forward_diff.ma10"
# datapath = "testdata_t2400.mat" 
# datapath = f"data.mat" 
name= sys.argv[3]
kanamatf = scipy.io.loadmat(kana_loadpath)


kana = kanamatf['samp']
# print(kana.shape)
kana = kana[-1]
# print(kana.shape)
kana = kana[:,0,:] + 1j*kana[:,1,:]
# print(np.round(kana[:15,0].real).astype(int))
print(kana[:10,0])
print(np.var(kana))
print(np.mean(kana))
datamatf = scipy.io.loadmat(dat_loadpath)

data = datamatf['bpsk']
# print(data.shape)
data = data[:,0,:] + 1j*data[:,1,:]
# data = data[:,0,:]
# data = data[data.shape[0]-kana.shape[0]:]

# data = data[:kana.shape[0]]
print(data[:10,0])

print(np.std(kana-data))
print(np.mean(kana-data))
# print(data.shape)
# print(data.size)
# print(data[0].size)
# data = np.asarray(data).astype(int)


# data = np.real(data)
sample_n = data.size
# print(sample_n)

x = np.real(kana)
datax = np.real(data)
print(np.mean(x))
print(np.var(x))
print("--")
y = np.imag(kana)
datay = np.imag(data)
print(np.mean(y))
print(np.var(y))
print("=========")

indices = np.where((datax == 1) & (datay == 1), True, False)
print(len(indices))
plt.scatter(x[indices==True], y[indices==True], color='blue', s=1)
indices = np.where((datax == 1) & (datay == -1), True, False)
plt.scatter(x[indices==True], y[indices==True], color='red', s=1)
indices = np.where((datax == -1) & (datay == -1), True, False)
plt.scatter(x[indices==True], y[indices==True], color='green', s=1)
indices = np.where((datax == -1) & (datay == 1), True, False)
plt.scatter(x[indices==True], y[indices==True], color='purple', s=1)


indices = np.where((datax == 1) & (datay == 0), True, False)
plt.scatter(x[indices==True], y[indices==True], color='blue', s=1)
indices = np.where((datax == -1) & (datay == 0), True, False)
plt.scatter(x[indices==True], y[indices==True], color='red', s=1)

plt.ylim(-1.5, 1.5) 
plt.xlim(-1.5, 1.5) 
plt.ylabel('Imaginary') 
plt.xlabel('Real') 

plt.savefig(name) 

# diff = kana-data
# print(diff[0])
# diff = [np.sqrt(x.real*x.real+x.imag*x.imag) for x in diff]
diff = np.abs(kana-data)
mag = np.abs(data)
print (np.std(diff))
# print(data)
# print (np.mean(mag))
# print (np.mean(mag)/np.mean(diff))
# print(diff.shape)
# print('---')
# print(diff[0])
# err_avg = np.sum(diff)/sample_n
# print(err_avg)