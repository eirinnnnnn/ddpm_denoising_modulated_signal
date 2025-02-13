import scipy.io
import numpy as np
import matplotlib.pyplot as plt 
import sys
import math


dat_loadpath = sys.argv[1]

snr_id = np.arange(1000, 10000, 2000)
ber = [1e-4,0.0007380110807251914,0.0027595232915083832,0.003999580121834425,0.011610125557004558]
lmmse = scipy.io.loadmat("./qam_BER_lmmse.mat")
lmmse = lmmse["BER"][0]
lmmse = lmmse[:4]
snr_beta = scipy.io.loadmat(dat_loadpath + "beta_snr.mat")
snr_beta = snr_beta["snr"][0]
plt.semilogy(-1*snr_beta[snr_id], ber, '-go', label = "DDPM")
plt.semilogy(np.arange(0, 11, 3), lmmse, "-bo", label = "LMMSE")
plt.legend()
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.savefig("ouo.png")