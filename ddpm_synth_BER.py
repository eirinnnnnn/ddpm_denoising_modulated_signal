
import scipy.io
import numpy as np
import matplotlib.pyplot as plt 
import sys
import math

np.set_printoptions(precision=None, threshold=10000000, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None, sign=None, floatmode=None)

dat_loadpath = sys.argv[1]
kana_loadpath = sys.argv[1]
ber = []
dss = []
del_var = [1, 10, 100, 1000]
vardiff = 1000
snr_id = np.arange(1000, 8000, 1000)
snr_id_a = np.arange(500, 10000, 500)
for t in snr_id:
    correct_bit = 0
    dss_range = 0
    for numid in range(1):
        numid = 0
        kanafile = f"kana_t{t}_dt{vardiff}_{numid}.mat"
        kanamatf = scipy.io.loadmat(kana_loadpath+kanafile)
        kana = kanamatf['samp']
        kana = kana[-1]
        kana = kana[:,0,:] + 1j*kana[:,1,:]
        # print(kana.shape)

        datafile = f"testdata_t{t}_dt{vardiff}_{numid}.mat"
        datamatf = scipy.io.loadmat(dat_loadpath+datafile)
        data = datamatf['bpsk']
        data = data[:,0,:] + 1j*data[:,1,:]
        # print(data[:10,0])
        dss_range  = dss_range + np.mean(np.abs(kana-data))
        # print(np.std(kana-data))
        # print(np.mean(kana-data))
        sample_n = data.size

        x = np.real(kana)
        datax = np.real(data)
        y = np.imag(kana)
        datay = np.imag(data)
        # print("=========")
        # numerate = [[1/math.sqrt(2),1/math.sqrt(2)], [-1/math.sqrt(2),1/math.sqrt(2)], [-1/math.sqrt(2),-1/math.sqrt(2)], [1/math.sqrt(2),-1/math.sqrt(2)]]
        numerate = [[1,1], [-1,1], [-1,-1], [1,-1]]
        # numerate = [[1,0], [-1,0]]
        for ii in range (len(numerate)):    
            distance = []
            xx, yy = numerate[ii]          
            indices = np.where((datax == xx) & (datay == yy), True, False)
            coor = [x[indices==True], y[indices==True]]
            coor = np.asarray(coor)
            coor = coor.T
            for xi in range (len(numerate)):
                xxx, yyx = numerate[xi]
                orig = [xxx, yyx]
                orig = np.asarray(orig)
                orig = orig.T
                orig = np.tile(orig, (coor.shape[0], 1))
                distance.append(np.square(coor[:, 0] - orig[:, 0]) + np.square(coor[:,1] - orig[:, 1]))
            distance = np.asarray(distance)
            distance = distance.T 
            # print(distance.shape)
            for enum in range(coor.shape[0]):
                # if (enum==0): print(distance[enum])
                if (min(distance[enum]) == distance[enum][ii]): correct_bit = correct_bit + 1
            # print(correct_bit)

        # print(correct_bit)
    dss.append(dss_range/1)
    ber.append(1-correct_bit/1600000)
print(ber)
print(dss)
scipy.io.savemat(dat_loadpath + f"ddpm_ber_dt{vardiff}.mat", {'ber': ber})
scipy.io.savemat(dat_loadpath + f"ddpm_dss_dt{vardiff}.mat", {'dss': dss})
snr_beta = scipy.io.loadmat(dat_loadpath + "beta_snr.mat")
snr_beta = snr_beta["snr"][0]
print(snr_beta[snr_id])
ddpm_snr = []
for i in range (1000, 5400, 500):
    ddpm_snr.append(10*math.log10(1/(0.0001*i)))

lmmse = scipy.io.loadmat(dat_loadpath + "../lmmse/qam_BER_lmmse.mat")
lmmse = lmmse["BER"][0]
lmmse = lmmse[:5]
# ddpm = scipy.io.loadmat(dat_loadpath + "lmmse/bpsk_ddpm_ber.mat")
# ddpm = ddpm["ber"][0]
# print(ddpm)
# print(lmmse)
plt.cla()
# plt.plot(ddpm_snr, ddpm, '-ro', label = "ddpm, const beta")
plt.plot(np.arange(0, 13, 3), lmmse, '-bo', label = "lmmse")
plt.plot(-1*snr_beta[snr_id], ber, '-go', label = "ddpm, lin beta")
# plt.plot(snr_lin[snr_id], ber_lin, '-mo', label = "ddpm_QAM, exp beta")

plt.legend()
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.title("Bit Error Rate")
plt.savefig(dat_loadpath + "ber_cmp.png") 
# ber_lin = scipy.io.loadmat("./beta_exp_qpsk_0509/ddpm_ber.mat")
ber_lin = ber 
snr_lin = snr_beta
# snr_lin = scipy.io.loadmat("./beta_exp_qpsk_0509/beta_snr.mat")
# ber_lin = scipy.io.loadmat("./beta_lin_qpsk_0509/ddpm_ber.mat")
# snr_lin = scipy.io.loadmat("./beta_lin_qpsk_0509/beta_snr.mat")
# snr_lin = snr_lin["snr"][0]
# ber_lin = ber_lin["ber"][0]

plt.cla()
lmmse_dss = scipy.io.loadmat(dat_loadpath + "../lmmse/bpsk_dss_lmmse.mat")
lmmse_dss = lmmse_dss["dss"][0]
lmmse_dss = lmmse_dss[:5]
plt.plot(np.arange(0, 13, 3), lmmse_dss, '-bo', label = "lmmse")
plt.plot(-1*snr_beta[snr_id], dss, '-go', label = "ddpm, lin beta")
# plt.plot(-1*snr_lin[snr_id], ber_lin, '-mo', label = "ddpm, exp beta")
# plt.plot(-1*snr_beta[snr_id], ber, '-go', label = "ddpm_QAM, lin beta")
plt.legend()
plt.xlabel("SNR(dB)")
plt.ylabel("distance squared sum")
plt.title("Distance squared sum")
plt.savefig(dat_loadpath + "dss_cmp.png") 
plt.savefig(kana_loadpath+f"dss.png")

plt.cla()
mis_lmmse = scipy.io.loadmat(dat_loadpath + "../lmmse/qam_BER_lmmse_mis.mat")
mis_lmmse = mis_lmmse["BER"][0]
# mis_lmmse = lmmse[:5]
plt.plot(np.arange(0, 13, 1), mis_lmmse, '-go', label = "lmmse, mismatched")
plt.plot(np.arange(0, 13, 3), lmmse, '-bo', label = "lmmse")
# plt.plot(-1*snr_beta[snr_id], dss, '-go', label = "ddpm, lin beta")
# plt.plot(-1*snr_lin[snr_id], ber_lin, '-mo', label = "ddpm, exp beta")
# plt.plot(-1*snr_beta[snr_id], ber, '-go', label = "ddpm_QAM, lin beta")
plt.legend()
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.title("LMMSE mismatched")
plt.savefig(dat_loadpath + "qam_ber_mis_lmmse.png") 

plt.cla()
lmmse = scipy.io.loadmat(dat_loadpath + "../lmmse/qpsk_simp_ddpm_ber.mat")
lmmse = lmmse["ber"][0]
# lmmse = lmmse[:5]
snr_b = scipy.io.loadmat(dat_loadpath + "../lmmse/qpsk_simp_ddpm_beta.mat")
snr_b = snr_b["snr"][0]
plt.plot(-1*snr_b[snr_id_a], lmmse, '-bo', label = "ddpm, mismatched")
plt.plot(-1*snr_beta[snr_id], ber, '-go', label = "ddpm")

plt.legend()
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.title("Different T, constant beta schedules")
plt.savefig(kana_loadpath+f"ber_t_cmp.png")
