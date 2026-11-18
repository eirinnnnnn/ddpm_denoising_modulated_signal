"""Plot stored DDPM outputs (test/) on the corrected Es/N0 axis against LMMSE and
repetition-ML baselines; overlays fixed_eval/ser_fixed.mat from eval_fixed.py if present.

Usage: python plot_fixed.py test fixed_eval/ser_combined.png
"""
import sys, numpy as np, scipy.io as sio
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d, out = sys.argv[1], sys.argv[2]
beta, L = 1e-4, 16
Q = norm.sf
snr, ser = [], []
for t in [2000,3000,4000,5000,6000,7000,8000,8900]:
    x = sio.loadmat(f"{d}/testdata_t{t}_dt0_0.mat")['bpsk']
    k = sio.loadmat(f"{d}/kana_t{t}_dt0_0.mat")['samp'][-1]
    abar = (1-beta)**(t+1); s2 = (1-abar)/abar
    snr.append(10*np.log10(1/s2)); ser.append(np.mean(np.any(np.sign(k) != x, axis=1)))
snr, ser = np.array(snr), np.array(ser)
g = np.linspace(-3, 8, 300); a = 10**(g/20)
lmmse = 1-(1-Q(a))**2; opt = 1-(1-Q(np.sqrt(L)*a))**2
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.semilogy(g, lmmse, 'b-', lw=1.8, label='LMMSE, per-sample slicing')
ax.semilogy(g, opt, 'k-', lw=1.8, label=f'ML for length-{L} repetition (optimal)')
ax.semilogy(g, 1-(1-Q(np.sqrt(L)*10**((g-2)/20)))**2, 'k:', lw=1.2, label='optimal + 2 dB')
m = ser > 0
ax.semilogy(snr[m], ser[m], 'go', ms=7, label='DDPM, reverse from SNR-matched step (100k seq.)')
for s in snr[~m]:
    ax.annotate('', xy=(s, 1.2e-6), xytext=(s, 4e-6), arrowprops=dict(arrowstyle='->', color='g'))
ax.text(snr[~m].mean(), 5e-6, f'DDPM: 0 errors\nin {x.shape[0]:,} seq.', color='g', ha='center', fontsize=8)
ax.annotate('', xy=(0, 1.58e-3), xytext=(0, lmmse[np.argmin(abs(g-0))]),
            arrowprops=dict(arrowstyle='<->', color='gray'))
import os
f = os.path.join(os.path.dirname(out), 'ser_fixed.mat')
if os.path.exists(f):
    r = sio.loadmat(f); e, sd = r['esn0_dB'].ravel(), r['ser_ddpm'].ravel(); n = int(r['n_seq'].ravel()[0])
    ax.semilogy(e[sd > 0], sd[sd > 0], 'mx', ms=9, mew=2, label=f'DDPM, fresh run with eval_fixed.py ({n//1000}k seq.)')
    ax.legend(fontsize=8, loc='center right')
ax.set_ylim(1e-6, 1); ax.set_xlim(-3, 8)
ax.set_xlabel(r'$E_s/N_0$ (dB)'); ax.set_ylabel('QPSK symbol error rate')
ax.set_title(f'Length-{L} QPSK repetition over AWGN (corrected SNR axis)', fontsize=10)
ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8, loc='center right')
fig.tight_layout(); fig.savefig(out, dpi=150)
print(np.c_[snr, ser])
