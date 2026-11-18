"""Corrected evaluation of the SNR-matched DDPM detector.

Idea (unchanged): the AWGN channel output is treated as the forward-diffusion
state x_t at the step t whose noise level matches the channel SNR, and the
learned reverse process is run from t (not from pure noise at T) down to 0.

Fixes relative to ddpm_synth_BER.py / quant_ddpm_PSK.py:
  1. SNR axis.  With x_t = sqrt(abar_t) x + sqrt(1-abar_t) n, rescaling gives
     y = x + w with per-real-dim noise variance s2 = (1-abar_t)/abar_t, so
     Es/N0 = 1/s2 for the +-1 +-j QPSK points (Es = 2, N0 = 2 s2).
     The old axis used -10log10(sum beta), which is off by up to ~2 dB.
  2. Off-by-one.  alphas_cumprod[t] is the noise level of x_t, so the reverse
     loop must start at step t (old code started at t-1).
  3. Baselines on the same noise realisation.  The training/test data
     (N_PSK(..., sgn=True)) repeats ONE QPSK symbol over all positions, i.e. a
     length-L repetition code.  Per-sample slicing (= LMMSE + slicing) ignores
     that structure; the optimal detector averages the L samples then slices.
  4. Error rate is QPSK symbol error rate (SER), labelled as such.

Usage: python eval_fixed.py <model_dir> <out_dir> [n_seq] [esn0_dB ...]
"""
import os
import sys
import time

import numpy as np
import scipy.io as sio
import torch

import denoising_diffusion_pytorch_1d as ddm

model_dir, out_dir = sys.argv[1], sys.argv[2]
n_seq = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
esn0_list = [float(v) for v in sys.argv[4:]] or [-4, -3, -2, -1, 0, 1, 2]
os.makedirs(out_dir, exist_ok=True)
torch.set_num_threads(os.cpu_count())
torch.manual_seed(0)
rng = np.random.default_rng(0)
device = "cpu"

pkg = sio.loadmat(os.path.join(model_dir, "fin_model.mat"))
L = int(pkg["seq_len"][0][0])
betas = torch.from_numpy(pkg["b"][0]).double()
T = int(pkg["t"][0][0])

model = ddm.n_MLP(dim=L, channels=2, n=3, resnet_block_groups=1,
                  sinusoidal_pos_emb_theta=10000)
model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"),
                                 map_location=device))
model.eval()

alphas = 1.0 - betas
abar = torch.cumprod(alphas, 0)
abar_prev = torch.cat([torch.ones(1, dtype=abar.dtype), abar[:-1]])
post_var = betas * (1 - abar_prev) / (1 - abar)
s2_of_t = ((1 - abar) / abar).numpy()


c_eps = (betas / torch.sqrt(1 - abar)).tolist()
c_out = (1 / torch.sqrt(alphas)).tolist()
c_sig = torch.sqrt(post_var).tolist()


@torch.inference_mode()
def reverse_from(xt, t):
    x = xt
    b = x.shape[0]
    tt = torch.empty((b,), dtype=torch.long)
    for i in range(t, -1, -1):
        tt.fill_(i)
        eps = model(x, tt)
        x = (x - c_eps[i] * eps) * c_out[i]
        if i > 0:
            x += c_sig[i] * torch.randn_like(x)
    return x


def qpsk_repetition(n):
    sym = rng.integers(0, 2, size=(n, 2, 1)) * 2 - 1  # +-1 per real dim
    return np.repeat(sym, L, axis=2).astype(np.float64)


def ser(xhat, x):
    # a QPSK symbol is wrong if either real dimension is wrong
    return np.mean(np.any(np.sign(xhat) != x, axis=1))


rows = []
print(f"{'Es/N0':>6} {'t':>5} | {'DDPM':>9} {'per-sample':>10} {'rep-ML':>9} | sec")
for target in esn0_list:
    t = int(np.argmin(np.abs(10 * np.log10(1 / s2_of_t) - target)))
    assert t < T
    s2 = s2_of_t[t]
    esn0 = 10 * np.log10(1 / s2)
    x = qpsk_repetition(n_seq)
    y = x + np.sqrt(s2) * rng.standard_normal(x.shape)       # channel output

    ser_sym = ser(y, x)                                       # LMMSE + slicing
    ser_rep = ser(np.repeat(y.mean(axis=2, keepdims=True), L, axis=2), x)

    t0 = time.time()
    xt = torch.from_numpy(np.sqrt(abar[t].item()) * y).float()  # receiver scaling
    outs = [reverse_from(xt[k:k + 5000], t) for k in range(0, n_seq, 5000)]
    xhat = torch.cat(outs).numpy()
    ser_ddpm = ser(xhat, x)
    dt = time.time() - t0
    rows.append([esn0, t, ser_ddpm, ser_sym, ser_rep])
    print(f"{esn0:6.2f} {t:5d} | {ser_ddpm:9.2e} {ser_sym:10.2e} {ser_rep:9.2e} | {dt:.0f}",
          flush=True)

rows = np.array(rows)
sio.savemat(os.path.join(out_dir, "ser_fixed.mat"),
            {"esn0_dB": rows[:, 0], "t": rows[:, 1], "ser_ddpm": rows[:, 2],
             "ser_persample": rows[:, 3], "ser_repML": rows[:, 4], "n_seq": n_seq})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.figure()
for col, lab, sty in [(3, "per-sample slicing (LMMSE)", "-bo"),
                      (4, f"ML for length-{L} repetition (optimal)", "-ks"),
                      (2, "DDPM, reverse from SNR-matched t", "-go")]:
    m = rows[:, col] > 0
    plt.semilogy(rows[m, 0], rows[m, col], sty, label=lab)
plt.xlabel("Es/N0 (dB)")
plt.ylabel("QPSK symbol error rate")
plt.title(f"{n_seq} sequences per point")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.savefig(os.path.join(out_dir, "ser_fixed.png"), dpi=150)
print("saved", out_dir)
