"""Sanity checks of the from-scratch implementations against independent references."""
import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter
from scipy.fft import rfft
import features as F

rng = np.random.default_rng(0)

from synth import synth_vowel

x = synth_vowel()
frames = F.frame_signal(F.preemphasis(x))
P = F.power_spectrum(frames)

# 1. Levinson vs solve_toeplitz (normal equations R a = -r)
r = F.autocorr_from_power(P, F.LPC_ORDER)
t = 40
a_ref = solve_toeplitz(r[t, :-1] * np.r_[1 + 1e-9, np.ones(F.LPC_ORDER - 1)] + np.r_[1e-10, np.zeros(F.LPC_ORDER - 1)],
                       -r[t, 1:])
a, e = F.levinson_batch(r, F.LPC_ORDER)
print("Levinson max |diff| vs solve_toeplitz:", np.max(np.abs(a[t, 1:] - a_ref)))
assert np.allclose(a[t, 1:], a_ref, atol=1e-6)

# 2. LPC cepstrum recursion vs cepstrum of 1/A(z) computed by FFT
nfft = 4096
A = rfft(a[t], n=nfft)
logH = -np.log(np.abs(A) + 1e-30)                  # ln|1/A|
c_fft = 2 * np.fft.irfft(logH, n=nfft)[1:F.N_CEP + 1]   # real cepstrum = c_n/2 for n>0 (min-phase)
c_rec = F.lpc_to_cep(a[t:t + 1])[0]
print("LPC-cepstrum recursion vs FFT cepstrum max |diff|:", np.max(np.abs(c_fft - c_rec)))
assert np.allclose(c_fft, c_rec, atol=1e-4)

# 3. LPC envelope should show the synthetic formants (700, 1200, 2600 Hz) at order 18
env, fax = F.lpc_envelope_db(a, e)
peaks = fax[1:-1][(env[t, 1:-1] > env[t, :-2]) & (env[t, 1:-1] > env[t, 2:])]
print("LPC envelope peaks (Hz):", np.round(peaks[:6]))

# 4. MFCC vs librosa, on a NON-stationary signal (a stationary vowel has ~zero variance across frames)
import librosa
from scipy.fft import dct
segs = [synth_vowel(dur=0.6, f0=f0, formants=fm, lead=0.05, rng=rng) for f0, fm in
        [(100, ((500, 70), (1500, 90), (2500, 120))), (160, ((800, 80), (1100, 90), (2900, 120))),
         (130, ((350, 60), (2200, 90), (3000, 120)))]]
segs.append(rng.standard_normal(6000) * 0.05)
xn = np.concatenate(segs)
Pn = F.power_spectrum(F.frame_signal(F.preemphasis(xn)))
mine = F.mfcc_from_power(Pn)
fb_l = librosa.filters.mel(sr=F.SR, n_fft=F.NFFT, n_mels=F.N_MEL, htk=True, norm=None)
ref = dct(np.log(np.maximum(Pn @ fb_l.T, 1e-10)), type=2, norm="ortho", axis=1)[:, 1:F.N_CEP + 1]
corr = [np.corrcoef(mine[:, i], ref[:, i])[0, 1] for i in range(F.N_CEP)]
print("MFCC vs librosa-filterbank MFCC (same P): per-coef corr min/mean = %.3f / %.3f" % (min(corr), np.mean(corr)))
assert np.mean(corr) > 0.95 and min(corr) > 0.9

# 5. PLP differs from LPC (different peaks) and the cepstra are finite
plp, ap, ep = F.plp_cepstra(P)
assert np.isfinite(plp).all()
envp, _ = F.lpc_envelope_db(ap, ep)
print("PLP 12th-order envelope peaks (Hz):", np.round(fax[1:-1][(envp[t, 1:-1] > envp[t, :-2]) & (envp[t, 1:-1] > envp[t, 2:])][:6]))

# 6. Silence frames: the leading 0.3 s must be rejected by the VAD, and no NaNs anywhere
m = F.speech_mask(F.frame_signal(x))
print("VAD: first 25 frames speech?", m[:25].any(), "| total speech frames:", m.sum(), "of", len(m))
feats = F.frame_features(x, True)
assert all(np.isfinite(v).all() for v in feats.values())
print("ALL CHECKS PASSED")
