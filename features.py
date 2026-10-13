"""
Frame-synchronous speech features written from first principles (NumPy/SciPy only).

All three feature streams (MFCC, LPC cepstra, PLP cepstra) use the SAME framing grid:
16 kHz, 25 ms Hamming frames (400 samples), 10 ms hop (160 samples), 512-point FFT
(>= frame length, so no frame samples are discarded).
"""
import math
import numpy as np
import soundfile as sf
from scipy.fft import dct, rfft, irfft
from scipy.signal import resample_poly

SR = 16000
FRAME = 400          # 25 ms
HOP = 160            # 10 ms
NFFT = 512
N_MEL = 26
N_CEP = 13           # C1..C13 (C0 excluded)
LPC_ORDER = 18       # fs[kHz] + 2 rule of thumb
PLP_ORDER = 12
PREEMPH = 0.97
VAD_DB = 35.0        # frames within 35 dB of the loudest frame are "speech"
STREAMS = ("MFCC", "LPCC", "PLP")


# ----------------------------------------------------------------- I/O
def load_audio(path, sr=SR, rms=0.1):
    """Read a WAV file, average channels (no arbitrary channel choice), resample, RMS-normalise."""
    x, fs = sf.read(path, always_2d=True)
    x = x.mean(axis=1)
    if fs != sr:
        g = math.gcd(int(fs), int(sr))
        x = resample_poly(x, sr // g, fs // g)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12) * rms


def add_noise(x, snr_db, rng):
    """White Gaussian noise at a given whole-utterance SNR (dB)."""
    p_sig = np.mean(x ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return x + rng.standard_normal(len(x)) * math.sqrt(p_noise)


# ----------------------------------------------------------------- front end
def preemphasis(x, a=PREEMPH):
    """y[n] = x[n] - a x[n-1]; first sample kept."""
    return np.append(x[0], x[1:] - a * x[:-1])


def frame_signal(x, frame=FRAME, hop=HOP):
    """Split into frames; the final partial frame is zero-padded (so no samples are dropped)."""
    n = 1 + int(math.ceil(max(len(x) - frame, 0) / hop))
    pad = (n - 1) * hop + frame - len(x)
    xp = np.pad(x, (0, pad))
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return xp[idx]


def power_spectrum(frames, nfft=NFFT):
    """One-sided POWER spectrum |X|^2 of Hamming-windowed frames."""
    w = np.hamming(frames.shape[1])
    return np.abs(rfft(frames * w, n=nfft, axis=1)) ** 2


def speech_mask(frames, thresh_db=VAD_DB):
    """Simple energy VAD: frame energy within thresh_db of the loudest frame."""
    e = 10 * np.log10(np.sum(frames ** 2, axis=1) + 1e-12)
    return e > (e.max() - thresh_db)


# ----------------------------------------------------------------- MFCC
def hz2mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel2hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_filt=N_MEL, nfft=NFFT, sr=SR, fmin=0.0, fmax=None):
    """Triangular filters equally spaced on the mel scale between fmin and fmax (default 0..sr/2)."""
    fmax = fmax or sr / 2
    mpts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_filt + 2)
    fpts = mel2hz(mpts)
    bins = np.floor((nfft + 1) * fpts / sr).astype(int)
    fb = np.zeros((n_filt, nfft // 2 + 1))
    for i in range(1, n_filt + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb


_FB = mel_filterbank()


def mfcc_from_power(P, n_cep=N_CEP):
    """Power spectrum -> mel filterbank energies -> natural log (floored) -> DCT-II (orthonormal)
    -> C1..C13. No liftering, no deltas, C0 excluded."""
    E = P @ _FB.T
    logE = np.log(np.maximum(E, 1e-10))
    return dct(logE, type=2, norm="ortho", axis=1)[:, 1:n_cep + 1]


# ----------------------------------------------------------------- LPC
def autocorr_from_power(P, order, nfft=NFFT):
    """Autocorrelation r[0..order] from the power spectrum (Wiener-Khinchin)."""
    r = irfft(P, n=nfft, axis=1)[:, :order + 1]
    return r


def levinson_batch(r, order):
    """Levinson-Durbin for many frames at once. r: (T, order+1).
    Returns a (T, order+1) with a[:,0]==1 (A(z)=1+sum a_k z^-k) and the error energy e (T,)."""
    r = r.copy()
    r[:, 0] = r[:, 0] * (1 + 1e-9) + 1e-10        # tiny white-noise correction (silent frames)
    T = r.shape[0]
    a = np.zeros((T, order + 1))
    a[:, 0] = 1.0
    e = r[:, 0].copy()
    for i in range(1, order + 1):
        acc = np.sum(a[:, :i] * r[:, i:0:-1], axis=1)
        k = -acc / e
        prev = a[:, :i + 1].copy()
        a[:, 1:i + 1] = prev[:, 1:i + 1] + k[:, None] * prev[:, i - 1::-1]
        e = e * (1 - k ** 2)
        e = np.maximum(e, 1e-20)
    return a, e


def lpc_to_cep(a, n_cep=N_CEP):
    """Recursive LPC -> cepstrum (gain term excluded): c_n = -a_n - sum_{k<n} (k/n) c_k a_{n-k}."""
    T, p1 = a.shape
    p = p1 - 1
    c = np.zeros((T, n_cep + 1))
    for n in range(1, n_cep + 1):
        acc = np.zeros(T)
        for k in range(1, n):
            if n - k <= p:
                acc += (k / n) * c[:, k] * a[:, n - k]
        an = a[:, n] if n <= p else 0.0
        c[:, n] = -an - acc
    return c[:, 1:]


def lpc_from_power(P, order=LPC_ORDER):
    r = autocorr_from_power(P, order)
    return levinson_batch(r, order)


def lpc_envelope_db(a, e, nfft=NFFT, sr=SR):
    """All-pole spectral envelope in dB, 10*log10(G^2/|A(f)|^2), and its frequency axis."""
    A = rfft(a, n=nfft, axis=1)
    env = 10 * np.log10(e[:, None] / (np.abs(A) ** 2 + 1e-20) + 1e-20)
    return env, np.arange(nfft // 2 + 1) * sr / nfft


# ----------------------------------------------------------------- PLP (Hermansky 1990)
def hz2bark(f):
    return 6.0 * np.arcsinh(f / 600.0)


def bark2hz(z):
    return 600.0 * np.sinh(z / 6.0)


def plp_filterbank(nfft=NFFT, sr=SR):
    """Critical-band (Bark) integration, 1-Bark spacing, trapezoid-like Hermansky shape."""
    nfilts = int(math.ceil(hz2bark(sr / 2))) + 1
    step = hz2bark(sr / 2) / (nfilts - 1)
    binbarks = hz2bark(np.arange(nfft // 2 + 1) * sr / nfft)
    mids = np.arange(nfilts) * step
    wts = np.zeros((nfilts, nfft // 2 + 1))
    for i, mid in enumerate(mids):
        lof = binbarks - mid - 0.5
        hif = binbarks - mid + 0.5
        wts[i] = 10 ** np.minimum(0.0, np.minimum(hif, -2.5 * lof))
    return wts, mids


_PLP_FB, _PLP_MIDS = plp_filterbank()


def plp_cepstra(P, order=PLP_ORDER, n_cep=N_CEP):
    """Full PLP: Bark integration -> equal-loudness weighting -> cube-root intensity-loudness
    compression -> inverse DFT (autocorrelation) -> all-pole fit -> cepstra."""
    bands = P @ _PLP_FB.T                                    # critical-band integration
    f = bark2hz(_PLP_MIDS)
    fsq = f ** 2
    eql = ((fsq / (fsq + 1.6e5)) ** 2) * ((fsq + 1.44e6) / (fsq + 9.61e6))   # equal-loudness curve
    y = (bands * eql[None, :]) ** 0.33                       # intensity-loudness power law
    y[:, 0] = y[:, 1]                                        # replicate edge bands (as in RASTA-PLP)
    y[:, -1] = y[:, -2]
    sym = np.concatenate([y, y[:, -2:0:-1]], axis=1)         # symmetric auditory spectrum
    r = np.real(np.fft.ifft(sym, axis=1))[:, :order + 1]     # autocorrelation of auditory spectrum
    a, e = levinson_batch(r, order)
    return lpc_to_cep(a, n_cep), a, e


# ----------------------------------------------------------------- per-utterance API
def frame_features(x, preemph=True):
    """Frame-synchronous features for one 16 kHz signal.
    Speech-frame selection (VAD) is computed on the signal BEFORE pre-emphasis, so the same
    frames are compared with/without pre-emphasis. Returns dict of (T_speech, 13) arrays."""
    frames0 = frame_signal(x)
    mask = speech_mask(frames0)
    xs = preemphasis(x) if preemph else x
    P = power_spectrum(frame_signal(xs))
    a, e = lpc_from_power(P)
    plp, _, _ = plp_cepstra(P)
    out = {"MFCC": mfcc_from_power(P), "LPCC": lpc_to_cep(a), "PLP": plp}
    return {k: v[mask] for k, v in out.items()}


def utterance_vector(feats):
    """Per-stream utterance descriptor = [mean(13), std(13)] over speech frames."""
    return {k: np.concatenate([v.mean(0), v.std(0)]) for k, v in feats.items()}
