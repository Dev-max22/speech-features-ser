"""Synthetic signals for testing the pipeline (NOT used for any reported result)."""
import numpy as np
from scipy.signal import lfilter
import soundfile as sf
import os

SR = 16000

def synth_vowel(sr=SR, dur=1.5, f0=120, formants=((700, 80), (1200, 90), (2600, 120)), lead=0.3, rng=None):
    rng = rng or np.random.default_rng(0)
    n = int(dur * sr)
    src = np.zeros(n); src[::int(sr / f0)] = 1.0
    y = src
    for fc, bw in formants:
        r = np.exp(-np.pi * bw / sr); th = 2 * np.pi * fc / sr
        y = lfilter([1.0], [1, -2 * r * np.cos(th), r * r], y)
    y = np.concatenate([np.zeros(int(lead * sr)), y])
    y = y / np.sqrt(np.mean(y ** 2)) * 0.1
    return y + rng.standard_normal(len(y)) * 1e-4


def make_fake_ravdess(out_dir, n_actors=8, reps=2, seed=0):
    """Fake RAVDESS-named files (03-01-EMO-INT-STMT-REP-ACTOR.wav). Emotion changes f0 and formant
    positions, actor shifts them too, so a speaker-independent classifier should beat chance."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    for actor in range(1, n_actors + 1):
        a_f0 = rng.uniform(0.8, 1.3); a_fm = rng.uniform(0.9, 1.1)
        for emo in range(1, 9):
            for rep in range(1, reps + 1):
                f0 = 110 * a_f0 * (1 + 0.12 * (emo - 4.5)) * rng.uniform(0.95, 1.05)
                fs = a_fm * (1 + 0.05 * (emo - 4.5))
                fm = ((650 * fs, 80), (1250 * fs, 90), (2500 * fs, 120))
                x = synth_vowel(dur=1.2, f0=f0, formants=fm, lead=0.2, rng=rng)
                name = f"03-01-{emo:02d}-01-01-{rep:02d}-{actor:02d}.wav"
                sf.write(os.path.join(out_dir, name), x, SR)

if __name__ == "__main__":
    import sys
    make_fake_ravdess(sys.argv[1] if len(sys.argv) > 1 else "fake_ravdess")
