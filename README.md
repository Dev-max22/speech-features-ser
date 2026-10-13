# Frame-synchronous MFCC / LPC / PLP comparison (RAVDESS speech emotion recognition)

Open-source (NumPy/SciPy) replacement for the MATLAB pipeline in the original manuscript. Every stage
(framing, mel filterbank, Levinson-Durbin LPC, LPC-cepstrum recursion, Bark filterbank, equal-loudness
weighting, cube-root compression, PLP all-pole fit) is implemented in `features.py`.

## Settings (state these in the paper's Methods)
| Item | Value |
|---|---|
| Audio | RAVDESS speech, 24 actors, 8 emotions; channels averaged; resampled to 16 kHz; RMS-normalised |
| Pre-emphasis | y[n] = x[n] - 0.97 x[n-1], applied to the whole signal (on/off compared) |
| Framing | 25 ms Hamming (400 samples), 10 ms hop (160); final partial frame zero-padded |
| FFT | 512 points (>= frame length: no samples discarded); power spectrum |
| MFCC | 26 triangular mel filters, 0-8 kHz, natural log (floor 1e-10), DCT-II orthonormal, C1-C13 (C0 excluded), no liftering, no deltas |
| LPC | autocorrelation method, Levinson-Durbin, order 18 (fs[kHz]+2), per frame; 13 LPC cepstra |
| PLP | Bark integration (1-Bark spacing) -> equal-loudness -> cube-root -> inverse DFT -> order-12 all-pole -> 13 cepstra |
| Speech frames | energy within 35 dB of the loudest frame (VAD on the signal before pre-emphasis) |
| Utterance descriptor | mean and std of each stream over speech frames (26-d per stream) |
| Fusion | early fusion = concatenation; z-scoring fitted on training folds only |
| Classifier | RBF-SVM, C=10, gamma='scale', no tuning |
| Split | leave-one-actor-out (24 folds, speaker-independent) |
| Noise | white Gaussian, SNR 20/10/0 dB; training protocols: `clean` and `multi` |
| Statistics | paired Wilcoxon signed-rank over the 24 per-actor accuracies, Holm-corrected per family |

## Run
```bash
pip install -r requirements.txt
python test_features.py                       # unit checks (should end with ALL CHECKS PASSED)

# 1. download RAVDESS "Audio_Speech_Actors_01-24.zip" (Zenodo; Livingstone & Russo, 2018, PLoS ONE 13(5):e0196391) and unzip
python run_experiment.py --data_dir /path/to/RAVDESS --out results
python make_figures.py --wav /path/to/RAVDESS/Actor_03/03-01-05-01-01-01-03.wav --out figures --results results
```
Outputs: `results/summary.csv`, `results/per_actor_results.csv`, `results/significance.csv`,
`figures/fig1..4_*.png` (300 dpi, white background), `figures/table1_mfcc_stats.csv`, `figures/table_spectral_tilt.csv`.

## Caption notes
* Fig. 3: envelopes are level-aligned to the DFT mean for shape comparison. The PLP envelope is fitted after
  cube-root loudness compression, so its dB range is intentionally much smaller than LPC's.
* Fig. 4: error bars are standard error of the mean over the 24 actors; dotted line = chance (12.5 %).

## Validation done during development (synthetic signals only; no reported result depends on these)
* Levinson-Durbin matches `scipy.linalg.solve_toeplitz` to 1e-13.
* LPC-cepstrum recursion matches the FFT-derived cepstrum of 1/A(z) to 1e-15.
* Order-18 LPC recovers synthetic formants at 719/1188/2594 Hz (true 700/1200/2600).
* MFCC matches librosa's (HTK mel, same power spectrum) with per-coefficient correlation >= 0.96 on a non-stationary signal.
* Spectral-slope measure reproduces the analytic slope change of the pre-emphasis filter (5.21 vs 5.22 dB/octave).
