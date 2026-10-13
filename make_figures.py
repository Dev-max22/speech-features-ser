"""
Publication figures/tables for the revised manuscript (white background, 300 dpi, time in seconds,
shared colour limits for compared panels, labelled colour bars).

Usage:  python make_figures.py --wav /path/to/one_RAVDESS_clip.wav --out figures [--results results]
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import features as F

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 9,
                     "savefig.dpi": 300, "figure.dpi": 100})
HOPS = F.HOP / F.SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True); ap.add_argument("--out", default="figures")
    ap.add_argument("--results", default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    x = F.load_audio(a.wav); xe = F.preemphasis(x)
    t = np.arange(len(x)) / F.SR
    mask = F.speech_mask(F.frame_signal(x))

    # ---------------- Fig 1: waveforms (shared amplitude axis, seconds)
    fig, ax = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True, sharey=True)
    ax[0].plot(t, x, lw=0.4, color="k"); ax[0].set_title("(a) Original speech")
    ax[1].plot(t, xe, lw=0.4, color="k"); ax[1].set_title(r"(b) Pre-emphasised ($\alpha$ = 0.97)")
    for a_ in ax: a_.set_ylabel("Amplitude (a.u.)")
    ax[1].set_xlabel("Time (s)"); fig.tight_layout(); fig.savefig(f"{a.out}/fig1_waveforms.png"); plt.close(fig)

    # ---------------- features (speech frames only for statistics)
    def all_frames(sig):
        P = F.power_spectrum(F.frame_signal(sig)); return P
    P0, P1 = all_frames(x), all_frames(xe)
    M0, M1 = F.mfcc_from_power(P0), F.mfcc_from_power(P1)

    # ---------------- Fig 2: MFCC maps, SHARED colour limits
    vmin = min(M0.min(), M1.min()); vmax = max(M0.max(), M1.max())
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    for a_, M, ttl in zip(ax, (M0, M1), ("(a) Original", "(b) Pre-emphasised")):
        im = a_.imshow(M.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis",
                       extent=[0, M.shape[0] * HOPS, 0.5, F.N_CEP + 0.5])
        a_.set_title(ttl); a_.set_xlabel("Time (s)")
    ax[0].set_ylabel("Cepstral coefficient index")
    cb = fig.colorbar(im, ax=ax, label="MFCC value (dimensionless)"); fig.savefig(f"{a.out}/fig2_mfcc_maps.png", bbox_inches="tight"); plt.close(fig)

    # ---------------- Fig 3: spectral envelopes at the highest-energy speech frame
    e = 10 * np.log10(np.sum(F.frame_signal(x) ** 2, axis=1) + 1e-12); e[~mask] = -np.inf
    k = int(np.argmax(e))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for a_, P, ttl in zip(ax, (P0, P1), ("(a) Original", "(b) Pre-emphasised")):
        dft = 10 * np.log10(P[k] + 1e-20)
        al, el = F.lpc_from_power(P[k:k + 1]); lpc_env, fax = F.lpc_envelope_db(al, el)
        _, ap_, ep = F.plp_cepstra(P[k:k + 1]); plp_env, _ = F.lpc_envelope_db(ap_, ep)
        lpc_env = lpc_env[0] - lpc_env[0].mean() + dft.mean()
        plp_env = plp_env[0] - plp_env[0].mean() + dft.mean()       # levels aligned to the DFT mean for shape comparison
        a_.plot(fax / 1000, dft, color="0.75", lw=0.8, label="DFT (frame)")
        a_.plot(fax / 1000, lpc_env, color="C0", lw=1.5, label=f"LPC, order {F.LPC_ORDER}")
        a_.plot(fax / 1000, plp_env, color="C3", lw=1.5, ls="--", label=f"PLP, order {F.PLP_ORDER}")
        a_.set_title(ttl); a_.set_xlabel("Frequency (kHz)"); a_.set_xlim(0, 8)
    ax[0].set_ylabel("Level (dB, envelopes level-aligned)"); ax[0].legend(loc="upper right")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig3_envelopes.png"); plt.close(fig)

    # ---------------- Table 1 (replacement): per-coefficient mean/std across speech frames
    rows = []
    for i in range(F.N_CEP):
        rows.append({"coef": f"C{i+1}",
                     "raw_mean": M0[mask, i].mean(), "raw_std": M0[mask, i].std(),
                     "preemph_mean": M1[mask, i].mean(), "preemph_std": M1[mask, i].std()})
    pd.DataFrame(rows).round(3).to_csv(f"{a.out}/table1_mfcc_stats.csv", index=False)

    # ---------------- Pre-emphasis quantified: spectral slope & centroid
    f = np.arange(F.NFFT // 2 + 1) * F.SR / F.NFFT; band = (f >= 100) & (f <= 8000)
    def slope_centroid(P):
        sl = [np.polyfit(np.log2(f[band]), 10 * np.log10(p[band] + 1e-20), 1)[0] for p in P[mask]]
        ce = (P[mask] * f).sum(1) / (P[mask].sum(1) + 1e-20)
        return np.array(sl), ce
    s0, c0 = slope_centroid(P0); s1, c1 = slope_centroid(P1)
    pd.DataFrame({"measure": ["spectral slope (dB/octave, 100-8000 Hz)", "spectral centroid (Hz)"],
                  "raw_mean": [s0.mean(), c0.mean()], "raw_std": [s0.std(), c0.std()],
                  "preemph_mean": [s1.mean(), c1.mean()], "preemph_std": [s1.std(), c1.std()]}
                 ).round(2).to_csv(f"{a.out}/table_spectral_tilt.csv", index=False)

    # ---------------- Fig 4: accuracy vs condition (from run_experiment.py output)
    if a.results:
        S = pd.read_csv(os.path.join(a.results, "summary.csv")); S["cond"] = S["cond"].astype(str)
        protos = list(S["train"].unique()); conds = ["clean"] + [c for c in S["cond"].unique() if c != "clean"]
        conds = ["clean"] + sorted([c for c in conds if c != "clean"], key=lambda v: -float(v))
        fig, ax = plt.subplots(1, len(protos), figsize=(3.6 * len(protos) + 0.8, 3.4), sharey=True, squeeze=False)
        for a_, tm in zip(ax[0], protos):
            for cfg in S.config.unique():
                d = S[(S.train == tm) & (S.preemph) & (S.config == cfg)].set_index("cond").loc[conds]
                a_.errorbar(range(len(conds)), d.acc_mean, yerr=d.acc_std / np.sqrt(24), marker="o", ms=3, lw=1, capsize=2, label=cfg)
            a_.set_xticks(range(len(conds))); a_.set_xticklabels([c if c == "clean" else f"{c} dB" for c in conds])
            a_.set_title(f"Training: {'clean only' if tm == 'clean' else 'multi-condition'}"); a_.set_xlabel("Test condition (SNR)")
            a_.axhline(1 / 8, color="k", ls=":", lw=0.8)
        ax[0][0].set_ylabel("Accuracy (mean over actors; bars = s.e.m.)"); ax[0][-1].legend(fontsize=7, loc="upper right")
        fig.tight_layout(); fig.savefig(f"{a.out}/fig4_accuracy_vs_snr.png"); plt.close(fig)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
