"""
Speaker-independent speech-emotion recognition on RAVDESS (speech) comparing MFCC, LPC-cepstra and
PLP-cepstra, individually and fused, with/without pre-emphasis, clean and noisy test conditions.

Protocol
  * Features: frame-synchronous (16 kHz, 25/10 ms), per-utterance mean+std over speech frames (26-d/stream).
  * Fusion: early fusion = concatenation of utterance descriptors; z-scoring is fitted on TRAINING folds only.
  * Classifier: RBF-SVM, fixed hyper-parameters (C=10, gamma='scale'), no tuning (so no tuning leakage).
  * Split: leave-one-actor-out (24 folds) -> no actor ever appears in both train and test.
  * Noise: white Gaussian noise (SNR 20/10/0 dB) added to the audio, with two training protocols:
      - 'clean': train on clean speech only, test on clean + noisy (worst-case mismatch);
      - 'multi': train on clean + all noisy versions of the TRAINING actors, test on the held-out actor
                 at each condition (multi-condition training).
  * Statistics: paired Wilcoxon signed-rank over the per-actor accuracies, Holm-corrected within each family.

Usage:  python run_experiment.py --data_dir /path/to/RAVDESS --out results
"""
import argparse, glob, hashlib, itertools, os, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import features as F

EMOTIONS = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
CONFIGS = {
    "MFCC": ("MFCC",), "LPCC": ("LPCC",), "PLP": ("PLP",),
    "MFCC+LPCC": ("MFCC", "LPCC"), "MFCC+PLP": ("MFCC", "PLP"), "LPCC+PLP": ("LPCC", "PLP"),
    "MFCC+LPCC+PLP": ("MFCC", "LPCC", "PLP"),
}


def parse_name(path):
    p = os.path.basename(path)[:-4].split("-")
    if len(p) != 7 or p[1] != "01":          # keep speech (vocal channel 01); drop songs
        return None
    return dict(path=path, emotion=int(p[2]), actor=int(p[6]))


def seed_for(path, snr):
    return int(hashlib.md5(f"{os.path.basename(path)}|{snr}".encode()).hexdigest()[:8], 16)


def extract_clip(path, snrs):
    """All (pre-emphasis, condition) feature vectors for one clip."""
    x = F.load_audio(path)
    out = {}
    for cond in ["clean"] + list(snrs):
        xc = x if cond == "clean" else F.add_noise(x, cond, np.random.default_rng(seed_for(path, cond)))
        for pe in (False, True):
            out[(pe, cond)] = F.utterance_vector(F.frame_features(xc, preemph=pe))
    return out


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--snrs", type=float, nargs="*", default=[20, 10, 0])
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--protocols", nargs="*", default=["clean", "multi"], choices=["clean", "multi"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    meta = [m for m in (parse_name(p) for p in sorted(glob.glob(os.path.join(a.data_dir, "**", "*.wav"), recursive=True))) if m]
    assert meta, "No RAVDESS speech files (03-01-*-*-*-*-*.wav) found."
    y = np.array([m["emotion"] for m in meta]); g = np.array([m["actor"] for m in meta])
    print(f"{len(meta)} clips, {len(set(g))} actors, {len(set(y))} emotions")

    t0 = time.time()
    feats = Parallel(n_jobs=a.jobs, verbose=5)(delayed(extract_clip)(m["path"], a.snrs) for m in meta)
    print(f"feature extraction done in {time.time() - t0:.0f}s")

    conds = ["clean"] + list(a.snrs)
    def X_for(pe, cond, cfg):
        return np.array([np.concatenate([f[(pe, cond)][s] for s in CONFIGS[cfg]]) for f in feats])

    rows = []
    for pe, cfg in itertools.product((False, True), CONFIGS):
        Xte = {c: X_for(pe, c, cfg) for c in conds}
        for tr, te in LeaveOneGroupOut().split(Xte["clean"], y, g):
            for tm in a.protocols:
                if tm == "clean":
                    Xfit, yfit = Xte["clean"][tr], y[tr]
                else:
                    Xfit = np.vstack([Xte[c][tr] for c in conds]); yfit = np.tile(y[tr], len(conds))
                clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", random_state=0))
                clf.fit(Xfit, yfit)
                for c in conds:
                    pred = clf.predict(Xte[c][te])
                    rows.append(dict(train=tm, preemph=pe, config=cfg, cond=str(c), actor=int(g[te][0]),
                                     acc=float(np.mean(pred == y[te])),
                                     f1=float(f1_score(y[te], pred, average="macro", labels=sorted(EMOTIONS), zero_division=0))))
        print("done", "preemph" if pe else "raw", cfg, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "per_actor_results.csv"), index=False)

    summ = (df.groupby(["train", "preemph", "config", "cond"])[["acc", "f1"]].agg(["mean", "std"]).round(4))
    summ.columns = ["acc_mean", "acc_std", "f1_mean", "f1_std"]; summ = summ.reset_index()
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    # ---- significance (paired over actors; Holm within each family)
    def paired(d1, d2):
        m = d1.merge(d2, on="actor", suffixes=("_1", "_2"))
        diff = m["acc_1"] - m["acc_2"]
        p = 1.0 if np.allclose(diff, 0) else wilcoxon(m["acc_1"], m["acc_2"]).pvalue
        return diff.mean(), p
    srows = []
    for tm in a.protocols:
        dt = df[df.train == tm]
        for pe in (False, True):                                    # family A: fusion vs MFCC baseline
            for cond in map(str, conds):
                fam = []
                for cfg in CONFIGS:
                    if cfg == "MFCC": continue
                    d1 = dt[(dt.preemph == pe) & (dt.config == cfg) & (dt.cond == cond)]
                    d2 = dt[(dt.preemph == pe) & (dt.config == "MFCC") & (dt.cond == cond)]
                    md, p = paired(d1, d2); fam.append(dict(train=tm, family="fusion_vs_MFCC", preemph=pe, cond=cond,
                                                            comparison=f"{cfg} - MFCC", mean_diff_acc=md, p=p))
                ph = holm([r["p"] for r in fam])
                for r, pa in zip(fam, ph): r["p_holm"] = pa
                srows += fam
        for cond in map(str, conds):                                # family B: pre-emphasis effect
            fam = []
            for cfg in CONFIGS:
                d1 = dt[(dt.preemph) & (dt.config == cfg) & (dt.cond == cond)]
                d0 = dt[(~dt.preemph) & (dt.config == cfg) & (dt.cond == cond)]
                md, p = paired(d1, d0); fam.append(dict(train=tm, family="preemph_effect", preemph="on-off", cond=cond,
                                                        comparison=cfg, mean_diff_acc=md, p=p))
            ph = holm([r["p"] for r in fam])
            for r, pa in zip(fam, ph): r["p_holm"] = pa
            srows += fam
    pd.DataFrame(srows).round(4).to_csv(os.path.join(a.out, "significance.csv"), index=False)

    # ---- console tables
    for tm in a.protocols:
        for pe, lab in ((True, "ON"), (False, "OFF")):
            piv = summ[(summ.train == tm) & (summ.preemph == pe)].pivot(index="config", columns="cond", values="acc_mean")[[str(c) for c in conds]]
            print(f"\nAccuracy | train={tm} | pre-emphasis {lab} (mean over actors):\n", piv.round(3))
    print(f"\nChance level = {1/len(set(y)):.3f}.  Results written to {a.out}/")


if __name__ == "__main__":
    main()
