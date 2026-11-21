"""Multi-subject pilot analysis: real vs generated (LivePortrait, lossless), per subject.

    python pilot/analyze_batch.py

For each subject and condition, in non-overlapping 10 s windows (POS):
  HR error  = |HR(video pulse) - HR(finger-sensor PPG)|
  SNR@HR    = power at the TRUE heart rate (+ harmonic) vs rest of band, in dB
  wave r    = best correlation with the finger-sensor PPG within a +-0.5 s shift
Writes pilot/output/batch_results.csv and pilot/output/batch_snr.png.
"""
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import rppg_demo as d  # noqa: E402

WIN = 10.0


def signals(folder):
    video = next(p for p in folder.glob("vid.*") if p.suffix in (".avi", ".mp4"))
    times, rgbs, n, fps = d.cached_trace(video)
    t_gt, ppg = d.load_gt(folder, n, fps)
    fs_rec = (len(ppg) - 1) / t_gt[-1]
    idx = np.rint(times * fps).astype(int); keep = idx < len(ppg)
    times, rgbs, t_gt = idx[keep] / fs_rec, rgbs[keep], np.arange(len(ppg)) / fs_rec
    tu = np.arange(0, min(times[-1], t_gt[-1]), 1 / d.FS)
    return d.bandpass(d.pos(d.resample(times, rgbs, tu))), d.bandpass(np.interp(tu, t_gt, ppg))


def score(folder):
    pred, gt = signals(folder)
    w, nf = int(WIN * d.FS), int(8 * WIN * d.FS)
    f = np.fft.rfftfreq(nf, 1 / d.FS); band = (f >= d.LO) & (f <= d.HI)
    err, snr, r = [], [], []
    for s in range(0, len(pred) - w + 1, w):
        p, g = pred[s:s + w], gt[s:s + w]
        f0 = d.heart_rate(g)[0] / 60
        err.append(abs(d.heart_rate(p)[0] - 60 * f0))
        P = np.abs(np.fft.rfft(p * np.hanning(w), nf)) ** 2
        near = (abs(f - f0) <= .1) | (abs(f - 2 * f0) <= .1)
        snr.append(10 * np.log10(P[band & near].sum() / P[band & ~near].sum()))
        r.append(max(np.corrcoef(np.roll(p, k), g)[0, 1] for k in range(-15, 16)))
    return np.mean(err), np.mean(snr), np.mean(r), len(err)


def main():
    subjects = sorted(p.name for p in (ROOT / "eval_batch" / "generated").iterdir() if any(p.glob("vid.*")))
    rows = []
    for s in subjects:
        re_, rs, rr, nw = score(ROOT / "eval_batch" / "real" / s)
        ge, gs, gr, _ = score(ROOT / "eval_batch" / "generated" / s)
        rows.append((s, nw, re_, ge, rs, gs, rr, gr))

    print(f"{'subject':>9} | {'HR error bpm':^17} | {'SNR at true HR dB':^19} | {'waveform r':^13}")
    print(f"{'':>9} | {'real':>7} {'generated':>9} | {'real':>8} {'generated':>10} | {'real':>5} {'gen.':>6}")
    for s, nw, re_, ge, rs, gs, rr, gr in rows:
        print(f"{s:>9} | {re_:7.1f} {ge:9.1f} | {rs:+8.1f} {gs:+10.1f} | {rr:5.2f} {gr:6.2f}")
    a = np.array([r[2:] for r in rows])
    m, sd = a.mean(0), a.std(0, ddof=1)
    print(f"{'mean±sd':>9} | {m[0]:4.1f}±{sd[0]:<4.1f} {m[1]:5.1f}±{sd[1]:<4.1f} | {m[2]:+5.1f}±{sd[2]:<4.1f} {m[3]:+6.1f}±{sd[3]:<4.1f}"
          f" | {m[4]:.2f}±{sd[4]:.2f} {m[5]:.2f}±{sd[5]:.2f}")
    drop = a[:, 2] - a[:, 3]
    print(f"\nSNR drop real -> generated: {drop.mean():.1f} ± {drop.std(ddof=1):.1f} dB; lower in generated for "
          f"{(drop > 0).sum()}/{len(drop)} subjects")
    print(f"paired t-test p = {ttest_rel(a[:, 2], a[:, 3]).pvalue:.4f} | Wilcoxon (one-sided) p = "
          f"{wilcoxon(a[:, 2], a[:, 3], alternative='greater').pvalue:.4f}  (n = {len(drop)}; 0.031 is the smallest possible)")

    out = ROOT / "output"
    with open(out / "batch_results.csv", "w", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(["subject", "windows", "hr_err_real", "hr_err_gen", "snr_real_db", "snr_gen_db", "r_real", "r_gen"])
        cw.writerows([[r[0], r[1], *[round(x, 2) for x in r[2:]]] for r in rows])

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(rows))
    ax.bar(x - 0.19, a[:, 2], 0.36, color="#0F6B7A", label="real video")
    ax.bar(x + 0.19, a[:, 3], 0.36, color="#D1495B", label="generated (LivePortrait, lossless)")
    ax.axhline(0, color="#64748B", lw=0.8)
    ax.set_xticks(x, [s.replace("subject", "subject ") for s in subjects])
    ax.set_ylabel("pulse SNR at true heart rate (dB)")
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    [ax.spines[k].set_visible(False) for k in ("top", "right")]
    fig.tight_layout(); fig.savefig(out / "batch_snr.png", dpi=200)
    print(f"saved {out / 'batch_results.csv'} and {out / 'batch_snr.png'}")


if __name__ == "__main__":
    main()
