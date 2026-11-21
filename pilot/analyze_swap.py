"""Face-swap pilot analysis (FaceFusion + HyperSwap, lossless), alongside the LivePortrait result.

    python pilot/analyze_swap.py

1. Does the pulse survive?  swapped video vs the real target clip, scored against the TARGET's finger sensor.
2. Whose pulse is it?       the swapped video's pulse vs the finger sensor of the TARGET (original person)
                            and of the SOURCE (whose face was pasted in). An identity leak shows up as a
                            match with the target only.
Writes pilot/output/swap_results.csv and pilot/output/swap_snr.png.
"""
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import ttest_rel

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
import rppg_demo as d  # noqa: E402
from analyze_batch import score, signals  # noqa: E402

pairs = dict(l.split() for l in (ROOT / "swap_upload" / "pairs.txt").read_text().splitlines() if l.strip())
done = [t for t in pairs if any((ROOT / "eval_swap" / "swapped" / t).glob("vid.*"))]
if not done:
    raise SystemExit("No swapped videos yet in pilot/eval_swap/swapped/<subject>/vid.mp4")


def match_with(pred, gt):
    """Mean over 10 s windows: HR error and best waveform correlation (+-0.5 s) against a given sensor."""
    w, n = int(10 * d.FS), min(len(pred), len(gt))
    err, r = [], []
    for s in range(0, n - w + 1, w):
        p, g = pred[s:s + w], gt[s:s + w]
        err.append(abs(d.heart_rate(p)[0] - d.heart_rate(g)[0]))
        r.append(max(np.corrcoef(np.roll(p, k), g)[0, 1] for k in range(-15, 16)))
    return np.mean(err), np.mean(r)


rows = []
for t in done:
    src = pairs[t]
    tgt_err, tgt_snr, tgt_r, nw = score(ROOT / "eval_swap" / "target" / t)
    sw_err, sw_snr, sw_r, _ = score(ROOT / "eval_swap" / "swapped" / t)
    lp = ROOT / "eval_batch" / "generated" / t
    lp_snr = score(lp)[1] if any(lp.glob("vid.*")) else np.nan
    pred, _ = signals(ROOT / "eval_swap" / "swapped" / t)
    _, gt_source = signals(ROOT / "eval_swap" / "target" / src)       # the face donor's finger sensor
    src_err, src_r = match_with(pred, gt_source)
    rows.append((t, src, nw, tgt_err, sw_err, tgt_snr, sw_snr, lp_snr, tgt_r, sw_r, src_err, src_r))

print("1) Does the pulse survive the face swap?  (scored against the original person's finger sensor)")
print(f"{'video':>9} {'face of':>9} | {'HR error bpm':^16} | {'SNR at true HR dB':^27} | {'waveform r':^12}")
print(f"{'':>9} {'':>9} | {'real':>6} {'swapped':>8} | {'real':>7} {'swapped':>8} {'LivePortrait':>10} | {'real':>5} {'swap':>5}")
for t, src, nw, te, se, ts, ss, ls, tr, sr, _, _ in rows:
    print(f"{t:>9} {src:>9} | {te:6.1f} {se:8.1f} | {ts:+7.1f} {ss:+8.1f} {ls:+10.1f} | {tr:5.2f} {sr:5.2f}")
a = np.array([r[3:] for r in rows], float)
m = np.nanmean(a, 0)
print(f"{'mean':>19} | {m[0]:6.1f} {m[1]:8.1f} | {m[2]:+7.1f} {m[3]:+8.1f} {m[4]:+10.1f} | {m[5]:5.2f} {m[6]:5.2f}")
if len(rows) > 1:
    print(f"Pulse SNR drops by {np.mean(a[:, 2] - a[:, 3]):.1f} dB from real to swapped; lower in "
          f"{int((a[:, 2] > a[:, 3]).sum())}/{len(rows)} subjects "
          f"(paired t-test p = {ttest_rel(a[:, 2], a[:, 3]).pvalue:.3f}, n = {len(rows)})")

print("\n2) Whose pulse is it?  swapped video's pulse vs each person's finger sensor")
print(f"{'video':>9} | {'original person (video)':^24} | {'face donor':^22}")
print(f"{'':>9} | {'HR error':>10} {'waveform r':>12} | {'HR error':>9} {'waveform r':>11}")
for t, src, *_x, se, _ss, _ls, _tr, sr, sre, srr in [(r[0], r[1], r[4], r[6], r[7], r[8], r[9], r[10], r[11]) for r in rows]:
    print(f"{t:>9} | {se:10.1f} {sr:12.2f} | {sre:9.1f} {srr:11.2f}   ({src})")
print("A match with the original person and not with the donor means the swapped face carries the "
      "original person's pulse.")

out = ROOT / "output"
with open(out / "swap_results.csv", "w", newline="") as fh:
    cw = csv.writer(fh)
    cw.writerow(["target", "face_from", "windows", "hr_err_real", "hr_err_swapped", "snr_real_db", "snr_swapped_db",
                 "snr_liveportrait_db", "r_real", "r_swapped", "hr_err_vs_donor", "r_vs_donor"])
    cw.writerows([[r[0], r[1], r[2], *[round(float(x), 2) for x in r[3:]]] for r in rows])

fig, ax = plt.subplots(figsize=(7.4, 3.4))
x, w = np.arange(len(rows)), 0.26
ax.bar(x - w, a[:, 2], w, color="#0F6B7A", label="real video")
ax.bar(x, a[:, 3], w, color="#E09F3E", label="face swap (HyperSwap)")
ax.bar(x + w, a[:, 4], w, color="#D1495B", label="photo animation (LivePortrait)")
ax.axhline(0, color="#64748B", lw=0.8)
ax.set_xticks(x, [r[0].replace("subject", "subject ") for r in rows])
ax.set_ylabel("pulse SNR at true heart rate (dB)")
ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
[ax.spines[k].set_visible(False) for k in ("top", "right")]
fig.tight_layout(); fig.savefig(out / "swap_snr.png", dpi=200)
print(f"\nsaved {out / 'swap_results.csv'} and {out / 'swap_snr.png'}")
