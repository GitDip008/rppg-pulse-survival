"""Two simple, non-generative baselines inspired by prior work, for the 5 UBFC subjects.

    python pilot/simple_baselines.py

A) SCRAMBLE (cf. pixel-redistribution anonymization: Chen et al. 2022; Wang, Shan et al., JBHI 2025)
   One fixed random permutation of the pixels inside the face box, applied to every frame of the real
   lossless clip. Identity is destroyed; the average colour, and so the pulse, should survive.
   Scored with a fixed face box (roi.txt), and so is a matching real-video control.

B) PUT THE PULSE BACK (cf. PulseEdit, Chen et al., IEEE TIFS 2022)
   Take the real video's skin-colour pulse (band-passed per-channel change of the forehead/cheek mean)
   and multiply it onto the face box of the LivePortrait (lossless) video. The pulse should return,
   but it is, by construction, the ORIGINAL person's pulse.

Outputs (lossless FFV1) go to pilot/eval_simple/{real_fixedroi,scrambled,pulse_back}/<subject>/.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import rppg_demo as d  # noqa: E402

rng = np.random.default_rng(1)
subjects = ["subject1", "subject3", "subject4", "subject5", "subject8"]
OUT = ROOT / "eval_simple"
ff = imageio_ffmpeg.get_ffmpeg_exe()


def frames(path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


def writer(path, w, h, fps):
    """Lossless FFV1 via ffmpeg (raw BGR frames on stdin)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                             "-r", f"{fps}", "-i", "-", "-c:v", "ffv1", str(path)], stdin=subprocess.PIPE)


def median_box(path, n=10):
    boxes = [d.detect(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for i, f in zip(range(n * 30), frames(path)) if i % 30 == 0]
    return np.median([b for b in boxes if b is not None], axis=0).astype(int)


def setup(cond, s, gt):
    dd = OUT / cond / s
    shutil.rmtree(dd, ignore_errors=True); dd.mkdir(parents=True)
    shutil.copy(gt, dd / "ground_truth.txt")
    return dd


for s in subjects:
    real = ROOT / "eval_batch" / "real" / s / "vid.avi"
    gen = ROOT / "eval_batch" / "generated" / s / "vid.mp4"
    gt = ROOT / "eval_batch" / "real" / s / "ground_truth.txt"
    cap = cv2.VideoCapture(str(real)); fps = cap.get(5); W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
    x, y, w, h = median_box(real)
    roi_txt = f"{x} {y} {w} {h}"

    # --- control: the real video, scored with the same fixed box
    dd = setup("real_fixedroi", s, gt)
    shutil.copy(real, dd / "vid.avi"); (dd / "roi.txt").write_text(roi_txt)

    # --- A) scramble the face box with one fixed permutation
    perm = np.random.default_rng(0).permutation(w * h)
    dd = setup("scrambled", s, gt); (dd / "roi.txt").write_text(roi_txt)
    p = writer(dd / "vid.avi", W, H, fps)
    for f in frames(real):
        patch = f[y:y + h, x:x + w].reshape(-1, 3)
        f[y:y + h, x:x + w] = patch[perm].reshape(h, w, 3)
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    if s == "subject1":
        cv2.imwrite(str(ROOT / "output" / "scrambled_subject1.png"), f)

    # --- B) put the real pulse back onto the LivePortrait face
    times, rgbs, n, _ = d.extract_trace(real)                      # real skin colour per frame (RGB)
    rel = rgbs / rgbs.mean(axis=0) - 1                              # relative colour change
    pulse = np.stack([d.bandpass(np.interp(np.arange(n), np.rint(times * fps), rel[:, k])) for k in range(3)], 1)
    gbox = median_box(gen)
    gx, gy, gw, gh = gbox
    dd = setup("pulse_back", s, gt)
    p = writer(dd / "vid.avi", W, H, fps)
    for i, f in enumerate(frames(gen)):
        if i < n:
            gain = 1 + pulse[i][::-1]                               # RGB -> BGR
            reg = f[gy:gy + gh, gx:gx + gw].astype(np.float32) * gain
            # The pulse is ~0.2 grey levels, below one 8-bit step: dither (as camera noise does) before rounding.
            reg += rng.uniform(-0.5, 0.5, reg.shape)
            f[gy:gy + gh, gx:gx + gw] = np.clip(np.rint(reg), 0, 255).astype(np.uint8)
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    print(f"{s}: face box {roi_txt} | scrambled + pulse-back written | pulse amplitude ~{np.abs(pulse).max() * 100:.2f}% "
          f"of skin colour", flush=True)

print("\nNext: python rppg_demo.py --eval pilot/eval_simple/<condition>")
