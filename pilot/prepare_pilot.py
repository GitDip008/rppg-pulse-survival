"""Real-vs-generated pilot, step 1: cut a clip from a UBFC subject and lay out the eval folders.

    python pilot/prepare_pilot.py                    # subject1, first 30 s
    python pilot/prepare_pilot.py subject5 30

Creates
    pilot/upload/source.png          first frame (the still photo LivePortrait animates)
    pilot/upload/driving.mp4         the 30 s clip (drives the motion)
    pilot/eval/real_lossless/        the clip without compression           + ground truth
    pilot/eval/real_h264/            the exact clip uploaded (H.264, crf 10) + ground truth
    pilot/eval/generated/            put LivePortrait's output here as vid.mp4 (ground truth ready)

Then: upload the two files in pilot/upload to the Colab notebook, put its result in
pilot/eval/generated/vid.mp4, and run   python rppg_demo.py --eval pilot/eval
"""
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent
subject = sys.argv[1] if len(sys.argv) > 1 else "subject1"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
src = ROOT.parent / "dataset" / "DATASET_2" / subject
ff = imageio_ffmpeg.get_ffmpeg_exe()

cap = cv2.VideoCapture(str(src / "vid.avi"))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
rows = (src / "ground_truth.txt").read_text().strip().splitlines()
n = min(int(round(seconds * fps)), len(rows[0].split()))

up, ev = ROOT / "upload", ROOT / "eval"
for d in (up, ev / "real_lossless", ev / "real_h264", ev / "generated"):
    d.mkdir(parents=True, exist_ok=True)


def ffmpeg(*args):
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src / "vid.avi"), *args], check=True)


ffmpeg("-frames:v", "1", str(up / "source.png"))
ffmpeg("-frames:v", str(n), "-c:v", "libx264", "-crf", "10", "-preset", "slow", "-pix_fmt", "yuv420p",
       str(up / "driving.mp4"))
ffmpeg("-frames:v", str(n), "-c:v", "ffv1", str(ev / "real_lossless" / "vid.avi"))
shutil.copy(up / "driving.mp4", ev / "real_h264" / "vid.mp4")

gt = "\n".join(" ".join(r.split()[:n]) for r in rows) + "\n"   # same first n samples of PPG, HR, time
for d in ("real_lossless", "real_h264", "generated"):
    (ev / d / "ground_truth.txt").write_text(gt)

for p in (up / "driving.mp4", ev / "real_lossless" / "vid.avi"):
    c = cv2.VideoCapture(str(p)); m = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
    assert m == n, f"{p.name}: {m} frames, expected {n}"
print(f"{subject}: {n} frames ({n / fps:.1f} s at {fps:.2f} fps)")
print(f"upload these 2 files to Colab:  {up / 'source.png'}  ({(up / 'source.png').stat().st_size / 1e3:.0f} KB)"
      f"  and  {up / 'driving.mp4'}  ({(up / 'driving.mp4').stat().st_size / 1e6:.0f} MB)")
