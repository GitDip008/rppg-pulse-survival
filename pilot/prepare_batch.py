"""Multi-subject pilot: cut the same 30 s clip from several UBFC subjects.

    python pilot/prepare_batch.py                 # subjects 1 3 4 5 8

Creates
    pilot/batch_upload.zip                        upload this one file to Colab (<subject>_source.png + _driving.mp4)
    pilot/eval_batch/real/<subject>/              real clip, lossless (FFV1)           + ground truth
    pilot/eval_batch/generated/<subject>/         ground truth; put the generated vid.mp4 here
Subject 1's lossless LivePortrait result from the first pilot is reused, so only new subjects go to Colab.

Then:  python rppg_demo.py --eval pilot/eval_batch/real
       python rppg_demo.py --eval pilot/eval_batch/generated
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
SECONDS = 30.0
subjects = [f"subject{s}" for s in (sys.argv[1:] or ["1", "3", "4", "5", "8"])]
ff = imageio_ffmpeg.get_ffmpeg_exe()
up, ev = ROOT / "batch_upload", ROOT / "eval_batch"
up.mkdir(exist_ok=True)
reused = ROOT / "eval" / "generated_lossless" / "vid.mp4"      # subject1, first pilot

for s in subjects:
    src = ROOT.parent / "dataset" / "DATASET_2" / s
    cap = cv2.VideoCapture(str(src / "vid.avi")); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    rows = (src / "ground_truth.txt").read_text().strip().splitlines()
    n = min(int(round(SECONDS * fps)), len(rows[0].split()))
    gt = "\n".join(" ".join(r.split()[:n]) for r in rows) + "\n"
    for cond in ("real", "generated"):
        (ev / cond / s).mkdir(parents=True, exist_ok=True)
        (ev / cond / s / "ground_truth.txt").write_text(gt)
    run = lambda *a: subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src / "vid.avi"), *a], check=True)
    run("-frames:v", str(n), "-c:v", "ffv1", str(ev / "real" / s / "vid.avi"))
    if s == "subject1" and reused.exists():
        shutil.copy(reused, ev / "generated" / s / "vid.mp4")
        print(f"{s}: {n} frames | generated video reused from the first pilot"); continue
    run("-frames:v", "1", str(up / f"{s}_source.png"))
    run("-frames:v", str(n), "-c:v", "libx264", "-crf", "10", "-preset", "slow", "-pix_fmt", "yuv420p",
        str(up / f"{s}_driving.mp4"))
    print(f"{s}: {n} frames ({n / fps:.1f} s) | clip ready for Colab")

zp = ROOT / "batch_upload.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:          # mp4 is already compressed
    for f in sorted(up.iterdir()):
        z.write(f, f.name)
print(f"\nUpload this to Colab: {zp}  ({zp.stat().st_size / 1e6:.0f} MB)")
