"""Face-swap pilot: keep each real video, swap in another subject's face (FaceFusion + HyperSwap).

    python pilot/prepare_swap.py

Face swapping keeps the target video's own pixels, so the target clips must be lossless:
H.264 crf 10 already erased subject 4's pulse. To keep the upload small, each 30 s clip is
cropped to a fixed 320x320 box around the face and encoded losslessly (H.264 qp 0, yuv444).

Pairs (target video <- source face), rotating through the 5 subjects:
    subject1 <- subject3, subject3 <- subject4, subject4 <- subject5, subject5 <- subject8, subject8 <- subject1
Creates
    pilot/swap_upload.zip            target_<s>.mp4 + source_<s>.png + pairs.txt
    pilot/eval_swap/target/<s>/      the exact uploaded clip + ground truth (baseline)
    pilot/eval_swap/swapped/<s>/     ground truth of the TARGET person; put the swapped vid.mp4 here
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import rppg_demo as d  # noqa: E402

CROP = 320
subjects = ["subject1", "subject3", "subject4", "subject5", "subject8"]
pairs = {t: subjects[(i + 1) % len(subjects)] for i, t in enumerate(subjects)}   # target <- source
ff = imageio_ffmpeg.get_ffmpeg_exe()
up = ROOT / "swap_upload"
shutil.rmtree(up, ignore_errors=True); up.mkdir()


def face_crop(video, n):
    """Fixed CROPxCROP box centred on the median face position over the clip."""
    cap = cv2.VideoCapture(str(video)); W, H = int(cap.get(3)), int(cap.get(4)); centres = []
    for i in range(0, n, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        b = d.detect(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) if ok else None
        if b is not None:
            centres.append((b[0] + b[2] / 2, b[1] + b[3] / 2))
    cap.release()
    cx, cy = np.median(centres, axis=0)
    x = int(np.clip(cx - CROP / 2, 0, W - CROP)) // 2 * 2
    y = int(np.clip(cy - CROP / 2, 0, H - CROP)) // 2 * 2
    return x, y


for s in subjects:
    src = ROOT.parent / "dataset" / "DATASET_2" / s / "vid.avi"
    gt = ROOT / "eval_batch" / "real" / s / "ground_truth.txt"            # same first 30 s
    n = len(gt.read_text().split("\n")[0].split())
    x, y = face_crop(src, n)
    vf = f"crop={CROP}:{CROP}:{x}:{y}"
    run = lambda *a: subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-vf", vf, *a], check=True)
    run("-frames:v", str(n), "-c:v", "libx264", "-qp", "0", "-preset", "veryslow", "-pix_fmt", "yuv444p",
        str(up / f"target_{s}.mp4"))
    run("-frames:v", "1", str(up / f"source_{s}.png"))
    for cond in ("target", "swapped"):
        dd = ROOT / "eval_swap" / cond / s
        shutil.rmtree(dd, ignore_errors=True); dd.mkdir(parents=True)
        shutil.copy(gt, dd / "ground_truth.txt")
    shutil.copy(up / f"target_{s}.mp4", ROOT / "eval_swap" / "target" / s / "vid.mp4")
    print(f"{s} <- face of {pairs[s]} | crop at ({x},{y}) | {n} frames | "
          f"{(up / f'target_{s}.mp4').stat().st_size / 1e6:.0f} MB")

(up / "pairs.txt").write_text("".join(f"{t} {src}\n" for t, src in pairs.items()))
zp = ROOT / "swap_upload.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
    for f in sorted(up.iterdir()):
        z.write(f, f.name)
print(f"\nUpload this to Colab: {zp}  ({zp.stat().st_size / 1e6:.0f} MB)")
