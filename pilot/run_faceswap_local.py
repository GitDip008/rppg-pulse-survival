"""Run the face-swap pilot locally: FaceFusion + HyperSwap on the laptop GPU (DirectML).

    python pilot/run_faceswap_local.py              # all 5 pairs from pilot/swap_upload/pairs.txt
    python pilot/run_faceswap_local.py --test       # 60 frames of the first pair, to check settings and speed

Needs FaceFusion in E:/things/facefusion (own Python 3.13 venv, onnxruntime-directml, ffmpeg/ffprobe in bin/).
Writes pilot/eval_swap/swapped/<target>/vid.mp4 (lossless: libx264rgb, quality 100) and checks the encoder.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2

FF = Path("E:/things/facefusion")
ROOT = Path(__file__).resolve().parent
UP = ROOT / "swap_upload"
env = dict(os.environ, PATH=str(FF / "bin") + os.pathsep + str(FF / ".venv" / "Scripts") + os.pathsep + os.environ["PATH"])
test = "--test" in sys.argv


def encoder(path):
    head = Path(path).read_bytes()[:300000].decode("latin1")
    m = re.search(r"options:(.*?)\x00", head, re.S)
    return dict(re.findall(r"(crf|qp|rc)=([\w.]+)", m.group(1))) if m else "not x264"


pairs = [l.split() for l in (UP / "pairs.txt").read_text().splitlines() if l.strip()]
for target, source in pairs[:1] if test else pairs:
    out = ROOT / ("swap_test.mp4" if test else f"eval_swap/swapped/{target}/vid.mp4")
    cmd = [str(FF / ".venv" / "Scripts" / "python.exe"), "facefusion.py", "headless-run",
           "--source-paths", str(UP / f"source_{source}.png"),
           "--target-path", str(UP / f"target_{target}.mp4"),
           "--output-path", str(out),
           "--processors", "face_swapper",
           "--face-swapper-model", "hyperswap_1a_256",
           "--face-swapper-pixel-boost", "256x256",
           "--face-selector-mode", "one",
           "--execution-providers", "directml",
           "--temp-frame-format", "png",
           "--output-video-encoder", "libx264rgb",
           "--output-video-quality", "100",
           "--output-video-preset", "medium"] + (["--trim-frame-end", os.environ.get("TEST_FRAMES", "60")] if test else [])
    t0 = time.time()
    r = subprocess.run(cmd, cwd=FF, env=env, capture_output=True, text=True)
    if r.returncode or not out.exists():
        print(f"{target}: FAILED (exit {r.returncode})\n{(r.stdout + r.stderr)[-3000:]}", flush=True)
        continue
    c = cv2.VideoCapture(str(out)); n, fps = int(c.get(7)), c.get(5); c.release()
    dt = time.time() - t0
    print(f"{target} <- face of {source}: {n} frames in {dt:.0f}s ({n / dt:.1f} fps) | "
          f"{out.stat().st_size * 8 / (n / fps) / 1000:.0f} kbit/s | encoder {encoder(out)}", flush=True)
