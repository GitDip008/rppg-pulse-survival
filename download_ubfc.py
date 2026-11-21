"""Download UBFC-rPPG (Bobbia et al., Pattern Recognition Letters 2017) into ./dataset.

    python download_ubfc.py                      # small files + DATASET_2 subjects 1 3 4 5 8
    python download_ubfc.py 9 10 11              # add more DATASET_2 subjects
    python download_ubfc.py all                  # all 42 DATASET_2 subjects (~63 GB)

Source: the authors' public Google Drive folder linked from
https://sites.google.com/view/ybenezeth/ubfcrppg . Cite the paper above if you use the data.
Files already on disk are skipped, so the script can be re-run after an interruption.
"""
import sys
from pathlib import Path

import gdown

FOLDER = "https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk"
OUT = Path(__file__).parent / "dataset"
SMALL = {"readme.txt", "Agreement.xlsx", "ubfcrppg_data_processor.py", "ubfcrppg_data_processor.m"}

args = sys.argv[1:] or ["1", "3", "4", "5", "8"]
files = gdown.download_folder(FOLDER, skip_download=True, quiet=True)
want = [f for f in files if f.path in SMALL or (
    f.path.replace("\\", "/").startswith("DATASET_2/") and
    (args == ["all"] or f.path.replace("\\", "/").split("/")[1] in {f"subject{s}" for s in args}))]
want.sort(key=lambda f: f.path not in SMALL)          # small files first

failed = []
for i, f in enumerate(want, 1):
    dst = OUT / f.path.replace("\\", "/")
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[{i}/{len(want)}] have {dst.relative_to(OUT)}"); continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{i}/{len(want)}] {dst.relative_to(OUT)}", flush=True)
    try:
        ok = gdown.download(id=f.id, output=str(dst), quiet=True)
    except Exception as e:                       # Drive's "too many downloads" quota raises here
        ok = False
        print(f"   blocked: {str(e).strip().splitlines()[-1][:120]}", flush=True)
    if not ok:
        dst.unlink(missing_ok=True)
        failed.append(f.path)
print("done" if not failed else f"done, {len(failed)} file(s) blocked; re-run later, or download them in a "
      f"browser from {FOLDER} into {OUT}")
