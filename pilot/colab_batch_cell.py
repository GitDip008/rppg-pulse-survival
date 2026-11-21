# Paste this whole file into ONE new Colab cell, below the install and weights cells.
# It uploads pilot/batch_upload.zip, runs LivePortrait losslessly on every subject,
# saves the results to Google Drive and downloads batch_out.zip.
%cd /content/LivePortrait
import re, pathlib, glob, os, zipfile, shutil, subprocess, cv2
from google.colab import files

# 1) lossless output (crf/qp 0); safe to run again
for p in pathlib.Path('src').rglob('*.py'):
    s = p.read_text()
    s2 = re.sub(r"crf\s*=\s*\d+", "crf=0", s)
    s2 = re.sub(r"(['\"]crf['\"]\s*,\s*)\d+", r"\g<1>0", s2)
    s2 = re.sub(r"(['\"]-crf['\"]\s*,\s*)['\"]?\d+['\"]?", r"\g<1>'0'", s2)
    if s2 != s:
        p.write_text(s2)

# 2) upload pilot/batch_upload.zip (105 MB)
up = files.upload()
shutil.rmtree('batch_in', ignore_errors=True)
zipfile.ZipFile(next(iter(up))).extractall('batch_in')

# 3) generate each subject
os.makedirs('batch_out', exist_ok=True)
for src in sorted(glob.glob('batch_in/*_source.png')):
    s = os.path.basename(src).replace('_source.png', '')
    shutil.rmtree('animations', ignore_errors=True)
    r = subprocess.run(['python', 'inference.py', '-s', src, '-d', f'batch_in/{s}_driving.mp4',
                        '--flag_crop_driving_video'], capture_output=True, text=True)
    outs = [x for x in glob.glob('animations/*.mp4') if not x.endswith('_concat.mp4')]
    if r.returncode or not outs:
        print(f'{s}: FAILED\n', r.stderr[-1500:]); continue
    shutil.copy(outs[0], f'batch_out/{s}_generated.mp4')
    c = cv2.VideoCapture(outs[0]); n, fps = int(c.get(7)), c.get(5)
    print(f'{s}: {n} frames, {os.path.getsize(outs[0]) * 8 / (n / fps) / 1000:.0f} kbit/s')

# 4) zip, copy to Drive if it is connected, download
shutil.make_archive('/content/batch_out', 'zip', 'batch_out')
if os.path.ismount('/content/drive'):
    dst = '/content/drive/MyDrive/Oulu/project_researcher/liveportrait_results'
    os.makedirs(dst, exist_ok=True); shutil.copy('/content/batch_out.zip', dst)
    print('copied to Drive:', dst)
files.download('/content/batch_out.zip')
