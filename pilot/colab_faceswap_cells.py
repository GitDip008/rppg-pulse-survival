# Face-swap pilot on Colab (T4 GPU). FaceFusion + HyperSwap (hyperswap_1a_256).
# Two cells: paste each block into its own Colab cell and run in order.

# ============================ CELL A: install FaceFusion (3-6 min) ============================
!git clone -q https://github.com/facefusion/facefusion.git /content/facefusion
%cd /content/facefusion
!pip install -q -r requirements.txt
!pip uninstall -y -q onnxruntime
!pip install -q onnxruntime-gpu
!python -c "import onnxruntime as o; print('providers:', o.get_available_providers())"
# expect 'CUDAExecutionProvider' in the list

# ============================ CELL B: swap all 5 subjects (15-30 min) ============================
%cd /content/facefusion
import os, re, glob, shutil, zipfile, subprocess, cv2
from google.colab import files

up = files.upload()                                  # choose pilot/swap_upload.zip (295 MB)
shutil.rmtree('swap_in', ignore_errors=True)
zipfile.ZipFile(next(iter(up))).extractall('swap_in')
os.makedirs('swap_out', exist_ok=True)

def encoder_settings(path):
    head = open(path, 'rb').read(300000).decode('latin1')
    m = re.search(r'options:(.*?)\x00', head, re.S)
    return dict(re.findall(r'(crf|qp|rc)=([\w.]+)', m.group(1))) if m else 'not x264'

for line in open('swap_in/pairs.txt'):
    target, source = line.split()
    out = f'swap_out/{target}_swapped.mp4'
    cmd = ['python', 'facefusion.py', 'headless-run',
           '--source-paths', f'swap_in/source_{source}.png',
           '--target-path', f'swap_in/target_{target}.mp4',
           '--output-path', out,
           '--processors', 'face_swapper',
           '--face-swapper-model', 'hyperswap_1a_256',
           '--face-swapper-pixel-boost', '256x256',
           '--face-selector-mode', 'one',
           '--execution-providers', 'cuda',
           '--temp-frame-format', 'png',
           '--output-video-encoder', 'libx264rgb',
           '--output-video-quality', '100',
           '--output-video-preset', 'medium']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode or not os.path.exists(out):
        print(f'{target}: FAILED\n', (r.stdout + r.stderr)[-2000:]); continue
    c = cv2.VideoCapture(out); n, fps = int(c.get(7)), c.get(5); c.release()
    print(f'{target} <- {source}: {n} frames, {os.path.getsize(out) * 8 / (n / fps) / 1000:.0f} kbit/s, '
          f'encoder {encoder_settings(out)}')

shutil.make_archive('/content/swap_out', 'zip', 'swap_out')
if os.path.ismount('/content/drive'):
    dst = '/content/drive/MyDrive/Oulu/project_researcher/liveportrait_results'
    os.makedirs(dst, exist_ok=True); shutil.copy('/content/swap_out.zip', dst); print('copied to Drive:', dst)
files.download('/content/swap_out.zip')
