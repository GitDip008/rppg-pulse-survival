"""Webcam rPPG demo: reads your pulse from face video with the POS method.

    python rppg_demo.py                       # live webcam
    python rppg_demo.py --save demo.mp4       # live webcam, and record the window for the slides
    python rppg_demo.py --video clip.mp4      # play a video file through the live view
    python rppg_demo.py --eval DATASET_2      # score against ground truth (UBFC-rPPG layout), no window
    python rppg_demo.py --selftest            # check the signal pipeline on synthetic data

Keys: q or Esc to quit.

Pipeline: face box (OpenCV Haar, on a half-size frame) -> forehead + cheek patches -> mean RGB
per frame -> resample to a steady 30 Hz -> POS (Wang et al., IEEE TBME 2017) -> band-pass
0.7-3 Hz -> strongest frequency = heart rate. SNR = power at the heart rate and its first
harmonic vs the rest of the band (de Haan & Jeanne, 2013).

--eval expects UBFC-rPPG folders: each subject folder holds vid.avi plus ground_truth.txt
(DATASET_2: line 1 PPG, line 2 HR, line 3 time in s) or gtdump.xmp (DATASET_1: CSV of
time_ms, HR, SpO2, PPG). Heart rate is computed the same way from the predicted pulse and the
finger-sensor PPG in non-overlapping windows, then MAE, RMSE and Pearson r are reported.
"""
import argparse
import collections
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

FS = 30.0             # resampling rate (Hz); webcams jitter, so we resample to a fixed clock
WIN_S = 10.0          # seconds of signal used for the live estimate
MIN_S = 5.0           # seconds needed before showing a heart rate
LO, HI = 0.7, 3.0     # pulse band in Hz (42-180 bpm)
DET_EVERY = 10        # run face detection every N frames (the slow part)
DET_SCALE = 0.5       # detect on a half-size frame
EST_EVERY_S = 0.5     # recompute the live heart rate twice a second


# ---------------------------------------------------------------- signal processing
def pos(rgb, fs=FS):
    """POS: project normalised RGB onto the plane orthogonal to skin tone. rgb: (N, 3)."""
    n, l = len(rgb), int(1.6 * fs)
    if n < l:
        return np.zeros(n)
    w = np.lib.stride_tricks.sliding_window_view(rgb, l, axis=0)   # (n-l+1, 3, l)
    c = w / w.mean(axis=2, keepdims=True)
    s1 = c[:, 1] - c[:, 2]                                          # G - B
    s2 = c[:, 1] + c[:, 2] - 2 * c[:, 0]                            # G + B - 2R
    p = s1 + (s1.std(axis=1, keepdims=True) / (s2.std(axis=1, keepdims=True) + 1e-9)) * s2
    p -= p.mean(axis=1, keepdims=True)
    h = np.zeros(n)
    for k in range(l):                                              # overlap-add
        h[k:k + n - l + 1] += p[:, k]
    return h


def chrom(rgb, fs=FS):
    """CHROM (de Haan & Jeanne, IEEE TBME 2013): two chrominance signals, combined per window."""
    n, l = len(rgb), int(1.6 * fs)
    if n < l:
        return np.zeros(n)
    c = np.lib.stride_tricks.sliding_window_view(rgb, l, axis=0)
    c = c / c.mean(axis=2, keepdims=True)
    x = 3 * c[:, 0] - 2 * c[:, 1]                                   # 3R - 2G
    y = 1.5 * c[:, 0] + c[:, 1] - 1.5 * c[:, 2]                     # 1.5R + G - 1.5B
    s = x - (x.std(axis=1, keepdims=True) / (y.std(axis=1, keepdims=True) + 1e-9)) * y
    s = (s - s.mean(axis=1, keepdims=True)) * np.hanning(l)
    h = np.zeros(n)
    for k in range(l):
        h[k:k + n - l + 1] += s[:, k]
    return h


def green(rgb, fs=FS):
    """GREEN (Verkruysse et al., Optics Express 2008): the normalised green channel."""
    return rgb[:, 1] / rgb[:, 1].mean() - 1


METHODS = {"pos": pos, "chrom": chrom, "green": green}


def bandpass(x, fs=FS):
    b, a = butter(3, [LO / (fs / 2), HI / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def heart_rate(x, fs=FS):
    """Return (bpm, snr_db) from a band-passed pulse signal."""
    n = 8 * len(x)                                   # zero-pad for finer frequency bins
    f = np.fft.rfftfreq(n, 1 / fs)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    band = (f >= LO) & (f <= HI)
    f0 = f[band][np.argmax(p[band])]
    near = (np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.1)
    return 60 * f0, 10 * np.log10(p[band & near].sum() / (p[band & ~near].sum() + 1e-12))


def resample(times, values, tu):
    v = np.asarray(values, float)
    return np.stack([np.interp(tu, times, v[:, k]) for k in range(v.shape[1])], 1) if v.ndim == 2 \
        else np.interp(tu, times, v)


def estimate(times, rgbs, method="pos"):
    """Live: resample the raw trace to FS, extract the pulse, return (pulse, bpm, snr) or None."""
    t = np.asarray(times)
    if t[-1] - t[0] < MIN_S:
        return None
    tu = np.arange(t[0], t[-1], 1 / FS)
    pulse = bandpass(METHODS[method](resample(t, rgbs, tu)))
    return (pulse, *heart_rate(pulse))


def window_hrs(pred, gt, win_s=10.0, fs=FS, step_s=None):
    """Heart rate per window for prediction and ground truth (non-overlapping unless step_s < win_s)."""
    n, step = int(win_s * fs), int((step_s or win_s) * fs)
    out = [(*heart_rate(pred[s:s + n]), heart_rate(gt[s:s + n])[0]) for s in range(0, len(pred) - n + 1, step)]
    return np.array(out).reshape(-1, 3)              # columns: hr_pred, snr, hr_gt


def metrics(hr_p, hr_g):
    e = hr_p - hr_g
    r = np.corrcoef(hr_p, hr_g)[0, 1] if len(e) > 1 and hr_g.std() > 0 and hr_p.std() > 0 else float("nan")
    return np.abs(e).mean(), np.sqrt((e ** 2).mean()), r


# ---------------------------------------------------------------- face + skin regions
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
ROIS = [(0.30, 0.08, 0.70, 0.22),   # forehead   (x0, y0, x1, y1) as fractions of the face box
        (0.16, 0.50, 0.38, 0.72),   # left cheek
        (0.62, 0.50, 0.84, 0.72)]   # right cheek


def detect(gray):
    small = cv2.resize(gray, None, fx=DET_SCALE, fy=DET_SCALE)
    faces = CASCADE.detectMultiScale(small, 1.2, 5, minSize=(50, 50))
    if not len(faces):
        return None
    return (np.asarray(max(faces, key=lambda b: b[2] * b[3])) / DET_SCALE).astype(int)


def roi_rects(box):
    x, y, w, h = box
    return [(int(x + a * w), int(y + b * h), int(x + c * w), int(y + d * h)) for a, b, c, d in ROIS]


def skin_rgb(frame, box):
    pix = np.concatenate([frame[y0:y1, x0:x1].reshape(-1, 3) for x0, y0, x1, y1 in roi_rects(box)])
    return pix[:, ::-1].mean(axis=0) if len(pix) else None      # BGR -> RGB


class FaceTracker:
    """Detect every DET_EVERY frames and smooth the box between detections."""
    def __init__(self):
        self.box, self.i = None, 0

    def update(self, frame):
        if self.i % DET_EVERY == 0 or self.box is None:
            found = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if found is not None:
                self.box = found if self.box is None else (0.7 * np.asarray(self.box) + 0.3 * found).astype(int)
        self.i += 1
        return self.box


# ---------------------------------------------------------------- drawing
def panel(width, height, pulse, bpm, snr, status, fps, method):
    img = np.full((height, width, 3), (51, 42, 14), np.uint8)       # BGR of #0E2A33
    put = lambda s, xy, sc, col, th=1: cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
    put(f"rPPG  |  {method.upper()} method  |  {fps:4.1f} fps", (24, 40), 0.65, (200, 215, 220))
    if pulse is None:
        put(status, (24, height // 2), 0.85, (143, 169, 178), 2)
        return img
    put(f"{bpm:5.1f} bpm", (24, 120), 2.0, (255, 255, 255), 3)
    put(f"signal quality (SNR): {snr:+.1f} dB", (24, 165), 0.65, (143, 169, 178))
    y0, y1 = 210, height - 30
    seg = pulse[-int(WIN_S * FS):-int(0.5 * FS)]                    # drop the filter's edge wobble
    seg = (seg - seg.min()) / (np.ptp(seg) + 1e-9)
    pts = np.stack([np.linspace(24, width - 24, len(seg)), y1 - seg * (y1 - y0)], 1).astype(np.int32)
    cv2.polylines(img, [pts], False, (91, 73, 209), 2, cv2.LINE_AA)  # #D1495B
    put(f"last {WIN_S:.0f} s", (24, height - 8), 0.5, (143, 169, 178))
    return img


# ---------------------------------------------------------------- live view
def open_capture(source):
    if isinstance(source, str):
        return cv2.VideoCapture(source)
    # ponytail: DirectShow opens faster and often gives a higher frame rate than MSMF on Windows
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def run(source, save, method="pos"):
    cap = open_capture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source!r}")
    is_file = isinstance(source, str)
    fps_file = cap.get(cv2.CAP_PROP_FPS) or 30.0
    times = collections.deque(maxlen=int(WIN_S * 60))   # room for up to 60 fps
    rgbs = collections.deque(maxlen=int(WIN_S * 60))
    face, frame_i, t0 = FaceTracker(), 0, time.perf_counter()
    bpms, snrs, writer, result = collections.deque(maxlen=15), [], None, None
    last_est, fps, t_prev = -1.0, 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if fps else 1 / max(now - t_prev, 1e-6)
        t_prev = now
        t = frame_i / fps_file if is_file else now - t0
        frame_i += 1
        box = face.update(frame)
        status = "Looking for a face..."
        if box is not None:
            rgb = skin_rgb(frame, box)
            if rgb is not None:
                times.append(t); rgbs.append(rgb)
            for x0, y0, x1, y1 in roi_rects(box):
                cv2.rectangle(frame, (x0, y0), (x1, y1), (91, 73, 209), 2)
            status = "Hold still, collecting signal..."
            if t - last_est >= EST_EVERY_S and len(times) > 10:
                last_est = t
                result = estimate(times, rgbs, method)
                if result:
                    bpms.append(result[1]); snrs.append(result[2])

        pulse, bpm, snr = (result[0], float(np.median(bpms)), result[2]) if result else (None, 0, 0)
        out = np.hstack([frame, panel(int(frame.shape[0] * 0.9), frame.shape[0], pulse, bpm, snr, status, fps, method)])
        if save and writer is None:
            writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps_file if is_file else max(fps, 10),
                                     (out.shape[1], out.shape[0]))
        if writer:
            writer.write(out)
        cv2.imshow("rPPG demo", out)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    if bpms:
        print(f"Heart rate (median of last estimates): {np.median(bpms):.1f} bpm | mean SNR: {np.mean(snrs):+.1f} dB"
              f" | average {fps:.1f} fps")


# ---------------------------------------------------------------- ground-truth evaluation
def load_gt(folder, n_frames, fps):
    """Return (times_s, ppg) from a UBFC-rPPG subject folder."""
    folder = Path(folder)
    if (folder / "ground_truth.txt").exists():                        # DATASET_2
        rows = [np.array(l.split(), float) for l in (folder / "ground_truth.txt").read_text().strip().splitlines()]
        ppg, t = rows[0], (rows[2] if len(rows) > 2 else None)
    elif (folder / "gtdump.xmp").exists():                            # DATASET_1
        a = np.loadtxt(folder / "gtdump.xmp", delimiter=",")
        ppg, t = a[:, 3], a[:, 0] / 1000.0
    else:
        raise FileNotFoundError(f"no ground_truth.txt or gtdump.xmp in {folder}")
    if t is None or len(t) != len(ppg):
        t = np.arange(len(ppg)) / fps if len(ppg) == n_frames else np.linspace(0, n_frames / fps, len(ppg))
    return t - t[0], ppg


def extract_trace(video):
    """Mean skin RGB for every frame of a video, without opening a window.
    A roi.txt ("x y w h") next to the video fixes the face box (for videos a detector can't read)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    roi = Path(video).with_name("roi.txt")
    fixed = np.array(roi.read_text().split(), int) if roi.exists() else None
    face, times, rgbs, i = FaceTracker(), [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        box = fixed if fixed is not None else face.update(frame)
        rgb = skin_rgb(frame, box) if box is not None else None
        if rgb is not None:
            times.append(i / fps); rgbs.append(rgb)
        i += 1
    cap.release()
    return np.array(times), np.array(rgbs), i, fps


def cached_trace(video):
    """extract_trace, cached next to the video: reading GBs of video is slow, the trace is tiny."""
    cache = Path(video).with_name("trace.npz")
    if cache.exists() and cache.stat().st_mtime > Path(video).stat().st_mtime:
        z = np.load(cache)
        return z["times"], z["rgbs"], int(z["n"]), float(z["fps"])
    times, rgbs, n, fps = extract_trace(video)
    np.savez(cache, times=times, rgbs=rgbs, n=n, fps=fps)
    return times, rgbs, n, fps


def evaluate(path, win_s, methods=tuple(METHODS), step_s=None):
    path = Path(path)
    videos = [path] if path.is_file() else sorted(p for p in path.rglob("vid.*") if p.suffix.lower() in (".avi", ".mp4"))
    if not videos:
        raise SystemExit(f"No vid.avi / vid.mp4 found under {path}")
    rows, pooled = [], {m: ([], []) for m in methods}
    for v in videos:
        t0 = time.perf_counter()
        times, rgbs, n, fps = cached_trace(v)
        name = v.parent.name
        if len(times) < MIN_S * fps:
            print(f"{name:>12}: face not found often enough, skipped"); continue
        t_gt, ppg = load_gt(v.parent, n, fps)
        if abs(n - len(ppg)) <= max(2, 0.02 * len(ppg)):
            # One sensor sample per frame (UBFC). Treat frames as evenly spaced (the per-sample
            # timestamps carry ~10 ms logging jitter, incl. 0 ms steps), but take the rate from the
            # recording's real duration, not the container fps, which re-encoders and generators
            # sometimes rewrite (e.g. 30 -> 25).
            fs_rec = (len(ppg) - 1) / t_gt[-1]
            idx = np.rint(times * fps).astype(int)
            keep = idx < len(ppg)
            times, rgbs, t_gt = idx[keep] / fs_rec, rgbs[keep], np.arange(len(ppg)) / fs_rec
        tu = np.arange(max(times[0], t_gt[0]), min(times[-1], t_gt[-1]), 1 / FS)
        rgb, gt = resample(times, rgbs, tu), bandpass(resample(t_gt, ppg, tu))
        line = []
        for m in methods:
            w = window_hrs(bandpass(METHODS[m](rgb)), gt, win_s, step_s=step_s)
            if not len(w):
                continue
            mae, rmse, r = metrics(w[:, 0], w[:, 2])
            rows.append([name, m, len(w), round(mae, 2), round(rmse, 2), round(r, 3), round(w[:, 1].mean(), 2)])
            pooled[m][0].append(w[:, 0]); pooled[m][1].append(w[:, 2])
            line.append(f"{m.upper()} {mae:5.2f}")
        print(f"{name:>12}: MAE (bpm)  " + "  |  ".join(line) + f"   [{time.perf_counter() - t0:.0f}s]")
    if not rows:
        raise SystemExit("Nothing evaluated.")
    n_vid = len({r_[0] for r_ in rows})
    overlap = f" every {step_s:g}s (overlapping, so windows are not independent)" if step_s and step_s < win_s else ""
    print(f"\nOVERALL: {n_vid} videos, {win_s:.0f}s windows{overlap}, heart rate from contact PPG as ground truth")
    print(f"{'method':>8} | {'MAE bpm':>8} | {'RMSE bpm':>8} | {'Pearson r':>9}")
    for m in methods:
        if pooled[m][0]:
            mae, rmse, r = metrics(np.concatenate(pooled[m][0]), np.concatenate(pooled[m][1]))
            rows.append(["OVERALL", m, sum(len(a) for a in pooled[m][0]), round(mae, 2), round(rmse, 2), round(r, 3), ""])
            print(f"{m.upper():>8} | {mae:8.2f} | {rmse:8.2f} | {r:9.3f}")
    with open("eval_results.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["subject", "method", "windows", "mae_bpm", "rmse_bpm", "pearson_r", "mean_snr_db"])
        cw.writerows(rows)
    print("Saved eval_results.csv")


# ---------------------------------------------------------------- self-check
def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    # 1) POS + heart rate on a jittery 25 fps trace with lighting drift
    t = np.arange(0, 30, 1 / 25.0) + rng.normal(0, 0.004, 750)
    hr_true = 72 + 6 * (t > 15)                                       # 72 bpm, then 78 bpm
    phase = 2 * np.pi * np.cumsum(hr_true / 60) / 25.0
    pulse = 0.004 * np.sin(phase)
    light = 1 + 0.05 * np.sin(2 * np.pi * 0.05 * t)
    rgb = np.stack([light * (0.70 + 0.5 * pulse), light * (0.50 + pulse), light * (0.40 + 0.6 * pulse)], 1)
    rgb = 255 * (rgb + rng.normal(0, 0.002, rgb.shape))
    _, bpm, _ = estimate(t[:375], rgb[:375])
    assert abs(bpm - 72) < 2, bpm
    for m in ("chrom", "green"):
        _, b, _ = estimate(t[:375], rgb[:375], m)
        assert abs(b - 72) < 2, (m, b)
    # 2) windowed evaluation against a "finger sensor" signal
    tu = np.arange(t[0], t[-1], 1 / FS)
    pred = bandpass(pos(resample(t, rgb, tu)))
    gt = bandpass(np.interp(tu, t, np.sin(phase)))
    w = window_hrs(pred, gt, 10)
    mae, rmse, _ = metrics(w[:, 0], w[:, 2])
    assert mae < 2 and rmse < 2.5, (mae, rmse)
    # 3) UBFC DATASET_2 ground-truth parser
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "ground_truth.txt").write_text(" ".join(map(str, np.sin(phase[:300]))) + "\n" +
                                                  " ".join(["72"] * 300) + "\n" + " ".join(map(str, t[:300])) + "\n")
        tg, pg = load_gt(d, 300, 25.0)
        assert len(tg) == len(pg) == 300 and tg[0] == 0
    print(f"selftest ok: POS/CHROM/GREEN all ~72 bpm (POS {bpm:.1f}) | windowed eval MAE {mae:.2f} bpm, RMSE {rmse:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="video file instead of the webcam (live view)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--save", help="record the demo window to this .mp4")
    ap.add_argument("--eval", help="UBFC-rPPG folder (or one vid.avi) to score against ground truth")
    ap.add_argument("--eval-win", type=float, default=10.0, help="evaluation window in seconds (default 10)")
    ap.add_argument("--eval-step", type=float, help="window step in s (default: no overlap); smoother, not more data")
    ap.add_argument("--method", choices=METHODS, default="pos", help="live view: pulse extraction method")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.eval:
        evaluate(a.eval, a.eval_win, step_s=a.eval_step)
    else:
        run(a.video or a.camera, a.save, a.method)
