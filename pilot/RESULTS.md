# Pilot: does the pulse survive face generation? (4 Oct 2026)

**Setup.** UBFC-rPPG subject 1 (Bobbia et al., 2017), first 30 s (878 frames). LivePortrait (2024),
self-reenactment: the first frame is animated with the motion of the same video, so the face region of
every output frame is generated. Pulse extracted with POS; ground truth is the finger pulse sensor.
Scored in 10 s windows.

| Video | Encoding (verified in file) | Data rate | POS heart-rate error | SNR at true HR | Waveform r |
|---|---|---|---|---|---|
| Real | lossless (FFV1) | 60,000 kbit/s | 0.38 bpm | +5.5 dB | 0.85 |
| Real | H.264 crf 10 | 7,900 kbit/s | 0.00 bpm | -0.5 dB | 0.70 |
| Real | H.264 crf 18 | 830 kbit/s | 37.1 bpm | -7.9 dB | 0.46 |
| **Generated** | **lossless (qp 0)** | 2,040 kbit/s | **37.9 bpm** | **-10.6 dB** | **0.18** |
| Generated | H.264 crf 18 (LivePortrait default) | 140 kbit/s | 33.8 bpm | -13.0 dB | 0.20 |

Waveform r = best Pearson correlation with the finger-sensor PPG within a +-0.5 s shift.
Figure: `output/pulse_real_vs_generated.png` (one 8 s window).

**Findings**
1. The generated face carries no usable pulse, even saved losslessly: heart-rate error ~38 bpm,
   pulse power at the true heart rate ~16 dB below the real video, waveform correlation 0.18 vs 0.85.
2. Standard video compression alone (crf 18, a common default) also erases the pulse in this data.
   LivePortrait saves at crf 18 by default, so a pulse-survival study must save generated video
   losslessly, or it measures the encoder instead of the generator.
3. Likely mechanism: LivePortrait renders skin appearance from one still photo and only transfers
   motion, so the heartbeat's frame-to-frame colour change is never carried over.

**Limits.** One subject, one 30 s clip, one generator, one rPPG method. A first pilot, not a finding.
Next: more subjects, other generators (audio-driven, video diffusion), and generators conditioned on
the pulse (research plan, step 2).

---

# Update: 5 subjects (4 Oct 2026)

Same setup, UBFC-rPPG subjects 1, 3, 4, 5, 8 (30 s each), LivePortrait output saved losslessly
(qp 0, verified in every file). Script: `pilot/analyze_batch.py`; per-subject data: `output/batch_results.csv`;
figure: `output/batch_snr.png`. Non-overlapping 10 s windows (3 per subject), POS.

| Subject | HR error real | HR error generated | SNR real (dB) | SNR generated (dB) | r real | r generated |
|---|---|---|---|---|---|---|
| 1 | 0.4 | 37.9 | +5.5 | −10.6 | 0.85 | 0.18 |
| 3 | 2.6 | 42.0 | −0.5 | −11.9 | 0.68 | 0.28 |
| 4 | 0.2 | 56.3 | +0.8 | −10.2 | 0.79 | 0.24 |
| 5 | 0.4 | 31.1 | +5.7 | −11.4 | 0.82 | 0.17 |
| 8 | 0.5 | 27.8 | +4.7 | −10.2 | 0.86 | 0.22 |
| **mean ± sd** | **0.8 ± 1.0** | **39.0 ± 11.1** | **+3.2 ± 2.9** | **−10.9 ± 0.8** | **0.80 ± 0.07** | **0.22 ± 0.04** |

SNR at the true heart rate drops by 14.1 ± 2.8 dB, lower in 5/5 subjects
(paired t-test p = 0.0003; one-sided Wilcoxon p = 0.031, the smallest possible with n = 5).

**Still a pilot:** 5 subjects, 30 s each, one generator, self-reenactment only, classic rPPG (POS).
The t-test assumes roughly normal differences, which n = 5 cannot confirm.

---

# Update: face swapping and two simple baselines (4 Oct 2026, evening)

## ~~Face swap (first analysis)~~ INVALID: see correction below

*The numbers in this subsection were measured on wrongly decoded colours and are withdrawn.*

## Face swap: FaceFusion + HyperSwap (hyperswap_1a_256), run locally on a GTX 1060 (DirectML)
Each real 30 s clip (lossless, cropped 320x320 around the face) kept its own pixels and motion and got
another subject's face (1<-3, 3<-4, 4<-5, 5<-8, 8<-1). Output lossless (libx264rgb, qp 0, verified).
Scored against the ORIGINAL person's finger sensor. Script `pilot/analyze_swap.py`; data
`output/swap_results.csv`; figure `output/swap_snr.png` (real vs face swap vs LivePortrait).

| | Real | Face swap | LivePortrait |
|---|---|---|---|
| HR error (bpm) | 0.6 | 38.0 | 39.0 |
| Pulse SNR at true HR (dB) | +3.3 | −10.3 | −10.9 |
| Waveform r | 0.81 | 0.28 | 0.22 |

Pulse SNR 13.6 dB lower in 5/5 subjects (paired t-test p = 0.005). The swapped pulse matches neither the
original person (mean r 0.28) nor the face donor (mean r 0.21), so no identity leak: the pulse is lost,
not carried over. (A 2025 study found pulses in high-quality, person-specific deepfakes; one-shot
swappers like HyperSwap may differ.)

## Simple baselines (`pilot/simple_baselines.py`), 5 subjects, mean
| Condition | HR error | Pulse SNR | Waveform r |
|---|---|---|---|
| Real, fixed face box (control) | 0.5 bpm | +4.1 dB | 0.84 |
| A. Face scrambled (fixed pixel permutation; cf. pixel-redistribution anonymization) | 0.5 bpm | +2.4 dB | 0.81 |
| B. LivePortrait + real pulse multiplied back on (cf. PulseEdit), dithered | 4.2 bpm | +2.2 dB | 0.76 |

- A: non-generative scrambling keeps the pulse but destroys the face.
- B: the pulse can be restored post hoc, but it is the original person's pulse (identity-leak risk,
  cf. Sun, Li et al., IJCB 2024). A baseline that a pulse-preserving generator must beat.
- **The pulse in UBFC is ~0.2 grey levels, below one 8-bit step.** Without dithering, injecting it onto a
  clean frame mostly rounded it away. Real cameras keep it because sensor noise dithers it.
  Hypothesis (untested): generators output clean re-drawn pixels, so sub-level changes are quantized away.

---

# CORRECTION: face swap keeps the pulse, the original person's (4 Oct 2026, late evening)

**What went wrong.** FaceFusion's `libx264rgb` output stores the pixel data as G, B, R planes but labels the
stream `yuvj444p` (bt709). Every decoder (OpenCV and ffmpeg) then applies a YUV->RGB conversion and scrambles
the colours (green hair, magenta skin). The first face-swap analysis was run on those scrambled colours and
is invalid. Fix: decode the planes without conversion, reorder G,B,R -> BGR, save as lossless FFV1
(`eval_swap/swapped/<s>/vid.avi`; original kept as `x264rgb_original.mp4`). Verified: background pixel identical
to the real video (BGR 151, 161, 78). LivePortrait and real videos are standard yuv420p/yuv444p and unaffected.

**Corrected results** (5 subjects, POS, 10 s windows, scored against the original person's finger sensor):

| | Real | Face swap (HyperSwap) | Photo animation (LivePortrait) |
|---|---|---|---|
| HR error (bpm) | 0.6 | **0.4** | 39.0 |
| Pulse SNR at true HR (dB) | +3.3 | **+1.9** | −10.9 |
| Waveform r | 0.81 | **0.78** | 0.22 |

Face swap vs real SNR: −1.4 dB, not significant (paired t p = 0.17).

**Whose pulse?** The swapped video's pulse matches the ORIGINAL person, not the face donor:
waveform r 0.78 vs 0.33 (paired t p = 0.0004); HR error 0.4 vs 11.0 bpm (p = 0.004).
So the swapped face shows another person's identity but carries the original person's heartbeat, a
physiological signal shown to be a biometric (Sun, Li et al., IJCB 2024): an identity leak through the pulse.
Consistent with Seibold et al. (2025), "High-quality deepfakes have a heart!".

**What this changes**
- Generator design decides whether physiology survives: photo animation (appearance from one still photo)
  erases the pulse; face swapping (re-renders each real frame) keeps it.
- Face swapping is not a safe anonymizer for physiological data: the original pulse leaks.
- The earlier "quantization of clean generated pixels" explanation was proposed for the face-swap loss and is
  withdrawn; the LivePortrait mechanism (appearance from a single photo) stands.
- Research question sharpened: a realistic new face with a plausible pulse that does NOT identify the original
  person (step 3 of the plan).
