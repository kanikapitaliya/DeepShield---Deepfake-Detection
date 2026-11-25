"""
src/rppg.py
===========
CHROM-based rPPG physiological signal extraction — Model B feature pipeline.

Implements the exact pipeline from notebook 03_DeepShield_Model_B_rPPG_v2.ipynb:
  1. Sample video frames at ≤30 FPS
  2. Detect face with Haar cascade
  3. Extract central facial ROI (20–80% x, 20–75% y of bounding box)
  4. Apply skin-colour mask (HSV: S>20, S<220, V>40)
  5. Extract mean RGB signals
  6. Windowed CHROM chrominance estimation (3.2s windows, 50% overlap)
  7. Bandpass filter: 0.7–3.0 Hz (Butterworth order 3)
  8. Welch PSD for spectral features
  9. Extract 8 features in fixed order

Quality rejection: returns None + reason if the signal is insufficient.
Never fabricates features.

Feature order (must match training):
  ['pulse_rate_bpm_proxy', 'band_power', 'peak_power_ratio',
   'spectral_entropy', 'pulse_std', 'pulse_rms', 'pulse_skew', 'pulse_kurtosis']
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict
from scipy.signal import butter, sosfiltfilt, welch

logger = logging.getLogger(__name__)

RPPG_FEATURE_COLS = [
    'pulse_rate_bpm_proxy',
    'band_power',
    'peak_power_ratio',
    'spectral_entropy',
    'pulse_std',
    'pulse_rms',
    'pulse_skew',
    'pulse_kurtosis',
]

_haar_detector = None


def _get_haar():
    global _haar_detector
    if _haar_detector is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _haar_detector = cv2.CascadeClassifier(path)
        if _haar_detector.empty():
            _haar_detector = None
            logger.warning("Haar cascade not loaded; rPPG face track will fail.")
    return _haar_detector


def extract_rppg_features(
    video_path: str,
    max_fps: float = 30.0,
    min_seconds: float = 8.0,
) -> Tuple[Optional[Dict[str, float]], str]:
    """
    Extract 8 CHROM-rPPG features from a video.

    Args:
        video_path:   Path to the video file.
        max_fps:      Maximum sampling FPS (downsamples if native FPS > max_fps).
        min_seconds:  Minimum video duration in seconds required.

    Returns:
        (features_dict, status_string)
        features_dict: dict with 8 float features (or None if extraction fails)
        status_string: 'ok' on success, rejection reason on failure
    """
    features, status, _ = extract_rppg_with_signal(
        video_path, max_fps=max_fps, min_seconds=min_seconds
    )
    return features, status


def extract_rppg_with_signal(
    video_path: str,
    max_fps: float = 30.0,
    min_seconds: float = 8.0,
) -> Tuple[Optional[Dict[str, float]], str, Optional[Dict[str, object]]]:
    """
    Extract 8 CHROM-rPPG features along with the underlying pulse waveform and PSD.

    Returns:
        (features_dict, status_string, signal_data)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 'cannot_open', None

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not np.isfinite(fps) or fps < 8 or nframes < 2:
        cap.release()
        return None, 'invalid_fps_or_frames', None

    duration = nframes / fps
    if duration < min_seconds:
        cap.release()
        return None, 'video_too_short', None

    step = max(1, int(np.ceil(fps / max_fps)))
    sample_fps = fps / step

    detector = _get_haar()
    rgb_means = []
    face_hits = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            h, w = frame.shape[:2]
            if detector is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                if len(faces) > 0:
                    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                    # Central facial ROI (matches notebook)
                    x1, x2 = int(x + 0.20 * fw), int(x + 0.80 * fw)
                    y1, y2 = int(y + 0.20 * fh), int(y + 0.75 * fh)
                    roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if roi.size and roi.shape[0] >= 20 and roi.shape[1] >= 20:
                        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        mask = (
                            (hsv[:, :, 1] > 20) &
                            (hsv[:, :, 1] < 220) &
                            (hsv[:, :, 2] > 40)
                        )
                        if mask.sum() > 100:
                            bgr = roi[mask].astype(np.float64)
                            rgb_means.append([bgr[:, 2].mean(), bgr[:, 1].mean(), bgr[:, 0].mean()])
                            face_hits += 1
                        else:
                            rgb_means.append([np.nan] * 3)
                    else:
                        rgb_means.append([np.nan] * 3)
                else:
                    rgb_means.append([np.nan] * 3)
            else:
                rgb_means.append([np.nan] * 3)
        frame_idx += 1

    cap.release()

    arr = np.asarray(rgb_means, dtype=float)
    if len(arr) < int(min_seconds * sample_fps) or face_hits / max(1, len(arr)) < 0.80:
        return None, 'insufficient_face_track', None

    # Trim boundary NaN runs
    valid = np.isfinite(arr).all(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return None, 'no_valid_frames', None
    arr = arr[indices[0]:indices[-1] + 1]
    valid = np.isfinite(arr).all(axis=1)

    if len(arr) < int(min_seconds * sample_fps):
        return None, 'insufficient_continuous_duration', None

    # Check for long internal NaN gaps (>0.5 s)
    gaps = np.diff(np.flatnonzero(valid))
    if len(gaps) > 0 and gaps.max() - 1 > int(0.5 * sample_fps):
        return None, 'long_face_tracking_gap', None

    # Interpolate remaining NaN gaps
    rgb = _interpolate_nan(arr)

    if sample_fps <= 6:
        return None, 'invalid_sample_fps', None

    # Windowed CHROM (3.2 s, 50% overlap)
    try:
        pulse = _chrom_pulse(rgb, sample_fps)
    except Exception as e:
        logger.debug(f"CHROM failed: {e}")
        return None, 'chrom_failed', None

    if pulse is None:
        return None, 'chrom_failed', None

    if np.std(pulse) < 1e-6:
        return None, 'low_signal_variance', None

    # Welch PSD
    freqs, power = welch(
        pulse,
        fs=sample_fps,
        nperseg=min(len(pulse), max(64, int(sample_fps * 8)))
    )
    band = (freqs >= 0.7) & (freqs <= 3.0)
    if band.sum() < 3:
        return None, 'insufficient_spectral_resolution', None

    bf, bp = freqs[band], power[band]
    peak_i = int(np.argmax(bp))
    peak_hz = float(bf[peak_i])
    total_power = float(bp.sum() * (freqs[1] - freqs[0]) + 1e-12)
    peak_power = float(bp[peak_i])
    spectral_entropy = float(
        -(bp / (bp.sum() + 1e-12) * np.log2(bp / (bp.sum() + 1e-12) + 1e-12)).sum()
    )

    features = {
        'pulse_rate_bpm_proxy': peak_hz * 60.0,
        'band_power': total_power,
        'peak_power_ratio': peak_power / (float(bp.sum()) + 1e-12),
        'spectral_entropy': spectral_entropy,
        'pulse_std': float(np.std(pulse)),
        'pulse_rms': float(np.sqrt(np.mean(pulse ** 2))),
        'pulse_skew': float(_skew(pulse)),
        'pulse_kurtosis': float(_kurtosis(pulse)),
    }

    if not all(np.isfinite(v) for v in features.values()):
        return None, 'nonfinite_features', None

    signal_data = {
        'pulse': pulse,
        'sample_fps': float(sample_fps),
        'freqs': freqs,
        'power': power,
        'peak_hz': peak_hz,
    }
    return features, 'ok', signal_data


def _interpolate_nan(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values in a 2D array (column-wise)."""
    result = arr.copy()
    for col in range(arr.shape[1]):
        y = arr[:, col]
        nans = np.isnan(y)
        if nans.any() and (~nans).sum() >= 2:
            xp = np.where(~nans)[0]
            fp = y[~nans]
            result[:, col] = np.interp(np.arange(len(y)), xp, fp)
    return result


def _chrom_pulse(rgb: np.ndarray, sample_fps: float) -> Optional[np.ndarray]:
    """
    Windowed CHROM algorithm (de Haan & Jeanne, 2013).
    Returns reconstructed pulse signal or None on failure.
    """
    sos = butter(3, [0.7, 3.0], btype='bandpass', fs=sample_fps, output='sos')
    window = int(round(3.2 * sample_fps))
    if window > len(rgb):
        return None

    hop = max(1, window // 2)
    starts = list(range(0, len(rgb) - window + 1, hop))
    if not starts or starts[-1] != len(rgb) - window:
        starts.append(len(rgb) - window)

    pulse_sum = np.zeros(len(rgb))
    weights = np.zeros(len(rgb))
    taper = np.hanning(window)

    for start in starts:
        segment = rgb[start:start + window]
        R, G, B = (segment / np.maximum(segment.mean(axis=0), 1e-6)).T
        try:
            x = sosfiltfilt(sos, 3 * R - 2 * G)
            y = sosfiltfilt(sos, 1.5 * R + G - 1.5 * B)
        except ValueError:
            return None
        if np.std(y) < 1e-8:
            return None
        chrom = x - (np.std(x) / np.std(y)) * y
        pulse_sum[start:start + window] += taper * chrom
        weights[start:start + window] += taper

    keep = weights > 1e-6
    pulse = pulse_sum[keep] / weights[keep]
    pulse = pulse - np.mean(pulse)
    return pulse


def _skew(x: np.ndarray) -> float:
    """Pearson skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-12:
        return 0.0
    return float(((x - mu) ** 3).mean() / sigma ** 3)


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis."""
    n = len(x)
    if n < 4:
        return 0.0
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-12:
        return 0.0
    return float(((x - mu) ** 4).mean() / sigma ** 4) - 3.0
