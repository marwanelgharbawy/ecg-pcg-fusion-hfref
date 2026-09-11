from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class BeatDetection:
    peaks: np.ndarray
    polarity: int
    prominence_threshold: float
    rr_seconds: np.ndarray
    plausible_rr: np.ndarray
    status: str

    @property
    def plausible_rr_fraction(self) -> float:
        if not len(self.rr_seconds):
            return float("nan")
        return float(self.plausible_rr.mean())


def _finite_vector(values, name="signal"):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
    return values


def robust_scale(values):
    """Median-absolute-deviation scale, with standard deviation as fallback."""
    values = _finite_vector(values)
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    if scale <= np.finfo(float).eps:
        scale = values.std()
    return float(scale)


def detect_ecg_beats(
    values,
    fs,
    *,
    min_hr_bpm=30,
    max_hr_bpm=200,
    prominence_scale=1.25,
    minimum_prominence=1.5,
    minimum_beats=3,
    minimum_plausible_fraction=0.75,
):
    """Find candidate dominant ECG peaks while considering both polarities.

    The detector uses SciPy peak prominence on the centered signal and its
    inverse. It selects the polarity with the most plausible inter-peak
    intervals, breaking ties by median prominence. Peaks are candidates for
    alignment, not clinical R-wave or QRS annotations.
    """
    values = _finite_vector(values, "ECG")
    if fs <= 0 or not 0 < min_hr_bpm < max_hr_bpm:
        raise ValueError("fs must be positive and heart-rate bounds must be ordered.")
    if prominence_scale < 0 or minimum_prominence < 0 or minimum_beats < 2:
        raise ValueError("Prominence settings must be nonnegative and minimum_beats >= 2.")

    centered = values - np.median(values)
    threshold = max(float(minimum_prominence), prominence_scale * robust_scale(centered))
    minimum_distance = max(1, int(np.floor(fs * 60.0 / max_hr_bpm)))
    rr_min = 60.0 / max_hr_bpm
    rr_max = 60.0 / min_hr_bpm
    candidates = []

    for polarity in (1, -1):
        peaks, properties = signal.find_peaks(
            polarity * centered,
            distance=minimum_distance,
            prominence=threshold,
        )
        rr = np.diff(peaks) / fs
        plausible = (rr >= rr_min) & (rr <= rr_max)
        plausible_fraction = float(plausible.mean()) if len(rr) else 0.0
        median_prominence = (
            float(np.median(properties["prominences"])) if len(peaks) else 0.0
        )
        median_width = (
            float(np.median(signal.peak_widths(polarity * centered, peaks)[0]))
            if len(peaks)
            else float("inf")
        )
        sharpness = median_prominence / max(median_width, np.finfo(float).eps)
        enough = int(len(peaks) >= minimum_beats)
        candidates.append(
            (
                enough,
                plausible_fraction,
                sharpness,
                median_prominence,
                polarity,
                peaks,
                rr,
                plausible,
            )
        )

    _, plausible_fraction, _, _, polarity, peaks, rr, plausible = max(
        candidates, key=lambda item: item[:4]
    )
    if len(peaks) < minimum_beats:
        status = "insufficient_peaks"
    elif plausible_fraction < minimum_plausible_fraction:
        status = "implausible_intervals"
    else:
        status = "ok"

    return BeatDetection(
        peaks=np.asarray(peaks, dtype=int),
        polarity=int(polarity),
        prominence_threshold=threshold,
        rr_seconds=np.asarray(rr, dtype=float),
        plausible_rr=np.asarray(plausible, dtype=bool),
        status=status,
    )


def extract_aligned_beats(values, peaks, fs, *, pre_seconds=0.20, post_seconds=0.28):
    """Extract fixed windows aligned to candidate peaks; omit boundary windows."""
    values = _finite_vector(values, "ECG")
    peaks = np.asarray(peaks, dtype=int)
    if peaks.ndim != 1:
        raise ValueError("peaks must be one-dimensional.")
    if fs <= 0 or pre_seconds <= 0 or post_seconds <= 0:
        raise ValueError("fs and beat-window durations must be positive.")
    pre = int(round(pre_seconds * fs))
    post = int(round(post_seconds * fs))
    if pre < 1 or post < 1:
        raise ValueError("Beat window is shorter than one sample.")
    usable_peaks = peaks[(peaks - pre >= 0) & (peaks + post < len(values))]
    if not len(usable_peaks):
        return np.empty((0, pre + post + 1), dtype=float), usable_peaks
    beats = np.stack([values[peak - pre : peak + post + 1] for peak in usable_peaks])
    return beats, usable_peaks


def beat_morphology_features(beats, *, peak_index):
    """Summarize relative morphology of already standardized, aligned beats."""
    beats = np.asarray(beats, dtype=float)
    if beats.ndim != 2:
        raise ValueError("beats must have shape (beats, samples).")
    if not len(beats):
        return {
            "beat_to_median_rmse_z": float("nan"),
            "median_beat_peak_to_peak_z": float("nan"),
            "median_beat_post_pre_rms_ratio": float("nan"),
        }
    if not 0 <= peak_index < beats.shape[1]:
        raise ValueError("peak_index lies outside the beat window.")

    median_beat = np.median(beats, axis=0)
    beat_rmse = np.sqrt(np.mean((beats - median_beat) ** 2, axis=1))
    pre_rms = np.sqrt(np.mean(median_beat[:peak_index] ** 2)) if peak_index else np.nan
    post_rms = np.sqrt(np.mean(median_beat[peak_index + 1 :] ** 2))
    ratio = post_rms / pre_rms if np.isfinite(pre_rms) and pre_rms > 1e-12 else np.nan
    return {
        "beat_to_median_rmse_z": float(np.median(beat_rmse)),
        "median_beat_peak_to_peak_z": float(np.ptp(median_beat)),
        "median_beat_post_pre_rms_ratio": float(ratio),
    }


def _longest_constant_run(values):
    if len(values) < 2:
        return len(values)
    changes = np.flatnonzero(np.diff(values) != 0) + 1
    edges = np.concatenate(([0], changes, [len(values)]))
    return int(np.diff(edges).max())


def raw_ecg_quality_features(values, fs):
    """Return acquisition-quality proxies in raw digital WAV units.

    Baseline and high-frequency ratios are component RMS divided by centered
    signal RMS. Full-scale fraction assumes normalized WAV decoding. Constant
    runs are reported directly rather than automatically called dropouts.
    """
    values = _finite_vector(values, "raw ECG")
    if fs <= 80:
        raise ValueError("ECG sample rate must exceed twice the 40 Hz analysis cutoff.")
    centered = values - np.median(values)
    total_rms = float(np.sqrt(np.mean(centered**2)))
    if total_rms <= np.finfo(float).eps:
        return {
            "baseline_drift_rms_ratio": float("nan"),
            "high_frequency_rms_ratio": float("nan"),
            "full_scale_sample_fraction": float(np.mean(np.abs(values) >= 0.999)),
            "longest_constant_run_s": len(values) / fs,
            "quality_status": "constant_signal",
        }

    baseline = signal.sosfiltfilt(
        signal.butter(3, 0.5, btype="lowpass", fs=fs, output="sos"), centered
    )
    high_frequency = signal.sosfiltfilt(
        signal.butter(3, 40.0, btype="highpass", fs=fs, output="sos"), centered
    )
    full_scale_fraction = float(np.mean(np.abs(values) >= 0.999))
    longest_run_s = _longest_constant_run(values) / fs
    flags = []
    if full_scale_fraction > 0:
        flags.append("full_scale_samples")
    if longest_run_s >= 0.5:
        flags.append("long_constant_run")
    return {
        "baseline_drift_rms_ratio": float(np.sqrt(np.mean(baseline**2)) / total_rms),
        "high_frequency_rms_ratio": float(
            np.sqrt(np.mean(high_frequency**2)) / total_rms
        ),
        "full_scale_sample_fraction": full_scale_fraction,
        "longest_constant_run_s": float(longest_run_s),
        "quality_status": ";".join(flags) if flags else "no_strict_flag",
    }


def summarize_processed_ecg(
    values,
    fs,
    *,
    pre_seconds=0.20,
    post_seconds=0.28,
    detector_kwargs=None,
):
    """Summarize candidate beats and relative morphology in a processed ECG."""
    detection = detect_ecg_beats(values, fs, **(detector_kwargs or {}))
    beats, usable_peaks = extract_aligned_beats(
        values,
        detection.peaks,
        fs,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
    )
    plausible_rr = detection.rr_seconds[detection.plausible_rr]
    if detection.status == "ok" and len(plausible_rr):
        heart_rate = 60.0 / np.median(plausible_rr)
        rr_cv = 100.0 * np.std(plausible_rr, ddof=1) / np.mean(plausible_rr) if len(plausible_rr) > 1 else np.nan
    else:
        heart_rate = np.nan
        rr_cv = np.nan
    if detection.status == "ok":
        morphology = beat_morphology_features(
            beats, peak_index=int(round(pre_seconds * fs))
        )
    else:
        morphology = beat_morphology_features(
            np.empty((0, beats.shape[1])), peak_index=int(round(pre_seconds * fs))
        )
    return {
        "detected_beats": int(len(detection.peaks)),
        "usable_beats": int(len(usable_peaks)),
        "candidate_peak_polarity": detection.polarity,
        "prominence_threshold_z": detection.prominence_threshold,
        "plausible_rr_fraction": detection.plausible_rr_fraction,
        "estimated_heart_rate_bpm": float(heart_rate),
        "rr_cv_percent_short_recording": float(rr_cv),
        "detection_status": detection.status,
        **morphology,
    }


def summarize_ecg(
    values,
    raw_values,
    fs,
    *,
    pre_seconds=0.20,
    post_seconds=0.28,
    detector_kwargs=None,
):
    """Combine candidate-beat, timing, morphology, and raw-quality summaries."""
    processed = summarize_processed_ecg(
        values,
        fs,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        detector_kwargs=detector_kwargs,
    )
    quality = raw_ecg_quality_features(raw_values, fs)
    return {**processed, **quality}


def pcg_envelope(values, fs, *, smoothing_hz=20.0):
    """Smoothed Hilbert magnitude for descriptive same-site PCG comparison."""
    values = _finite_vector(values, "PCG")
    if fs <= 0 or not 0 < smoothing_hz < fs / 2:
        raise ValueError("smoothing_hz must lie between zero and Nyquist.")
    magnitude = np.abs(signal.hilbert(values - np.median(values)))
    sos = signal.butter(3, smoothing_hz, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, magnitude)
