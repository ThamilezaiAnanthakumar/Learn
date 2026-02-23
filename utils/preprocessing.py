from dataclasses import dataclass
from math import gcd
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, medfilt, resample_poly


@dataclass
class PPGPreprocessor:
    fs_target: float = 125.0
    lowcut: float = 0.5
    highcut: float = 8.0
    filter_order: int = 4
    remove_dc: bool = True
    detrend: bool = True
    detrend_order: int = 2
    clip_std: Optional[float] = 5.0
    median_kernel: Optional[int] = None
    window_size: float = 10.0
    overlap: float = 0.0
    normalize: bool = True

    def __call__(self, ppg: np.ndarray, fs: float, length: float) -> np.ndarray:
        signal = np.asarray(ppg, dtype=np.float64).ravel()
        signal = self._truncate_or_pad(signal, fs, length)
        signal = self._resample(signal, fs)
        signal = self._clip_outliers(signal)
        signal = self._median_filter(signal)
        signal = self._bandpass_filter(signal)
        signal = self._remove_dc(signal)
        signal = self._detrend(signal)
        windows = self._window(signal)
        windows = self._normalize(windows)
        return windows

    def _truncate_or_pad(self, signal: np.ndarray, fs: float, length: float) -> np.ndarray:
        n_target = int(fs * length)
        if len(signal) >= n_target:
            return signal[:n_target]
        return np.pad(signal, (0, n_target - len(signal)), mode="edge")

    def _resample(self, signal: np.ndarray, fs: float) -> np.ndarray:
        if fs == self.fs_target:
            return signal
        fs_int = int(round(fs))
        fs_target_int = int(round(self.fs_target))
        g = gcd(fs_target_int, fs_int)
        return resample_poly(signal, fs_target_int // g, fs_int // g)

    def _clip_outliers(self, signal: np.ndarray) -> np.ndarray:
        if self.clip_std is None:
            return signal
        mu, sigma = signal.mean(), signal.std()
        if sigma == 0:
            return signal
        return np.clip(signal, mu - self.clip_std * sigma, mu + self.clip_std * sigma)

    def _median_filter(self, signal: np.ndarray) -> np.ndarray:
        if self.median_kernel is None:
            return signal
        kernel = self.median_kernel if self.median_kernel % 2 == 1 else self.median_kernel + 1
        return medfilt(signal, kernel_size=kernel)

    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        nyq = self.fs_target / 2.0
        b, a = butter(self.filter_order, [self.lowcut / nyq, self.highcut / nyq], btype="band")
        return filtfilt(b, a, signal)

    def _remove_dc(self, signal: np.ndarray) -> np.ndarray:
        if not self.remove_dc:
            return signal
        return signal - np.mean(signal)

    def _detrend(self, signal: np.ndarray) -> np.ndarray:
        if not self.detrend:
            return signal
        t = np.arange(len(signal), dtype=np.float64)
        coeffs = np.polyfit(t, signal, self.detrend_order)
        trend = np.polyval(coeffs, t)
        return signal - trend

    def _window(self, signal: np.ndarray) -> np.ndarray:
        win_samples = int(self.fs_target * self.window_size)
        step_samples = int(self.fs_target * (self.window_size - self.overlap))
        if step_samples <= 0:
            raise ValueError("overlap must be smaller than window_size")
        n_windows = (len(signal) - win_samples) // step_samples + 1
        if n_windows <= 0:
            raise ValueError(
                f"signal length {len(signal)} too short for "
                f"window_size={self.window_size}s at fs={self.fs_target}Hz"
            )
        shape = (n_windows, win_samples)
        strides = (step_samples * signal.strides[0], signal.strides[0])
        return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides).copy()

    def _normalize(self, windows: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return windows
        mins = windows.min(axis=1, keepdims=True)
        maxs = windows.max(axis=1, keepdims=True)
        rng = maxs - mins
        rng[rng == 0] = 1.0
        return (windows - mins) / rng
