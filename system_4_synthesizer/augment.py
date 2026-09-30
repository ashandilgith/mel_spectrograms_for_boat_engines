import numpy as np
import librosa
from scipy.signal import butter, lfilter

def add_gaussian_noise(y, noise_factor=0.015):
    """Simulates ambient water, wind, and pre-amp noise floor."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def shift_pitch(y, sr, n_steps):
    """Simulates variance in engine displacement or idle speed setting."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def stretch_time(y, rate):
    """Simulates varying rates of acceleration or deceleration bogs."""
    return librosa.effects.time_stretch(y=y, rate=rate)

def apply_lowpass_filter(y, sr, cutoff=3000):
    """Simulates a smartphone mic recording with the engine cowling ON."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, y)