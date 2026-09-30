import os
import glob
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

OUTPUT_DIR = "system_1_classifier/data/raw_audio"
SAMPLE_RATE = 22050
DURATION = 5.0  # seconds per generated sample

CLASSES = [
    "normal",
    "belt_pulley_squeal",
    "lean_bog_starvation",
    "four_stroking_rich",
    "misfire_backfire",
    "cylinder_drop",
    "rpm_ceiling_bog",
    "start_and_die"
]

# --- DSP FILTER UTILITIES ---

def lowpass(data, cutoff, sr=SAMPLE_RATE, order=4):
    nyquist = 0.5 * sr
    norm_cutoff = min(cutoff / nyquist, 0.99)
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def bandpass(data, lowcut, highcut, sr=SAMPLE_RATE, order=3):
    nyquist = 0.5 * sr
    low = max(lowcut / nyquist, 0.01)
    high = min(highcut / nyquist, 0.99)
    b, a = butter(order, [low, high], btype='band', analog=False)
    return lfilter(b, a, data)

def normalize(audio):
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        return np.float32(audio / peak * 0.90)
    return np.float32(audio)

# --- PROCEDURAL ACOUSTIC ENGINES (NO SEED REQUIRED) ---

def synth_normal(t, sr):
    """Steady combustion pulses with natural mechanical harmonics."""
    f0 = np.random.uniform(28.0, 34.0)  # Idle combustion frequency (~800-1000 RPM)
    signal = np.zeros_like(t)
    
    # Harmonic series (f0, 2f0, 3f0...)
    for h in range(1, 12):
        amp = (1.0 / (h ** 1.15)) * np.random.uniform(0.85, 1.15)
        signal += amp * np.sin(2 * np.pi * (h * f0) * t + np.random.uniform(0, 2 * np.pi))
        
    exhaust_rumble = lowpass(np.random.normal(0, 0.35, len(t)), cutoff=180, sr=sr)
    return normalize(signal + exhaust_rumble)

def synth_belt_squeal(t, sr):
    """Normal engine baseline with continuous high-frequency friction oscillation."""
    base = synth_normal(t, sr)
    squeal_f = np.random.uniform(3200.0, 4800.0)
    
    # Frequency flutter simulating belt slip across the pulley
    flutter = 40.0 * np.sin(2 * np.pi * np.random.uniform(8.0, 14.0) * t)
    squeal = np.sin(2 * np.pi * (squeal_f + flutter) * t)
    squeal_noise = bandpass(np.random.normal(0, 0.4, len(t)), squeal_f - 200, squeal_f + 200, sr=sr)
    
    return normalize(base * 0.6 + (squeal + squeal_noise) * 0.75)

def synth_lean_bog(t, sr):
    """Engine idles, attempts to rev, then collapses due to fuel starvation."""
    f0_idle = 30.0
    f_curve = np.zeros_like(t)
    amp_env = np.ones_like(t)
    
    bog_start = 1.5
    for i, ti in enumerate(t):
        if ti < bog_start:
            f_curve[i] = f0_idle
        elif ti < bog_start + 0.4:
            # Brief acceleration attempt
            f_curve[i] = f0_idle + 30.0 * (ti - bog_start) / 0.4
        else:
            # Sudden collapse to stall
            elapsed = ti - (bog_start + 0.4)
            decay = np.exp(-3.5 * elapsed)
            f_curve[i] = f0_idle * decay
            amp_env[i] = decay
            
    phase = 2 * np.pi * np.cumsum(f_curve) / sr
    engine = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    
    # Intake whoosh sound during throttle tip-in
    whoosh_raw = np.random.normal(0, 1.0, len(t))
    whoosh_mask = np.exp(-0.5 * ((t - (bog_start + 0.1)) / 0.15) ** 2)
    whoosh = bandpass(whoosh_raw, 300, 1200, sr=sr) * whoosh_mask * 1.5
    
    return normalize(engine * amp_env + whoosh)

def synth_four_stroking(t, sr):
    """Flooded 2-stroke firing only on alternate cycles (halved f0 + muddy rumble)."""
    f0 = np.random.uniform(26.0, 30.0)
    sub_f0 = f0 / 2.0  # Dropped cycle creates heavy sub-harmonic
    
    signal = 1.0 * np.sin(2 * np.pi * sub_f0 * t) + 0.7 * np.sin(2 * np.pi * f0 * t)
    for h in range(3, 8):
        signal += (0.4 / h) * np.sin(2 * np.pi * (h * sub_f0) * t)
        
    rich_chug = lowpass(np.random.normal(0, 0.6, len(t)), cutoff=250, sr=sr)
    return normalize(signal + rich_chug)

def synth_misfire_backfire(t, sr):
    """Steady engine interrupted by sudden, sharp shockwave transients."""
    base = synth_normal(t, sr)
    pop_times = [1.2, 2.7, 4.1]
    spikes = np.zeros_like(t)
    
    for pt in pop_times:
        if pt < DURATION:
            decay_env = np.exp(-np.maximum(0, t - pt) * 45.0) * (t >= pt)
            pop = np.random.normal(0, 1.0, len(t)) * decay_env * 4.0
            spikes += pop
            
    return normalize(base * 0.7 + spikes)

def synth_cylinder_drop(t, sr):
    """Multi-cylinder engine with one dead cylinder creating an asymmetric lope."""
    f0 = np.random.uniform(30.0, 36.0)
    cycle_time = 1.0 / f0
    signal = np.zeros_like(t)
    
    # Asymmetric 3-cylinder cycle: Pulse 1, Pulse 2, DEAD (weak exhaust puff only)
    for i, ti in enumerate(t):
        phase_in_cycle = (ti % cycle_time) / cycle_time
        if phase_in_cycle < 0.33:
            signal[i] = np.sin(2 * np.pi * phase_in_cycle * 3)
        elif phase_in_cycle < 0.66:
            signal[i] = np.sin(2 * np.pi * phase_in_cycle * 3)
        else:
            signal[i] = 0.12 * np.sin(2 * np.pi * phase_in_cycle * 3)  # Dead cylinder
            
    roughness = lowpass(np.random.normal(0, 0.3, len(t)), cutoff=220, sr=sr)
    return normalize(signal + roughness)

def synth_rpm_ceiling_bog(t, sr):
    """Engine revs up but clamps at ~3,000 RPM with erratic fuel flutter."""
    f_curve = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < 2.0:
            f_curve[i] = 30.0 + (65.0 - 30.0) * (ti / 2.0)
        else:
            # Clamped ceiling with 3.5Hz fuel flutter
            flutter = 3.0 * np.sin(2 * np.pi * 3.5 * ti)
            f_curve[i] = 65.0 + flutter
            
    phase = 2 * np.pi * np.cumsum(f_curve) / sr
    engine = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.3 * np.sin(3 * phase)
    rush = bandpass(np.random.normal(0, 0.3, len(t)), 400, 1800, sr=sr)
    return normalize(engine + rush)

def synth_start_and_die(t, sr):
    """Cranks, fires up, then immediately dies due to lack of sustained fuel pressure."""
    f_curve = np.zeros_like(t)
    amp_env = np.zeros_like(t)
    
    crank_end = 0.8
    fire_end = 1.8
    stall_end = 3.0
    
    for i, ti in enumerate(t):
        if ti < crank_end:
            # Starter motor pulses
            f_curve[i] = 16.0
            amp_env[i] = 0.4
        elif ti < fire_end:
            # Engine catches
            f_curve[i] = 45.0 + 15.0 * np.sin(2 * np.pi * (ti - crank_end))
            amp_env[i] = 0.95
        elif ti < stall_end:
            # Fuel cuts out, RPM drops to zero
            elapsed = ti - fire_end
            decay = np.exp(-3.0 * elapsed)
            f_curve[i] = 45.0 * decay
            amp_env[i] = decay
        else:
            # Silent
            f_curve[i] = 0.0
            amp_env[i] = 0.0
            
    phase = 2 * np.pi * np.cumsum(f_curve) / sr
    engine = np.sin(phase) + 0.4 * np.sin(2 * phase)
    return normalize(engine * amp_env)

# --- DISPATCH TABLE ---
GENERATORS = {
    "normal": synth_normal,
    "belt_pulley_squeal": synth_belt_squeal,
    "lean_bog_starvation": synth_lean_bog,
    "four_stroking_rich": synth_four_stroking,
    "misfire_backfire": synth_misfire_backfire,
    "cylinder_drop": synth_cylinder_drop,
    "rpm_ceiling_bog": synth_rpm_ceiling_bog,
    "start_and_die": synth_start_and_die,
}

# --- CLI MODES ---

def clean_synthetic_audio():
    """Deletes all generated .wav files while leaving the directory structure intact."""
    print(f"Cleaning all generated .wav files from {OUTPUT_DIR}...")
    deleted_count = 0
    for class_name in CLASSES:
        class_folder = os.path.join(OUTPUT_DIR, class_name)
        if os.path.exists(class_folder):
            files = glob.glob(os.path.join(class_folder, "*.wav"))
            for f in files:
                os.remove(f)
                deleted_count += 1
    print(f"Cleanup complete. Removed {deleted_count} files. Folders remain ready.")

def generate_procedural_dataset(samples_per_class=5):
    """Synthesizes realistic audio variations for every class from mathematical equations."""
    print(f"Generating procedural audio ({samples_per_class} samples per class)...")
    total_generated = 0
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    
    for class_name in CLASSES:
        class_folder = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_folder, exist_ok=True)
        gen_func = GENERATORS[class_name]
        
        for idx in range(samples_per_class):
            audio = gen_func(t, SAMPLE_RATE)
            out_file = os.path.join(class_folder, f"proc_{class_name}_{idx}.wav")
            sf.write(out_file, audio, SAMPLE_RATE)
            total_generated += 1
            
        print(f"Generated {samples_per_class} files for '{class_name}'.")
        
    print(f"Synthesis complete. Generated {total_generated} procedural .wav files ready for CNN training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piranaware Procedural Synthesizer & Cleanup Engine")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['synthesize', 'clean'],
        help="Use 'synthesize' to generate procedural audio, or 'clean' to wipe audio files."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of procedural audio samples to create per class (default: 5)."
    )
    
    args = parser.parse_args()
    
    if args.mode == "clean":
        clean_synthetic_audio()
    elif args.mode == "synthesize":
        generate_procedural_dataset(samples_per_class=args.count)