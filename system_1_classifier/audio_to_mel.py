import os
import glob
import numpy as np
import librosa

SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION_CHUNK = 1.0

CLASSES = [
    "normal", "belt_pulley_squeal", "lean_bog_starvation",
    "four_stroking_rich", "misfire_backfire", "cylinder_drop",
    "rpm_ceiling_bog", "start_and_die"
]

def wav_to_mel_chunks(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")

    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    chunk_samples = int(SAMPLE_RATE * DURATION_CHUNK)
    total_chunks = len(y) // chunk_samples
    spectrogram_chunks = []
    
    for i in range(total_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        y_slice = y[start:end]
        
        mel = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        
        if mel_norm.shape[1] > 44:
            mel_norm = mel_norm[:, :44]
        elif mel_norm.shape[1] < 44:
            mel_norm = np.pad(mel_norm, ((0, 0), (0, 44 - mel_norm.shape[1])), 'constant')
            
        spectrogram_chunks.append(mel_norm)
        
    return np.array(spectrogram_chunks)

def process_dataset(raw_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for class_name in CLASSES:
        input_path = os.path.join(raw_dir, class_name)
        output_path = os.path.join(output_dir, class_name)
        os.makedirs(output_path, exist_ok=True)
        
        for idx, wav_path in enumerate(glob.glob(os.path.join(input_path, "*.wav"))):
            chunks = wav_to_mel_chunks(wav_path)
            for c_idx, chunk in enumerate(chunks):
                np.save(os.path.join(output_path, f"{class_name}_{idx}_c{c_idx}.npy"), chunk)

if __name__ == "__main__":
    process_dataset("system_1_classifier/data/raw_audio", "system_1_classifier/data/spectrograms")