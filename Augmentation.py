import librosa
import numpy as np
import os
from scipy.io.wavfile import write


# ---------- AUGMENTATION FUNCTIONS ----------
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def time_shift(y, shift_max=0.2):
    shift = np.random.randint(int(len(y) * shift_max))
    return np.roll(y, shift)


def change_pitch(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def change_speed(y, rate=1.2):
    return librosa.effects.time_stretch(y=y, rate=rate)


# ---------- PATHS AND PARAMETERS ----------
dataset_dir = "dataset"  # Original recordings
augmented_dir = "augmented_dataset"  # Augmented files go here
sr = 16000  # Sample rate

if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# ---------- PROCESS DATASET ----------
for word in os.listdir(dataset_dir):
    word_path = os.path.join(dataset_dir, word)
    save_word_path = os.path.join(augmented_dir, word)
    if not os.path.exists(save_word_path):
        os.makedirs(save_word_path)

    for file in os.listdir(word_path):
        file_path = os.path.join(word_path, file)
        y, sr = librosa.load(file_path, sr=sr)

        # 1. Original
        write(os.path.join(save_word_path, file), sr, (y * 32767).astype(np.int16))

        # 2. Add noise
        y_noise = add_noise(y)
        write(os.path.join(save_word_path, file.replace(".wav", "_noise.wav")), sr, (y_noise * 32767).astype(np.int16))

        # 3. Pitch shift
        y_pitch = change_pitch(y, sr, n_steps=2)
        write(os.path.join(save_word_path, file.replace(".wav", "_pitch.wav")), sr, (y_pitch * 32767).astype(np.int16))

        # 4. Time stretch / speed change
        try:
            y_speed = change_speed(y, rate=1.2)
            # Match original length
            min_len = min(len(y_speed), len(y))
            y_speed = y_speed[:min_len]
            write(os.path.join(save_word_path, file.replace(".wav", "_speed.wav")), sr,
                  (y_speed * 32767).astype(np.int16))
        except Exception as e:
            print(f"Skipping speed change for {file}: {e}")

print("Augmentation finished!")
