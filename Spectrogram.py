import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

augmented_dir = "augmented_dataset"
spectrogram_dir = "spectrograms"

if not os.path.exists(spectrogram_dir):
    os.makedirs(spectrogram_dir)

for word in os.listdir(augmented_dir):
    word_path = os.path.join(augmented_dir, word)
    save_word_path = os.path.join(spectrogram_dir, word)
    if not os.path.exists(save_word_path):
        os.makedirs(save_word_path)

    for file in os.listdir(word_path):
        file_path = os.path.join(word_path, file)
        y, sr = librosa.load(file_path, sr=16000)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(2, 2))  # Small image for CNN
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
        plt.axis('off')  # Remove axes
        save_path = os.path.join(save_word_path, file.replace(".wav", ".png"))
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
