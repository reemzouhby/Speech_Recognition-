import sounddevice as sd
from scipy.io.wavfile import write
import os


fs = 16000  # sample rate
seconds = 2
words = ["stop", "go", "start", "yes", "no"]
num_samples = 30
dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

for word in words:
    word_dir = os.path.join(dataset_dir, word)
    if not os.path.exists(word_dir):
        os.makedirs(word_dir)

    print(f"\nRecording word: {word}")
    for i in range(1, num_samples+1):
        input(f"Press Enter to record {word} sample {i}...")
        print("Recording...")
        audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        filename = os.path.join(word_dir, f"{word}_{i}.wav")
        write(filename, fs, audio)
        print(f"Saved: {filename}")
