# 🎤 Speech Command Recognition

A machine learning project that recognizes spoken voice commands using deep learning. This application records audio input, converts it to mel-spectrograms, and uses a convolutional neural network to classify spoken words.

## 📋 Features

- **Audio Recording**: Real-time audio recording with adjustable duration
- **Data Augmentation**: Multiple augmentation techniques to enhance training data
  - Noise addition
  - Pitch shifting
  - Time stretching/speed change
  - Original recordings
- **Spectrogram Generation**: Converts audio to mel-spectrograms for CNN input
- **Deep Learning Model**: CNN-based classifier trained on augmented spectrograms
- **Web Interface**: User-friendly Streamlit application for real-time predictions
- **Confidence Threshold**: Smart prediction with confidence scoring
- **Multi-word Recognition**: Trained on 5 command words: stop, go, start, yes, no

## 🛠️ Project Structure

```
├── record_audio.py           # Data collection script
├── Augmentation.py        # Audio augmentation pipeline
├── speech_model1.h5          # Trained Keras model
├── streamlit.py                    # Streamlit web application
├── dataset/                  # Original audio recordings
├── augmented_dataset/        # Augmented audio files
└── spectrograms/             # Generated mel-spectrograms
```

## 📦 Installation

### Requirements
- Python 3.7+
- pip package manager

### Dependencies

Install required packages:

```bash
pip install streamlit sounddevice scipy librosa numpy tensorflow scikit-learn matplotlib
```

### Individual Package Details

- **streamlit**: Web app framework
- **sounddevice**: Audio recording
- **scipy**: WAV file handling
- **librosa**: Audio feature extraction
- **numpy**: Numerical computing
- **tensorflow**: Deep learning (includes Keras)
- **scikit-learn**: Machine learning utilities
- **matplotlib**: Visualization

## 🚀 Usage

### 1. Recording Training Data

Record audio samples for each command word:

```bash
python record_audio.py
```

This script will:
- Create a `dataset/` directory
- Prompt you to record 30 samples per word
- Save WAV files organized by word

**Process**:
- Press Enter to start each recording
- Speak the command word clearly
- Recordings are 2 seconds each at 16kHz sample rate

### 2. Augmenting the Dataset

Expand your training data with augmentation:

```bash
python Augmentation.py
```

Generates 4 versions of each audio sample:
- Original audio
- Audio with added noise
- Pitch-shifted audio
- Speed-changed audio

**Output**: `augmented_dataset/` with ~120 samples per word (30 × 4)

### 3. Generating Spectrograms

Convert augmented audio to mel-spectrograms:

```bash
python generate_spectrograms.py
```

**Output**: `spectrograms/` directory with PNG images (128×128 pixels)

These images serve as input for the CNN model.

### 4. Training the Model

Train the CNN on generated spectrograms (custom training script required):

```bash
# Use the spectrogram images to train a CNN model
# Model should output 5 classes (stop, go, start, yes, no)
# Save as: speech_model1.h5
```

### 5. Running the Web App

Launch the interactive Streamlit application:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

**Features**:
- Record 2-second audio clips
- Real-time spectrogram visualization
- Instant predictions with confidence scores
- Reset button to clear results

## 📊 How It Works

### Pipeline Overview

```
Audio Recording → Signal Processing → Mel-Spectrogram → CNN → Prediction
```

### Step 1: Audio Recording
- Sample rate: 16,000 Hz
- Duration: 2 seconds
- Channels: Mono (1)
- Format: WAV

### Step 2: Audio Augmentation
Creates variations to improve model generalization:
- **Noise**: Adds random Gaussian noise (factor: 0.005)
- **Pitch Shift**: Changes pitch by ±2 semitones
- **Time Stretch**: Changes speed/tempo (rate: 1.2x)
- **Silence Trimming**: Removes leading/trailing silence

### Step 3: Mel-Spectrogram
Converts audio to visual representation:
- Type: Mel-frequency spectrogram
- Mel bands: 128
- Format: 128×128 pixel grayscale images
- Normalization: Power-to-dB scale

### Step 4: CNN Classification
- Input: 128×128 grayscale images
- Architecture: Convolutional Neural Network
- Output: 5 classes with confidence scores
- Confidence Threshold: 0.6 (default)

## 📈 Model Architecture

Expected CNN structure:

```
Input (128, 128, 1)
    ↓
Conv2D (32 filters, 3×3)
    ↓
MaxPool (2×2)
    ↓
Conv2D (64 filters, 3×3)
    ↓
MaxPool (2×2)
    ↓
Flatten
    ↓
Dense (128 units)
    ↓
Dropout (0.5)
    ↓
Dense (5 units, softmax)
    ↓
Output: [stop, go, start, yes, no]
```

## 🎯 Performance Optimization

### Data Collection Tips
- Record in a quiet environment
- Speak clearly at consistent volume
- Vary speaking speed and tone
- Ensure good microphone quality

### Model Tuning
- **Confidence Threshold**: Adjust `threshold = 0.6` in `app.py`
  - Lower for more lenient predictions
  - Higher for stricter predictions
- **Augmentation**: Adjust augmentation factors for more/less variation
- **Architecture**: Modify CNN layers for accuracy vs. speed tradeoff

## 🔧 Technical Details

### Audio Processing Parameters
```python
fs = 16000              # Sample rate (Hz)
seconds = 2             # Recording duration
n_mels = 128           # Mel frequency bands
img_size = (128, 128)  # CNN input size
```

### Augmentation Factors
```python
noise_factor = 0.005   # Noise amplitude
shift_max = 0.2        # Time shift percentage
n_steps = 2            # Pitch shift semitones
speed_rate = 1.2       # Time stretch factor
```

## 📝 Code Snippets

### Recording Audio
```python
import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000
seconds = 2
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
write("output.wav", fs, audio)
```

### Generating Spectrograms
```python
import librosa
import numpy as np

y, sr = librosa.load("audio.wav", sr=16000)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)
```

### Making Predictions
```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("speech_model1.h5")
prediction = model.predict(image_array)
confidence = np.max(prediction)
class_label = np.argmax(prediction)
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No audio device found" | Check microphone connection; ensure sounddevice can detect devices |
| "Model not found" | Verify `speech_model1.h5` exists in current directory |
| "Low accuracy" | Collect more training samples; increase augmentation; check audio quality |
| "Slow predictions" | Reduce model size; use quantization; check system resources |
| "Confidence always low" | Lower threshold; retrain with more varied data |

## 📚 Libraries Used

| Library | Purpose |
|---------|---------|
| **Streamlit** | Web interface framework |
| **TensorFlow/Keras** | Deep learning model |
| **Librosa** | Audio feature extraction |
| **SoundDevice** | Real-time audio recording |
| **Scikit-learn** | Machine learning utilities |
| **Matplotlib** | Visualization |
| **NumPy** | Numerical operations |
| **SciPy** | Audio file I/O |

## 🎓 Learning Resources

- [Librosa Documentation](https://librosa.org/)
- [TensorFlow Audio Processing](https://www.tensorflow.org/tutorials/audio)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Mel-Spectrograms Explained](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

## 🔮 Future Enhancements

- [ ] Add multi-language support
- [ ] Implement real-time continuous recognition
- [ ] Add voice activity detection (VAD)
- [ ] Support for custom words/commands
- [ ] Model quantization for mobile deployment
- [ ] WebRTC for browser-based recording
- [ ] Confidence calibration
- [ ] User accuracy metrics/analytics


## 👤 Author

-Reem AL-Zouhby 

-Sourour Hammoud 
## Some screnshoots of our App 
 ![Image](app.png)

## 🙏 Acknowledgments

- TensorFlow/Keras for the deep learning framework
- Librosa for audio processing tools
- Streamlit for the web interface framework
- Inspired by keyword spotting research and TinyML projects

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review code comments for specific details
3. Verify all dependencies are installed correctly
4. Ensure microphone and model files are accessible

---

**Note**: Ensure microphone permissions are granted to the application. Vary your voice, speak clearly, and test predictions in similar acoustic environments to where the model was trained.
