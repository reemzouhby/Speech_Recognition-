import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

# ---------------- LOAD MODEL ----------------
model = load_model("speech_model1.h5")
labels = ["stop", "go", "start", "yes", "no"]
le = LabelEncoder()
le.fit(labels)

# ---------------- APP LAYOUT ----------------
st.set_page_config(page_title="🎤 Voice Recognition Demo", layout="centered")
st.title("🎤 Speech Recognition Web App")
st.write("Say one of your words: **stop**, **go**, **start**, **yes**, **no**")
st.write("The model will say 'I don't know' if it is not confident enough.")

# Session state to clear previous outputs
if "recorded" not in st.session_state:
    st.session_state.recorded = False

threshold = 0.6  # minimum confidence to accept prediction

# ---------------- RECORD BUTTON ----------------
if st.button("Record"):

    # Clear previous output
    if st.session_state.recorded:
        st.empty()
    st.session_state.recorded = True

    fs = 16000
    seconds = 2

    st.info("Recording...")
    audio = sd.rec(int(seconds*fs), samplerate=fs, channels=1)
    sd.wait()
    write("temp.wav", fs, audio)
    st.success("Recording done!")

    # Load and trim silence
    y, sr = librosa.load("temp.wav", sr=fs)
    y, _ = librosa.effects.trim(y)

    # Convert to Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Display spectrogram
    st.subheader("Spectrogram")
    fig, ax = plt.subplots(figsize=(4,4))
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax, cmap='magma')
    ax.axis('off')
    st.pyplot(fig)

    # Prepare image for CNN
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = load_img(buf, target_size=(128,128), color_mode="grayscale")
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    max_prob = np.max(prediction)
    pred_index = np.argmax(prediction)

    if max_prob < threshold:
        pred_label = "I don't know"
    else:
        pred_label = le.inverse_transform([pred_index])[0]

    # Display prediction and confidence
    st.success(f"Predicted: **{pred_label}**")
    st.progress(int(max_prob*100))
    st.write(f"Confidence: {max_prob*100:.1f}%")
