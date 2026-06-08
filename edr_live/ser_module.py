"""
ser_module.py  —  Speech Emotion Recognition (LSTM + MFCC)
────────────────────────────────────────────────────────────
Dataset used for training : RAVDESS
Features                  : 40 MFCC coefficients (librosa)
Model                     : LSTM → Dropout → Dense → Softmax
────────────────────────────────────────────────────────────
"""

import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# ── librosa ──────────────────────────────────────────────────────────
try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    print("[SER] librosa missing → pip install librosa")

# ── sounddevice ──────────────────────────────────────────────────────
try:
    import sounddevice as sd
    MIC_OK = True
except (ImportError, OSError):
    MIC_OK = False
    print("[SER] sounddevice / PortAudio missing → mic disabled")

# ── TensorFlow ───────────────────────────────────────────────────────
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                         Input, Reshape)
    from tensorflow.keras.utils import to_categorical
    TF_OK = True
except ImportError:
    TF_OK = False
    print("[SER] TensorFlow missing → pip install tensorflow")


EMOTIONS    = ["Angry", "Disgust", "Fear", "Happy",
               "Neutral", "Sad", "Surprise"]
N_MFCC      = 40
SAMPLE_RATE = 22050
DURATION    = 3        # seconds captured per mic sample


class SERModule:
    """
    Speech Emotion Recognition.

    Quick start (no RAVDESS, just see the system work):
        ser = SERModule()              # uses random mock
        emo, conf = ser.predict_mock() # instant fake prediction

    Full training on RAVDESS:
        ser = SERModule()
        ser.train("path/to/ravdess/")
        ser.save("models/ser_lstm.h5")

    Load trained model:
        ser = SERModule(model_path="models/ser_lstm.h5")
        emo, conf = ser.predict_file("audio.wav")
        emo, conf = ser.predict_mic()
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self._load(model_path)
        elif model_path:
            print(f"[SER] No model at '{model_path}'. Call .train() first.")

    # ── MFCC extraction ──────────────────────────────────────────────
    def _mfcc(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray | None:
        if not LIBROSA_OK:
            return None
        try:
            feat = librosa.feature.mfcc(y=audio.astype(np.float32),
                                         sr=sr, n_mfcc=N_MFCC)
            return np.mean(feat.T, axis=0).astype(np.float32)
        except Exception as e:
            print(f"[SER] MFCC error: {e}")
            return None

    # ── Build LSTM model ─────────────────────────────────────────────
    def build(self):
        if not TF_OK:
            return
        self.model = Sequential([
            Input(shape=(N_MFCC,)),
            Reshape((1, N_MFCC)),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.25),
            Dense(len(EMOTIONS), activation="softmax"),
        ])
        self.model.compile(optimizer="adam",
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])
        print("[SER] LSTM model built.")
        return self.model

    # ── Train on RAVDESS ─────────────────────────────────────────────
    def train(self, data_dir: str, epochs: int = 50, batch: int = 32):
        """
        Train on RAVDESS .wav files.
        RAVDESS emotion codes in filename (3rd field):
          01=neutral 02=calm→neutral 03=happy 04=sad
          05=angry   06=fearful     07=disgust 08=surprise
        """
        if not (LIBROSA_OK and TF_OK):
            print("[SER] Cannot train: librosa or TF missing.")
            return None

        rmap = {"01":"Neutral","02":"Neutral","03":"Happy","04":"Sad",
                "05":"Angry","06":"Fear","07":"Disgust","08":"Surprise"}

        X, y = [], []
        print(f"[SER] Scanning {data_dir} …")
        for root, _, files in os.walk(data_dir):
            for f in files:
                if not f.endswith(".wav"):
                    continue
                parts = f.replace(".wav","").split("-")
                if len(parts) < 3:
                    continue
                label = rmap.get(parts[2])
                if label not in EMOTIONS:
                    continue
                try:
                    audio, sr = librosa.load(
                        os.path.join(root, f),
                        sr=SAMPLE_RATE, duration=DURATION)
                    feat = self._mfcc(audio, sr)
                    if feat is not None:
                        X.append(feat)
                        y.append(EMOTIONS.index(label))
                except Exception:
                    pass

        if not X:
            print("[SER] No files loaded. Check data_dir.")
            return None

        print(f"[SER] {len(X)} samples loaded. Training …")
        X = np.array(X, dtype=np.float32)
        Y = to_categorical(np.array(y), num_classes=len(EMOTIONS))

        if self.model is None:
            self.build()

        history = self.model.fit(X, Y, epochs=epochs,
                                  batch_size=batch,
                                  validation_split=0.2,
                                  verbose=1)
        print(f"[SER] Done. Val acc: "
              f"{history.history['val_accuracy'][-1]*100:.1f}%")
        return history

    # ── Inference ────────────────────────────────────────────────────
    def _run(self, feat: np.ndarray) -> tuple[str, float]:
        probs = self.model.predict(feat.reshape(1, -1), verbose=0)[0]
        idx   = int(np.argmax(probs))
        return EMOTIONS[idx], round(float(probs[idx]), 4)

    def predict_file(self, path: str) -> tuple[str, float]:
        """Predict from a .wav file."""
        if not LIBROSA_OK or self.model is None:
            return self.predict_mock()
        try:
            audio, sr = librosa.load(path, sr=SAMPLE_RATE,
                                      duration=DURATION)
            feat = self._mfcc(audio, sr)
            if feat is None:
                return "Neutral", 0.10
            return self._run(feat)
        except Exception as e:
            print(f"[SER] File predict error: {e}")
            return "Neutral", 0.10

    def predict_mic(self, duration: int = DURATION) -> tuple[str, float]:
        """Record from microphone and predict."""
        if not MIC_OK:
            return self.predict_mock()
        try:
            print(f"[SER] 🎙 Recording {duration}s …")
            rec = sd.rec(int(duration * SAMPLE_RATE),
                         samplerate=SAMPLE_RATE,
                         channels=1, dtype="float32")
            sd.wait()
            audio = rec.flatten()
            if not LIBROSA_OK or self.model is None:
                return self.predict_mock()
            feat = self._mfcc(audio)
            if feat is None:
                return "Neutral", 0.10
            return self._run(feat)
        except Exception as e:
            print(f"[SER] Mic error: {e}")
            return self.predict_mock()

    def predict_array(self, audio: np.ndarray) -> tuple[str, float]:
        """Predict from numpy audio array."""
        if not LIBROSA_OK or self.model is None:
            return self.predict_mock()
        feat = self._mfcc(audio)
        if feat is None:
            return "Neutral", 0.10
        return self._run(feat)

    @staticmethod
    def predict_mock() -> tuple[str, float]:
        """Returns a random prediction (demo without model)."""
        import random
        return (random.choice(EMOTIONS),
                round(random.uniform(0.42, 0.91), 3))

    # ── Save / Load ──────────────────────────────────────────────────
    def save(self, path: str = "models/ser_lstm.h5"):
        if self.model is None:
            print("[SER] Nothing to save.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"[SER] Model saved → {path}")

    def _load(self, path: str):
        if not TF_OK:
            return
        try:
            self.model = load_model(path)
            print(f"[SER] Model loaded ← {path}")
        except Exception as e:
            print(f"[SER] Load failed: {e}")
