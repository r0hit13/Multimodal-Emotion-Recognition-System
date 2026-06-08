# EDR Live — Multimodal Emotion Recognition
**Galgotias University | Md Adil (22SCSE1012699) | Rohit Kumar Ram (22SCSE1012599)**

---

## ▶ STEP-BY-STEP SETUP (Follow exactly)

### Step 1 — Check Python version
Open CMD or VS Code Terminal:
```
python --version
```
Must be **Python 3.10 or 3.11**. Download from https://www.python.org

---

### Step 2 — Open project in VS Code
- Extract the zip
- In VS Code → File → Open Folder → select `edr_live/`

---

### Step 3 — Create virtual environment
In VS Code Terminal (`Ctrl + backtick`):
```bash
python -m venv venv
```
Then activate it:
```bash
# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate
```
You should see `(venv)` at the start of the terminal line.

---

### Step 4 — Install all packages
```bash
pip install -r requirements.txt
```
⚠️ This takes 5–10 minutes. Wait for it to finish.

---

### Step 5 — RUN THE LIVE SYSTEM
```bash
python main.py
```
This opens your **webcam** and shows real-time emotion detection.

---

## Controls inside the live window

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot |
| `T` | Increase τ threshold by 0.05 |
| `R` | Decrease τ threshold by 0.05 |
| `P` | Pause / Resume |

---

## All Run Commands

```bash
# 1. Instant terminal test (no webcam, runs in 1 second)
python test_edr.py

# 2. Live webcam + mic (main demo)
python main.py

# 3. Different webcam (if camera 0 doesn't work)
python main.py --camera 1

# 4. Change EDR threshold
python main.py --tau 0.15

# 5. With trained speech model
python main.py --model models/ser_lstm.h5
```

---

## Train the Speech Model (optional but gives real speech emotion)

### Download RAVDESS dataset (free):
1. Go to: https://zenodo.org/record/1188976
2. Download the `Audio_Speech_Actors_01-24.zip` files
3. Extract all `.wav` files into the folder: `data/ravdess/`

### Train:
```bash
python train_ser.py --data data/ravdess/ --epochs 50
```

### Use trained model:
```bash
python main.py --model models/ser_lstm.h5
```

---

## Project Files

```
edr_live/
│
├── main.py             ← MAIN LIVE SYSTEM (webcam + mic + EDR)
├── edr_framework.py    ← EDR algorithm (the core innovation)
├── ser_module.py       ← Speech LSTM model (MFCC + LSTM)
├── train_ser.py        ← Train speech model on RAVDESS
├── test_edr.py         ← Terminal test (no webcam needed)
├── requirements.txt    ← All pip packages
│
├── models/             ← Trained model saved here after training
├── data/               ← Put RAVDESS dataset here
│   └── ravdess/
└── screenshots/        ← Auto-created when you press S
```

---

## How the System Works

```
┌─────────────────────────────────────────────────────────┐
│                  EDR Live System                        │
│                                                         │
│  Webcam → DeepFace CNN → Ef (emotion), Cf (confidence) │
│                                  ↓                      │
│                          EDR Framework                  │
│                                  ↑                      │
│  Mic → MFCC → LSTM    → Es (emotion), Cs (confidence)  │
└─────────────────────────────────────────────────────────┘

EDR Algorithm:
  1. Ef == Es?     → OUTPUT agreed emotion         (AGREEMENT)
  2. Δ = |Cf - Cs|
  3. Δ > τ?        → OUTPUT higher-confidence one  (DOMINANT)
  4. Δ ≤ τ?        → OUTPUT "Uncertain"            (UNCERTAIN)
```

---

## Results from Paper (Table I)

| Method              | Accuracy | F1-Score |
|---------------------|----------|----------|
| Unimodal (Face)     | 72.5%    | 0.72     |
| Unimodal (Speech)   | 64.0%    | 0.63     |
| Standard Fusion     | 79.2%    | 0.78     |
| **Proposed EDR**    | **84.6%**| **0.84** |

---

## Troubleshooting

**Camera not opening:**
```bash
python main.py --camera 1    # try camera index 1 or 2
```

**deepface slow on first run:**
DeepFace downloads its model weights on first use (~100 MB). Wait for it.

**No microphone / PortAudio error:**
The system still works with mock speech values. Only webcam is required.

**TensorFlow not installing:**
```bash
pip install tensorflow-cpu    # use CPU-only version
```
