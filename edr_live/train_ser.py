"""
train_ser.py  —  Train LSTM Speech Emotion Model on RAVDESS
═══════════════════════════════════════════════════════════
USAGE:
  python train_ser.py
  python train_ser.py --data data/ravdess/ --epochs 60

After training:
  python main.py --model models/ser_lstm.h5
═══════════════════════════════════════════════════════════

GETTING RAVDESS DATA (free download):
  1. Go to https://zenodo.org/record/1188976
  2. Download the Audio_Speech_Actors zip files
  3. Extract all .wav files into:  data/ravdess/
  4. Run this script
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ser_module import SERModule, EMOTIONS


def plot_and_save(history, out="models"):
    os.makedirs(out, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13,4))
    fig.suptitle("LSTM Speech Emotion Recognition — Training",
                 fontsize=13, fontweight="bold")

    a1.plot(history.history["accuracy"],     label="Train", color="#4f8ef7")
    a1.plot(history.history["val_accuracy"], label="Val",   color="#f77c4f")
    a1.set_title("Accuracy"); a1.set_xlabel("Epoch")
    a1.set_ylabel("Accuracy"); a1.legend(); a1.grid(alpha=0.3)

    a2.plot(history.history["loss"],     label="Train", color="#4f8ef7")
    a2.plot(history.history["val_loss"], label="Val",   color="#f77c4f")
    a2.set_title("Loss"); a2.set_xlabel("Epoch")
    a2.set_ylabel("Loss"); a2.legend(); a2.grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out, "training_plot.png")
    plt.savefig(p, dpi=150)
    print(f"[TRAIN] Plot saved → {p}")
    plt.close()


def main(args):
    print("\n" + "═"*52)
    print("  SER LSTM Training")
    print("  Dataset : RAVDESS")
    print("═"*52)

    if not os.path.isdir(args.data):
        print(f"\n  ERROR: '{args.data}' not found.")
        print("  Download RAVDESS from:")
        print("  https://zenodo.org/record/1188976")
        print("  Extract .wav files into: data/ravdess/\n")
        sys.exit(1)

    ser = SERModule()
    ser.build()
    history = ser.train(args.data, epochs=args.epochs, batch=args.batch)

    if history is None:
        sys.exit(1)

    ser.save("models/ser_lstm.h5")
    plot_and_save(history)

    final = history.history["val_accuracy"][-1]
    print(f"\n[TRAIN] ✓ Final val accuracy : {final*100:.2f}%")
    print("[TRAIN] ✓ Model → models/ser_lstm.h5")
    print("[TRAIN] ✓ Now run:")
    print("          python main.py --model models/ser_lstm.h5\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="data/ravdess/")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch",  type=int, default=32)
    main(ap.parse_args())
