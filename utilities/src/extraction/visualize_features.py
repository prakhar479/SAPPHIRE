# Run inside a notebook cell or as 'python visual...py --audio file --lyrics file'
import argparse
import matplotlib.pyplot as plt
from Extraction.process import analyze


p = argparse.ArgumentParser()
p.add_argument("--audio")
p.add_argument("--out", default="vis.png")
args, _ = p.parse_known_args()
feats = analyze(audio_path=args.audio)
# Extract pitch/chroma/loudness
pitch = feats.get("melody", {}).get("pitch_contour", {})
chroma = feats.get("acoustic", {}).get("chroma_matrix", None)
loud = feats.get("acoustic", {}).get("loudness", {})


# simple plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
if pitch:
    axes[0].plot(pitch.get("times", []), pitch.get("frequency_hz", []))
    axes[0].set_title("Pitch contour (Hz)")
if chroma is not None:
    axes[1].imshow(chroma, aspect="auto", origin="lower")
    axes[1].set_title("Chroma")
if loud:
    axes[2].plot(loud.get("times", []), loud.get("loudness_db", []))
    axes[2].set_title("Loudness (approx dB/LUFS)")
plt.tight_layout()
plt.show()
