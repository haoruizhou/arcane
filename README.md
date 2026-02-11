# Arcane

AI-Powered Photo Culling App for macOS (Metal) and Nvidia (CUDA).

## Features
- **Fast RAW Loading**: Uses `rawpy` for high-fidelity image decoding.
- **AI Analysis**: 
    - Face detection and boxing.
    - Eye openness check.
    - Focus scoring.
- **Efficient Culling**:
    - Keyboard shortcuts for ratings (1-5) and flags (P/N/X).
    - Writes standard XMP sidecar files compatible with Lightroom/Capture One.

## Installation

Ensure you have Python 3.10+ installed.

```bash
pip install .
```

## Usage

```bash
python src/main.py
```
