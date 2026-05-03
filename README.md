````markdown
# SpeechEvalKit

<p align="center">
  <b>🎧 SpeechEvalKit</b><br>
  A unified Python package for objective and subjective speech evaluation.
</p>

<p align="center">
  <a href="https://pypi.org/project/speechevalkit/">
    <img src="https://img.shields.io/pypi/v/speechevalkit.svg" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/speechevalkit/">
    <img src="https://img.shields.io/pypi/pyversions/speechevalkit.svg" alt="Python Versions">
  </a>
  <a href="https://pypi.org/project/speechevalkit/">
    <img src="https://img.shields.io/pypi/l/speechevalkit.svg" alt="License">
  </a>
  <a href="https://pypi.org/project/speechevalkit/">
    <img src="https://img.shields.io/pypi/dm/speechevalkit.svg" alt="Downloads">
  </a>
</p>

---

## Project Description

**SpeechEvalKit** is a lightweight, research-friendly Python toolkit for evaluating speech generation, speech enhancement, text-to-speech, voice conversion, dysarthric speech synthesis, and ASR systems.

It supports both:

1. **Objective evaluation** using metrics such as PESQ, STOI, SI-SDR, MCD, cosine similarity, WER, CER, and MER.
2. **Subjective evaluation** using a Django-based MOS / SMOS web interface with subject registration, admin dashboard, experiment-wise CSV logging, Plotly graphs, and waveform playback.

SpeechEvalKit is designed to be simple enough for quick experiments and flexible enough for real research studies.

---

## Table of Contents

- [Documentation](#documentation)
- [Installation](#installation)
  - [Using PyPI](#using-pypi)
  - [Using Anaconda](#using-anaconda)
  - [Building From Source](#building-from-source)
- [System Dependencies](#system-dependencies)
  - [Linux apt-get](#linux-apt-get)
  - [Linux yum](#linux-yum)
  - [Mac](#mac)
  - [Windows](#windows)
- [Quick Start](#quick-start)
- [Python API](#python-api)
- [Command Line Interface](#command-line-interface)
- [Supported Metrics](#supported-metrics)
- [MOS and SMOS Web GUI](#mos-and-smos-web-gui)
- [Admin Dashboard](#admin-dashboard)
- [Experiment Outputs](#experiment-outputs)
- [Recommended Metric Sets](#recommended-metric-sets)
- [Optional Dependencies](#optional-dependencies)
- [Troubleshooting](#troubleshooting)
- [Discussion](#discussion)
- [Citing](#citing)
- [License](#license)

---

## Documentation

SpeechEvalKit documentation is currently provided through this README and package examples.

The package is built around one simple API:

```python
from speechevalkit import evaluate

results = evaluate(
    ref_dir="ground_truth/",
    pred_dir="generated/",
    metrics=["pesq", "stoi", "mcd", "si_sdr", "cosine"]
)

print(results["summary"])
````

SpeechEvalKit automatically handles:

* directory scanning
* filename matching
* audio loading
* mono conversion
* resampling
* waveform alignment
* corrupted file skipping
* batch processing
* progress bars
* JSON / CSV output
* optional multiprocessing
* optional fuzzy filename matching

For subjective evaluation, SpeechEvalKit provides a Django web interface:

```python
from speechevalkit.gui import launch_gui

launch_gui(
    audio_dir="generated/",
    ref_dir="ground_truth/",
    mode="smos",
    recursive=True,
    host="0.0.0.0",
    port=8765,
)
```

Back To Top ↥

---

## Installation

SpeechEvalKit can be installed using PyPI, Anaconda, or directly from source.

### Using PyPI

Install the latest stable version from PyPI:

```bash
python -m pip install speechevalkit
```

For full functionality, install optional dependencies:

```bash
python -m pip install pesq pystoi jiwer django
```

Recommended full install:

```bash
python -m pip install speechevalkit
python -m pip install pesq pystoi jiwer django
```

Check installation:

```bash
python -c "from speechevalkit import evaluate; print('SpeechEvalKit API OK')"
python -c "from speechevalkit.gui import launch_gui; print('SpeechEvalKit GUI OK')"
```

Back To Top ↥

---

### Using Anaconda

Create a clean environment:

```bash
conda create -n speechevalkit python=3.11 -y
conda activate speechevalkit
```

Install SpeechEvalKit:

```bash
python -m pip install speechevalkit
```

Install optional metric and GUI dependencies:

```bash
python -m pip install pesq pystoi jiwer django
```

Install audio system libraries through conda-forge:

```bash
conda install -c conda-forge ffmpeg libsndfile -y
```

Test:

```bash
python -c "from speechevalkit import evaluate; print('OK')"
```

Back To Top ↥

---

### Building From Source

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/speechevalkit.git
cd speechevalkit
```

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
```

Install in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Install optional dependencies:

```bash
python -m pip install pesq pystoi jiwer django
```

Build package:

```bash
python -m pip install build twine
python -m build
twine check dist/*
```

Back To Top ↥

---

## System Dependencies

SpeechEvalKit uses Python libraries for audio loading and processing. For broader audio format support, you may need FFmpeg and libsndfile.

### Linux apt-get

Ubuntu / Debian:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv ffmpeg libsndfile1 python3-tk
```

Then install SpeechEvalKit:

```bash
python3 -m pip install speechevalkit
```

Optional dependencies:

```bash
python3 -m pip install pesq pystoi jiwer django
```

Back To Top ↥

---

### Linux yum

CentOS / RHEL / Fedora-like systems:

```bash
sudo yum install -y python3 python3-pip ffmpeg libsndfile
```

If FFmpeg is unavailable in the default repository, enable RPM Fusion or EPEL depending on your distribution.

Then install:

```bash
python3 -m pip install speechevalkit
```

Optional dependencies:

```bash
python3 -m pip install pesq pystoi jiwer django
```

Back To Top ↥

---

### Mac

Install system dependencies with Homebrew:

```bash
brew install ffmpeg libsndfile
```

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install SpeechEvalKit:

```bash
python -m pip install --upgrade pip
python -m pip install speechevalkit
```

Optional dependencies:

```bash
python -m pip install pesq pystoi jiwer django
```

Back To Top ↥

---

### Windows

Create and activate a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\activate
```

Install SpeechEvalKit:

```powershell
python -m pip install --upgrade pip
python -m pip install speechevalkit
```

Install optional dependencies:

```powershell
python -m pip install pesq pystoi jiwer django
```

For broader audio format support, install FFmpeg and add it to your system PATH.

Test installation:

```powershell
python -c "from speechevalkit import evaluate; print('API OK')"
python -c "from speechevalkit.gui import launch_gui; print('GUI OK')"
```

Back To Top ↥

---

## Quick Start

### Objective Speech Evaluation

Directory structure:

```text
ground_truth/
├── utt001.wav
├── utt002.wav
└── utt003.wav

generated/
├── utt001.wav
├── utt002.wav
└── utt003.wav
```

Run:

```python
from speechevalkit import evaluate

results = evaluate(
    ref_dir="ground_truth/",
    pred_dir="generated/",
    metrics=["pesq", "stoi", "mcd", "si_sdr", "cosine"],
    sample_rate=16000,
    show_progress=True,
)

print(results["summary"])
```

Example output:

```python
{
    "pesq": 3.42,
    "stoi": 0.91,
    "mcd": 5.23,
    "si_sdr": 14.72,
    "cosine": 0.96
}
```

Back To Top ↥

---

## Python API

### Basic Evaluation

```python
from speechevalkit import evaluate

results = evaluate(
    ref_dir="ground_truth/",
    pred_dir="generated/",
    metrics=["pesq", "stoi", "mcd", "si_sdr", "cosine"]
)

print(results)
```

### Save Results

```python
from speechevalkit import evaluate

results = evaluate(
    ref_dir="ground_truth/",
    pred_dir="generated/",
    metrics=["pesq", "stoi", "mcd"],
    save_json="results.json",
    save_csv="results.csv",
)
```

### Recursive Directory Evaluation

```python
from speechevalkit import evaluate

results = evaluate(
    ref_dir="ground_truth/",
    pred_dir="generated/",
    metrics=["pesq", "stoi"],
    recursive=True,
)
```

### Fuzzy Filename Matching

Useful when generated filenames are slightly different.

```python
from speechevalkit import evaluate

results = evaluate(
    ref_dir="ground_truth/",
    pred_dir="generated/",
    metrics=["pesq", "stoi"],
    fuzzy_match=True,
    match_threshold=0.75,
)
```

Example:

```text
ground_truth/
└── speaker01_utt001.wav

generated/
└── speaker01_utt001_generated.wav
```

### Parallel Evaluation

```python
from speechevalkit import evaluate

results = evaluate(
    ref_dir="ground_truth/",
    pred_dir="generated/",
    metrics=["pesq", "stoi", "mcd", "si_sdr"],
    num_workers="auto",
)
```

Manual worker count:

```python
results = evaluate(
    ref_dir="ground_truth/",
    pred_dir="generated/",
    metrics=["pesq", "stoi"],
    num_workers=4,
)
```

### Text Evaluation

```python
from speechevalkit.evaluate import evaluate_text

results = evaluate_text(
    ref_text_dir="reference_text/",
    pred_text_dir="predicted_text/",
    metrics=["wer", "cer", "mer"],
)

print(results["summary"])
```

Back To Top ↥

---

## Command Line Interface

Basic CLI:

```bash
speechevalkit --ref ground_truth/ --pred generated/ --metrics pesq stoi
```

All common metrics:

```bash
speechevalkit \
  --ref ground_truth/ \
  --pred generated/ \
  --metrics pesq stoi mcd si_sdr cosine
```

Recursive matching:

```bash
speechevalkit \
  --ref ground_truth/ \
  --pred generated/ \
  --metrics pesq stoi \
  --recursive
```

Save outputs:

```bash
speechevalkit \
  --ref ground_truth/ \
  --pred generated/ \
  --metrics pesq stoi mcd \
  --save-json results.json \
  --save-csv results.csv
```

Use fuzzy matching:

```bash
speechevalkit \
  --ref ground_truth/ \
  --pred generated/ \
  --metrics pesq stoi \
  --fuzzy-match \
  --match-threshold 0.75
```

Use multiprocessing:

```bash
speechevalkit \
  --ref ground_truth/ \
  --pred generated/ \
  --metrics pesq stoi mcd \
  --num-workers auto
```

Back To Top ↥

---

## Supported Metrics

### Audio Metrics

| Metric   | Description                                | Direction        |
| -------- | ------------------------------------------ | ---------------- |
| `pesq`   | Perceptual Evaluation of Speech Quality    | Higher is better |
| `stoi`   | Short-Time Objective Intelligibility       | Higher is better |
| `si_sdr` | Scale-Invariant Signal-to-Distortion Ratio | Higher is better |
| `mcd`    | Mel-Cepstral Distortion                    | Lower is better  |
| `cosine` | MFCC / embedding-style cosine similarity   | Higher is better |

### Text Metrics

| Metric | Description          | Direction       |
| ------ | -------------------- | --------------- |
| `wer`  | Word Error Rate      | Lower is better |
| `cer`  | Character Error Rate | Lower is better |
| `mer`  | Match Error Rate     | Lower is better |

### Subjective Metrics

| Metric | Description                                  |
| ------ | -------------------------------------------- |
| `mos`  | Mean Opinion Score for quality / naturalness |
| `smos` | Speaker Similarity Mean Opinion Score        |

Back To Top ↥

---

## MOS and SMOS Web GUI

SpeechEvalKit includes a Django-based subjective evaluation website.

Install Django:

```bash
python -m pip install django
```

### MOS Web GUI

MOS is used for evaluating naturalness or speech quality.

```python
from speechevalkit.gui import launch_gui

launch_gui(
    audio_dir="generated/",
    mode="mos",
    recursive=True,
    host="0.0.0.0",
    port=8765,
)
```

### SMOS Web GUI

SMOS is used for speaker similarity evaluation.

```python
from speechevalkit.gui import launch_gui

launch_gui(
    audio_dir="converted_or_generated/",
    ref_dir="ground_truth/",
    mode="smos",
    recursive=True,
    host="0.0.0.0",
    port=8765,
)
```

The web GUI supports:

* subject name entry
* age entry
* gender entry
* organization entry
* auto-increment subject numbering
* mandatory scoring of every WAV file
* admin-controlled shuffle
* final remark submission
* waveform-only audio player
* click waveform to play from timestamp
* played waveform segment in blue
* remaining waveform segment in light grey
* mobile-friendly UI
* desktop-friendly UI
* experiment-wise CSV saving
* subject-wise CSV saving
* admin dashboard
* Plotly graphs

Back To Top ↥

---

## Admin Dashboard

Default admin login:

```text
Username: ravindra
Password: speechevalkit
```

Admin URL:

```text
http://127.0.0.1:8765/admin/login
```

If running on a network, SpeechEvalKit prints a LAN URL such as:

```text
http://192.168.x.x:8765/admin/login
```

The admin dashboard includes:

* overall experiment dashboard
* subject dropdown menu
* subject-specific dashboard
* subject-wise CSV download
* ratings CSV download
* global CSV download
* subject information CSV download
* WAV file summary CSV download
* score distribution graph
* gender distribution graph
* organization distribution graph
* file-wise mean score graph
* admin settings panel

Admin settings include:

* shuffle WAV order
* allow page refresh / continuation
* show subject file navigation

Subjects cannot enable or disable shuffle. Shuffle is admin-only.

Back To Top ↥

---

## Experiment Outputs

Every web GUI run creates a fresh experiment folder.

Example MOS output:

```text
results/experiments/
└── exp_001_2026-05-03_12-30-10_mos/
    ├── mos_ratings.csv
    ├── subjects.csv
    ├── global_subject_results.csv
    ├── mos_summary.csv
    ├── wav_file_score_summary.csv
    ├── admin_settings.json
    └── django_session.sqlite3
```

Example SMOS output:

```text
results/experiments/
└── exp_002_2026-05-03_12-45-22_smos/
    ├── smos_ratings.csv
    ├── subjects.csv
    ├── global_subject_results.csv
    ├── smos_summary.csv
    ├── wav_file_score_summary.csv
    ├── admin_settings.json
    └── django_session.sqlite3
```

### `mos_ratings.csv` / `smos_ratings.csv`

Contains one row per WAV rating.

### `subjects.csv`

Contains subject demographic information.

### `global_subject_results.csv`

Contains one row per subject with overall subject score.

### `wav_file_score_summary.csv`

Contains per-WAV mean, standard deviation, mean ± std, and confidence interval.

### `mos_summary.csv` / `smos_summary.csv`

Contains experiment-level summary statistics.

Back To Top ↥

---

## Recommended Metric Sets

### Speech Enhancement

```python
metrics = ["pesq", "stoi", "si_sdr", "cosine"]
```

### Text-to-Speech

```python
metrics = ["mcd", "cosine", "stoi"]
```

Recommended subjective evaluation:

```text
MOS
```

### Voice Conversion

```python
metrics = ["mcd", "cosine", "si_sdr"]
```

Recommended subjective evaluation:

```text
SMOS
```

### ASR

```python
metrics = ["wer", "cer", "mer"]
```

### Dysarthric Speech Synthesis

```python
metrics = ["pesq", "stoi", "mcd", "cosine"]
```

Recommended subjective evaluation:

```text
MOS and SMOS
```

Back To Top ↥

---

## Optional Dependencies

| Feature          | Install                        |
| ---------------- | ------------------------------ |
| True PESQ        | `python -m pip install pesq`   |
| True STOI        | `python -m pip install pystoi` |
| WER / CER / MER  | `python -m pip install jiwer`  |
| Web GUI          | `python -m pip install django` |
| Audio decoding   | `ffmpeg`, `libsndfile`         |
| Plotly dashboard | Loaded from CDN in browser     |

Recommended full setup:

```bash
python -m pip install speechevalkit
python -m pip install pesq pystoi jiwer django
```

Back To Top ↥

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'speechevalkit'`

Install the package:

```bash
python -m pip install speechevalkit
```

For local development:

```bash
python -m pip install -e .
```

Make sure your structure is:

```text
project_root/
├── pyproject.toml
├── README.md
└── speechevalkit/
    ├── __init__.py
    ├── evaluate.py
    ├── cli.py
    ├── gui.py
    ├── metrics/
    └── utils/
```

### `ModuleNotFoundError: No module named 'django'`

Install Django:

```bash
python -m pip install django
```

### Audio files are not loading

Install FFmpeg and libsndfile.

Ubuntu:

```bash
sudo apt install ffmpeg libsndfile1
```

Conda:

```bash
conda install -c conda-forge ffmpeg libsndfile -y
```

### PyPI upload says version already exists

PyPI does not allow uploading the same version twice.

Update `pyproject.toml`:

```toml
version = "0.2.1"
```

Then rebuild:

```bash
rm -rf dist build *.egg-info speechevalkit.egg-info
python -m build
twine check dist/*
twine upload dist/*
```

### Browser cannot open LAN URL

Allow the port:

```bash
sudo ufw allow 8765/tcp
```

Then open the printed LAN URL on another device connected to the same Wi-Fi.

### Subject cannot submit

This is expected if not all WAV files are rated.

SpeechEvalKit requires each subject to rate every WAV file before the final submit page is available.

Back To Top ↥

---

## Discussion

For questions, issues, bugs, or feature requests, use the GitHub issue tracker:

```text
https://github.com/YOUR_USERNAME/speechevalkit/issues
```

For PyPI package information:

```text
https://pypi.org/project/speechevalkit/
```

Back To Top ↥

---

## Citing

If you use SpeechEvalKit in academic work, please cite:

```bibtex
@software{speechevalkit2026,
  title = {SpeechEvalKit: A Unified Speech Evaluation Toolkit for Speech Generation, Enhancement, Voice Conversion, ASR, MOS, and SMOS Evaluation},
  author = {YOUR NAME},
  year = {2026},
  url = {https://pypi.org/project/speechevalkit/},
  version = {0.2.0}
}
```

Recommended citations for related metrics:

```bibtex
@inproceedings{rix2001pesq,
  title = {Perceptual evaluation of speech quality (PESQ), a new method for speech quality assessment of telephone networks and codecs},
  author = {Rix, Antony W. and Beerends, John G. and Hollier, Michael P. and Hekstra, Andries P.},
  booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing},
  year = {2001}
}

@article{taal2011stoi,
  title = {An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech},
  author = {Taal, Cees H. and Hendriks, Richard C. and Heusdens, Richard and Jensen, Jesper},
  journal = {IEEE Transactions on Audio, Speech, and Language Processing},
  year = {2011}
}

@inproceedings{leroux2019sdr,
  title = {SDR half-baked or well done?},
  author = {Le Roux, Jonathan and Wisdom, Scott and Erdogan, Hakan and Hershey, John R.},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing},
  year = {2019}
}

@inproceedings{mittag2021nisqa,
  title = {NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets},
  author = {Mittag, Gabriel and Naderi, Babak and Chehadi, Assmaa and Möller, Sebastian},
  booktitle = {Interspeech},
  year = {2021}
}

@inproceedings{reddy2021dnsmos,
  title = {DNSMOS: A Non-Intrusive Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors},
  author = {Reddy, Chandan K. A. and Gopal, Vishak and Cutler, Ross},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing},
  year = {2021}
}
```

Back To Top ↥

---

## License

MIT License.

Copyright © 2026 SpeechEvalKit.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the terms of the MIT License.

Back To Top ↥

```
```
