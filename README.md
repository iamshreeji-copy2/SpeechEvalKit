# SpeechEvalKit

<p align="center">
  <b>🎧 SpeechEvalKit</b><br>
  A Python package for speech generation, enhancement, voice conversion, ASR, MOS, and SMOS evaluation.
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

## Overview

**SpeechEvalKit** is a lightweight and extensible Python toolkit for evaluating speech systems.

It is designed for:

- speech generation
- text-to-speech
- speech enhancement
- voice conversion
- dysarthric speech synthesis
- ASR transcript evaluation
- subjective MOS / SMOS listening tests

SpeechEvalKit provides:

- a simple Python API
- command-line evaluation
- batch directory evaluation
- automatic audio loading and resampling
- waveform alignment
- fuzzy filename matching
- multiprocessing support
- JSON and CSV logging
- colorful terminal output
- Django-based MOS / SMOS web GUI
- admin dashboard for subjective tests
- Plotly-based visualization
- subject-wise and experiment-wise CSV export

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

Documentation is currently available through this README and package examples.

The package supports both objective and subjective speech evaluation workflows.

Example use cases:

```python
from speechevalkit import evaluate

results = evaluate(
    ref_dir="ground_truth/",
    pred_dir="generated/",
    metrics=["pesq", "stoi", "mcd", "si_sdr", "cosine"]
)

print(results["summary"])