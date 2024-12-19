# Sensorlytics

**Sensorlytics** is a Python library for optical spectrum analysis. It simplifies the processing, visualization, and analysis of spectral data, making it useful for:

- Optical experiments
- Material characterization
- Sensing applications

The library supports saving and loading data in common formats such as JSON and Parquet, promoting scientific reproducibility.

## Features

- Read and process multiple `.txt` files containing spectral data.
- Generate plots for visualizing spectral data and extracted features.
- Compute smoothed mean values and fringe visibility.
- Save and load data in JSON and Parquet formats.
- Find resonant wavelengths using peak detection methods.
- Histogram and PDF (Probability Density Function) calculations.

## Installation

To use **Sensorlytics**, you need Python 3.8 or higher. Install the required dependencies:

```bash
pip install process_spectra pandas numpy matplotlib seaborn pyarrow scipy
