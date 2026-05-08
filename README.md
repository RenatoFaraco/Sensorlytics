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

## Installation & Setup

To ensure an isolated environment and avoid dependency conflicts, follow the steps below to set up your virtual environment.

### Create the Virtual Environment

```bash
python -m venv venv
```
### Activate the Virtual Enviroment
```bash
venv\Scripts\activate
```

### Instal Dependencies

To use **Sensorlytics**, you need Python 3.8 or higher!
With the environment active (you will see the (venv) prefix in your terminal), install the required libraries:

```bash
pip install --upgrade pip
pip install process_spectra pandas numpy matplotlib seaborn pyarrow scipy
```
