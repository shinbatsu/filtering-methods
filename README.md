# Wiener-Kalman Filtering and Analysis

This project provides a framework for filtering, analyzing, and visualizing nonlinear signals with noise using Wiener and Extended Kalman filters. It handles signals of the form λ = f(x) with saturation and variable observation noise.

## Features

- AR(1) signal generation and processing
- Nonlinear transformations with saturation
- Wiener filtering with PSD-based frequency response
- Extended Kalman filtering with linearization
- SNR, MSE, and improvement metrics
- Autocorrelation analysis
- Noise variance variation analysis
- Export results to Excel
- Comprehensive visualization

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Install dependencies using uv and the pyproject.toml file:
   ```bash
   uv install
   ```

## Usage

1. Configure experiment parameters in `config.py`.
2. Run the main script:
   ```bash
   python main.py
   ```
   
3. Outputs:
   - Excel file with data, filter parameters, SNR metrics, autocorrelation, and noise variation analysis
   - Plots of filter performance, extended analyses, and SNR stepwise evolution