
import numpy as np

def calculate_acf(signal_data, max_lag=20):
    """Calculate the autocorrelation function"""
    
    acf_values = []

    for lag in range(max_lag + 1):
        if lag == 0:
            corr = 1.0
        else:
            corr = np.corrcoef(signal_data[:-lag], signal_data[lag:])[0, 1]
            if np.isnan(corr):
                corr = 0.0
        acf_values.append(corr)

    return acf_values

