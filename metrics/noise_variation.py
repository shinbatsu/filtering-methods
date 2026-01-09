import numpy as np
import pandas as pd
from filters.kalman import kalman_filter
from filters.wiener import wiener_filter

def analyze_noise_variation(lambda_true, n0_data, n1_data, a):
    """Analysis of the impact of observation noise variance variation"""

    alpha_values = np.arange(1.0, 10.5, 0.5)
    results = []

    for alpha in alpha_values:
        n0_scaled = alpha * n0_data
        var_n0_scaled = alpha**2 * np.var(n0_data)

        y_scaled = lambda_true + n0_scaled

        lambda_wiener_scaled, _ = wiener_filter(
            y_scaled, n0_data, n1_data, a, R=var_n0_scaled
        )

        lambda_ekf_scaled, _, _, _, _, _, _, _ = kalman_filter(
            y_scaled, n0_data, n1_data, a, R=var_n0_scaled
        )

        mse_wiener_scaled = np.mean((lambda_wiener_scaled - lambda_true)**2)
        mse_ekf_scaled = np.mean((lambda_ekf_scaled - lambda_true)**2)

        signal_power = np.mean(lambda_true**2)
        snr_in_scaled = signal_power / var_n0_scaled
        snr_out_wiener_scaled = signal_power / mse_wiener_scaled
        snr_out_ekf_scaled = signal_power / mse_ekf_scaled

        improvement_wiener_scaled = snr_out_wiener_scaled / snr_in_scaled
        improvement_ekf_scaled = snr_out_ekf_scaled / snr_in_scaled

        results.append({
            'alpha': alpha,
            'var_n0_scaled': var_n0_scaled,
            'mse_wiener': mse_wiener_scaled,
            'mse_ekf': mse_ekf_scaled,
            'snr_in': snr_in_scaled,
            'snr_out_wiener': snr_out_wiener_scaled,
            'snr_out_ekf': snr_out_ekf_scaled,
            'improvement_wiener': improvement_wiener_scaled,
            'improvement_ekf': improvement_ekf_scaled
        })

    return pd.DataFrame(results)
