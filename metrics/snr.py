import numpy as np

def calculate_snr_metrics(lambda_true, lambda_wiener, lambda_ekf, n0_data):
    """Calculation of all SNR metrics and improvement"""

    signal_power = np.mean(lambda_true**2)

    mse_wiener = np.mean((lambda_wiener - lambda_true)**2)
    mse_ekf = np.mean((lambda_ekf - lambda_true)**2)

    snr_in = signal_power / np.var(n0_data)
    snr_out_wiener = signal_power / mse_wiener
    snr_out_ekf = signal_power / mse_ekf

    improvement_wiener = snr_out_wiener / snr_in
    improvement_ekf = snr_out_ekf / snr_in

    metrics = {
        'signal_power': signal_power,
        'mse_wiener': mse_wiener,
        'mse_ekf': mse_ekf,
        'snr_in': snr_in,
        'snr_out_wiener': snr_out_wiener,
        'snr_out_ekf': snr_out_ekf,
        'improvement_wiener': improvement_wiener,
        'improvement_ekf': improvement_ekf
    }

    return metrics