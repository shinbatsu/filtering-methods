import numpy as np
from scipy.fft import fft, fftfreq, ifft

def wiener_filter(y, n0_data, n1_data, a, R=None):
    """
    Wiener filter for nonlinear signal λ = f(x).
    
    Args:
        y: Observed signal.
        n0_data: Reference data for observation noise.
        n1_data: Reference data for process noise.
        a: AR(1) process parameter.
        R: Observation noise variance.
    """
    N = len(y)
    b = a ** 2
    
    var_n1 = np.var(n1_data)
    var_n0 = np.var(n0_data) if R is None else R
    
    var_x = _calculate_state_variance(var_n1, b)
    var_lambda = _estimate_lambda_variance(y, var_n0)
    
    psd_lambda = _calculate_psd_lambda(N, var_lambda, b)
    frequency_response = _design_wiener_filter(psd_lambda, var_n0)
    filtered_signal = _apply_wiener_filter(y, frequency_response)
    var_lambda_v = np.var(filtered_signal)
    
    parameters = _build_wiener_parameters(a, b, var_n0, var_n1, var_x, var_lambda, var_lambda_v)
    
    return filtered_signal, {
        'a': parameters['a'],
        'b': parameters['b'],
        'R': parameters['R'],
        'var_n1': parameters['var_n1'],
        'var_x': parameters['var_x'],
        'var_lambda': parameters['var_lambda'],
        'var_lambda_v': parameters['var_lambda_v'],
        'H_freq_response': frequency_response
    }


def _calculate_state_variance(var_n1, b):
    """
    Calculate state variance for AR(1) process.
    
    Args:
        var_n1: Process noise variance.
        b: a^2 parameter.
    """
    return var_n1 / (1 - b) if b < 1 else var_n1


def _estimate_lambda_variance(y, var_n0):
    """
    Estimate signal variance from observations.
    
    Args:
        y: Observed signal.
        var_n0: Observation noise variance.
    """
    var_y = np.var(y)
    var_lambda = var_y - var_n0
    
    if var_lambda <= 0:
        var_lambda = max(var_y, 1e-10)
    
    return var_lambda


def _calculate_psd_lambda(N, var_lambda, b):
    """
    Calculate power spectral density of λ signal.
    
    Args:
        N: Signal length.
        var_lambda: Signal variance.
        b: a^2 parameter.
    """
    frequencies = fftfreq(N)
    psd = np.zeros(N, dtype=complex)
    
    for i, f in enumerate(frequencies):
        denominator = 1 - b * np.exp(-1j * 2 * np.pi * f)
        psd[i] = var_lambda * (1 - b ** 2) / (np.abs(denominator) ** 2)
    
    return psd


def _design_wiener_filter(psd_lambda, var_n0):
    """
    Design Wiener filter frequency response.
    
    Args:
        psd_lambda: Signal power spectral density.
        var_n0: Observation noise variance.
    """
    return psd_lambda / (psd_lambda + var_n0)


def _apply_wiener_filter(signal, frequency_response):
    """
    Apply filter in frequency domain.
    
    Args:
        signal: Input signal.
        frequency_response: Filter frequency response.
    """
    signal_fft = fft(signal)
    filtered_fft = frequency_response * signal_fft
    return np.real(ifft(filtered_fft))


def _build_wiener_parameters(a, b, var_n0, var_n1, var_x, var_lambda, var_lambda_v):
    """
    Build Wiener filter parameters dictionary.
    
    Args:
        a: AR(1) process parameter.
        b: a^2 parameter.
        var_n0: Observation noise variance.
        var_n1: Process noise variance.
        var_x: State variance.
        var_lambda: Signal variance.
        var_lambda_v: Filtered signal variance.
    """
    return {
        'a': a,
        'b': b,
        'R': var_n0,
        'var_n1': var_n1,
        'var_x': var_x,
        'var_lambda': var_lambda,
        'var_lambda_v': var_lambda_v
    }


def calculate_snr_improvement(original_signal, filtered_signal, noise_variance):
    """
    Calculate SNR improvement in decibels.
    
    Args:
        original_signal: Original noisy signal.
        filtered_signal: Filtered signal.
        noise_variance: Noise variance.
    """
    original_snr = 10 * np.log10(np.var(original_signal) / noise_variance)
    filtered_snr = 10 * np.log10(np.var(filtered_signal) / noise_variance)
    return filtered_snr - original_snr