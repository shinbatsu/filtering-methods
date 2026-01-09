# main.py
import numpy as np
import pandas as pd

# Подключение своих модулей
from excel.loader import load_or_generate_noise
from models.nonlinear import nonlinear_f
from filters.wiener import wiener_filter
from filters.kalman import kalman_filter
from metrics.snr import calculate_snr_metrics
from metrics.acf import calculate_acf
from metrics.noise_variation import analyze_noise_variation
from views.plots import (
    plot_filter_comparison,
    plot_extended_analysis,
    plot_snr_steps
)
from config import *

n1_data, n0_data = load_or_generate_noise('variant.xlsx', N)

x_true = np.zeros(N)
x_true[0] = n1_data[0]
for k in range(1, N):
    x_true[k] = a * x_true[k-1] + n1_data[k]

lambda_true = nonlinear_f(x_true)
y_observed = lambda_true + n0_data


lambda_wiener, wiener_params = wiener_filter(y_observed,n0_data,n1_data, a)
lambda_ekf, x_hat, P_ekf, kalman_params, x_minus, P_minus, H_jac, K = kalman_filter(
    y_observed, n0_data, n1_data, a
)

snr_metrics = calculate_snr_metrics(lambda_true, lambda_wiener, lambda_ekf, n0_data)
acf_values = calculate_acf(y_observed, max_lag=20)
noise_variation_df = analyze_noise_variation(lambda_true, n0_data, n1_data, a)


var_n1 = np.var(n1_data)
var_n0 = np.var(n0_data)
var_x_true = np.var(x_true)
var_lambda_true = np.var(lambda_true)
var_y = np.var(y_observed)

from excel.exporter import export_full_analysis

export_full_analysis(
    N=N,
    preview_points=preview_points,
    n1_data=n1_data,
    n0_data=n0_data,
    x_true=x_true,
    lambda_true=lambda_true,
    y_observed=y_observed,
    lambda_wiener=lambda_wiener,
    lambda_ekf=lambda_ekf,
    x_hat=x_hat,
    P_ekf=P_ekf,
    x_minus=x_minus,
    P_minus=P_minus,
    H_jac=H_jac,
    K=K,
    wiener_params=wiener_params,
    kalman_params=kalman_params,
    snr_metrics=snr_metrics,
    acf_values=acf_values,
    noise_variation_df=noise_variation_df,
    var_n1=var_n1,
    var_n0=var_n0,
    var_x_true=var_x_true,
    var_lambda_true=var_lambda_true,
    var_y=var_y
)
plot_filter_comparison(
    lambda_true, y_observed, lambda_wiener, lambda_ekf, 
    x_hat, P_ekf, K, wiener_params, N, preview_points
)

plot_extended_analysis(
    acf_values, noise_variation_df, lambda_true, y_observed,
    lambda_wiener, lambda_ekf, snr_metrics
)

snr_out_kalman = plot_snr_steps(
    lambda_true, lambda_ekf,
    first_steps=50
)

better_filter = "Винер" if snr_metrics['improvement_wiener'] > snr_metrics['improvement_ekf'] else "Калман"

print("\n=== ПОЛНАЯ СВОДКА РЕЗУЛЬТАТОВ ===")
print(f"Количество точек: {N}")
print(f"Лучший фильтр по улучшению SNR: {better_filter}")
print(f"MSE Винера: {snr_metrics['mse_wiener']:.6f}, MSE Калмана: {snr_metrics['mse_ekf']:.6f}")
print(f"ACF(0): {acf_values[0]:.4f}, ACF(1): {acf_values[1]:.4f}")
