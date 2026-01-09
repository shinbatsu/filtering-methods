# visualization/plots.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftfreq

from config import output_folder
import os

def plot_filter_comparison(lambda_true, y_observed, lambda_wiener, lambda_ekf, 
                           x_hat, P_ekf, K, wiener_params, N, preview_points=100):
    """Сравнение фильтров, ошибки и параметры Калмана, частотная характеристика Винера."""
    
    plt.figure(figsize=(20, 15))
    
    # 1. Сравнение фильтрации (первые preview_points точек)
    plt.subplot(3, 2, 1)
    plt.plot(lambda_true[:preview_points], 'b-', label='Истинный λ', linewidth=2)
    plt.plot(y_observed[:preview_points], 'r-', label='Наблюдения y', alpha=0.7, linewidth=1)
    plt.plot(lambda_wiener[:preview_points], 'g-', label='Фильтр Винера', linewidth=1.5)
    plt.plot(lambda_ekf[:preview_points], 'm-', label='Фильтр Калмана', linewidth=1.5)
    plt.xlabel('Время, k'); plt.ylabel('Амплитуда')
    plt.title('СРАВНЕНИЕ ФИЛЬТРАЦИИ')
    plt.legend(); plt.grid(True)
    
    # 2. Ошибка фильтрации
    plt.subplot(3, 2, 2)
    plt.plot(lambda_wiener[:preview_points] - lambda_true[:preview_points], 'g-', label='Ошибка Винера', alpha=0.7)
    plt.plot(lambda_ekf[:preview_points] - lambda_true[:preview_points], 'm-', label='Ошибка Калмана', alpha=0.7)
    plt.xlabel('Время, k'); plt.ylabel('Ошибка')
    plt.title('Ошибка фильтрации')
    plt.legend(); plt.grid(True)
    
    # 3. Параметры Калмана (первые preview_points точек)
    plt.subplot(3, 2, 3)
    plt.plot(P_ekf[:preview_points], 'b-', label='Ковариация P', linewidth=2)
    plt.plot(K[:preview_points], 'r-', label='Коэффициент K', linewidth=2)
    plt.xlabel('Время, k'); plt.ylabel('Значение')
    plt.title('ПАРАМЕТРЫ КАЛМАНА')
    plt.legend(); plt.grid(True)
    
    # 4. Частотная характеристика Винера
    plt.subplot(3, 2, 4)
    freq = fftfreq(N)
    H_magnitude = np.abs(wiener_params['H_freq_response'])
    plt.plot(freq[:N//2], H_magnitude[:N//2], 'g-', linewidth=2)
    plt.xlabel('Частота'); plt.ylabel('|H(f)|')
    plt.title('ЧАСТОТНАЯ ХАРАКТЕРИСТИКА ВИНЕРА')
    plt.grid(True)
    
    # 5. Оценка состояния Калманом
    plt.subplot(3, 2, 5)
    plt.plot(lambda_true[:preview_points], 'b-', label='Истинное x', linewidth=2)
    plt.plot(x_hat[:preview_points], 'm-', label='Оценка Калмана', linewidth=1.5)
    plt.xlabel('Время, k'); plt.ylabel('Состояние')
    plt.title('ОЦЕНКА СОСТОЯНИЯ КАЛМАНОМ')
    plt.legend(); plt.grid(True)
    
    # 6. Распределение ошибок (все данные)
    plt.subplot(3, 2, 6)
    plt.hist(lambda_wiener - lambda_true, bins=50, alpha=0.5, label='Ошибки Винера', color='green', density=True)
    plt.hist(lambda_ekf - lambda_true, bins=50, alpha=0.5, label='Ошибки Калмана', color='magenta', density=True)
    plt.xlabel('Ошибка'); plt.ylabel('Плотность вероятности')
    plt.title('Распределение ошибок')
    plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "filter_analysis.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_extended_analysis(acf_values, noise_variation_df, lambda_true, y_observed, 
                           lambda_wiener, lambda_ekf, snr_metrics):
    """Дополнительный анализ: автокорреляция, SNR, MSE, сравнение фильтров."""
    
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    plt.stem(range(len(acf_values)), acf_values, basefmt=" ")
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Лаг'); plt.ylabel('ACF'); plt.title('АВТОКОРРЕЛЯЦИОННАЯ ФУНКЦИЯ y'); plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(noise_variation_df['alpha'], noise_variation_df['improvement_wiener'], 'g-', label='Винер', linewidth=2)
    plt.plot(noise_variation_df['alpha'], noise_variation_df['improvement_ekf'], 'm-', label='Калман', linewidth=2)
    plt.xlabel('α'); plt.ylabel('Улучшение SNR')
    plt.title('УЛУЧШЕНИЕ SNR ПРИ ВАРИАЦИИ ШУМА'); plt.legend(); plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(lambda_true[:100], 'b-', label='Истинный λ', linewidth=2)
    plt.plot(y_observed[:100], 'r-', label='Наблюдения y', alpha=0.5)
    plt.plot(lambda_wiener[:100], 'g-', label='Фильтр Винера')
    plt.plot(lambda_ekf[:100], 'm-', label='Фильтр Калмана')
    plt.xlabel('Время, k'); plt.ylabel('Амплитуда')
    plt.title('СРАВНЕНИЕ ФИЛЬТРАЦИИ'); plt.legend(); plt.grid(True)
    
    # 4. MSE при вариации шума
    plt.subplot(2, 3, 4)
    plt.semilogy(noise_variation_df['alpha'], noise_variation_df['mse_wiener'], 'g-', label='Винер', linewidth=2)
    plt.semilogy(noise_variation_df['alpha'], noise_variation_df['mse_ekf'], 'm-', label='Калман', linewidth=2)
    plt.xlabel('α'); plt.ylabel('MSE (лог. шкала)'); plt.title('MSE ПРИ ВАРИАЦИИ ШУМА'); plt.legend(); plt.grid(True)
    
    # 5. SNR на выходе при вариации шума
    plt.subplot(2, 3, 5)
    plt.semilogy(noise_variation_df['alpha'], noise_variation_df['snr_out_wiener'], 'g-', label='Винер', linewidth=2)
    plt.semilogy(noise_variation_df['alpha'], noise_variation_df['snr_out_ekf'], 'm-', label='Калман', linewidth=2)
    plt.semilogy(noise_variation_df['alpha'], noise_variation_df['snr_in'], 'k--', label='Входной SNR', alpha=0.7)
    plt.xlabel('α'); plt.ylabel('SNR (лог. шкала)'); plt.title('SNR ПРИ ВАРИАЦИИ ШУМА'); plt.legend(); plt.grid(True)
    
    # 6. Сравнение производительности фильтров
    plt.subplot(2, 3, 6)
    metrics = ['Улучшение SNR', 'MSE', 'SNR на выходе']
    wiener_values = [snr_metrics['improvement_wiener'], snr_metrics['mse_wiener'], snr_metrics['snr_out_wiener']]
    ekf_values = [snr_metrics['improvement_ekf'], snr_metrics['mse_ekf'], snr_metrics['snr_out_ekf']]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    plt.bar(x_pos - width/2, wiener_values, width, label='Винер', color='green', alpha=0.7)
    plt.bar(x_pos + width/2, ekf_values, width, label='Калман', color='magenta', alpha=0.7)
    plt.xlabel('Метрика'); plt.ylabel('Значение'); plt.title('СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ')
    plt.xticks(x_pos, metrics); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "extended_analysis.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_snr_steps(lambda_true, lambda_ekf, first_steps=50):
    """График SNR на выходе Калмана по шагам."""

    eps = 1e-6
    signal_power = np.mean(lambda_true**2)
    error_ekf_sq = (lambda_ekf - lambda_true)**2
    snr_out_kalman = signal_power / np.maximum(error_ekf_sq, eps)
    
    plt.figure(figsize=(10, 5))
    plt.plot(snr_out_kalman[:first_steps], 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Шаг k'); plt.ylabel('SNR на выходе')
    plt.title('SNR на выходе фильтра Калмана (первые 50 шагов)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "kalman_snr.png"), dpi=300)
    plt.show()
    
    # Возвращаем полный массив SNR для Excel/анализа
    return snr_out_kalman