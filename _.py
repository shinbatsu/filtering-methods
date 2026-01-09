import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fftfreq

# Основная программа
print("=== РАСШИРЕННАЯ СИСТЕМА ФИЛЬТРАЦИИ С АНАЛИЗОМ SNR И ACF ===\n")

# Параметры
a = 0.813
N = 1000

print(f"РАБОТА С {N} ЗНАЧЕНИЯМИ...")

# Загрузка или генерация данных
n1_data, n0_data = ('variant.xlsx', N)

print("ВЫЧИСЛЕНИЕ ИСХОДНОГО ПРОЦЕССА...")

# Вычисление исходного процесса
x_true = np.zeros(N)
x_true[0] = n1_data[0]
for k in range(1, N):
    x_true[k] = a * x_true[k-1] + n1_data[k]

lambda_true = nonlinear_f(x_true)
y_observed = lambda_true + n0_data

print("ПРИМЕНЕНИЕ ФИЛЬТРА ВИНЕРА...")
lambda_wiener, wiener_params = comprehensive_wiener_filter(y_observed, a)

print("ПРИМЕНЕНИЕ ФИЛЬТРА КАЛМАНА...")
lambda_ekf, x_hat, P_ekf, kalman_params, x_minus, P_minus, H_jac, K = comprehensive_kalman_filter(y_observed, n1_data, a)

# ВЫЧИСЛЕНИЕ ДОПОЛНИТЕЛЬНЫХ ПАРАМЕТРОВ
print("ВЫЧИСЛЕНИЕ ДОПОЛНИТЕЛЬНЫХ ПАРАМЕТРОВ...")

# 1. Метрики SNR и улучшения
snr_metrics = calculate_snr_metrics(lambda_true, y_observed, lambda_wiener, lambda_ekf, n0_data)

# 2. Автокорреляционная функция
acf_values = calculate_acf(y_observed, max_lag=20)

# 3. Анализ вариации дисперсии шума
noise_variation_df = analyze_noise_variation(lambda_true, n1_data, n0_data, a )

# Базовые статистики
var_n1 = np.var(n1_data)
var_n0 = np.var(n0_data)
var_x_true = np.var(x_true)
var_lambda_true = np.var(lambda_true)
var_y = np.var(y_observed)

# СОЗДАНИЕ ПОДРОБНОГО EXCEL ФАЙЛА
print("\nСОЗДАНИЕ ПОДРОБНОГО EXCEL ФАЙЛА...")

# Основные данные (первые 100 точек)
preview_points = min(100, N)
main_data_preview = pd.DataFrame({
    'k': np.arange(preview_points),
    'n1': n1_data[:preview_points], 'n0': n0_data[:preview_points],
    'x_true': x_true[:preview_points], 'lambda_true': lambda_true[:preview_points],
    'y_observed': y_observed[:preview_points], 'lambda_wiener': lambda_wiener[:preview_points],
    'lambda_ekf': lambda_ekf[:preview_points], 'x_hat_ekf': x_hat[:preview_points],
    'P_ekf': P_ekf[:preview_points], 'x_minus_ekf': x_minus[:preview_points],
    'P_minus_ekf': P_minus[:preview_points], 'H_jac_ekf': H_jac[:preview_points],
    'K_ekf': K[:preview_points]
})

# Полные данные (все 1000 точек) - ДОБАВЛЯЕМ НОВЫЕ ПЕРЕМЕННЫЕ
main_data_full = pd.DataFrame({
    'k': np.arange(N), 'n1': n1_data, 'n0': n0_data,
    'x_true': x_true, 'lambda_true': lambda_true, 'y_observed': y_observed,
    'lambda_wiener': lambda_wiener, 'lambda_ekf': lambda_ekf,
    'x_hat_ekf': x_hat, 'P_ekf': P_ekf,
    'x_minus_ekf': x_minus, 'P_minus_ekf': P_minus,  # ДОБАВЛЕНО
    'H_jac_ekf': H_jac, 'K_ekf': K                   # ДОБАВЛЕНО
})

# Параметры фильтров
wiener_params_df = pd.DataFrame({
    'Параметр': [
        'a', 'b = a²', 'R = Var(n0)', 'Var(n1)',
        'Var(x)', 'Var(λ)', 'Var(λv)', 'MSE Винера'
    ],
    'Значение': [
        wiener_params['a'],
        wiener_params['b'],
        wiener_params['R'],
        wiener_params['var_n1'],
        wiener_params['var_x'],
        wiener_params['var_lambda'],
        wiener_params['var_lambda_v'],
        snr_metrics['mse_wiener']
    ],
    'Описание': [
        'Параметр линейного преобразования',
        'Параметр модели для спектральной плотности',
        'Дисперсия шума наблюдения',
        'Дисперсия формирующего шума',
        'Дисперсия процесса x',
        'Дисперсия процесса λ',
        'Дисперсия выхода фильтра Винера',
        'Среднеквадратичная ошибка'
    ]
})

kalman_params_df = pd.DataFrame({
    'Параметр': [
        'a', 'Q = Var(n1)', 'R = Var(n0)',
        'x₀', 'P₀', 'Var(λk)', 'P_final', 'MSE Калмана'
    ],
    'Значение': [
        kalman_params['a'],
        kalman_params['Q'],
        kalman_params['R'],
        kalman_params['x0'],
        kalman_params['P0'],
        kalman_params['var_lambda_ekf'],
        kalman_params['final_P'],
        snr_metrics['mse_ekf']
    ],
    'Описание': [
        'Параметр линейного преобразования',
        'Дисперсия шума процесса',
        'Дисперсия шума наблюдения',
        'Начальная оценка состояния',
        'Начальная ковариация',
        'Дисперсия выхода фильтра Калмана',
        'Финальная ковариация',
        'Среднеквадратичная ошибка'
    ]
})

# НОВЫЙ ЛИСТ: Пояснения к переменным Калмана
kalman_variables_df = pd.DataFrame({
    'Переменная': ['x_minus_ekf', 'P_minus_ekf', 'H_jac_ekf', 'K_ekf'],
    'Описание': [
        'Априорная оценка состояния (прогноз до измерения)',
        'Априорная ковариация ошибки',
        'Якобиан функции наблюдения',
        'Коэффициент усиления Калмана'
    ],
    'Формула': [
        'x_minus = a · x_hat[k-1]',
        'P_minus = a² · P[k-1] + Q',
        'H_jac = 2 · x_minus',
        'K = P_minus · H_jac / (H_jac² · P_minus + R)'
    ]
})

# НОВЫЙ ЛИСТ: SNR и улучшение
snr_analysis_df = pd.DataFrame({
    'Метрика': [
        'Мощность сигнала E[λ²]',
        'Дисперсия шума наблюдения Var(n0)',
        'SNR на входе (Signal_power / Var(n0))',
        'MSE фильтра Винера',
        'SNR на выходе Винера',
        'Коэффициент улучшения Винера',
        'MSE фильтра Калмана',
        'SNR на выходе Калмана',
        'Коэффициент улучшения Калмана'
    ],
    'Значение': [
        snr_metrics['signal_power'],
        var_n0,
        snr_metrics['snr_in'],
        snr_metrics['mse_wiener'],
        snr_metrics['snr_out_wiener'],
        snr_metrics['improvement_wiener'],
        snr_metrics['mse_ekf'],
        snr_metrics['snr_out_ekf'],
        snr_metrics['improvement_ekf']
    ],
    'Описание': [
        'Средняя мощность сигнала λ',
        'Дисперсия шума наблюдения',
        'Отношение сигнал/шум до фильтрации',
        'Среднеквадратичная ошибка фильтра Винера',
        'Отношение сигнал/шум после фильтра Винера',
        'Во сколько раз улучшилось SNR после фильтра Винера',
        'Среднеквадратичная ошибка фильтра Калмана',
        'Отношение сигнал/шум после фильтра Калмана',
        'Во сколько раз улучшилось SNR после фильтра Калмана'
    ]
})

# НОВЫЙ ЛИСТ: Автокорреляционная функция
acf_df = pd.DataFrame({
    'Lag': range(21),
    'ACF': acf_values,
    'Описание': ['Автокорреляция с лагом ' + str(i) for i in range(21)]
})

# НОВЫЙ ЛИСТ: Статистика процессов
process_stats_df = pd.DataFrame({
    'Процесс': ['n1', 'n0', 'x_true', 'lambda_true', 'y_observed', 'lambda_wiener', 'lambda_ekf'],
    'Дисперсия': [var_n1, var_n0, var_x_true, var_lambda_true, var_y,
                 wiener_params['var_lambda_v'], kalman_params['var_lambda_ekf']],
    'Среднее': [np.mean(n1_data), np.mean(n0_data), np.mean(x_true), np.mean(lambda_true),
               np.mean(y_observed), np.mean(lambda_wiener), np.mean(lambda_ekf)],
    'Стандартное отклонение': [np.std(n1_data), np.std(n0_data), np.std(x_true), np.std(lambda_true),
                              np.std(y_observed), np.std(lambda_wiener), np.std(lambda_ekf)]
})

# Формулы фильтров
formulas_df = pd.DataFrame({
    'Фильтр': ['Винер', 'Калман'],
    'Формулы': [
        'PSD_λ(f) = Var(λ)·(1-b²)/|1-b·exp(-j2πf)|²\n'
        'H(f) = PSD_λ(f)/(PSD_λ(f)+R)\n'
        'λv = real(IFFT(H·FFT(y)))',

        'x_minus = a·x_hat_{k-1}\n'
        'P_minus = a²·P_{k-1}+Q\n'
        'H_jac = 2·x_minus\n'
        'K = P_minus·H_jac/(H_jac²·P_minus+R)\n'
        'x_hat_k = x_minus + K·(y_k - x_minus²)\n'
        'P_k = P_minus - K·H_jac·P_minus\n'
        'λk = x_hat_k²'
    ]
})

# Создание Excel файла
output_filename = f'analysis_{N}_points.xlsx'

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    # Основные данные
    main_data_preview.to_excel(writer, sheet_name='Данные (первые 100)', index=False)
    main_data_full.to_excel(writer, sheet_name='Все данные', index=False)

    # Параметры фильтров
    wiener_params_df.to_excel(writer, sheet_name='Параметры Винера', index=False)
    kalman_params_df.to_excel(writer, sheet_name='Параметры Калмана', index=False)
    kalman_variables_df.to_excel(writer, sheet_name='Переменные Калмана', index=False)  # НОВЫЙ ЛИСТ

    # НОВЫЕ ЛИСТЫ
    snr_analysis_df.to_excel(writer, sheet_name='SNR анализ', index=False)
    acf_df.to_excel(writer, sheet_name='Автокорреляция', index=False)
    noise_variation_df.to_excel(writer, sheet_name='Вариация шума', index=False)

    # Статистика и формулы
    process_stats_df.to_excel(writer, sheet_name='Статистика процессов', index=False)
    formulas_df.to_excel(writer, sheet_name='Формулы фильтров', index=False)

print(f"Файл создан: '{output_filename}'")

# СОЗДАНИЕ ПЕРВОГО ФАЙЛА ГРАФИКОВ (comprehensive_filter_analysis)
print("\nСОЗДАНИЕ ПЕРВОГО ФАЙЛА ГРАФИКОВ...")

plt.figure(figsize=(20, 15))

# График 1: Сравнение фильтрации (первые 100 точек)
plt.subplot(3, 2, 1)
plt.plot(lambda_true[:100], 'b-', label='Истинный λ', linewidth=2)
plt.plot(y_observed[:100], 'r-', label='Наблюдения y', alpha=0.7, linewidth=1)
plt.plot(lambda_wiener[:100], 'g-', label='Фильтр Винера', linewidth=1.5)
plt.plot(lambda_ekf[:100], 'm-', label='Фильтр Калмана', linewidth=1.5)
plt.xlabel('Время, k')
plt.ylabel('Амплитуда')
plt.title('СРАВНЕНИЕ ФИЛЬТРАЦИИ (первые 100 точек)')
plt.legend()
plt.grid(True)

# График 2: Ошибка фильтрации
plt.subplot(3, 2, 2)
error_wiener = lambda_wiener - lambda_true
error_ekf = lambda_ekf - lambda_true
plt.plot(error_wiener[:100], 'g-', label='Ошибка Винера', alpha=0.7)
plt.plot(error_ekf[:100], 'm-', label='Ошибка Калмана', alpha=0.7)
plt.xlabel('Время, k')
plt.ylabel('Ошибка')
plt.title('Ошибка фильтрации')
plt.legend()
plt.grid(True)

# График 3: Параметры Калмана (первые 100 точек)
plt.subplot(3, 2, 3)
plt.plot(P_ekf[:100], 'b-', label='Ковариация P', linewidth=2)
plt.plot(K[:100], 'r-', label='Коэффициент K', linewidth=2)
plt.xlabel('Время, k')
plt.ylabel('Значение')
plt.title('ПАРАМЕТРЫ КАЛМАНА (первые 100 точек)')
plt.legend()
plt.grid(True)

# График 4: Частотная характеристика Винера
plt.subplot(3, 2, 4)
freq = fftfreq(N)
H_magnitude = np.abs(wiener_params['H_freq_response'])
plt.plot(freq[:N//2], H_magnitude[:N//2], 'g-', linewidth=2)
plt.xlabel('Частота')
plt.ylabel('|H(f)|')
plt.title('ЧАСТОТНАЯ ХАРАКТЕРИСТИКА ВИНЕРА')
plt.grid(True)

# График 5: Оценка состояния Калманом (первые 100 точек)
plt.subplot(3, 2, 5)
plt.plot(x_true[:100], 'b-', label='Истинное x', linewidth=2)
plt.plot(x_hat[:100], 'm-', label='Оценка Калмана', linewidth=1.5)
plt.xlabel('Время, k')
plt.ylabel('Состояние')
plt.title('ОЦЕНКА СОСТОЯНИЯ КАЛМАНОМ (первые 100 точек)')
plt.legend()
plt.grid(True)

# График 6: Распределение ошибок (все данные)
plt.subplot(3, 2, 6)
errors_wiener = lambda_wiener - lambda_true
errors_ekf = lambda_ekf - lambda_true
plt.hist(errors_wiener, bins=50, alpha=0.5, label='Ошибки Винера', color='green', density=True)
plt.hist(errors_ekf, bins=50, alpha=0.5, label='Ошибки Калмана', color='magenta', density=True)
plt.xlabel('Ошибка')
plt.ylabel('Плотность вероятности')
plt.title('Распределение ошибок (все данные)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'comprehensive_filter_analysis_{N}_points.png', dpi=300, bbox_inches='tight')
plt.show()

# СОЗДАНИЕ ВТОРОГО ФАЙЛА ГРАФИКОВ (extended_analysis)
print("\nСОЗДАНИЕ ВТОРОГО ФАЙЛА ГРАФИКОВ...")

plt.figure(figsize=(18, 12))

# График 1: Автокорреляционная функция
plt.subplot(2, 3, 1)
plt.stem(range(21), acf_values, basefmt=" ")
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('Лаг')
plt.ylabel('ACF')
plt.title('АВТОКОРРЕЛЯЦИОННАЯ ФУНКЦИЯ y')
plt.grid(True)

# График 2: Улучшение SNR при вариации шума
plt.subplot(2, 3, 2)
plt.plot(noise_variation_df['alpha'], noise_variation_df['improvement_wiener'],
         'g-', label='Винер', linewidth=2)
plt.plot(noise_variation_df['alpha'], noise_variation_df['improvement_ekf'],
         'm-', label='Калман', linewidth=2)
plt.xlabel('Коэффициент усиления шума (α)')
plt.ylabel('Улучшение SNR')
plt.title('УЛУЧШЕНИЕ SNR ПРИ ВАРИАЦИИ ШУМА')
plt.legend()
plt.grid(True)

# График 3: Сравнение фильтрации (первые 100 точек)
plt.subplot(2, 3, 3)
plt.plot(lambda_true[:100], 'b-', label='Истинный λ', linewidth=2)
plt.plot(y_observed[:100], 'r-', label='Наблюдения y', alpha=0.5)
plt.plot(lambda_wiener[:100], 'g-', label='Фильтр Винера')
plt.plot(lambda_ekf[:100], 'm-', label='Фильтр Калмана')
plt.xlabel('Время, k')
plt.ylabel('Амплитуда')
plt.title('СРАВНЕНИЕ ФИЛЬТРАЦИИ')
plt.legend()
plt.grid(True)

# График 4: MSE при вариации шума
plt.subplot(2, 3, 4)
plt.semilogy(noise_variation_df['alpha'], noise_variation_df['mse_wiener'],
            'g-', label='Винер', linewidth=2)
plt.semilogy(noise_variation_df['alpha'], noise_variation_df['mse_ekf'],
            'm-', label='Калман', linewidth=2)
plt.xlabel('Коэффициент усиления шума (α)')
plt.ylabel('MSE (лог. шкала)')
plt.title('MSE ПРИ ВАРИАЦИИ ШУМА')
plt.legend()
plt.grid(True)

# График 5: SNR на выходе при вариации шума
plt.subplot(2, 3, 5)
plt.semilogy(noise_variation_df['alpha'], noise_variation_df['snr_out_wiener'],
            'g-', label='Винер', linewidth=2)
plt.semilogy(noise_variation_df['alpha'], noise_variation_df['snr_out_ekf'],
            'm-', label='Калман', linewidth=2)
plt.semilogy(noise_variation_df['alpha'], noise_variation_df['snr_in'],
            'k--', label='Входной SNR', alpha=0.7)
plt.xlabel('Коэффициент усиления шума (α)')
plt.ylabel('SNR (лог. шкала)')
plt.title('SNR ПРИ ВАРИАЦИИ ШУМА')
plt.legend()
plt.grid(True)

# График 6: Сравнение производительности
plt.subplot(2, 3, 6)
metrics = ['Улучшение SNR', 'MSE', 'SNR на выходе']
wiener_values = [snr_metrics['improvement_wiener'], snr_metrics['mse_wiener'], snr_metrics['snr_out_wiener']]
ekf_values = [snr_metrics['improvement_ekf'], snr_metrics['mse_ekf'], snr_metrics['snr_out_ekf']]

x_pos = np.arange(len(metrics))
width = 0.35

plt.bar(x_pos - width/2, wiener_values, width, label='Винер', color='green', alpha=0.7)
plt.bar(x_pos + width/2, ekf_values, width, label='Калман', color='magenta', alpha=0.7)
plt.xlabel('Метрика')
plt.ylabel('Значение')
plt.title('СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ')
plt.xticks(x_pos, metrics)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'extended_analysis_{N}_points.png', dpi=300, bbox_inches='tight')
plt.show()

# ВЫВОД СВОДКИ
print("\n" + "="*70)
print("ПОЛНАЯ СВОДКА РЕЗУЛЬТАТОВ")
print("="*70)

print("\nОСНОВНЫЕ СТАТИСТИКИ:")
print(f"Количество точек: {N}")
print(f"Мощность сигнала E[λ²]: {snr_metrics['signal_power']:.6f}")
print(f"Дисперсия шума наблюдения: {var_n0:.6f}")
print(f"SNR на входе: {snr_metrics['snr_in']:.2f}")

print("\nПРОИЗВОДИТЕЛЬНОСТЬ ФИЛЬТРОВ:")
print(f"MSE Винера: {snr_metrics['mse_wiener']:.8f}")
print(f"MSE Калмана: {snr_metrics['mse_ekf']:.8f}")
print(f"Улучшение SNR Винера: {snr_metrics['improvement_wiener']:.2f} раз")
print(f"Улучшение SNR Калмана: {snr_metrics['improvement_ekf']:.2f} раз")

better_filter = "Винер" if snr_metrics['improvement_wiener'] > snr_metrics['improvement_ekf'] else "Калман"
print(f"\nЛУЧШИЙ ФИЛЬТР ПО УЛУЧШЕНИЮ SNR: {better_filter}")

print("\nАНАЛИЗ АВТОКОРРЕЛЯЦИИ:")
print(f"ACF(0): {acf_values[0]:.4f}")
print(f"ACF(1): {acf_values[1]:.4f}")
print(f"ACF(5): {acf_values[5]:.4f}")
print(f"ACF(10): {acf_values[10]:.4f}")

print("\nФАЙЛЫ СОХРАНЕНЫ:")
print(f"Данные: '{output_filename}'")
print(f"График 1: 'comprehensive_filter_analysis_{N}_points.png'")
print(f"График 2: 'extended_analysis_{N}_points.png'")

print("\nНОВЫЕ ЛИСТЫ В EXCEL:")
print("1. 'SNR анализ' - отношения сигнал/шум и улучшения")
print("2. 'Автокорреляция' - ACF значений с лагами 0-20")
print("3. 'Вариация шума' - анализ при изменении дисперсии шума")
print("4. 'Переменные Калмана' - пояснения к x_minus, P_minus, H_jac, K")

print("\nДОПОЛНИТЕЛЬНЫЕ ПЕРЕМЕННЫЕ КАЛМАНА:")
print("✓ x_minus_ekf - априорная оценка состояния")
print("✓ P_minus_ekf - априорная ковариация ошибки")
print("✓ H_jac_ekf - матрица Якоби наблюдения")
print("✓ K_ekf - коэффициент усиления Калмана")

# ============================================================
# ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ: SNR НА ВЫХОДЕ КАЛМАНА ПО ШАГАМ
# ============================================================

print("\nРАСЧЁТ SNR НА ВЫХОДЕ ФИЛЬТРА КАЛМАНА ПО ШАГАМ...")

eps = 1e-6  # защита от деления на ноль

# Мощность сигнала
signal_power = np.mean(lambda_true**2)

# Ошибка фильтра Калмана по шагам
error_ekf_sq = (lambda_ekf - lambda_true)**2

# SNR на выходе фильтра Калмана по шагам
snr_out_kalman = signal_power / np.maximum(error_ekf_sq, eps)

snr_kalman_df = pd.DataFrame({
    'k': np.arange(N),
    'lambda_true': lambda_true,
    'lambda_ekf': lambda_ekf,
    'Ошибка²': error_ekf_sq,
    'SNR_выход_Калмана': snr_out_kalman,
    'K_Калмана': K
})

# Асимптотическое значение SNR (усреднение последних 30% шагов)
steady_start = int(0.7 * N)
snr_out_steady = np.mean(snr_out_kalman[steady_start:])
print(f"Асимптотическое значение SNR на выходе Калмана: {snr_out_steady:.2f}")

# Средний коэффициент усиления Калмана
# Убираем нули, чтобы ⟨K⟩ было реалистичным
K_nonzero = K[K > 1e-6]  # фильтруем практически нулевые
K_mean_kalman = np.mean(K_nonzero) if len(K_nonzero) > 0 else np.mean(K)
K_mean_wiener = np.mean(np.abs(wiener_params['H_freq_response']))

print(f"Средний коэффициент Калмана ⟨K⟩: {K_mean_kalman:.4f}")
print(f"Средний коэффициент Винера ⟨H⟩: {K_mean_wiener:.4f}")

# График первых 50 шагов
plt.figure(figsize=(10, 5))
plt.plot(snr_out_kalman[:50], 'b-o', linewidth=2, markersize=4)
plt.xlabel('Шаг k')
plt.ylabel('SNR на выходе')
plt.title('SNR на выходе фильтра Калмана (первые 50 шагов)')
plt.grid(True)
plt.tight_layout()
plt.savefig('snr_out_kalman_first_50.png', dpi=300)
plt.show()

# Сохраняем в Excel
with pd.ExcelWriter(output_filename, engine='openpyxl', mode='a') as writer:
    snr_kalman_df.to_excel(writer, sheet_name='SNR_Калмана_по_шагам', index=False)
