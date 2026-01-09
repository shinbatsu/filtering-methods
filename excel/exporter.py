# io/excel_export.py
import pandas as pd
import numpy as np
from config import output_folder
import os

def export_full_analysis(
    N,
    preview_points,
    n1_data,
    n0_data,
    x_true,
    lambda_true,
    y_observed,
    lambda_wiener,
    lambda_ekf,
    x_hat,
    P_ekf,
    x_minus,
    P_minus,
    H_jac,
    K,
    wiener_params,
    kalman_params,
    snr_metrics,
    acf_values,
    noise_variation_df,
    var_n1,
    var_n0,
    var_x_true,
    var_lambda_true,
    var_y
):

    main_data_preview = pd.DataFrame({
        'k': np.arange(preview_points),
        'n1': n1_data[:preview_points],
        'n0': n0_data[:preview_points],
        'x_true': x_true[:preview_points],
        'lambda_true': lambda_true[:preview_points],
        'y_observed': y_observed[:preview_points],
        'lambda_wiener': lambda_wiener[:preview_points],
        'lambda_ekf': lambda_ekf[:preview_points],
        'x_hat_ekf': x_hat[:preview_points],
        'P_ekf': P_ekf[:preview_points],
        'x_minus_ekf': x_minus[:preview_points],
        'P_minus_ekf': P_minus[:preview_points],
        'H_jac_ekf': H_jac[:preview_points],
        'K_ekf': K[:preview_points]
    })

    main_data_full = pd.DataFrame({
        'k': np.arange(N),
        'n1': n1_data,
        'n0': n0_data,
        'x_true': x_true,
        'lambda_true': lambda_true,
        'y_observed': y_observed,
        'lambda_wiener': lambda_wiener,
        'lambda_ekf': lambda_ekf,
        'x_hat_ekf': x_hat,
        'P_ekf': P_ekf,
        'x_minus_ekf': x_minus,
        'P_minus_ekf': P_minus,
        'H_jac_ekf': H_jac,
        'K_ekf': K
    })

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
        ]
    })

    acf_df = pd.DataFrame({
        'Lag': np.arange(len(acf_values)),
        'ACF': acf_values,
        'Описание': ['Автокорреляция с лагом ' + str(i) for i in range(len(acf_values))]
    })

    process_stats_df = pd.DataFrame({
        'Процесс': ['n1', 'n0', 'x_true', 'lambda_true', 'y_observed', 'lambda_wiener', 'lambda_ekf'],
        'Дисперсия': [var_n1, var_n0, var_x_true, var_lambda_true, var_y,
                     wiener_params['var_lambda_v'], kalman_params['var_lambda_ekf']],
        'Среднее': [np.mean(n1_data), np.mean(n0_data), np.mean(x_true), np.mean(lambda_true),
                   np.mean(y_observed), np.mean(lambda_wiener), np.mean(lambda_ekf)],
        'Стандартное отклонение': [np.std(n1_data), np.std(n0_data), np.std(x_true), np.std(lambda_true),
                                  np.std(y_observed), np.std(lambda_wiener), np.std(lambda_ekf)]
    })

    formulas_df = pd.DataFrame({
        'Фильтр': ['Винер', 'Калман'],
        'Формулы': [
            'PSD_λ(f) = Var(λ)·(1-b²)/|1-b·exp(-j2πf)|²\nH(f) = PSD_λ(f)/(PSD_λ(f)+R)\nλv = real(IFFT(H·FFT(y)))',
            'x_minus = a·x_hat_{k-1}\nP_minus = a²·P_{k-1}+Q\nH_jac = 2·x_minus\nK = P_minus·H_jac/(H_jac²·P_minus+R)\nx_hat_k = x_minus + K·(y_k - x_minus²)\nP_k = P_minus - K·H_jac·P_minus\nλk = x_hat_k²'
        ]
    })

    with pd.ExcelWriter(os.path.join(output_folder,"data.xlsx"), engine='openpyxl') as writer:
        main_data_preview.to_excel(writer, sheet_name='Данные (первые 100)', index=False)
        main_data_full.to_excel(writer, sheet_name='Все данные', index=False)
        wiener_params_df.to_excel(writer, sheet_name='Параметры Винера', index=False)
        kalman_params_df.to_excel(writer, sheet_name='Параметры Калмана', index=False)
        kalman_variables_df.to_excel(writer, sheet_name='Переменные Калмана', index=False)
        snr_analysis_df.to_excel(writer, sheet_name='SNR анализ', index=False)
        acf_df.to_excel(writer, sheet_name='Автокорреляция', index=False)
        noise_variation_df.to_excel(writer, sheet_name='Вариация шума', index=False)
        process_stats_df.to_excel(writer, sheet_name='Статистика процессов', index=False)
        formulas_df.to_excel(writer, sheet_name='Формулы фильтров', index=False)
