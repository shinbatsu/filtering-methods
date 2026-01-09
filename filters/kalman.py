import numpy as np
from models.nonlinear import nonlinear_f, nonlinear_f_derivative

def kalman_filter(y_observed, n1, n0, a=0.813, R=None):
    """
    Extended Kalman filter for λ = f(x) with saturation.

    Args:
        y_observed: Observed signal (np.array)
        n1: Process noise (np.array)
        n0: Observation noise (np.array)
        a: State transition coefficient
        R: Observation noise variance (if None, computed from n0)
    """
    N = len(y_observed)

    Q = np.var(n1)
    var_n0 = np.var(n0) if R is None else R

    x_hat = np.zeros(N)
    P = np.zeros(N)
    x_hat[0] = 0.0
    P[0] = Q / (1 - a**2)

    lambda_ekf = np.zeros(N)
    lambda_ekf[0] = nonlinear_f(x_hat[0])

    x_minus_arr = np.zeros(N)
    P_minus_arr = np.zeros(N)
    H_jac_arr = np.zeros(N)
    K_arr = np.zeros(N)

    for k in range(1, N):
        x_minus = a * x_hat[k-1]
        P_minus = a**2 * P[k-1] + Q

        H_jac = nonlinear_f_derivative(x_minus)

        denom = H_jac**2 * P_minus + var_n0
        K = 0.0 if denom == 0 else P_minus * H_jac / denom

        innovation = y_observed[k] - nonlinear_f(x_minus)
        x_hat[k] = x_minus + K * innovation
        P[k] = P_minus - K * H_jac * P_minus
        lambda_ekf[k] = nonlinear_f(x_hat[k])

        x_minus_arr[k] = x_minus
        P_minus_arr[k] = P_minus
        H_jac_arr[k] = H_jac
        K_arr[k] = K

    var_lambda_ekf = np.var(lambda_ekf)
    kalman_params = {
        'a': a,
        'Q': Q,
        'R': var_n0,
        'x0': x_hat[0],
        'P0': P[0],
        'var_lambda_ekf': var_lambda_ekf,
        'final_P': P[-1]
    }

    return lambda_ekf, x_hat, P, kalman_params, x_minus_arr, P_minus_arr, H_jac_arr, K_arr
