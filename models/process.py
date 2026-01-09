import numpy as np

from models.nonlinear import nonlinear_f
from excel.loader import load_or_generate_noise
from .nonlinear import nonlinear_f

def generate_process(n1_file: str, N: int, a: float):
    n1_data, n0_data = load_or_generate_noise(n1_file, N)

    x_true = np.zeros(N)
    x_true[0] = n1_data[0]
    for k in range(1, N):
        x_true[k] = a * x_true[k-1] + n1_data[k]

    lambda_true = nonlinear_f(x_true)
    y_observed = lambda_true + n0_data

    return n1_data, n0_data, x_true, lambda_true, y_observed
