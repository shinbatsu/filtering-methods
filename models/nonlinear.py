import numpy as np

def nonlinear_f(x):
    return np.where(
        x < -0.5, -0.5,
        np.where(x > 0.5, 0.5, x**3)
    )

def nonlinear_f_derivative(x):
    return np.where(np.abs(x) <= 0.5, 3 * x**2, 0.0)
