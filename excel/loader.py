# io/loader.py

import numpy as np
import pandas as pd


def load_or_generate_noise(
    filename: str = "variant.xlsx",
    n_points: int = 1000,
    var_n1: float = 0.78874,
    var_n0: float = 0.79123
):
    """
    Loads n1 and n0 from Excel.
    If the file is missing or contains insufficient data, fills in / generates the missing values.
    """

    try:
        data = pd.read_excel(filename)
        n1 = data["n1"].values
        n0 = data["n0"].values

        if len(n1) < n_points:
            additional = n_points - len(n1)
            n1_add = np.random.normal(0, np.sqrt(var_n1), additional)
            n0_add = np.random.normal(0, np.sqrt(var_n0), additional)
            n1 = np.concatenate([n1, n1_add])
            n0 = np.concatenate([n0, n0_add])

        return n1[:n_points], n0[:n_points]

    except Exception:
        n1 = np.random.normal(0, np.sqrt(var_n1), n_points)
        n0 = np.random.normal(0, np.sqrt(var_n0), n_points)

        df = pd.DataFrame({"n1": n1, "n0": n0})
        df.to_excel(filename, index=False)

        return n1, n0
