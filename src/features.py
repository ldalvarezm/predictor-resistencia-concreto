"""Variables derivadas para el predictor de resistencia del concreto."""
import numpy as np
import pandas as pd

COMPONENTES = ["cemento", "escoria", "ceniza", "agua",
               "superplast", "agr_grueso", "agr_fino"]
BASE = COMPONENTES + ["edad"]
DERIVADAS = ["ligantes", "a_lig", "a_c", "frac_adiciones",
             "frac_arena", "log_edad"]


def agregar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de df con las variables derivadas añadidas.

    Son transformaciones fila a fila, sin parámetros aprendidos de los datos,
    así que no introducen fuga de información entre entrenamiento y prueba.
    """
    faltan = [c for c in BASE if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas: {faltan}")

    out = df.copy()
    out["ligantes"] = out["cemento"] + out["escoria"] + out["ceniza"]
    out["a_lig"] = out["agua"] / out["ligantes"]
    out["a_c"] = out["agua"] / out["cemento"]
    out["frac_adiciones"] = (out["escoria"] + out["ceniza"]) / out["ligantes"]
    out["frac_arena"] = out["agr_fino"] / (out["agr_fino"] + out["agr_grueso"])
    out["log_edad"] = np.log(out["edad"])
    return out