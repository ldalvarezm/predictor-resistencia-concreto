"""App de Streamlit: predictor de resistencia a compresión del concreto."""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBRegressor

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.features import BASE, COMPONENTES, DERIVADAS, agregar_variables  # noqa: E402

TODAS = BASE + DERIVADAS
RMSE_TIPICO = 5.6                      # MPa, validación cruzada por grupos de mezcla
MASA_MIN, MASA_MAX = 2194.6, 2551.0    # kg/m³ observados en el dataset
EDADES_CURVA = [3, 7, 14, 28, 56, 90, 180, 365]

st.set_page_config(page_title="Predictor de resistencia del concreto", layout="wide")


@st.cache_resource
def cargar_modelo():
    modelo = XGBRegressor()
    modelo.load_model(RAIZ / "models" / "xgb_final.json")
    return modelo


def predecir(modelo, filas):
    datos = agregar_variables(pd.DataFrame(filas))
    return modelo.predict(datos[TODAS])


modelo = cargar_modelo()

st.title("Predictor de resistencia a compresión del concreto")
st.caption("Modelo XGBoost entrenado con el dataset de I-Cheng Yeh (UCI): "
           "1030 ensayos, 1005 tras eliminar duplicados exactos.")
st.warning("Herramienta educativa y de investigación. No sustituye ensayos de "
           "laboratorio ni el diseño de mezclas según norma.")

with st.sidebar:
    st.header("Dosificación (kg/m³)")
    mezcla = {
        "cemento": st.slider("Cemento", 102.0, 540.0, 350.0, 5.0),
        "escoria": st.slider("Escoria de alto horno", 0.0, 360.0, 0.0, 5.0),
        "ceniza": st.slider("Ceniza volante", 0.0, 200.0, 0.0, 5.0),
        "agua": st.slider("Agua", 122.0, 247.0, 180.0, 1.0),
        "superplast": st.slider("Superplastificante", 0.0, 32.0, 0.0, 0.5),
        "agr_grueso": st.slider("Agregado grueso", 801.0, 1145.0, 1000.0, 5.0),
        "agr_fino": st.slider("Agregado fino", 594.0, 993.0, 800.0, 5.0),
    }
    edad = st.number_input("Edad (días)", min_value=3, max_value=365,
                           value=28, step=1)
mezcla["edad"] = int(edad)

pred = float(predecir(modelo, [mezcla])[0])
ligantes = mezcla["cemento"] + mezcla["escoria"] + mezcla["ceniza"]
a_lig = mezcla["agua"] / ligantes
masa = sum(mezcla[c] for c in COMPONENTES)

c1, c2, c3 = st.columns(3)
c1.metric("Resistencia estimada", f"{pred:.1f} MPa")
c2.metric("Agua / ligantes", f"{a_lig:.2f}")
c3.metric("Masa total", f"{masa:.0f} kg/m³")
st.caption(f"Error típico del modelo: ±{RMSE_TIPICO} MPa (RMSE en validación por "
           "grupos de mezcla). Es una referencia, no un intervalo de confianza calibrado.")

avisos = []
if not MASA_MIN <= masa <= MASA_MAX:
    avisos.append(f"La suma de componentes ({masa:.0f} kg/m³) queda fuera del rango "
                  f"de los datos de entrenamiento ({MASA_MIN:.0f}–{MASA_MAX:.0f}): "
                  "la dosificación puede no ser realista y la predicción es menos fiable.")
if a_lig > 0.75 or a_lig < 0.25:
    avisos.append("La relación agua/ligantes está en una zona con muy pocos datos "
                  "(fuera de 0.25–0.75).")
if edad < 7:
    avisos.append("Hay pocos ensayos a edades menores de 7 días: predicción menos fiable.")
if pred > 75:
    avisos.append("Por encima de ~75 MPa el modelo tiene muy pocos datos y tiende "
                  "a subestimar.")
for aviso in avisos:
    st.warning(aviso)

curva = predecir(modelo, [{**mezcla, "edad": e} for e in EDADES_CURVA])
fig = go.Figure()
fig.add_trace(go.Scatter(x=EDADES_CURVA, y=curva + RMSE_TIPICO, mode="lines",
                         line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=EDADES_CURVA, y=curva - RMSE_TIPICO, mode="lines",
                         line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(70,130,180,0.2)",
                         name=f"±{RMSE_TIPICO} MPa", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=EDADES_CURVA, y=curva, mode="lines+markers",
                         name="Predicción"))
fig.update_xaxes(type="log", title="Edad (días, escala log)",
                 tickvals=EDADES_CURVA,
                 ticktext=[str(e) for e in EDADES_CURVA])
fig.update_yaxes(title="Resistencia (MPa)", rangemode="tozero")
fig.update_layout(title="Desarrollo de resistencia de esta mezcla")
st.plotly_chart(fig, width="stretch")

with st.expander("Limitaciones del modelo"):
    st.markdown(
        "- Datos de concretos convencionales y de alta resistencia (2–83 MPa), "
        "no UHPC.\n"
        "- Entrenado con ensayos de 17 fuentes; no incluye tipo de cemento, "
        "curado ni tamaño de probeta.\n"
        "- Sin datos a 1–2 días y con pocos a más de 180 días.\n"
        "- Describe asociaciones del conjunto de datos, no efectos causales: "
        "cambiar una sola variable lejos de mezclas típicas es extrapolar."
    )