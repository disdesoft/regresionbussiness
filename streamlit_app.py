
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from pathlib import Path

# Compatibilidad para RMSE sin warnings
try:
    from sklearn.metrics import root_mean_squared_error as _rmse_fn
    def RMSE(y_true, y_pred):
        return _rmse_fn(y_true, y_pred)
except Exception:
    from sklearn.metrics import mean_squared_error as _mse_fn
    import numpy as np
    def RMSE(y_true, y_pred):
        return np.sqrt(_mse_fn(y_true, y_pred))


st.set_page_config(page_title="Inscripciones vs Ventas por Asesor", layout="wide")
st.markdown(
    "<style> .big-cta .stButton>button {font-size:1.2rem;padding:0.9rem 1.4rem;border-radius:12px;font-weight:700;box-shadow:0 4px 14px rgba(0,0,0,.15);} </style>",
    unsafe_allow_html=True
)
st.title("Inscripciones vs Ventas por Asesor — Análisis y Proyección")
st.caption("Programa: Ingeniería de Sistemas y Computación • Asignatura: Business Intelligence")
st.caption("Alumno: Fabián Valero • Docente: Ivon Forero")

@st.cache_data(show_spinner=False)
def load_excel(file_bytes_or_path):
    xls = pd.ExcelFile(file_bytes_or_path)
    sheets = {name: xls.parse(name) for name in xls.sheet_names}
    return sheets

def find_cols(patterns, columns):
    found = []
    for c in columns:
        s = str(c).lower()
        if any(p in s for p in patterns):
            found.append(c)
    return found

def infer_flags(df, col_ins_fecha, col_ins_valor, col_ins_num, col_matr_fecha, col_matr_flag, cols_valores_recibos):
    if col_ins_fecha and col_ins_fecha in df.columns:
        ins_flag = df[col_ins_fecha].notna()
    elif col_ins_valor and col_ins_valor in df.columns:
        ins_flag = pd.to_numeric(df[col_ins_valor], errors="coerce").fillna(0) > 0
    elif col_ins_num and col_ins_num in df.columns:
        ins_flag = df[col_ins_num].notna()
    else:
        ins_flag = pd.Series(False, index=df.index)

    if col_matr_flag and col_matr_flag in df.columns:
        v = df[col_matr_flag].astype(str).str.strip().str.lower()
        venta_flag = v.isin(["si", "sí", "1", "true", "verdadero", "x", "matriculado"])
    elif col_matr_fecha and col_matr_fecha in df.columns:
        venta_flag = df[col_matr_fecha].notna()
    elif cols_valores_recibos:
        sub = pd.DataFrame({c: pd.to_numeric(df.get(c), errors="coerce") for c in cols_valores_recibos if c in df.columns})
        venta_flag = (sub.fillna(0) > 0).any(axis=1) if len(sub.columns) else pd.Series(False, index=df.index)
    else:
        venta_flag = pd.Series(False, index=df.index)
    return ins_flag.fillna(False), venta_flag.fillna(False)

st.sidebar.header("1) Datos")
use_included = st.sidebar.checkbox("Usar base incluida", value=True, help="Usa el archivo base integrado en el proyecto.")
uploaded = st.sidebar.file_uploader("O sube otra base (.xlsx)", type=["xlsx"])

data_source = None
if uploaded is not None:
    data_source = uploaded
    st.sidebar.success("Usando base subida.")
elif use_included:
    included_path = "base_incluida.xlsx"
    if Path(included_path).exists():
        data_source = included_path
        st.sidebar.success("Usando base incluida.")
    else:
        st.sidebar.warning("No se encontró la base incluida. Sube un archivo o añade base_incluida.xlsx.")
else:
    st.info("Selecciona una fuente de datos para continuar.")

if data_source:
    try:
        sheets = load_excel(data_source)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()

    hoja = st.selectbox("Hoja a analizar", list(sheets.keys()))
    df = sheets[hoja].copy()
    cols = list(df.columns)

    st.sidebar.header("2) Mapeo de columnas")
    sugerido_asesor = find_cols(["asesor", "coordinador", "vendedor", "gestor"], cols)
    sugerido_ins_fecha = find_cols(["fecha de pago inscrip", "pago inscrip"], cols)
    if not sugerido_ins_fecha and "Fecha de Pago Inscripción".lower() in [c.lower() for c in cols]:
        sugerido_ins_fecha = ["Fecha de Pago Inscripción"]
    sugerido_ins_valor = find_cols(["pago recibo inscrip", "valor inscrip"], cols)
    if not sugerido_ins_valor and "Pago Recibo Inscripción".lower() in [c.lower() for c in cols]:
        sugerido_ins_valor = ["Pago Recibo Inscripción"]
    sugerido_ins_num = find_cols(["no. recibo de inscrip", "n° recibo inscrip", "numero recibo inscrip"], cols)
    if not sugerido_ins_num and "No. Recibo de Inscripción".lower() in [c.lower() for c in cols]:
        sugerido_ins_num = ["No. Recibo de Inscripción"]

    sugerido_matr_fecha = find_cols(["fecha de pago matric", "pago matric"], cols)
    if not sugerido_matr_fecha and "Fecha de Pago Matricula".lower() in [c.lower() for c in cols]:
        sugerido_matr_fecha = ["Fecha de Pago Matricula"]
    sugerido_matr_flag = find_cols(["matrícula académica", "matricula academica", "matriculado"], cols)
    if not sugerido_matr_flag and "Matrícula Académica".lower() in [c.lower() for c in cols]:
        sugerido_matr_flag = ["Matrícula Académica"]
    sugerido_recibos_valor = find_cols(["valor recibo", "saldo recibo", "recibo 1", "recibo 2", "recibo 3", "recibo 4"], cols)
    if not sugerido_recibos_valor and "Valor Recibo 1".lower() in [c.lower() for c in cols]:
        sugerido_recibos_valor = ["Valor Recibo 1"]

    col_asesor = st.sidebar.selectbox("Columna de Asesor(a)", options=cols, index=cols.index(sugerido_asesor[0]) if sugerido_asesor else 0)
    col_ins_fecha = st.sidebar.selectbox("Fecha pago Inscripción (opcional)", options=["(ninguna)"] + cols, index=(["(ninguna)"]+cols).index(sugerido_ins_fecha[0]) if sugerido_ins_fecha else 0)
    col_ins_valor = st.sidebar.selectbox("Valor/Pago Inscripción (opcional)", options=["(ninguna)"] + cols, index=(["(ninguna)"]+cols).index(sugerido_ins_valor[0]) if sugerido_ins_valor else 0)
    col_ins_num = st.sidebar.selectbox("N° Recibo Inscripción (opcional)", options=["(ninguna)"] + cols, index=(["(ninguna)"]+cols).index(sugerido_ins_num[0]) if sugerido_ins_num else 0)

    col_matr_fecha = st.sidebar.selectbox("Fecha pago Matrícula (opcional)", options=["(ninguna)"] + cols, index=(["(ninguna)"]+cols).index(sugerido_matr_fecha[0]) if sugerido_matr_fecha else 0)
    col_matr_flag = st.sidebar.selectbox("Matrícula Académica / Matriculado (opcional)", options=["(ninguna)"] + cols, index=(["(ninguna)"]+cols).index(sugerido_matr_flag[0]) if sugerido_matr_flag else 0)
    cols_valores_recibos = st.sidebar.multiselect("Valores de Recibos (opcional)", options=cols, default=[c for c in sugerido_recibos_valor if c in cols][:6])

    col_ins_fecha = None if col_ins_fecha == "(ninguna)" else col_ins_fecha
    col_ins_valor = None if col_ins_valor == "(ninguna)" else col_ins_valor
    col_ins_num = None if col_ins_num == "(ninguna)" else col_ins_num
    col_matr_fecha = None if col_matr_fecha == "(ninguna)" else col_matr_fecha
    col_matr_flag = None if col_matr_flag == "(ninguna)" else col_matr_flag

    st.subheader("Ejecución")
    uplift = st.slider("Escenario de crecimiento de inscripciones (%)", min_value=0, max_value=100, value=20, step=5)
    st.markdown('<div class="big-cta">', unsafe_allow_html=True)
    run = st.button("🚀 Analizar y predecir ventas", help="Ejecuta la regresión/ comparación y genera proyección por asesor")
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        ins_flag, venta_flag = infer_flags(df, col_ins_fecha, col_ins_valor, col_ins_num, col_matr_fecha, col_matr_flag, cols_valores_recibos)
        work = pd.DataFrame({
            "Asesor": df[col_asesor].astype(str).str.strip(),
            "InscripcionFlag": ins_flag,
            "VentaFlag": venta_flag
        })

        agg = work.groupby("Asesor").agg(
            inscripciones=("InscripcionFlag", "sum"),
            ventas=("VentaFlag", "sum"),
            total_registros=("InscripcionFlag", "count")
        ).reset_index()

        agg = agg[(agg["inscripciones"] > 0) | (agg["ventas"] > 0)]

        st.subheader("Datos agregados por asesor")
        st.dataframe(agg.sort_values(["inscripciones","ventas"], ascending=False), use_container_width=True)

        var_x = np.var(agg["inscripciones"].values) if len(agg) > 0 else 0.0
        var_y = np.var(agg["ventas"].values) if len(agg) > 0 else 0.0
        can_regress = (len(agg) >= 2 and var_x > 0 and var_y > 0)

        if not can_regress:
            st.warning("No es posible ajustar una regresión lineal (señal insuficiente). Se usa comparación y proyección por tasa global.")
            c1, c2 = st.columns(2)
            with c1:
                fig_b1, ax_b1 = plt.subplots()
                ax_b1.bar(agg["Asesor"], agg["inscripciones"])
                ax_b1.set_title("Inscripciones por asesor")
                ax_b1.set_xlabel("Asesor")
                ax_b1.set_ylabel("Inscripciones")
                plt.setp(ax_b1.get_xticklabels(), rotation=60, ha="right")
                st.pyplot(fig_b1, use_container_width=True)
            with c2:
                fig_b2, ax_b2 = plt.subplots()
                ax_b2.bar(agg["Asesor"], agg["ventas"])
                ax_b2.set_title("Ventas por asesor")
                ax_b2.set_xlabel("Asesor")
                ax_b2.set_ylabel("Ventas")
                plt.setp(ax_b2.get_xticklabels(), rotation=60, ha="right")
                st.pyplot(fig_b2, use_container_width=True)

            tasa_global = (agg["ventas"].sum() / agg["inscripciones"].sum()) if agg["inscripciones"].sum() > 0 else 0.0
            agg["inscripciones_escenario"] = (agg["inscripciones"] * (1 + uplift/100.0)).round().astype(int)
            agg["prediccion_ventas"] = (agg["inscripciones_escenario"] * tasa_global).round(0).astype(int)
            st.subheader("Proyección por asesor (tasa global)")
            st.dataframe(agg[["Asesor","inscripciones","ventas","inscripciones_escenario","prediccion_ventas"]], use_container_width=True)

            fig_c, ax_c = plt.subplots()
            x = np.arange(len(agg))
            width = 0.35
            ax_c.bar(x - width/2, agg["ventas"], width, label="Ventas actuales")
            ax_c.bar(x + width/2, agg["prediccion_ventas"], width, label="Ventas proyectadas")
            ax_c.set_xticks(x)
            ax_c.set_xticklabels(agg["Asesor"], rotation=60, ha="right")
            ax_c.set_title(f"Ventas actuales vs proyectadas (+{uplift}%)")
            ax_c.set_xlabel("Asesor")
            ax_c.set_ylabel("Ventas")
            ax_c.legend()
            st.pyplot(fig_c, use_container_width=True)

            st.download_button(
                "Descargar proyección (CSV)",
                data=agg[["Asesor","inscripciones","ventas","inscripciones_escenario","prediccion_ventas"]].to_csv(index=False).encode("utf-8"),
                file_name="proyeccion_asesores.csv",
                mime="text/csv"
            )
        else:
            X = agg[["inscripciones"]].astype(float).values
            y = agg["ventas"].astype(float).values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)

            r2 = r2_score(y, y_pred)
            rmse = RMSE(y, y_pred)
            slope = float(model.coef_[0])
            intercept = float(model.intercept_)

            st.subheader("Resultados de la regresión lineal")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R²", f"{r2:.3f}")
            m2.metric("RMSE", f"{rmse:.3f}")
            m3.metric("Pendiente (β₁)", f"{slope:.3f}")
            m4.metric("Intercepto (β₀)", f"{intercept:.3f}")

            fig1, ax1 = plt.subplots()
            ax1.scatter(agg["inscripciones"], agg["ventas"])
            x_line = np.linspace(agg["inscripciones"].min(), agg["inscripciones"].max(), 100).reshape(-1, 1)
            y_line = model.predict(x_line)
            ax1.plot(x_line, y_line)
            ax1.set_xlabel("Número de inscripciones (por asesor)")
            ax1.set_ylabel("Número de ventas (por asesor)")
            ax1.set_title("Inscripciones vs Ventas por Asesor (recta de regresión)")
            st.pyplot(fig1, use_container_width=True)

            resid = y - y_pred
            fig2, ax2 = plt.subplots()
            ax2.scatter(y_pred, resid)
            ax2.axhline(0)
            ax2.set_xlabel("Ventas predichas")
            ax2.set_ylabel("Residuo")
            ax2.set_title("Diagrama de residuos")
            st.pyplot(fig2, use_container_width=True)

            agg["ventas_pred_actual"] = (slope * agg["inscripciones"] + intercept).clip(lower=0)
            agg["inscripciones_escenario"] = (agg["inscripciones"] * (1 + uplift/100.0)).round().astype(int)
            agg["ventas_pred_escenario"] = (slope * agg["inscripciones_escenario"] + intercept).clip(lower=0)

            st.subheader(f"Proyección por asesor (escenario +{uplift}%)")
            st.dataframe(
                agg[["Asesor","inscripciones","ventas","ventas_pred_actual","inscripciones_escenario","ventas_pred_escenario"]].round(2),
                use_container_width=True
            )

            fig3, ax3 = plt.subplots()
            x = np.arange(len(agg))
            width = 0.35
            ax3.bar(x - width/2, agg["ventas"], width, label="Ventas actuales")
            ax3.bar(x + width/2, agg["ventas_pred_escenario"], width, label="Ventas proyectadas")
            ax3.set_xticks(x)
            ax3.set_xticklabels(agg["Asesor"], rotation=60, ha="right")
            ax3.set_title(f"Ventas actuales vs proyectadas (+{uplift}%)")
            ax3.set_xlabel("Asesor")
            ax3.set_ylabel("Ventas")
            ax3.legend()
            st.pyplot(fig3, use_container_width=True)

            st.subheader("Descargas")
            st.download_button(
                "Descargar datos agregados (CSV)",
                data=agg.to_csv(index=False).encode("utf-8"),
                file_name="agregado_asesores.csv",
                mime="text/csv"
            )
            st.download_button(
                "Descargar proyección (CSV)",
                data=agg[["Asesor","inscripciones","ventas","ventas_pred_actual","inscripciones_escenario","ventas_pred_escenario"]].to_csv(index=False).encode("utf-8"),
                file_name="proyeccion_asesores.csv",
                mime="text/csv"
            )

            st.markdown(
                f"**Interpretación:** β₁={slope:.3f} sugiere el cambio esperado en ventas ante una unidad adicional de inscripciones por asesor. "
                f"Con R²={r2:.3f}, el modelo explica esa proporción de la variabilidad de ventas. "
                f"El escenario de +{uplift}% en inscripciones refleja el potencial de crecimiento por asesor."
            )
