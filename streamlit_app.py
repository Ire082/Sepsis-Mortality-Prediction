# Importar librerías necesarias para la aplicación Streamlit
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import os

# ⚙️ Configuración de parámetros de la aplicación
CONFIG = {
    'PREDICTORS': [
        'sofa_coag', 'sofa_liver', 'sofa_renal', 'sofa_cv_bp', 'sofa_cv_hr',
        'sofa_cns', 'lactate_score', 'wbc_score', 'fluid_balance_score'
    ],
    'TARGET': 'hospital_expire_flag',
    'MODEL_PATH': 'rf_model_mortality.joblib',
    'SCALER_PATH': 'scaler.joblib'
}

# Configurar la página de Streamlit (DEBE SER LA PRIMERA LLAMADA)
st.set_page_config(page_title="Predicción de Mortalidad por Sepsis", layout="wide")
st.title("Aplicación de Predicción de Mortalidad por Sepsis")

# 📊 Sección para cargar datos
st.header("Cargar Datos")
if st.button("Cargar Datos de Ejemplo"):
    try:
        df_clean = pd.read_csv('sample_data.csv')
        st.session_state['df'] = df_clean
        st.success("¡Datos cargados exitosamente! ✅")
        st.dataframe(df_clean.head())
    except Exception as e:
        st.error(f"Error al cargar datos: {e} ❌")

# 🔍 Sección para seleccionar predictores
st.header("Seleccionar Predictores")
selected_predictors = st.multiselect(
    "Elige las variables para predecir",
    options=CONFIG['PREDICTORS'],
    default=CONFIG['PREDICTORS']
)

# 🤖 Sección para cargar modelo y escalador
st.header("Modelo")
try:
    model = joblib.load(CONFIG['MODEL_PATH'])
    scaler = joblib.load(CONFIG['SCALER_PATH'])
    st.success("Modelo y escalador cargados exitosamente ✅")
except Exception as e:
    st.error(f"Error al cargar modelo o escalador: {e} ❌")

# 🚀 Sección para realizar predicciones
if st.button("Predecir"):
    if 'df' in st.session_state and selected_predictors:
        try:
            X = st.session_state['df'][selected_predictors]
            y_true = st.session_state['df'][CONFIG['TARGET']]
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]
            st.session_state['df']['Prediction'] = predictions
            st.session_state['df']['Probability'] = probabilities
            st.dataframe(st.session_state['df'][['subject_id', 'icustay_id', 'Prediction', 'Probability']])

            # 📈 Calcular métricas de evaluación
            metrics = {
                'Precision': precision_score(y_true, predictions),
                'Recall': recall_score(y_true, predictions),
                'F1-Score': f1_score(y_true, predictions),
                'AUC-ROC': auc(*roc_curve(y_true, probabilities)[:2])
            }
            st.subheader("Métricas de Evaluación")
            st.json(metrics)

            # 📊 Gráfico de barras para métricas
            fig_metrics = go.Figure(data=[
                go.Bar(x=list(metrics.keys()), y=list(metrics.values()), marker_color='#1f77b4')
            ])
            fig_metrics.update_layout(
                title="Métricas de Rendimiento del Modelo",
                xaxis_title="Métrica",
                yaxis_title="Puntuación"
            )
            st.plotly_chart(fig_metrics)

            # 📉 Curva ROC
            fpr, tpr, _ = roc_curve(y_true, probabilities)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines', name=f'Curva ROC (AUC = {metrics["AUC-ROC"]:.2f})'
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Aleatorio'
            ))
            fig_roc.update_layout(
                title='Curva ROC',
                xaxis_title='Tasa de Falsos Positivos',
                yaxis_title='Tasa de Verdaderos Positivos'
            )
            st.plotly_chart(fig_roc)

            # 🚨 Alerta para pacientes de alto riesgo
            high_risk = st.session_state['df'][probabilities > 0.7]
            if not high_risk.empty:
                st.warning(f"¡Alerta: {len(high_risk)} pacientes con >70% de riesgo de mortalidad! ⚠️")
                st.dataframe(high_risk[['subject_id', 'icustay_id', 'Probability']])
        except Exception as e:
            st.error(f"Error al realizar predicciones: {e} ❌")
    else:
        st.error("Por favor, carga datos y selecciona predictores primero. ⚠️")
