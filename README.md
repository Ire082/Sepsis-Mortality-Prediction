# Predicción de Mortalidad por Sepsis

**Proyecto final del curso Organización y Gestión de la Información y Conocimientos Clínicos**  
**Universidad de Barcelona, Junio 2025**

Este proyecto desarrolla un sistema de aprendizaje automático para predecir la mortalidad hospitalaria en pacientes con sepsis utilizando la base de datos MIMIC-IV. Al integrar variables clínicas como puntuaciones SOFA y niveles de lactato, el sistema ayuda a los equipos médicos a identificar pacientes de alto riesgo para intervenciones oportunas.

## Equipo
- Maria Chamorro Doganoc
- Marc Gregoris Brugué
- Irene Olivero Martínez

## Relevancia clínica
La sepsis es una afección potencialmente mortal causada por una respuesta descontrolada a una infección, que puede provocar fallo orgánico y altas tasas de mortalidad. La predicción temprana del riesgo de mortalidad puede guiar decisiones clínicas, optimizar la asignación de recursos y salvar vidas. Este proyecto utiliza la base de datos MIMIC-IV, un conjunto de datos de UCI de acceso público, para entrenar modelos predictivos robustos con datos clínicos reales.

## Características
- **Procesamiento de datos**: Extrae y preprocesa datos de pacientes con sepsis de MIMIC-IV (por ejemplo, puntuaciones SOFA, niveles de lactato, presión arterial).
- **Modelos de aprendizaje automático**: Entrena modelos supervisados (Regresión Logística, Random Forest, XGBoost) para predecir el riesgo de mortalidad.
- **Aplicación web interactiva**: Una aplicación Streamlit permite a los usuarios:
  - Cargar datos de pacientes (mediante `sample_data.csv` o BigQuery).
  - Seleccionar predictores (por ejemplo, puntuaciones SOFA, lactato).
  - Ver predicciones y probabilidades de mortalidad.
  - Visualizar métricas de rendimiento (AUC-ROC, precisión, sensibilidad, F1-score) y curvas ROC.
  - Recibir alertas para pacientes de alto riesgo (>70% de probabilidad de mortalidad).
- **Despliegue**: Alojado en Hugging Face Space: [SepsisMortalityPrediction](https://huggingface.co/spaces/Irene082/SepsisMortalityPrediction).
- **Repositorio**: Todo el código y entregables están disponibles en GitHub: [Sepsis-Mortality-Prediction](https://github.com/Ire082/Sepsis-Mortality-Prediction).

## Arquitectura del sistema
El sistema sigue un flujo modular:
1. **Extracción de datos**: Consultas SQL extraen datos de pacientes con sepsis de MIMIC-IV a través de BigQuery.
2. **Preprocesamiento**: Gestiona valores faltantes (imputación con la media), normaliza datos y agrega por paciente/estancia en UCI.
3. **Modelado**: Entrena modelos Random Forest y XGBoost con optimización de hiperparámetros y validación cruzada.
4. **Despliegue**: Utiliza Streamlit para un frontend y backend interactivo, desplegado en Hugging Face.
5. **Visualización**: Muestra métricas y curvas ROC usando Plotly.

## Instalación y uso

### Requisitos previos
- Python 3.8+
- Dependencias listadas en `requirements.txt` (por ejemplo, pandas, scikit-learn, streamlit, plotly, xgboost)
- Acceso a Google BigQuery (opcional, para datos en vivo desde MIMIC-IV)

### Ejecutar el Notebook localmente
1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/Sepsis-Mortality-Prediction
   cd Sepsis-Mortality-Prediction
