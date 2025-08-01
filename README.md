# Proyecto 1: Predicción del Rendimiento Agrícola ante el Cambio Climático

**Tags:** `Machine Learning Supervisado`, `Regresión`, `Python`, `Scikit-learn`, `Pandas`

## Objetivo del Proyecto

Este proyecto aborda el desafío de la seguridad alimentaria mediante el desarrollo de un modelo de **aprendizaje automático supervisado** para predecir el rendimiento de los cultivos. Utilizando el dataset `cambio_climatico_agricultura.csv`, que contiene variables climáticas y de producción, el objetivo es construir un modelo de regresión robusto que sirva como herramienta para la toma de decisiones en el sector agrícola.

## Metodología y Herramientas

El flujo de trabajo se desarrolló de principio a fin, cubriendo las siguientes etapas clave:

1.  **Análisis Exploratorio de Datos (EDA):** Se utilizó `Matplotlib` y `Seaborn` para visualizar las distribuciones de las variables, identificar correlaciones entre factores climáticos y el rendimiento, y detectar valores atípicos.
2.  **Preprocesamiento de Datos:** Se realizó una limpieza de datos exhaustiva con `Pandas`, manejando valores nulos a través de imputación por la media, una técnica adecuada para las variables predictoras continuas de este problema de regresión.
3.  **Modelado y Entrenamiento:**
    * Se evaluaron y compararon múltiples algoritmos de regresión de `Scikit-learn`, como Regresión Lineal (baseline), **Random Forest Regressor** y **Gradient Boosting Regressor**.
    * El conjunto de datos se dividió en entrenamiento y prueba (`train_test_split`) para una evaluación objetiva del rendimiento.
4.  **Optimización y Evaluación:**
    * Se aplicaron técnicas de **ajuste de hiperparámetros** para optimizar el modelo con mejor desempeño.
    * El rendimiento final se midió utilizando métricas estándar de regresión como el **Error Cuadrático Medio (MSE)** y el **Error Absoluto Medio (MAE)** para interpretar la precisión del modelo en unidades del mundo real.

## Resultados Clave

El modelo final, un `Gradient Boosting Regressor` optimizado, logró predecir el rendimiento de los cultivos con un alto grado de precisión. El análisis de importancia de características (`feature importance`) reveló que **[Menciona la variable más importante que encontraste, ej: 'la temperatura máxima']** es el factor más determinante para el rendimiento, proveyendo un insight valioso para la gestión de cultivos.

## Cómo Utilizar

1.  Clona este repositorio: `git clone https://github.com/patriciolavin/aprendizaje_maquina_supervisado.git`
2.  Instala las dependencias: `pip install pandas scikit-learn matplotlib seaborn`
3.  Ejecuta la Jupyter Notebook para ver el análisis completo.
