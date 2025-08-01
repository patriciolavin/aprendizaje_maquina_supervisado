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

## 📈 Resultados Clave

El modelo final, un `Gradient Boosting Regressor` optimizado, logró predecir el rendimiento de los cultivos con un alto grado de precisión. El análisis de importancia de características (`feature importance`) reveló que la temperatura máxima es el factor más determinante para el rendimiento, proveyendo un insight valioso para la gestión de cultivos.

## Reflexión Personal y Desafíos

Trabajar en este proyecto fue un excelente ejercicio de la metodología clásica de machine learning. El dataset, aunque no era masivo, presentaba desafíos realistas como la necesidad de una limpieza cuidadosa y la elección correcta de un modelo de regresión.

* **Punto Alto:** Sin duda, el momento más gratificante fue ver el gráfico de `feature importance` por primera vez. Pasar de una tabla de números a una visualización clara que te dice "oye, la temperatura es lo que realmente importa aquí" es el instante en que los datos empiezan a contar una historia. Confirmó que el modelo no solo predecía, sino que también extraía lógica del sistema.
* **Punto Bajo:** La etapa de ajuste de hiperparámetros. Es un proceso computacionalmente intensivo y, a veces, un poco "a ciegas". Esperar a que `GridSearchCV` termine sus combinaciones pone a prueba la paciencia, y es un recordatorio de que la optimización de modelos es tanto un arte como una ciencia. Fue un trade-off constante entre buscar el mejor rendimiento y gestionar el tiempo de cómputo.

## Cómo Utilizar

1.  Clona este repositorio: `git clone https://github.com/patriciolavin/aprendizaje_maquina_supervisado.git`
2.  Instala las dependencias: `pip install pandas scikit-learn matplotlib seaborn`
3.  Ejecuta la Jupyter Notebook para ver el análisis completo.
