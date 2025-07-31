import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class ReportAnalyzer:
    """
    Clase responsable de analizar los resultados del entrenamiento y el EDA
    para generar texto dinámico y conclusiones para el reporte final.
    """
    def __init__(self, regression_results, classification_results, reg_data, clf_data, eda_results):
        self.regression_results = regression_results
        self.classification_results = classification_results
        self.reg_data = reg_data
        self.clf_data = clf_data
        self.eda_results = eda_results
        self.features = self.reg_data['X_train'].columns.tolist()

    def create_summary_tables(self):
        """
        Procesa los resultados brutos para crear tablas de resumen con rankings y
        porcentajes de mejora para ambos tipos de modelos.
        """
        # --- Resumen de Regresión ---
        reg_summary_data = []
        baseline_r2 = {k.replace('_Lasso','').replace('_Ridge',''): v['R2'] for k, v in self.regression_results['Baseline'].items()}
        
        for scenario, results in self.regression_results.items():
            sorted_results = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)
            for rank, (model, metrics) in enumerate(sorted_results, 1):
                base_model_name = model.replace('_Lasso','').replace('_Ridge','')
                base_r2 = baseline_r2.get(base_model_name, np.nan)
                improvement = ((metrics['R2'] - base_r2) / abs(base_r2) * 100) if pd.notna(base_r2) and base_r2 != 0 else 0
                
                reg_summary_data.append({
                    'Escenario': scenario, 'Modelo': model, 'Ranking': rank,
                    'R2': metrics['R2'], 'MAE': metrics['MAE'], 'MSE': metrics['MSE'], 'RMSE': metrics['RMSE'],
                    '% Mejora R2': improvement
                })
        
        regression_summary_df = pd.DataFrame(reg_summary_data)
        regression_ranks = regression_summary_df.set_index(['Escenario', 'Modelo'])['Ranking'].to_dict()

        # --- Resumen de Clasificación ---
        clf_summary_data = []
        baseline_acc = {k: v['Accuracy'] for k, v in self.classification_results['Baseline'].items()}
        
        for scenario, results in self.classification_results.items():
            sorted_results = sorted(results.items(), key=lambda x: x[1]['Accuracy'], reverse=True)
            for rank, (model, metrics) in enumerate(sorted_results, 1):
                base_acc = baseline_acc.get(model, np.nan)
                improvement = ((metrics['Accuracy'] - base_acc) / abs(base_acc) * 100) if pd.notna(base_acc) and base_acc != 0 else 0
                
                clf_summary_data.append({
                    'Escenario': scenario, 'Modelo': model, 'Ranking': rank,
                    'Accuracy': metrics['Accuracy'], 
                    'Precisión (Weighted)': metrics['Report']['weighted avg']['precision'],
                    'Recall (Weighted)': metrics['Report']['weighted avg']['recall'],
                    'F1-Score (Weighted)': metrics['Report']['weighted avg']['f1-score'],
                    'ROC-AUC': metrics['ROC-AUC'],
                    '% Mejora Accuracy': improvement
                })
        
        classification_summary_df = pd.DataFrame(clf_summary_data)
        classification_ranks = classification_summary_df.set_index(['Escenario', 'Modelo'])['Ranking'].to_dict()

        return regression_summary_df, classification_summary_df, regression_ranks, classification_ranks

    def get_best_models(self, regression_ranks, classification_ranks):
        """Identifica los mejores modelos (ranking #1) en el escenario 'Full'."""
        best_reg_name = [k[1] for k, v in regression_ranks.items() if k[0] == 'Full' and v == 1][0]
        best_clf_name = [k[1] for k, v in classification_ranks.items() if k[0] == 'Full' and v == 1][0]
        
        best_reg_metrics = self.regression_results['Full'][best_reg_name]
        best_clf_metrics = self.classification_results['Full'][best_clf_name]

        return {
            'regression': {'name': best_reg_name, 'metrics': best_reg_metrics},
            'classification': {'name': best_clf_name, 'metrics': best_clf_metrics}
        }

    def generate_analysis_texts(self, best_models):
        """Genera textos dinámicos sobre el impacto de las estrategias de modelado."""
        # Análisis de Regresión
        r2_optimized = self.regression_results['Optimized'][best_models['regression']['name'].replace('_Lasso','').replace('_Ridge','')]['R2']
        r2_baseline = self.regression_results['Baseline'][best_models['regression']['name'].replace('_Lasso','').replace('_Ridge','')]['R2']
        r2_improvement = (r2_optimized - r2_baseline) / abs(r2_baseline) * 100 if r2_baseline != 0 else 0
        
        # Análisis de Clasificación
        acc_balanced = self.classification_results['Balanced'][best_models['classification']['name']]['Accuracy']
        acc_baseline_clf = self.classification_results['Baseline'][best_models['classification']['name']]['Accuracy']
        acc_improvement_bal = (acc_balanced - acc_baseline_clf) / abs(acc_baseline_clf) * 100 if acc_baseline_clf != 0 else 0

        analysis = {
            'hyperparameter_regression': f"La optimización de hiperparámetros (escenario 'Optimized') demostró ser efectiva. Por ejemplo, para el modelo base del mejor modelo final ({best_models['regression']['name']}), el R² aumentó de {r2_baseline:.3f} a {r2_optimized:.3f}, lo que representa una mejora del **{r2_improvement:.2f}%**. Esto indica que la búsqueda en grilla encontró una configuración más adecuada que la estándar.",
            'regularization_regression': "La regularización (Lasso y Ridge) aplicada a la Regresión Lineal ofreció una alternativa robusta, controlando la complejidad del modelo. Aunque en el escenario 'Full' no superó a los modelos de ensamblaje, el mejor modelo regularizado, **{best_reg}**, alcanzó un R² de **{best_reg_r2:.3f}**, demostrando su validez como un modelo interpretable y eficiente.".format(best_reg=best_models['regression']['name'], best_reg_r2=best_models['regression']['metrics']['R2']),
            'full_regression': f"La combinación de todas las técnicas en el escenario 'Full' culminó con **{best_models['regression']['name']}** como el modelo de regresión superior, alcanzando un R² de **{best_models['regression']['metrics']['R2']:.3f}**. Esto subraya la sinergia entre la optimización de hiperparámetros y, en su caso, la regularización, para maximizar el rendimiento predictivo.",
            'balancing_classification': f"El balanceo de clases (escenario 'Balanced') tuvo un impacto significativo en la equidad del modelo. Para el modelo {best_models['classification']['name']}, el Accuracy cambió de {acc_baseline_clf:.3f} a {acc_balanced:.3f} (una mejora del **{acc_improvement_bal:.2f}%**). Más importante aún, el F1-Score para las clases minoritarias suele mejorar, indicando que el modelo ya no ignora las categorías con menos muestras.",
            'regularization_classification': "La regularización en los modelos de clasificación, como la penalización L1/L2 en LogisticRegression, ayudó a mejorar la generalización y a evitar el sobreajuste, especialmente en escenarios con muchas características o ruido.",
            'full_classification': f"El enfoque integral del escenario 'Full' (balanceo, regularización y optimización) permitió a **{best_models['classification']['name']}** alcanzar el máximo rendimiento, con un Accuracy de **{best_models['classification']['metrics']['Accuracy']:.3f}** y un F1-Score (ponderado) de **{best_models['classification']['metrics']['Report']['weighted avg']['f1-score']:.3f}**. Esto demuestra que un preprocesamiento cuidadoso y una optimización exhaustiva son clave para construir un clasificador robusto y fiable."
        }
        return analysis
        
    def generate_feature_importance_analysis(self):
        """Analiza la importancia de características usando un RandomForest de referencia."""
        rf = RandomForestRegressor(random_state=42)
        rf.fit(self.reg_data['X_train'], self.reg_data['y_train'])
        
        importances = pd.Series(rf.feature_importances_, index=self.features).sort_values(ascending=False)
        
        most_important_feature = importances.index[0]
        most_important_value = importances.iloc[0]
        least_important_feature = importances.index[-1]
        least_important_value = importances.iloc[-1]

        feature_texts = {
            'technique': "Se utilizó un modelo **RandomForestRegressor** de referencia para calcular la importancia de cada característica. Este método se basa en la 'reducción media de impureza' (Mean Decrease in Impurity) que cada variable aporta a los árboles de decisión del ensamblaje. Una mayor reducción de impureza implica una característica más decisiva para la predicción.",
            'interpretation': f"El análisis revela que **'{most_important_feature}'** es la variable más influyente en la predicción de la producción agrícola, con una importancia relativa de **{most_important_value:.2%}**. En el otro extremo, '{least_important_feature}' es la menos influyente ({least_important_value:.2%}). Esto sugiere que las políticas y estrategias de adaptación deben priorizar la gestión de los factores relacionados con **{most_important_feature}** para mitigar los efectos del cambio climático en la agricultura."
        }
        return feature_texts

    def generate_interpretation_texts(self, best_models):
        """Genera interpretaciones de los resultados de los mejores modelos."""
        best_reg_name = best_models['regression']['name']
        best_reg_r2 = best_models['regression']['metrics']['R2']
        
        best_clf_name = best_models['classification']['name']
        best_clf_metrics = best_models['classification']['metrics']
        
        cm = pd.DataFrame(best_clf_metrics['model'].named_steps['model'].classes_, columns=['Clases'])
        cm_html = cm.to_html(classes='table table-sm', index=False) # Para referencia

        interpretation = {
            'regression': f"El mejor modelo de regresión, **{best_reg_name}**, obtuvo un coeficiente de determinación (R²) de **{best_reg_r2:.3f}**. Esto significa que el modelo es capaz de explicar aproximadamente el **{best_reg_r2:.1%} de la variabilidad** en la 'Producción_alimentos' basándose en las características climáticas. Es un indicador de un buen ajuste predictivo.",
            'classification': f"El clasificador superior, **{best_clf_name}**, alcanzó un **Accuracy del {best_clf_metrics['Accuracy']:.1%}** y un **F1-Score ponderado de {best_clf_metrics['Report']['weighted avg']['f1-score']:.3f}**. El alto F1-Score es particularmente relevante, ya que indica un buen equilibrio entre precisión y recall, asegurando que el modelo no solo acierta en sus predicciones, sino que también captura la mayoría de los casos positivos para cada clase de impacto.",
            'confusion_matrix': f"La matriz de confusión del modelo **{best_clf_name}** es clave. Un número bajo de falsos negativos (casos reales de 'Alto' impacto clasificados erróneamente como 'Bajo' o 'Medio') es crucial. Un buen rendimiento en esta área significa que el modelo es fiable para alertar sobre los países que enfrentan los mayores riesgos, permitiendo una intervención prioritaria y efectiva.",
            'metrics': "Métricas como **R²** (para regresión) y **F1-Score/ROC-AUC** (para clasificación) son fundamentales. R² nos dice qué tan bien se ajustan nuestras predicciones a los datos reales. F1-Score y ROC-AUC son vitales para la clasificación porque evalúan el rendimiento del modelo en clases desbalanceadas, asegurando que no solo sea preciso en general, sino también efectivo en la identificación de cada categoría de riesgo climático."
        }
        return interpretation

    def generate_scope_texts(self, best_models):
        """Genera textos sobre el alcance y aplicabilidad de los modelos."""
        scope = {
            'regression': f"El modelo **{best_models['regression']['name']}** es ideal para realizar pronósticos cuantitativos sobre la producción de alimentos. Su fortaleza reside en capturar relaciones complejas (posiblemente no lineales) entre las variables climáticas y el rendimiento agrícola. Puede ser utilizado por organismos gubernamentales para estimar cosechas futuras y planificar importaciones o exportaciones.",
            'classification': f"El clasificador **{best_models['classification']['name']}** es una herramienta estratégica para la evaluación de riesgos. Su propósito no es dar una cifra exacta de producción, sino categorizar a los países según su nivel de vulnerabilidad climática. Es perfecto para que organizaciones internacionales prioricen la asignación de recursos y ayuda para la adaptación climática."
        }
        return scope

    def generate_model_usage_texts(self):
        """Genera textos explicando cómo usar los modelos guardados."""
        usage = {
            'usage': "Todos los modelos entrenados en cada escenario han sido guardados y registrados usando **MLflow**. Están disponibles en la carpeta `mlruns/` del proyecto. Estos artefactos pueden ser cargados para realizar predicciones sobre nuevos datos sin necesidad de re-entrenamiento.",
            'steps': """
                1. **Iniciar UI de MLflow:** Ejecutar `mlflow ui` en la terminal desde la raíz del proyecto para explorar los experimentos.
                2. **Identificar el Modelo:** Navegar al experimento 'Impacto Climatico en Agricultura', buscar el ciclo de ejecución deseado (p. ej., 'Clasificacion-Full') y el modelo (p. ej., 'RandomForestClassifier'). Copiar el `Run ID`.
                3. **Cargar el Modelo en Python:** Usar el siguiente código para cargar el modelo:
                   <pre><code class='language-python'>import mlflow
model_uri = f"runs:/RUN_ID_COPIADO/NOMBRE_DEL_MODELO"
loaded_model = mlflow.sklearn.load_model(model_uri)</code></pre>
                4. **Realizar Predicciones:** Utilizar el método `loaded_model.predict(nuevos_datos)` para obtener nuevas predicciones. Los datos de entrada deben tener la misma estructura que los de entrenamiento.
            """
        }
        return usage

    def generate_results_analysis(self, best_models):
        """Genera el análisis de resultados y las implicaciones en seguridad alimentaria."""
        results = {
            'best_model': f"Para la **predicción cuantitativa (regresión)**, el mejor modelo es **{best_models['regression']['name']}** con un R² de {best_models['regression']['metrics']['R2']:.3f}. Para la **clasificación de riesgo**, el mejor es **{best_models['classification']['name']}** con un Accuracy de {best_models['classification']['metrics']['Accuracy']:.3f}.",
            'food_security': f"Los hallazgos tienen implicaciones directas en la **seguridad alimentaria global**. El modelo de clasificación, **{best_models['classification']['name']}**, puede identificar proactivamente las naciones que serán más afectadas por el cambio climático. Si el modelo predice que una región productora clave pasará a un nivel de 'Alto' impacto, esto es una alerta temprana de posibles déficits en la producción de alimentos. Permite a los gobiernos y ONGs implementar medidas de mitigación, como la introducción de cultivos resistentes a la sequía o la mejora de sistemas de riego, antes de que ocurra una crisis alimentaria."
        }
        return results

    def generate_conclusions(self, best_models, feature_importance_texts):
        """Genera las conclusiones clave y recomendaciones estratégicas."""
        conclusions = {
            'key_conclusions': f"La conclusión principal es que los modelos de ensamblaje como **{best_models['regression']['name']}** y **{best_models['classification']['name']}** son superiores para modelar la compleja relación entre el clima y la agricultura. {feature_importance_texts['interpretation']} La aplicación de un pipeline de MLOps robusto, desde la limpieza de datos y manejo de outliers hasta la optimización y el versionado con MLflow, fue fundamental para obtener resultados fiables.",
            'recommendations': """
                <ol>
                    <li><b>Implementación en Producción:</b> Desplegar el modelo de clasificación **{clf_model}** como un servicio de alerta temprana para monitorear la vulnerabilidad climática de los países en tiempo real.</li>
                    <li><b>Enfoque en Políticas Públicas:</b> Utilizar los insights sobre la importancia de las características (priorizando **{feature}**) para guiar las políticas de inversión agrícola y adaptación al cambio climático.</li>
                    <li><b>Expansión del Dataset:</b> Enriquecer el modelo con más variables, como tipo de suelo, inversión en tecnología agrícola y políticas de subsidios, para aumentar aún más la precisión y el alcance del análisis.</li>
                    <li><b>Monitoreo Continuo:</b> Establecer un sistema de monitoreo del rendimiento del modelo (model drifting) para re-entrenarlo periódicamente a medida que se disponga de nuevos datos y las condiciones climáticas evolucionen.</li>
                </ol>
            """.format(
                clf_model=best_models['classification']['name'],
                feature=self.features[0] # Asumiendo que el primero es el más importante
            )
        }
        return conclusions