import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
import io
import base64
from statsmodels.stats.outliers_influence import variance_inflation_factor

def fig_to_base64(fig):
    """Convierte una figura matplotlib a imagen base64 para embebido en HTML."""
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"<img src='data:image/png;base64,{img_base64}' class='img-fluid'>"

class DataVisualizer:
    def __init__(self, df, reports_dir, logger):
        self.df = df
        self.reports_dir = reports_dir
        self.logger = logger
    
    def generate_eda(self):
        """Genera análisis exploratorio de datos (EDA) y devuelve los componentes."""
        self.logger.info("Iniciando generación de Análisis Exploratorio de Datos (EDA).")
        
        # --- Cálculos para el EDA ---
        info_df = self.df.info(verbose=False) # Para obtener el resumen
        dimensions = f"Filas: {self.df.shape[0]}, Columnas: {self.df.shape[1]}"
        memory = f"Memoria utilizada: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        head = self.df.head().to_html(classes='table table-striped text-center', justify='center')
        desc = self.df.describe().to_html(classes='table table-striped text-center', justify='center')
        skewness = self.df.select_dtypes(include=[np.number]).skew().to_frame(name='Skewness').to_html(classes='table table-striped text-center', justify='center')
        kurtosis = self.df.select_dtypes(include=[np.number]).kurtosis().to_frame(name='Kurtosis').to_html(classes='table table-striped text-center', justify='center')
        
        # Percentiles (NUEVO)
        percentiles = self.df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).to_html(classes='table table-striped text-center', justify='center')

        # Multicolinealidad con VIF (NUEVO)
        numeric_cols_df = self.df.select_dtypes(include=np.number)
        vif_data = pd.DataFrame()
        vif_data["feature"] = numeric_cols_df.columns
        vif_data["VIF"] = [variance_inflation_factor(numeric_cols_df.values, i) for i in range(len(numeric_cols_df.columns))]
        multicollinearity = vif_data.to_html(classes='table table-striped text-center', justify='center', index=False)

        # --- Generación de Gráficos ---
        plots = {}
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        
        # Histograma para cada variable numérica
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(self.df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
            ax.set_title(f"Distribución de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            plots[f"hist_{col}"] = fig_to_base64(fig)

        # Matriz de Correlación
        corr = numeric_cols_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Matriz de Correlación")
        plots["correlation_matrix"] = fig_to_base64(fig)
        
        # Scatter plots
        if 'Producción_alimentos' in numeric_cols:
            for col in numeric_cols.drop('Producción_alimentos'):
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.regplot(data=self.df, x=col, y='Producción_alimentos', ax=ax, scatter_kws={"alpha":0.5})
                ax.set_title(f"Relación entre {col} y Producción de Alimentos")
                plots[f"scatter_{col}"] = fig_to_base64(fig)

        self.logger.info("Generación de EDA completada.")
        
        return {
            'dimensions': dimensions, 
            'memory': memory, 
            'head': head, 
            'desc': desc, 
            'skewness': skewness, 
            'kurtosis': kurtosis, 
            'plots': plots,
            'percentiles': percentiles,
            'multicollinearity': multicollinearity
        }
    
    def generate_regression_plots(self, regression_results, reg_data):
        self.logger.info("Generando gráficos para modelos de REGRESIÓN.")
        regression_plots = {}
        for scenario, results in regression_results.items():
            regression_plots[scenario] = {}
            for model_name, metrics in results.items():
                try:
                    regression_plots[scenario][model_name] = {
                        'residuals': self.plot_residuals(model_name, reg_data['y_test'], metrics['y_pred']),
                        'real_vs_pred': self.plot_real_vs_predicted(model_name, reg_data['y_test'], metrics['y_pred']),
                        'learning_curve': self.plot_learning_curve(model_name, metrics['model'], reg_data['X_train'], reg_data['y_train'])
                    }
                except Exception as e:
                    self.logger.error(f"FALLO al generar gráficos para {model_name} en {scenario}: {e}", exc_info=True)
                    regression_plots[scenario][model_name] = {}
        return regression_plots

    def generate_classification_plots(self, classification_results, clf_data):
        """Genera y organiza todos los gráficos para los modelos de clasificación."""
        self.logger.info("Generando gráficos para modelos de CLASIFICACIÓN.")
        classification_plots = {}
        
        # SOLUCIÓN: Obtener las etiquetas reales de los datos de prueba
        actual_labels = np.unique(np.concatenate([clf_data['y_test'], 
                                                 *[metrics['y_pred'] for results in classification_results.values() 
                                                for metrics in results.values()]]))
        
        # Log para debugging
        self.logger.info(f"Etiquetas detectadas en los datos: {actual_labels}")
        self.logger.info(f"Etiquetas originales proporcionadas: {clf_data.get('labels', 'No proporcionadas')}")
        
        for scenario, results in classification_results.items():
            classification_plots[scenario] = {}
            for model_name, metrics in results.items():
                try:
                    # Usar las etiquetas reales detectadas en lugar de las proporcionadas
                    classification_plots[scenario][model_name] = {
                        'confusion_matrix': self.plot_confusion_matrix(model_name, clf_data['y_test'], metrics['y_pred'], actual_labels),
                        'roc_curve': self.plot_roc_curve(model_name, metrics['model'], clf_data['X_test'], clf_data['y_test'], actual_labels),
                        'learning_curve': self.plot_learning_curve(model_name, metrics['model'], clf_data['X_train'], clf_data['y_train'], scoring='f1_weighted')
                    }
                except Exception as e:
                    self.logger.error(f"FALLO al generar gráficos para {model_name} en {scenario}: {e}", exc_info=True)
                    classification_plots[scenario][model_name] = {}
        return classification_plots
        
    def _generate_plot(self, plot_function, *args, **kwargs):
        """Función helper para generar gráficos y manejar errores."""
        model_name = kwargs.get('model_name', 'Desconocido')
        self.logger.debug(f"Generando gráfico '{plot_function.__name__}' para el modelo '{model_name}'.")
        try:
            fig = plot_function(*args, **kwargs)
            html_content = fig.to_html(full_html=False, include_plotlyjs='cdn')
            self.logger.debug(f"Gráfico '{plot_function.__name__}' para '{model_name}' generado (tamaño HTML: {len(html_content)} bytes).")
            return html_content
        except Exception as e:
            self.logger.error(f"Error al generar gráfico '{plot_function.__name__}' para el modelo '{model_name}': {e}", exc_info=True)
            return f"<p class='text-danger'>Error al generar gráfico {plot_function.__name__} para {model_name}: {e}</p>"

    def plot_residuals(self, model_name, y_true, y_pred):
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=y_pred, y=residuals, ax=ax, alpha=0.5)
        sns.regplot(x=y_pred, y=residuals, ax=ax, scatter=False, color='red', lowess=True)
        ax.set_title(f"Análisis de Residuos: {model_name}")
        ax.set_xlabel("Valores Predichos")
        ax.set_ylabel("Residuos")
        return fig_to_base64(fig)


    def _create_residuals_plot(self, y_true, y_pred, model_name):
        residuals = y_true - y_pred
        return px.scatter(x=y_pred, y=residuals, title=f"Análisis de Residuos: {model_name}", labels={'x': 'Valores Predichos', 'y': 'Residuos'}, trendline="lowess", trendline_color_override="red")

    def plot_real_vs_predicted(self, model_name, y_true, y_pred):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax.set_title(f"Valores Reales vs. Predichos: {model_name}")
        ax.set_xlabel("Valores Reales")
        ax.set_ylabel("Valores Predichos")
        return fig_to_base64(fig)

    def _create_real_vs_predicted_plot(self, y_true, y_pred, model_name):
        fig = px.scatter(x=y_true, y=y_pred, title=f"Valores Reales vs. Predichos: {model_name}", labels={'x': 'Valores Reales', 'y': 'Valores Predichos'})
        fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()], mode='lines', name='Línea Ideal', line=dict(color='red', dash='dash')))
        return fig

    def plot_learning_curve(self, model_name, model, X, y, cv=5, scoring='r2'):
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(train_sizes, train_mean, 'o-', label='Entrenamiento')
        ax.plot(train_sizes, test_mean, 'o-', label='Validación')
        ax.set_title(f"Curva de Aprendizaje: {model_name}")
        ax.set_xlabel("Cantidad de Datos de Entrenamiento")
        ax.set_ylabel(f"Score ({scoring.upper()})")
        ax.legend()
        return fig_to_base64(fig)

    def _create_learning_curve_plot(self, model, X, y, cv, scoring, model_name):
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Score de Entrenamiento'))
        fig.add_trace(go.Scatter(x=train_sizes, y=test_mean, mode='lines+markers', name='Score de Validación Cruzada'))
        fig.update_layout(title=f"Curva de Aprendizaje: {model_name}", xaxis_title="Ejemplos de Entrenamiento", yaxis_title=f"Score ({scoring.upper()})")
        return fig

    def plot_confusion_matrix(self, model_name, y_true, y_pred, labels):
        present_labels = list(set(y_true) | set(y_pred))
        filtered_labels = [label for label in labels if label in present_labels]

        if not filtered_labels:
            raise ValueError("Ninguna de las etiquetas especificadas está presente en y_true ni en y_pred.")

        cm = confusion_matrix(y_true, y_pred, labels=filtered_labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=filtered_labels, yticklabels=filtered_labels, ax=ax)
        ax.set_title(f"Matriz de Confusión: {model_name}")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor Real")
        return fig_to_base64(fig)

    def _create_confusion_matrix_plot(self, y_true, y_pred, labels, model_name):
        # SOLUCIÓN: Validar que las etiquetas existan en los datos
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        # Usar solo las etiquetas que realmente aparecen en los datos
        valid_labels = np.unique(np.concatenate([unique_true, unique_pred]))
        
        # Si se proporcionaron etiquetas específicas, usar solo las que están presentes
        if labels is not None:
            valid_labels = [label for label in labels if label in valid_labels]
        
        self.logger.info(f"Etiquetas válidas para matriz de confusión de {model_name}: {valid_labels}")
        
        cm = confusion_matrix(y_true, y_pred, labels=valid_labels)
        return px.imshow(cm, text_auto=True, title=f"Matriz de Confusión: {model_name}", 
                        x=[str(label) for label in valid_labels], 
                        y=[str(label) for label in valid_labels], 
                        color_continuous_scale='Blues', 
                        labels=dict(x="Predicción", y="Valor Real"))

    def plot_roc_curve(self, model_name, model, X_test, y_test, labels):
        y_score = model.predict_proba(X_test)
        fig, ax = plt.subplots(figsize=(6, 5))

        plotted = False
        for i, label in enumerate(labels):
            if label not in y_test.values:
                continue  # omite si la clase no está presente en y_test
            y_true_binary = (y_test == label).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")
            plotted = True

        if not plotted:
            raise ValueError("Ninguna de las clases está presente en y_test para curva ROC.")

        ax.plot([0, 1], [0, 1], 'k--', label='Azar')
        ax.set_title(f"Curva ROC: {model_name}")
        ax.set_xlabel("Tasa de Falsos Positivos")
        ax.set_ylabel("Tasa de Verdaderos Positivos")
        ax.legend()
        return fig_to_base64(fig)

    def _create_roc_curve_plot(self, model, X_test, y_test, labels, model_name):
        try:
            y_score = model.predict_proba(X_test)
            
            # SOLUCIÓN: Validar que las etiquetas coincidan con las clases del modelo
            model_classes = getattr(model, 'classes_', None)
            if model_classes is not None:
                valid_labels = model_classes
            else:
                valid_labels = np.unique(y_test)
            
            self.logger.info(f"Clases del modelo {model_name}: {valid_labels}")
            
            fig = go.Figure()
            for i, label in enumerate(valid_labels):
                if i < y_score.shape[1]:  # Verificar que el índice existe
                    y_true_binary = (y_test == label).astype(int)
                    fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Clase {label} (AUC = {roc_auc:.3f})'))
            
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Línea Base (Azar)', line=dict(dash='dash', color='grey')))
            fig.update_layout(title=f"Curva ROC (One-vs-Rest): {model_name}", xaxis_title="Tasa de Falsos Positivos (FPR)", yaxis_title="Tasa de Verdaderos Positivos (TPR)")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error al crear curva ROC para {model_name}: {e}")
            # Crear un gráfico vacío con mensaje de error
            fig = go.Figure()
            fig.add_annotation(text=f"Error al generar curva ROC: {str(e)}", 
                            x=0.5, y=0.5, showarrow=False, font=dict(size=14))
            fig.update_layout(title=f"Curva ROC (Error): {model_name}")
            return fig