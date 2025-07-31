import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

class ModelTrainer:
    def __init__(self, df, logger, pipeline_factory):
        self.df = df
        self.logger = logger
        self.pipeline_factory = pipeline_factory

    def _get_base_regression_models(self):
        """Retorna un diccionario de modelos de regresión base."""
        return {
            'LinearRegression': LinearRegression(),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
            'RandomForestRegressor': RandomForestRegressor(random_state=42),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'SVR': SVR()
        }

    def _get_base_classification_models(self, scenario_config):
        """Retorna un diccionario de modelos de clasificación base."""
        use_regularization = scenario_config['regularized']
        
        models = {
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'SVC': SVC(probability=True, random_state=42)
        }
        
        if use_regularization:
            models['LogisticRegression'] = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear', penalty='l1')
        else:
            models['LogisticRegression'] = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear', penalty='l2')
        
        return models

    def _get_param_grids(self, is_regression=True):
        """Retorna los grids de hiperparámetros para regresión o clasificación."""
        if is_regression:
            return {
                'LinearRegression_Lasso': {'model__alpha': [0.1, 1.0, 10.0]},
                'LinearRegression_Ridge': {'model__alpha': [0.1, 1.0, 10.0]},
                'DecisionTreeRegressor': {'model__max_depth': [3, 5, 10, None]},
                'RandomForestRegressor': {'model__n_estimators': [50, 100, 200], 'model__max_depth': [5, 10, None]},
                'KNeighborsRegressor': {'model__n_neighbors': [3, 5, 7, 9]},
                'SVR': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']}
            }
        else:
            return {
                'LogisticRegression': {'model__C': [0.1, 1, 10], 'model__penalty': ['l1', 'l2']},
                'DecisionTreeClassifier': {'model__max_depth': [3, 5, 10, None]},
                'RandomForestClassifier': {'model__n_estimators': [50, 100, 200], 'model__max_depth': [5, 10, None]},
                'KNeighborsClassifier': {'model__n_neighbors': [3, 5, 7, 9]},
                'SVC': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']}
            }

    def evaluate_regression(self, model, X_test, y_test):
        """Evalúa un modelo de regresión y retorna métricas."""
        y_pred = model.predict(X_test)
        metrics = {
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'y_pred': y_pred,
            'model': model
        }
        return metrics

    def evaluate_classification(self, model, X_test, y_test):
        """Evalúa un modelo de clasificación y retorna métricas."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_proba, multi_class='ovr'),
            'y_pred': y_pred,
            'model': model
        }
        return metrics

    def train_regression(self, scenarios):
        """Entrena y evalúa modelos de regresión."""
        self.logger.info("--- Iniciando Ciclo de Entrenamiento de REGRESIÓN ---")
        features = [col for col in self.df.columns if col != 'Producción_alimentos']
        X = self.df[features]
        y = self.df['Producción_alimentos']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        all_results = {}
        param_grids = self._get_param_grids(is_regression=True)

        for scenario in scenarios:
            self.logger.info(f"--- Escenario de Regresión: {scenario} ---")
            scenario_results = {}
            
            with mlflow.start_run(run_name=f"Regresion-{scenario}", nested=True):
                mlflow.log_param("scenario", scenario)

                models_to_train = self._get_base_regression_models()
                
                if scenario in ['Regularized', 'Full']:
                    del models_to_train['LinearRegression']
                    models_to_train['LinearRegression_Lasso'] = Lasso(random_state=42)
                    models_to_train['LinearRegression_Ridge'] = Ridge(random_state=42)

                for name, model in models_to_train.items():
                    with mlflow.start_run(run_name=name, nested=True) as model_run:
                        mlflow.log_param("model_name", name)
                        
                        pipe = self.pipeline_factory.create_regression_pipeline(model, scenario)
                        
                        use_optimization = scenario in ['Optimized', 'Full']
                        if use_optimization and name in param_grids:
                            final_pipe = GridSearchCV(pipe, param_grids[name], cv=5, scoring='r2', n_jobs=-1)
                        else:
                            final_pipe = pipe
                        
                        final_pipe.fit(X_train, y_train)
                        
                        if use_optimization and name in param_grids:
                            best_model_found = final_pipe.best_estimator_
                            mlflow.log_params(final_pipe.best_params_)
                        else:
                            best_model_found = final_pipe

                        metrics = self.evaluate_regression(best_model_found, X_test, y_test)
                        scenario_results[name] = metrics
                        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                        
                        signature = infer_signature(X_train, best_model_found.predict(X_train))
                        
                        mlflow.sklearn.log_model(
                            sk_model=best_model_found,
                            registered_model_name=name,
                            signature=signature,
                            input_example=X_train.head()
                        )

            all_results[scenario] = scenario_results

        self.logger.info("--- Ciclo de Entrenamiento de REGRESIÓN Completado ---")
        return all_results, {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    def train_classification(self, scenarios_config):
        """Entrena y evalúa modelos de clasificación."""
        self.logger.info("--- Iniciando Ciclo de Entrenamiento de CLASIFICACIÓN ---")
        
        features = [col for col in self.df.columns if col != 'Producción_alimentos']
        X = self.df[features]
        y_reg = self.df['Producción_alimentos']
        
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        
        self.logger.info("Creando target categórico 'Impacto_climatico' POST-split para evitar data leakage.")
        try:
            _, quantiles = pd.qcut(y_train_reg, q=3, labels=['Bajo', 'Medio', 'Alto'], retbins=True, duplicates='drop')
            self.logger.info(f"Cuantiles definidos desde train set: {quantiles}")
        except ValueError as e:
            self.logger.error(f"No se pudieron crear 3 cuantiles únicos. Error: {e}. Usando 2 cuantiles.")
            _, quantiles = pd.qcut(y_train_reg, q=2, labels=['Bajo', 'Alto'], retbins=True, duplicates='drop')

        labels = ['Bajo', 'Medio', 'Alto'] if len(quantiles) == 4 else ['Bajo', 'Alto']
        y_train = pd.cut(y_train_reg, bins=quantiles, labels=labels, include_lowest=True)
        y_test = pd.cut(y_test_reg, bins=quantiles, labels=labels, include_lowest=True)

        if y_test.isnull().any():
            self.logger.warning("Valores en y_test fuera del rango de entrenamiento detectados. Se imputarán a la moda.")
            y_test = y_test.fillna(y_train.mode()[0])

        all_results = {}
        param_grids = self._get_param_grids(is_regression=False)

        for scenario_config in scenarios_config:
            scenario_name = scenario_config['name']
            self.logger.info(f"--- Escenario de Clasificación: {scenario_name} ---")
            scenario_results = {}
            
            with mlflow.start_run(run_name=f"Clasificacion-{scenario_name}", nested=True):
                mlflow.log_params(scenario_config)

                models_to_train = self._get_base_classification_models(scenario_config)
                
                for name, model in models_to_train.items():
                    with mlflow.start_run(run_name=name, nested=True) as model_run:
                        mlflow.log_params({"model_name": name, **scenario_config})
                        
                        pipe = self.pipeline_factory.create_classification_pipeline(model, scenario_config)
                        
                        if scenario_config['optimized'] and name in param_grids:
                            final_pipe = GridSearchCV(pipe, param_grids[name], cv=5, scoring='f1_weighted', n_jobs=-1)
                        else:
                            final_pipe = pipe
                            
                        final_pipe.fit(X_train, y_train)

                        if scenario_config['optimized'] and name in param_grids:
                            best_model_found = final_pipe.best_estimator_
                            mlflow.log_params(final_pipe.best_params_)
                        else:
                            best_model_found = final_pipe
                        
                        metrics = self.evaluate_classification(best_model_found, X_test, y_test)
                        scenario_results[name] = metrics
                        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                        mlflow.log_dict(metrics['Report'], "classification_report.json")
                        
                        signature = infer_signature(X_train, best_model_found.predict(X_train))

                        mlflow.sklearn.log_model(
                            sk_model=best_model_found,
                            registered_model_name=name,
                            signature=signature,
                            input_example=X_train.head()
                        )

            all_results[scenario_name] = scenario_results

        self.logger.info("--- Ciclo de Entrenamiento de CLASIFICACIÓN Completado ---")
        return all_results, {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'labels': labels}