import os
import sys
import pandas as pd
import numpy as np
from src.utils.logger import setup_logger
from src.preprocessing.preprocess import DataPreprocessor
from src.visualization.visualize import DataVisualizer
from src.models.train_models import ModelTrainer
from src.pipelines.pipeline_factory import PipelineFactory
from src.analysis.dynamic_text import ReportAnalyzer
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import time
import mlflow
import mlflow.sklearn

def main():
    start_timer = time.time()
    
    # --- 1. CONFIGURACIÓN INICIAL ---
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    BAD_DATA_DIR = os.path.join(DATA_DIR, 'bad_data')
    LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
    REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
    TEMPLATES_DIR = os.path.join(ROOT_DIR, 'templates')
    
    TEST_SIZE = '20%'
    RANDOM_STATE = 42
    
    
    # Diccionario para traducir el día de la semana al español
    dias_semana = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
    }
    now = datetime.now()
    dia_en_ingles = now.strftime('%A')
    dia_en_espanol = dias_semana[dia_en_ingles]

    # Formatea la fecha completa como solicitaste
    fecha_reporte = f"{dia_en_espanol}, {now.strftime('%d-%m-%Y ; %H:%M:%S')}"
    
    
    logger = setup_logger(LOGS_DIR)
    logger.info("=" * 50)
    logger.info("INICIANDO EJECUCIÓN DEL PROYECTO DE ANÁLISIS CLIMÁTICO")
    logger.info("=" * 50)

    # --- 2. CONFIGURACIÓN DE MLFLOW ---
    mlflow.set_tracking_uri(f"file://{os.path.join(ROOT_DIR, 'mlruns')}")
    experiment_name = "Impacto Climatico en Agricultura"
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow configurado. Experimento: '{experiment_name}'.")

    # --- 3. CARGA Y PREPROCESAMIENTO DE DATOS ---
    # Recordando que el nombre del archivo es cambio_climatico_agricultura.csv
    data_path = os.path.join(DATA_DIR, 'raw', 'cambio_climatico_agricultura.csv')
    preprocessor = DataPreprocessor(data_path, PROCESSED_DIR, BAD_DATA_DIR, logger)
    df = preprocessor.load_data()
    preprocessor.check_data_quality() # Detecta y guarda datos malos, pero no modifica el df
    df_clean = preprocessor.handle_outliers() # MANEJO DE OUTLIERS
    df_processed = preprocessor.preprocess(df_clean) # Procesa el df limpio

    logger.info("Shape del dataset preprocesado y limpio: %d filas, %d columnas.", df_processed.shape[0], df_processed.shape[1])
    logger.debug("Primeras 5 filas del dataset procesado:\n%s", df_processed.head().to_string())

    # --- 4. ANÁLISIS EXPLORATORIO DE DATOS (EDA) ---
    visualizer = DataVisualizer(df_processed, REPORTS_DIR, logger)
    eda_results = visualizer.generate_eda()

    # --- 5. ENTRENAMIENTO DE MODELOS ---
    pipeline_factory = PipelineFactory(logger)
    trainer = ModelTrainer(df_processed, logger, pipeline_factory)

    # Definición explícita de escenarios para claridad y cumplimiento
    regression_scenarios = ['Baseline', 'Optimized', 'Regularized', 'Full']
    classification_scenarios_config = [
        {'name': 'Baseline', 'balanced': False, 'regularized': False, 'optimized': False},
        {'name': 'Balanced', 'balanced': True, 'regularized': False, 'optimized': False},
        {'name': 'Regularized', 'balanced': False, 'regularized': True, 'optimized': False},
        {'name': 'Optimized', 'balanced': False, 'regularized': False, 'optimized': True},
        {'name': 'Balanced_Regularized', 'balanced': True, 'regularized': True, 'optimized': False},
        {'name': 'Regularized_Optimized', 'balanced': False, 'regularized': True, 'optimized': True},
        {'name': 'Balanced_Optimized', 'balanced': True, 'regularized': False, 'optimized': True},
        {'name': 'Full', 'balanced': True, 'regularized': True, 'optimized': True},
    ]

    # Ejecución de los ciclos de entrenamiento
    with mlflow.start_run(run_name="CicloCompleto") as parent_run:
        mlflow.log_param("dataset_shape", df_processed.shape)
        
        regression_results, reg_data = trainer.train_regression(regression_scenarios)
        classification_results, clf_data = trainer.train_classification(classification_scenarios_config)

    # --- 6. GENERACIÓN DE VISUALIZACIONES POST-ENTRENAMIENTO ---
    logger.info("Generando visualizaciones de rendimiento de modelos.")
    regression_plots = visualizer.generate_regression_plots(regression_results, reg_data)
    classification_plots = visualizer.generate_classification_plots(classification_results, clf_data)

    # --- 7. ANÁLISIS DINÁMICO Y PREPARACIÓN DE CONTEXTO PARA REPORTE ---
    logger.info("Generando análisis y texto dinámico para el reporte.")
    analyzer = ReportAnalyzer(
        regression_results,
        classification_results,
        reg_data,
        clf_data,
        eda_results
    )

    # Generar todos los componentes dinámicos
    (   regression_summary_df, 
        classification_summary_df, 
        regression_ranks, 
        classification_ranks
    ) = analyzer.create_summary_tables()
    
    best_models = analyzer.get_best_models(regression_ranks, classification_ranks)
    analysis_texts = analyzer.generate_analysis_texts(best_models)
    feature_importance_texts = analyzer.generate_feature_importance_analysis()
    interpretation_texts = analyzer.generate_interpretation_texts(best_models)
    scope_texts = analyzer.generate_scope_texts(best_models)
    model_usage_texts = analyzer.generate_model_usage_texts()
    results_analysis_texts = analyzer.generate_results_analysis(best_models)
    conclusions_texts = analyzer.generate_conclusions(best_models, feature_importance_texts)
    
    # --- 8. GENERACIÓN DEL REPORTE HTML ---
    logger.info("Renderizando el reporte HTML final.")
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=True)
    template = env.get_template('report_template.html')

    template_context = {
        "fecha": fecha_reporte,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "eda": eda_results,
        "regression_results": regression_results,
        "classification_results": classification_results,
        "regression_plots": regression_plots,
        "classification_plots": classification_plots,
        "regression_summary_table": regression_summary_df.to_html(classes='table table-striped text-center', justify='center', escape=False),
        "classification_summary_table": classification_summary_df.to_html(classes='table table-striped text-center', justify='center', escape=False),
        "best_regression_model": best_models['regression']['name'],
        "best_classification_model": best_models['classification']['name'],
        "analysis": analysis_texts,
        "feature_importance": feature_importance_texts,
        "interpretation": interpretation_texts,
        "scope": scope_texts,
        "models": model_usage_texts,
        "results": results_analysis_texts,
        "conclusions": conclusions_texts
    }

    report_path = os.path.join(REPORTS_DIR, 'Reporte_Analisis_Impacto_Climatico.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(template.render(template_context))

    logger.info("="*50)
    logger.info(f"Reporte final generado exitosamente en: {report_path}")
    logger.info("Ejecución completada.")
    
    end_timer = time.time()
    running_time = end_timer - start_timer
    logger.info(f"Tiempo total de ejecución: {(running_time/60):.2f} minutos.")

if __name__ == "__main__":
    main()