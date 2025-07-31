from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import Lasso, Ridge

class PipelineFactory:
    def __init__(self, logger):
        self.logger = logger

    def create_regression_pipeline(self, model, scenario='Baseline'):
        """Crea un pipeline de preprocesamiento y modelado para regresión."""
        self.logger.debug(f"Creando pipeline de REGRESIÓN para modelo '{type(model).__name__}' en escenario '{scenario}'.")
        
        steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
        
        # Para escenarios regularizados, el modelo ya viene como Lasso o Ridge
        steps.append(('model', model))
        
        return Pipeline(steps)

    def create_classification_pipeline(self, model, scenario_config):
        """Crea un pipeline de preprocesamiento y modelado para clasificación."""
        scenario_name = scenario_config['name']
        use_balancing = scenario_config['balanced']
        
        self.logger.debug(f"Creando pipeline de CLASIFICACIÓN para '{type(model).__name__}' en escenario '{scenario_name}'.")
        
        # Usamos ImbPipeline para manejar correctamente los samplers de imblearn
        steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
        
        if use_balancing:
            self.logger.debug("Añadiendo SMOTE y RandomUnderSampler al pipeline.")
            steps.extend([
                ('oversample', SMOTE(random_state=42)), 
                ('undersample', RandomUnderSampler(random_state=42))
            ])
        
        steps.append(('model', model))

        return ImbPipeline(steps)