import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, data_path, processed_dir, bad_data_dir, logger):
        self.data_path = data_path
        self.processed_dir = processed_dir
        self.bad_data_dir = bad_data_dir
        self.logger = logger
        self.df = None
    
    def load_data(self):
        """Carga el dataset y registra información básica."""
        self.logger.info(f"Iniciando carga de datos desde '{self.data_path}'.")
        try:
            self.df = pd.read_csv(self.data_path)
            self.logger.info("Dataset cargado exitosamente.")
            self.logger.debug(f"Dimensiones iniciales: {self.df.shape[0]} filas, {self.df.shape[1]} columnas.")
            return self.df
        except FileNotFoundError:
            self.logger.error(f"Error crítico: El archivo no se encontró en '{self.data_path}'.")
            raise

    def check_data_quality(self):
        """Verifica la calidad de los datos y guarda registros problemáticos (sin modificar el df)."""
        self.logger.info("Iniciando verificación de calidad de datos.")
        os.makedirs(self.bad_data_dir, exist_ok=True)
        
        # Verificar y archivar valores nulos
        if self.df.isnull().sum().sum() > 0:
            nulls = self.df[self.df.isnull().any(axis=1)]
            nulls_path = os.path.join(self.bad_data_dir, "nulos_detectados.csv")
            nulls.to_csv(nulls_path, index=False)
            self.logger.warning(f"Detectados {len(nulls)} registros con valores nulos. Guardados en '{nulls_path}'.")
        else:
            self.logger.info("No se encontraron valores nulos.")

        # Verificar y archivar duplicados
        if self.df.duplicated().any():
            duplicates = self.df[self.df.duplicated()]
            dupes_path = os.path.join(self.bad_data_dir, "duplicados_detectados.csv")
            duplicates.to_csv(dupes_path, index=False)
            self.logger.warning(f"Detectados {len(duplicates)} registros duplicados. Guardados en '{dupes_path}'.")
        else:
            self.logger.info("No se encontraron registros duplicados.")
        
        self.logger.info("Verificación de calidad de datos completada.")

    def handle_outliers(self):
        """
        Detecta y elimina outliers usando el método IQR.
        Guarda los outliers eliminados y devuelve un DataFrame limpio.
        """
        self.logger.info("Iniciando detección y manejo de outliers.")
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        
        Q1 = self.df[numeric_cols].quantile(0.25)
        Q3 = self.df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_condition = ((self.df[numeric_cols] < (Q1 - 1.5 * IQR)) | (self.df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        
        outliers_df = self.df[outlier_condition]
        
        if not outliers_df.empty:
            outliers_path = os.path.join(self.bad_data_dir, "outliers_eliminados.csv")
            outliers_df.to_csv(outliers_path, index=False)
            self.logger.warning(f"Detectados y eliminados {len(outliers_df)} outliers (método IQR). Guardados en '{outliers_path}'.")
            
            df_clean = self.df[~outlier_condition]
            self.logger.info(f"Tamaño del dataset después de eliminar outliers: {df_clean.shape[0]} filas.")
        else:
            self.logger.info("No se detectaron outliers significativos para eliminar con el método IQR.")
            df_clean = self.df.copy()
            
        return df_clean

    def preprocess(self, df):
        """
        Ejecuta los pasos de preprocesamiento básicos sobre un DataFrame dado.
        """
        self.logger.info("Iniciando preprocesamiento del dataset limpio.")
        
        df_processed = df.copy()
        
        # --- INICIO DE LA CORRECCIÓN ---
        # Convierte columnas de características enteras a float para evitar errores
        # de esquema en inferencia si aparecen valores nulos en nuevos datos.
        feature_cols = [col for col in df_processed.columns if col != 'Producción_alimentos']
        
        for col in feature_cols:
            if pd.api.types.is_integer_dtype(df_processed[col]):
                self.logger.warning(f"Convirtiendo la columna entera '{col}' a float64 para prevenir errores de esquema en inferencia.")
                df_processed[col] = df_processed[col].astype('float64')
        # --- FIN DE LA CORRECCIÓN ---
        
        # Imputar valores nulos restantes (si los hubiera después de la limpieza)
        numeric_cols_missing = df_processed.select_dtypes(include=np.number).columns[df_processed.select_dtypes(include=np.number).isnull().any()]
        if not numeric_cols_missing.empty:
            self.logger.info("Aplicando imputación de datos con la mediana.")
            imputer = SimpleImputer(strategy='median')
            df_processed[numeric_cols_missing] = imputer.fit_transform(df_processed[numeric_cols_missing])
        
        # Eliminar columnas no deseadas para el modelado
        if 'País' in df_processed.columns:
            df_processed.drop(columns=['País'], inplace=True, errors='ignore')
            self.logger.info("Columna 'País' eliminada para el modelado.")

        # Guardar datos procesados
        os.makedirs(self.processed_dir, exist_ok=True)
        processed_path = os.path.join(self.processed_dir, "processed_data.csv")
        df_processed.to_csv(processed_path, index=False)
        self.logger.info(f"Datos preprocesados guardados en: {processed_path}")
        
        return df_processed