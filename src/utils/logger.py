import logging
import os
from datetime import datetime

def setup_logger(log_dir):
    """
    Configura un logger robusto para el proyecto.
    MEJORA: El formato ahora incluye el módulo y la función para una mejor trazabilidad.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"project_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    
    # Se define un nombre único para el logger para evitar conflictos
    logger = logging.getLogger("CambioClimaticoLogger")
    
    # Evitar que se agreguen handlers duplicados si la función es llamada múltiples veces
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.DEBUG)  # Nivel de captura general
    
    # MEJORA: Formato más detallado que incluye el origen del log.
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s] - %(message)s')
    
    # Handler para el archivo, captura desde el nivel DEBUG
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para la consola, captura desde el nivel INFO para no ser demasiado verboso
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger