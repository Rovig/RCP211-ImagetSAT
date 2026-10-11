"""
Système de logging centralisé pour le projet RCP211-ImagetSAT
"""

import copy
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Formatter qui ajoute des couleurs aux logs dans le terminal"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Vert
        'WARNING': '\033[33m',  # Jaune
        'ERROR': '\033[31m',    # Rouge
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Copie: le record est partagé avec les autres handlers (fichier)
        record = copy.copy(record)
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = 'RCP211',
    level: str = 'INFO',
    log_file: Optional[str] = None,
    log_dir: str = 'logs'
) -> logging.Logger:
    """
    Configure et retourne un logger
    
    Args:
        name: Nom du logger
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Nom du fichier de log (optionnel)
        log_dir: Répertoire pour les fichiers de log
        
    Returns:
        Logger configuré
    """
    logger = logging.getLogger(name)
    
    # Éviter les doublons de handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Format pour les logs
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Handler pour la console avec couleurs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(log_format, datefmt=date_format))
    logger.addHandler(console_handler)
    
    # Handler pour fichier si spécifié
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Ajouter la date au nom du fichier
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{timestamp}_{log_file}" if not log_file.startswith(timestamp) else log_file
        file_path = log_path / file_name
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {file_path}")
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Obtient un logger existant ou en crée un nouveau
    
    Args:
        name: Nom du logger (utilise le nom du module appelant par défaut)
        
    Returns:
        Logger
    """
    if name is None:
        # Obtenir le nom du module appelant
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'RCP211')
        else:
            name = 'RCP211'
    
    # Si le logger n'existe pas, le créer avec la configuration par défaut
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    
    return logger

