#!/usr/bin/env python3
"""
Module de génération d'images satellites à partir de cartes binaires
Utilise le ModelManager pour charger et utiliser les modèles
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional

from core.models.model_manager import ModelManager
from core.generation.environment_styles import EnvironmentStyler
from utils.logger import get_logger

logger = get_logger(__name__)


class SimpleSatelliteGenerator:
    """Générateur d'images satellites avec interface simplifiée"""
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialise le générateur
        
        Args:
            checkpoint_path: Chemin vers le checkpoint du modèle
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_manager = ModelManager()
        self.generator = self.model_manager.load_generator(checkpoint_path, self.device)
        self.styler = EnvironmentStyler()
        
    def generate(self, 
                 input_map_path: Optional[str] = None,
                 environment: str = 'default',
                 output_path: str = 'generated_satellite.png') -> str:
        """
        Génère une image satellite à partir d'une carte binaire
        
        Args:
            input_map_path: Chemin vers la carte binaire d'entrée
            environment: Type d'environnement ('urban', 'maritime', 'nature', 'default')
            output_path: Chemin de sortie pour l'image générée
            
        Returns:
            str: Chemin du fichier généré
        """
        # Création ou chargement de la carte d'entrée
        if input_map_path and Path(input_map_path).exists():
            input_map = self._load_and_process_map(input_map_path)
        else:
            logger.info("Création d'une carte aléatoire...")
            input_map = self._create_random_map()
        
        # Génération de l'image
        with torch.no_grad():
            fake_image = self.generator(input_map)
        
        # Conversion en image PIL
        fake_image_np = self._tensor_to_numpy(fake_image)
        fake_image_pil = Image.fromarray(fake_image_np)
        
        # Application du style d'environnement
        styled_image = self.styler.apply_style(fake_image_pil, environment)
        
        # Sauvegarde
        styled_image.save(output_path)
        return output_path
    
    def _load_and_process_map(self, map_path: str) -> torch.Tensor:
        """Charge et prétraite une carte binaire"""
        image = Image.open(map_path).convert('RGB')
        
        # Redimensionnement à la taille attendue par le générateur si nécessaire
        size = (self.generator.image_size, self.generator.image_size)
        if image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        
        # Conversion en tensor
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5  # Normalisation [-1, 1]
        
        return image_tensor.unsqueeze(0).to(self.device)
    
    def _create_random_map(self) -> torch.Tensor:
        """Crée une carte binaire aléatoire"""
        # Génération d'une carte simple avec des formes géométriques
        map_np = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Ajout de quelques rectangles (bâtiments)
        for _ in range(20):
            x1, y1 = np.random.randint(0, 200, 2)
            x2, y2 = x1 + np.random.randint(10, 50), y1 + np.random.randint(10, 50)
            map_np[y1:y2, x1:x2] = 255
        
        # Ajout de routes (lignes)
        for _ in range(5):
            if np.random.rand() > 0.5:  # Route horizontale
                y = np.random.randint(0, 256)
                map_np[y:y+3, :] = 128
            else:  # Route verticale
                x = np.random.randint(0, 256)
                map_np[:, x:x+3] = 128
        
        # Conversion en tensor
        map_tensor = torch.from_numpy(map_np).float()
        map_tensor = map_tensor.permute(2, 0, 1) / 255.0
        map_tensor = (map_tensor - 0.5) / 0.5
        
        return map_tensor.unsqueeze(0).to(self.device)
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convertit un tensor en array numpy pour PIL"""
        # Dénormalisation
        tensor = tensor.squeeze(0).cpu()
        tensor = tensor * 0.5 + 0.5
        tensor = torch.clamp(tensor, 0, 1)
        
        # Conversion en numpy et format PIL
        image_np = tensor.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        return image_np