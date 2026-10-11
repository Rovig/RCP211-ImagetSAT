#!/usr/bin/env python3
"""
Gestionnaire de modèles - Principes SOLID
"""

import torch
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod
from utils.logger import get_logger
from core.models.networks import (
    ImprovedGenerator, ImprovedDiscriminator,
    generator_from_state_dict, discriminator_from_state_dict
)

logger = get_logger(__name__)

class ModelLoader(ABC):
    """Interface pour le chargement de modèles (Interface Segregation)"""
    
    @abstractmethod
    def load(self, checkpoint_path: Optional[str], device: torch.device):
        pass

# Alias conservé pour le code existant: une seule définition, dans networks.py
OriginalGenerator = ImprovedGenerator


def _load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """Charge un checkpoint; toute erreur est propagée à l'appelant"""
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du checkpoint {checkpoint_path}: {e}") from e


class GeneratorLoader(ModelLoader):
    """Chargeur pour le générateur (Single Responsibility)"""
    
    def __init__(self, generator_class):
        self.generator_class = generator_class
    
    def load(self, checkpoint_path: Optional[str], device: torch.device):
        """Charge le générateur"""
        generator = self.generator_class()
        
        if checkpoint_path:
            generator = self._load_checkpoint_compatible(checkpoint_path, device)
            logger.info(f"Modèle chargé depuis: {checkpoint_path}")
        else:
            logger.warning("Attention: Utilisation d'un modèle non-entraîné (poids aléatoires)")
        
        generator.to(device)
        generator.eval()
        return generator
    
    def _load_checkpoint_compatible(self, checkpoint_path: str, device: torch.device):
        """Charge les poids en choisissant l'architecture (256 ou 512) d'après le checkpoint"""
        checkpoint = _load_checkpoint(checkpoint_path, device)
        return generator_from_state_dict(checkpoint['generator_state_dict'])


class ModelManager:
    """Gestionnaire principal des modèles - Point d'entrée unique"""
    
    def __init__(self):
        """Initialise le gestionnaire"""
        pass
    
    def load_generator(self, checkpoint_path: Optional[str], device: torch.device) -> nn.Module:
        """
        Charge un générateur depuis un checkpoint ou crée un nouveau
        
        Args:
            checkpoint_path: Chemin vers le checkpoint (optionnel)
            device: Device PyTorch (CPU/GPU)
            
        Returns:
            Modèle générateur prêt à l'emploi
        """
        if checkpoint_path:
            # Un checkpoint demandé mais inutilisable est une erreur: pas de repli
            # silencieux sur des poids aléatoires
            checkpoint = _load_checkpoint(checkpoint_path, device)
            generator = generator_from_state_dict(checkpoint['generator_state_dict'])
            logger.info(f"Générateur chargé depuis: {checkpoint_path}")
        else:
            generator = OriginalGenerator()
            logger.warning("Création d'un nouveau générateur avec poids aléatoires")
        
        generator.to(device)
        generator.eval()
        return generator
    
    def load_discriminator(self, checkpoint_path: Optional[str], device: torch.device) -> nn.Module:
        """
        Charge un discriminateur depuis un checkpoint ou crée un nouveau
        """
        if checkpoint_path:
            checkpoint = _load_checkpoint(checkpoint_path, device)
            discriminator = discriminator_from_state_dict(checkpoint['discriminator_state_dict'])
            logger.info(f"Discriminateur chargé depuis: {checkpoint_path}")
        else:
            discriminator = self.create_discriminator(device)
        
        discriminator.to(device)
        return discriminator
    
    def create_generator(self, device: torch.device, image_size: int = 256) -> nn.Module:
        """Crée un nouveau générateur"""
        generator = ImprovedGenerator(image_size=image_size)
        generator.to(device)
        return generator
    
    def create_discriminator(self, device: torch.device, image_size: int = 256) -> nn.Module:
        """Crée un nouveau discriminateur PatchGAN"""
        discriminator = ImprovedDiscriminator(image_size=image_size)
        discriminator.to(device)
        return discriminator
