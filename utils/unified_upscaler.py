#!/usr/bin/env python3
"""
Module unifié pour l'augmentation de résolution d'images satellites
Offre toutes les méthodes d'upscaling en une seule interface
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union, Optional
import torchvision.transforms as transforms
from pathlib import Path

from utils.device import get_device
from utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedImageUpscaler:
    """
    Upscaler unifié supportant plusieurs méthodes et formats
    Remplace image_upscaler.py et simple_upscaler.py
    """
    
    SUPPORTED_METHODS = ['bilinear', 'bicubic', 'nearest', 'lanczos', 'torch', 'srcnn', 'hybrid']
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        logger.info(f"Upscaler initialized on device: {self.device}")
    
    def upscale(self,
                input_path: Union[str, Path],
                output_path: Union[str, Path],
                scale_factor: int = 2,
                method: str = 'bilinear') -> Optional[str]:
        """
        Interface principale pour l'upscaling d'images
        
        Args:
            input_path: Chemin vers l'image d'entrée
            output_path: Chemin de sortie
            scale_factor: Facteur d'agrandissement (2, 3, 4, etc.)
            method: Méthode d'upscaling
            
        Returns:
            Chemin du fichier de sortie ou None si échec
        """
        try:
            logger.info(f"Upscaling {input_path} with factor {scale_factor} using {method}")
            
            # Validation
            if method not in self.SUPPORTED_METHODS:
                logger.error(f"Unsupported method: {method}")
                return None
            
            # Chargement de l'image
            image = self._load_image(input_path)
            if image is None:
                return None
            
            # Upscaling selon la méthode
            if method in ['bilinear', 'bicubic', 'nearest', 'lanczos']:
                upscaled = self._pil_upscale(image, scale_factor, method)
            elif method == 'torch':
                upscaled = self._torch_upscale(image, scale_factor)
            elif method == 'srcnn':
                upscaled = self._srcnn_upscale(image, scale_factor)
            elif method == 'hybrid':
                upscaled = self._hybrid_upscale(image, scale_factor)
            else:
                logger.error(f"Method {method} not implemented")
                return None
            
            # Sauvegarde
            if upscaled is not None:
                self._save_image(upscaled, output_path)
                logger.info(f"Upscaled image saved to {output_path}")
                return str(output_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Error during upscaling: {e}")
            return None
    
    def _load_image(self, path: Union[str, Path]) -> Optional[Image.Image]:
        """Charge une image depuis un fichier"""
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return None
    
    def _save_image(self, image: Union[Image.Image, torch.Tensor], path: Union[str, Path]):
        """Sauvegarde une image"""
        if isinstance(image, torch.Tensor):
            image = self._tensor_to_pil(image)
        image.save(path)
    
    def _pil_upscale(self, image: Image.Image, scale_factor: int, method: str) -> Image.Image:
        """Upscaling avec PIL"""
        width, height = image.size
        new_size = (width * scale_factor, height * scale_factor)
        
        resample_map = {
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'nearest': Image.NEAREST,
            'lanczos': Image.LANCZOS
        }
        
        return image.resize(new_size, resample_map[method])
    
    def _torch_upscale(self, image: Image.Image, scale_factor: int) -> Image.Image:
        """Upscaling avec PyTorch (interpolation bilinéaire)"""
        # Conversion en tensor
        tensor = self._pil_to_tensor(image).unsqueeze(0).to(self.device)
        
        # Upscaling
        upscaled = F.interpolate(
            tensor,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False
        )
        
        # Retour en PIL
        return self._tensor_to_pil(upscaled.squeeze(0))
    
    def _srcnn_upscale(self, image: Image.Image, scale_factor: int) -> Image.Image:
        """
        Upscaling avec SRCNN simplifié
        Note: Version simplifiée sans modèle pré-entraîné
        """
        # Pour l'instant, utilise bicubic comme placeholder
        logger.warning("SRCNN not fully implemented, using bicubic instead")
        return self._pil_upscale(image, scale_factor, 'bicubic')
    
    def _hybrid_upscale(self, image: Image.Image, scale_factor: int) -> Image.Image:
        """Upscaling hybride combinant plusieurs méthodes"""
        # Première passe avec bicubic
        intermediate = self._pil_upscale(image, scale_factor, 'bicubic')
        
        # Amélioration avec torch
        tensor = self._pil_to_tensor(intermediate).unsqueeze(0).to(self.device)
        
        # Légère netteté supplémentaire
        kernel = torch.tensor([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]], dtype=torch.float32).to(self.device)
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        sharpened = F.conv2d(tensor, kernel, padding=1, groups=3)
        sharpened = torch.clamp(sharpened * 0.1 + tensor * 0.9, 0, 1)
        
        return self._tensor_to_pil(sharpened.squeeze(0))
    
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convertit une image PIL en tensor"""
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform(image)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convertit un tensor en image PIL"""
        # S'assurer que le tensor est sur CPU
        tensor = tensor.cpu()
        
        # Clamp les valeurs
        tensor = torch.clamp(tensor, 0, 1)
        
        # Conversion
        transform = transforms.ToPILImage()
        return transform(tensor)


# Classes de compatibilité pour l'ancien code
class SimpleImageUpscaler(UnifiedImageUpscaler):
    """Alias pour compatibilité avec l'ancienne API SimpleImageUpscaler"""
    
    def upscale_image(self, image_path: str, output_path: str, 
                     scale_factor: int = 2, method: str = 'bilinear') -> Optional[str]:
        """Interface compatible avec l'ancienne API"""
        return self.upscale(image_path, output_path, scale_factor, method)

