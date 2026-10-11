#!/usr/bin/env python3
"""
Styles d'environnement appliqués aux images générées
"""

import numpy as np
from PIL import Image, ImageEnhance


class EnvironmentStyler:
    """Styleur simplifié pour images PIL"""
    
    def apply_style(self, image, environment_type: str):
        """Applique un style d'environnement à une image PIL"""
        if environment_type == 'default':
            return image
            
        if environment_type == 'urban':
            # Style urbain - contraste élevé, tons gris/béton
            enhancer = ImageEnhance.Contrast(image)
            styled = enhancer.enhance(1.5)  # Contraste beaucoup plus fort
            enhancer = ImageEnhance.Color(styled)
            styled = enhancer.enhance(0.6)  # Désaturation plus marquée
            enhancer = ImageEnhance.Brightness(styled)
            styled = enhancer.enhance(0.9)  # Légèrement plus sombre
            return styled
            
        elif environment_type == 'maritime':
            # Style maritime - dominante bleue prononcée
            img_array = np.array(image)
            # Renforcer le canal bleu et réduire rouge/vert
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 0.7, 0, 255)  # Moins de rouge
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 0.8, 0, 255)  # Moins de vert
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.4, 0, 255)  # Plus de bleu
            styled = Image.fromarray(img_array.astype(np.uint8))
            # Augmenter saturation pour effet plus marqué
            enhancer = ImageEnhance.Color(styled)
            styled = enhancer.enhance(1.6)
            return styled
            
        elif environment_type == 'nature':
            # Style nature - saturation verte intensive
            img_array = np.array(image)
            # Renforcer le canal vert
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 0.9, 0, 255)  # Moins de rouge
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.3, 0, 255)  # Plus de vert
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.8, 0, 255)  # Moins de bleu
            styled = Image.fromarray(img_array.astype(np.uint8))
            # Augmenter saturation et luminosité
            enhancer = ImageEnhance.Color(styled)
            styled = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Brightness(styled)
            styled = enhancer.enhance(1.2)
            return styled
            
        else:
            return image