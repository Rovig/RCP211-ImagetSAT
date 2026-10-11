#!/usr/bin/env python3
"""
Tests unitaires pour le module unified_upscaler
"""

import unittest
import tempfile
import os
from pathlib import Path
import numpy as np
from PIL import Image

from utils.unified_upscaler import UnifiedImageUpscaler, SimpleImageUpscaler


class TestUnifiedUpscaler(unittest.TestCase):
    """Tests pour le module unified_upscaler"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.upscaler = UnifiedImageUpscaler()
        self.test_image_size = (64, 64)
        
    def create_test_image(self, size=(64, 64), color=(255, 0, 0)):
        """Crée une image de test"""
        image = Image.new('RGB', size, color)
        return image
    
    def test_init(self):
        """Test de l'initialisation"""
        upscaler = UnifiedImageUpscaler()
        self.assertIsNotNone(upscaler.device)
        self.assertEqual(len(upscaler.SUPPORTED_METHODS), 7)
    
    def test_upscale_bilinear(self):
        """Test de l'upscaling bilinéaire"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Créer une image de test
            input_path = Path(tmpdir) / 'test_input.png'
            output_path = Path(tmpdir) / 'test_output.png'
            
            test_image = self.create_test_image()
            test_image.save(input_path)
            
            # Upscaler l'image
            result = self.upscaler.upscale(input_path, output_path, scale_factor=2, method='bilinear')
            
            # Vérifications
            self.assertIsNotNone(result)
            self.assertTrue(Path(result).exists())
            
            # Vérifier les dimensions
            upscaled_image = Image.open(result)
            self.assertEqual(upscaled_image.size, (128, 128))
    
    def test_upscale_multiple_methods(self):
        """Test de plusieurs méthodes d'upscaling"""
        methods_to_test = ['bilinear', 'bicubic', 'nearest', 'lanczos']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'test_input.png'
            test_image = self.create_test_image()
            test_image.save(input_path)
            
            for method in methods_to_test:
                with self.subTest(method=method):
                    output_path = Path(tmpdir) / f'test_output_{method}.png'
                    result = self.upscaler.upscale(input_path, output_path, scale_factor=2, method=method)
                    
                    self.assertIsNotNone(result)
                    self.assertTrue(Path(result).exists())
    
    def test_invalid_method(self):
        """Test avec une méthode invalide"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'test_input.png'
            output_path = Path(tmpdir) / 'test_output.png'
            
            test_image = self.create_test_image()
            test_image.save(input_path)
            
            result = self.upscaler.upscale(input_path, output_path, scale_factor=2, method='invalid')
            
            self.assertIsNone(result)
    
    def test_torch_upscale(self):
        """Test de l'upscaling avec PyTorch"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'test_input.png'
            output_path = Path(tmpdir) / 'test_output.png'
            
            test_image = self.create_test_image()
            test_image.save(input_path)
            
            result = self.upscaler.upscale(input_path, output_path, scale_factor=3, method='torch')
            
            self.assertIsNotNone(result)
            upscaled_image = Image.open(result)
            self.assertEqual(upscaled_image.size, (192, 192))
    
    def test_compatibility_simple_upscaler(self):
        """Test de compatibilité avec SimpleImageUpscaler"""
        simple_upscaler = SimpleImageUpscaler()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'test_input.png'
            output_path = Path(tmpdir) / 'test_output.png'
            
            test_image = self.create_test_image()
            test_image.save(str(input_path))
            
            result = simple_upscaler.upscale_image(
                str(input_path), str(output_path), scale_factor=2, method='bilinear'
            )
            
            self.assertIsNotNone(result)
            self.assertTrue(Path(result).exists())
    
    def test_scale_factors(self):
        """Test avec différents facteurs d'échelle"""
        scale_factors = [2, 3, 4]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'test_input.png'
            test_image = self.create_test_image((50, 50))
            test_image.save(input_path)
            
            for factor in scale_factors:
                with self.subTest(factor=factor):
                    output_path = Path(tmpdir) / f'test_output_{factor}x.png'
                    result = self.upscaler.upscale(input_path, output_path, scale_factor=factor)
                    
                    self.assertIsNotNone(result)
                    upscaled_image = Image.open(result)
                    expected_size = (50 * factor, 50 * factor)
                    self.assertEqual(upscaled_image.size, expected_size)


if __name__ == '__main__':
    unittest.main()