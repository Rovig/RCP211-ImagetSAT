#!/usr/bin/env python3
"""
Tests unitaires pour le module satellite_generator
"""

import unittest
import tempfile
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from core.generation.satellite_generator import SimpleSatelliteGenerator


class TestSatelliteGenerator(unittest.TestCase):
    """Tests pour le générateur d'images satellites"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Mock du ModelManager pour éviter de charger de vrais modèles
        self.mock_model_manager = patch('core.generation.satellite_generator.ModelManager')
        self.mock_model_manager_instance = self.mock_model_manager.start()
        
        # Mock du modèle générateur
        self.mock_generator = Mock()
        self.mock_generator.image_size = 256
        self.mock_generator.return_value = torch.randn(1, 3, 256, 256)
        
        # Mock du ModelManager pour retourner notre mock generator
        mock_manager = Mock()
        mock_manager.load_generator.return_value = self.mock_generator
        self.mock_model_manager_instance.return_value = mock_manager
        
        # Mock de l'EnvironmentStyler
        self.mock_styler = patch('core.generation.satellite_generator.EnvironmentStyler')
        self.mock_styler_instance = self.mock_styler.start()
        mock_styler_obj = Mock()
        mock_styler_obj.apply_style.side_effect = lambda img, style: img
        self.mock_styler_instance.return_value = mock_styler_obj
        
    def tearDown(self):
        """Nettoyage après chaque test"""
        self.mock_model_manager.stop()
        self.mock_styler.stop()
    
    def test_init(self):
        """Test de l'initialisation du générateur"""
        generator = SimpleSatelliteGenerator()
        
        # Vérifications
        self.assertIsNotNone(generator.device)
        self.assertIsNotNone(generator.model_manager)
        self.assertIsNotNone(generator.generator)
        self.assertIsNotNone(generator.styler)
    
    def test_init_with_checkpoint(self):
        """Test de l'initialisation avec un checkpoint"""
        checkpoint_path = "fake_checkpoint.pth"
        generator = SimpleSatelliteGenerator(checkpoint_path)
        
        # Vérifier que le checkpoint est passé au ModelManager
        generator.model_manager.load_generator.assert_called_once()
        call_args = generator.model_manager.load_generator.call_args
        self.assertEqual(call_args[0][0], checkpoint_path)
    
    def test_generate_with_random_map(self):
        """Test de génération avec carte aléatoire"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.png"
            
            generator = SimpleSatelliteGenerator()
            result = generator.generate(
                input_map_path=None,
                environment='default',
                output_path=str(output_path)
            )
            
            # Vérifications
            self.assertEqual(result, str(output_path))
            self.assertTrue(output_path.exists())
            
            # Vérifier l'image générée
            img = Image.open(output_path)
            self.assertEqual(img.size, (256, 256))
            self.assertEqual(img.mode, 'RGB')
    
    def test_generate_with_input_map(self):
        """Test de génération avec carte d'entrée"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Créer une carte d'entrée
            input_path = Path(tmpdir) / "input.png"
            input_image = Image.new('RGB', (256, 256), color=(255, 255, 255))
            input_image.save(input_path)
            
            output_path = Path(tmpdir) / "output.png"
            
            generator = SimpleSatelliteGenerator()
            result = generator.generate(
                input_map_path=str(input_path),
                environment='urban',
                output_path=str(output_path)
            )
            
            # Vérifications
            self.assertEqual(result, str(output_path))
            self.assertTrue(output_path.exists())
    
    def test_generate_with_different_environments(self):
        """Test de génération avec différents environnements"""
        environments = ['urban', 'maritime', 'nature', 'default']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SimpleSatelliteGenerator()
            
            for env in environments:
                with self.subTest(environment=env):
                    output_path = Path(tmpdir) / f"output_{env}.png"
                    
                    result = generator.generate(
                        environment=env,
                        output_path=str(output_path)
                    )
                    
                    self.assertTrue(output_path.exists())
                    # Vérifier que le style a été appliqué
                    generator.styler.apply_style.assert_called()
    
    def test_load_and_process_map(self):
        """Test du chargement et traitement d'une carte"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Créer une image de test
            test_path = Path(tmpdir) / "test_map.png"
            test_image = Image.new('RGB', (512, 512), color=(128, 128, 128))
            test_image.save(test_path)
            
            generator = SimpleSatelliteGenerator()
            tensor = generator._load_and_process_map(str(test_path))
            
            # Vérifications
            self.assertEqual(tensor.shape, (1, 3, 256, 256))  # Redimensionné
            self.assertTrue(torch.is_tensor(tensor))
            self.assertTrue(-1 <= tensor.min() <= tensor.max() <= 1)  # Normalisé
    
    def test_create_random_map(self):
        """Test de création de carte aléatoire"""
        generator = SimpleSatelliteGenerator()
        tensor = generator._create_random_map()
        
        # Vérifications
        self.assertEqual(tensor.shape, (1, 3, 256, 256))
        self.assertTrue(torch.is_tensor(tensor))
        self.assertTrue(-1 <= tensor.min() <= tensor.max() <= 1)
    
    def test_tensor_to_numpy(self):
        """Test de conversion tensor vers numpy"""
        generator = SimpleSatelliteGenerator()
        
        # Créer un tensor de test
        tensor = torch.rand(1, 3, 256, 256) * 2 - 1  # Valeurs entre -1 et 1
        
        # Convertir
        numpy_array = generator._tensor_to_numpy(tensor)
        
        # Vérifications
        self.assertEqual(numpy_array.shape, (256, 256, 3))
        self.assertEqual(numpy_array.dtype, np.uint8)
        self.assertTrue(0 <= numpy_array.min() <= numpy_array.max() <= 255)
    
    def test_device_selection(self):
        """Test de la sélection du device (CUDA, puis MPS, puis CPU)"""
        cases = [
            (False, False, 'cpu'),
            (False, True, 'mps'),
            (True, False, 'cuda'),
            (True, True, 'cuda'),
        ]
        
        for cuda, mps, expected in cases:
            with self.subTest(cuda=cuda, mps=mps):
                with patch('torch.cuda.is_available', return_value=cuda), \
                     patch('torch.backends.mps.is_available', return_value=mps):
                    generator = SimpleSatelliteGenerator()
                    self.assertEqual(generator.device.type, expected)
    
    def test_generate_with_missing_input_map(self):
        """Test qu'une carte d'entrée introuvable est une erreur"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.png"
            
            generator = SimpleSatelliteGenerator()
            with self.assertRaises(FileNotFoundError):
                generator.generate(
                    input_map_path=str(Path(tmpdir) / "absente.png"),
                    output_path=str(output_path)
                )
            
            self.assertFalse(output_path.exists())
    
    def test_generate_different_output_formats(self):
        """Test de génération avec différents formats de sortie"""
        formats = ['.png', '.jpg', '.jpeg']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SimpleSatelliteGenerator()
            
            for fmt in formats:
                with self.subTest(format=fmt):
                    output_path = Path(tmpdir) / f"output{fmt}"
                    
                    result = generator.generate(
                        output_path=str(output_path)
                    )
                    
                    self.assertTrue(Path(result).exists())
                    self.assertEqual(Path(result).suffix, fmt)


if __name__ == '__main__':
    unittest.main()