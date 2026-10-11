#!/usr/bin/env python3
"""
Tests unitaires pour le script generate_image
"""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch, Mock, MagicMock
import argparse

from core.generation.generate_image import main_simple


class TestGenerateImage(unittest.TestCase):
    """Tests pour le script de génération d'images"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Mock des modules externes
        self.mock_satellite_generator = patch('core.generation.generate_image.SimpleSatelliteGenerator')
        self.mock_upscaler = patch('core.generation.generate_image.SimpleImageUpscaler')
        
        # Démarrer les mocks
        self.mock_generator_class = self.mock_satellite_generator.start()
        self.mock_upscaler_class = self.mock_upscaler.start()
        
        # Configurer les instances mockées
        self.mock_generator_instance = Mock()
        self.mock_generator_instance.generate.return_value = 'generated_satellite.png'
        self.mock_generator_class.return_value = self.mock_generator_instance
        
        self.mock_upscaler_instance = Mock()
        self.mock_upscaler_instance.upscale_image.return_value = 'upscaled_image.png'
        self.mock_upscaler_class.return_value = self.mock_upscaler_instance
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        self.mock_satellite_generator.stop()
        self.mock_upscaler.stop()
    
    @patch('sys.argv', ['generate_image.py'])
    def test_default_generation(self):
        """Test de génération avec paramètres par défaut"""
        # Exécuter la fonction principale
        main_simple()
        
        # Vérifier que le générateur a été créé
        self.mock_generator_class.assert_called_once_with(None)
        
        # Vérifier que generate a été appelé avec les bons paramètres
        self.mock_generator_instance.generate.assert_called_once_with(
            None, 'urban', 'generated_satellite.png'
        )
    
    @patch('sys.argv', ['generate_image.py', '--checkpoint', 'model.pth', '--input_map', 'map.png'])
    def test_generation_with_checkpoint_and_map(self):
        """Test avec checkpoint et carte d'entrée"""
        main_simple()
        
        # Vérifier que le checkpoint est passé
        self.mock_generator_class.assert_called_once_with('model.pth')
        
        # Vérifier que la carte est utilisée
        self.mock_generator_instance.generate.assert_called_once_with(
            'map.png', 'urban', 'generated_satellite.png'
        )
    
    @patch('sys.argv', ['generate_image.py', '--environment', 'maritime', '--output', 'ocean.png'])
    def test_generation_with_environment(self):
        """Test avec environnement spécifique"""
        main_simple()
        
        # Vérifier l'environnement et le fichier de sortie
        self.mock_generator_instance.generate.assert_called_once_with(
            None, 'maritime', 'ocean.png'
        )
    
    @patch('sys.argv', ['generate_image.py', '--upscale', '2'])
    def test_generation_with_upscaling(self):
        """Test avec upscaling"""
        main_simple()
        
        # Vérifier que l'upscaler est appelé
        self.mock_upscaler_instance.upscale_image.assert_called_once()
        
        # Vérifier les paramètres de l'upscaling
        call_args = self.mock_upscaler_instance.upscale_image.call_args
        self.assertEqual(call_args[0][0], 'generated_satellite.png')  # input
        self.assertEqual(call_args[0][2], 2)  # scale factor
        self.assertEqual(call_args[0][3], 'bilinear')  # method par défaut
    
    @patch('sys.argv', ['generate_image.py', '--upscale', '4', '--upscale_method', 'bicubic'])
    def test_generation_with_custom_upscaling(self):
        """Test avec upscaling personnalisé"""
        main_simple()
        
        # Vérifier les paramètres personnalisés
        call_args = self.mock_upscaler_instance.upscale_image.call_args
        self.assertEqual(call_args[0][2], 4)  # scale factor
        self.assertEqual(call_args[0][3], 'bicubic')  # method
    
    @patch('sys.argv', ['generate_image.py', '--upscale', '2'])
    def test_upscaling_failure_handling(self):
        """Test de gestion d'échec d'upscaling"""
        # Simuler un échec d'upscaling
        self.mock_upscaler_instance.upscale_image.return_value = None
        
        # La fonction devrait s'exécuter sans erreur
        try:
            main_simple()
        except Exception as e:
            self.fail(f"main_simple() raised {e} unexpectedly!")
    
    def test_all_environment_options(self):
        """Test de tous les environnements disponibles"""
        environments = ['urban', 'maritime', 'nature', 'default']
        
        for env in environments:
            with self.subTest(environment=env):
                with patch('sys.argv', ['generate_image.py', '--environment', env]):
                    # Réinitialiser le mock
                    self.mock_generator_instance.generate.reset_mock()
                    
                    main_simple()
                    
                    # Vérifier que l'environnement correct est utilisé
                    call_args = self.mock_generator_instance.generate.call_args
                    self.assertEqual(call_args[0][1], env)
    
    def test_all_upscale_methods(self):
        """Test de toutes les méthodes d'upscaling"""
        methods = ['bilinear', 'bicubic', 'nearest', 'lanczos', 'torch']
        
        for method in methods:
            with self.subTest(method=method):
                with patch('sys.argv', ['generate_image.py', '--upscale', '2', '--upscale_method', method]):
                    # Réinitialiser le mock
                    self.mock_upscaler_instance.upscale_image.reset_mock()
                    
                    main_simple()
                    
                    # Vérifier que la méthode correcte est utilisée
                    call_args = self.mock_upscaler_instance.upscale_image.call_args
                    self.assertEqual(call_args[0][3], method)
    
    @patch('sys.argv', ['generate_image.py', '--output', 'custom_output.png', '--upscale', '3'])
    def test_upscaled_filename_generation(self):
        """Test de génération du nom de fichier upscalé"""
        self.mock_generator_instance.generate.return_value = 'custom_output.png'
        main_simple()
        
        # Vérifier que le nom de fichier upscalé est correct
        call_args = self.mock_upscaler_instance.upscale_image.call_args
        expected_output = 'custom_output_upscaled_3x.png'
        self.assertEqual(call_args[0][1], expected_output)
    
    @patch('sys.argv', ['generate_image.py', '--output', 'results/custom_output', '--upscale', '2'])
    def test_upscaled_filename_without_extension(self):
        """Test du nom de fichier upscalé quand la sortie n'a pas d'extension"""
        self.mock_generator_instance.generate.return_value = 'results/custom_output'
        main_simple()
        
        call_args = self.mock_upscaler_instance.upscale_image.call_args
        self.assertEqual(Path(call_args[0][1]), Path('results/custom_output_upscaled_2x'))
    
    @patch('sys.argv', ['generate_image.py', '--help'])
    def test_help_argument(self):
        """Test de l'argument --help"""
        # L'argument --help devrait lever SystemExit
        with self.assertRaises(SystemExit):
            main_simple()


if __name__ == '__main__':
    unittest.main()