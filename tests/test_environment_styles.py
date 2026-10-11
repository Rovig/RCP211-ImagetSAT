#!/usr/bin/env python3
"""
Tests unitaires pour le module environment_styles
"""

import unittest
import numpy as np
from PIL import Image
from unittest.mock import patch

from core.generation.environment_styles import EnvironmentStyler


class TestEnvironmentStyler(unittest.TestCase):
    """Tests pour le styliseur d'environnements"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.styler = EnvironmentStyler()
        # Créer une image de test de base
        self.test_image = Image.new('RGB', (100, 100), color=(128, 128, 128))
    
    def test_init(self):
        """Test de l'initialisation"""
        styler = EnvironmentStyler()
        self.assertIsInstance(styler, EnvironmentStyler)
    
    def test_apply_default_style(self):
        """Test du style par défaut (aucun changement)"""
        result = self.styler.apply_style(self.test_image, 'default')
        
        # L'image ne devrait pas changer avec le style default
        self.assertEqual(result.size, self.test_image.size)
        self.assertEqual(result.mode, self.test_image.mode)
        
        # Comparer les pixels (devrait être identique)
        original_array = np.array(self.test_image)
        result_array = np.array(result)
        np.testing.assert_array_equal(original_array, result_array)
    
    def test_apply_urban_style(self):
        """Test du style urbain"""
        result = self.styler.apply_style(self.test_image, 'urban')
        
        # Vérifications de base
        self.assertEqual(result.size, self.test_image.size)
        self.assertEqual(result.mode, 'RGB')
        
        # Le style urbain devrait réduire la saturation
        # Créer une image colorée pour mieux tester
        colorful_image = Image.new('RGB', (100, 100), color=(255, 100, 50))
        urban_result = self.styler.apply_style(colorful_image, 'urban')
        
        # Vérifier que l'image est moins saturée
        original_array = np.array(colorful_image)
        urban_array = np.array(urban_result)
        
        # La saturation devrait être réduite (les couleurs plus proches du gris)
        self.assertNotEqual(original_array.tolist(), urban_array.tolist())
    
    def test_apply_maritime_style(self):
        """Test du style maritime"""
        result = self.styler.apply_style(self.test_image, 'maritime')
        
        # Vérifications de base
        self.assertEqual(result.size, self.test_image.size)
        self.assertEqual(result.mode, 'RGB')
        
        # Le style maritime devrait augmenter les bleus
        result_array = np.array(result)
        original_array = np.array(self.test_image)
        
        # Vérifier que l'image a changé
        self.assertFalse(np.array_equal(original_array, result_array))
    
    def test_apply_nature_style(self):
        """Test du style nature"""
        result = self.styler.apply_style(self.test_image, 'nature')
        
        # Vérifications de base
        self.assertEqual(result.size, self.test_image.size)
        self.assertEqual(result.mode, 'RGB')
        
        # Le style nature devrait augmenter les verts
        result_array = np.array(result)
        original_array = np.array(self.test_image)
        
        # Vérifier que le canal vert a été amplifié
        self.assertGreater(result_array[:, :, 1].mean(), original_array[:, :, 1].mean())
    
    def test_invalid_style(self):
        """Test avec un style invalide"""
        # Un style invalide devrait retourner l'image originale
        result = self.styler.apply_style(self.test_image, 'invalid_style')
        
        # Devrait retourner l'image sans modification
        self.assertEqual(result.size, self.test_image.size)
        original_array = np.array(self.test_image)
        result_array = np.array(result)
        np.testing.assert_array_equal(original_array, result_array)
    
    def test_style_preservation_of_dimensions(self):
        """Test que les styles préservent les dimensions"""
        styles = ['urban', 'maritime', 'nature', 'default']
        sizes = [(50, 50), (256, 256), (512, 512)]
        
        for size in sizes:
            test_img = Image.new('RGB', size, color=(100, 150, 200))
            
            for style in styles:
                with self.subTest(style=style, size=size):
                    result = self.styler.apply_style(test_img, style)
                    self.assertEqual(result.size, size)
                    self.assertEqual(result.mode, 'RGB')
    
    def test_style_with_grayscale_image(self):
        """Test des styles avec une image en niveaux de gris"""
        # Créer une image grayscale
        gray_image = Image.new('L', (100, 100), color=128)
        
        # Convertir en RGB pour le test
        rgb_gray = gray_image.convert('RGB')
        
        styles = ['urban', 'maritime', 'nature']
        for style in styles:
            with self.subTest(style=style):
                result = self.styler.apply_style(rgb_gray, style)
                self.assertEqual(result.mode, 'RGB')
                self.assertEqual(result.size, (100, 100))
    
    def test_style_extreme_values(self):
        """Test avec des valeurs extrêmes"""
        # Image très claire
        bright_image = Image.new('RGB', (50, 50), color=(255, 255, 255))
        # Image très sombre
        dark_image = Image.new('RGB', (50, 50), color=(0, 0, 0))
        
        styles = ['urban', 'maritime', 'nature']
        
        for style in styles:
            with self.subTest(style=style, image_type='bright'):
                result = self.styler.apply_style(bright_image, style)
                # Vérifier que les valeurs restent dans les limites
                result_array = np.array(result)
                self.assertTrue((result_array >= 0).all())
                self.assertTrue((result_array <= 255).all())
            
            with self.subTest(style=style, image_type='dark'):
                result = self.styler.apply_style(dark_image, style)
                result_array = np.array(result)
                self.assertTrue((result_array >= 0).all())
                self.assertTrue((result_array <= 255).all())
    
    @patch('cv2.cvtColor')
    def test_hsv_conversion_error_handling(self, mock_cvtColor):
        """Test de la gestion d'erreur lors de la conversion HSV"""
        # Simuler une erreur dans cv2
        mock_cvtColor.side_effect = Exception("CV2 Error")
        
        # Le styler devrait gérer l'erreur gracieusement
        # et retourner l'image originale
        result = self.styler.apply_style(self.test_image, 'urban')
        
        # Vérifier que l'image est retournée malgré l'erreur
        self.assertIsInstance(result, Image.Image)
    
    def test_style_consistency(self):
        """Test que l'application répétée du même style donne le même résultat"""
        styles = ['urban', 'maritime', 'nature']
        
        for style in styles:
            with self.subTest(style=style):
                # Appliquer le style deux fois
                result1 = self.styler.apply_style(self.test_image, style)
                result2 = self.styler.apply_style(self.test_image, style)
                
                # Les résultats devraient être identiques
                array1 = np.array(result1)
                array2 = np.array(result2)
                np.testing.assert_array_equal(array1, array2)


if __name__ == '__main__':
    unittest.main()