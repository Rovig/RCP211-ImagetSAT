#!/usr/bin/env python3
"""
Tests unitaires pour le module model_manager
"""

import unittest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

from core.models.model_manager import ModelManager, OriginalGenerator, ModelLoader


class TestModelManager(unittest.TestCase):
    """Tests pour le gestionnaire de modèles"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.model_manager = ModelManager()
        self.device = torch.device('cpu')  # Utiliser CPU pour les tests
    
    def test_model_loader_interface(self):
        """Test que ModelLoader est bien une classe abstraite"""
        # Ne devrait pas pouvoir instancier ModelLoader directement
        with self.assertRaises(TypeError):
            ModelLoader()
    
    def test_original_generator_architecture(self):
        """Test de l'architecture du générateur"""
        generator = OriginalGenerator()
        
        # Vérifier que c'est un module PyTorch
        self.assertIsInstance(generator, nn.Module)
        
        # Vérifier la présence des couches principales
        self.assertTrue(hasattr(generator, 'down1'))
        self.assertTrue(hasattr(generator, 'down8'))
        self.assertTrue(hasattr(generator, 'up1'))
        self.assertTrue(hasattr(generator, 'up8'))
    
    def test_generator_forward_pass(self):
        """Test de la passe forward du générateur"""
        generator = OriginalGenerator()
        generator.eval()
        
        # Créer une entrée de test
        batch_size = 1
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        
        # Passe forward
        with torch.no_grad():
            output = generator(input_tensor)
        
        # Vérifications
        self.assertEqual(output.shape, (batch_size, 3, 256, 256))
        # Tanh activation devrait donner des valeurs entre -1 et 1
        self.assertTrue(output.min() >= -1.0)
        self.assertTrue(output.max() <= 1.0)
    
    def test_create_generator(self):
        """Test de création d'un nouveau générateur"""
        generator = self.model_manager.create_generator(self.device)
        
        # Vérifications
        self.assertIsInstance(generator, OriginalGenerator)
        # Vérifier que le modèle est sur le bon device
        next(generator.parameters()).device.type == self.device.type
    
    def test_create_generator_custom_size(self):
        """Test de création avec taille personnalisée"""
        generator_256 = self.model_manager.create_generator(self.device, image_size=256)
        generator_512 = self.model_manager.create_generator(self.device, image_size=512)
        
        # Les deux devraient être créés sans erreur
        self.assertIsInstance(generator_256, OriginalGenerator)
        self.assertIsInstance(generator_512, OriginalGenerator)
    
    def test_create_discriminator(self):
        """Test de création d'un discriminateur"""
        discriminator = self.model_manager.create_discriminator(self.device)
        
        # Vérifications de base
        self.assertIsInstance(discriminator, nn.Module)
        
        # Test de la passe forward
        input_tensor = torch.randn(1, 6, 256, 256)  # 3 + 3 canaux
        with torch.no_grad():
            output = discriminator(input_tensor)
        
        # La sortie devrait être un patch
        self.assertEqual(len(output.shape), 4)
        self.assertEqual(output.shape[1], 1)  # 1 canal de sortie
    
    @patch('torch.load')
    def test_load_generator_with_checkpoint(self, mock_load):
        """Test de chargement d'un générateur depuis un checkpoint"""
        # Préparer le mock du checkpoint
        mock_checkpoint = {
            'generator_state_dict': OriginalGenerator().state_dict(),
            'epoch': 100,
            'g_losses': [0.1, 0.2],
            'd_losses': [0.3, 0.4]
        }
        mock_load.return_value = mock_checkpoint
        
        # Charger le générateur
        generator = self.model_manager.load_generator('fake_checkpoint.pth', self.device)
        
        # Vérifications
        self.assertIsInstance(generator, OriginalGenerator)
        mock_load.assert_called_once_with('fake_checkpoint.pth', map_location=self.device, weights_only=True)
    
    def test_load_generator_without_checkpoint(self):
        """Test de chargement sans checkpoint (nouveau modèle)"""
        generator = self.model_manager.load_generator(None, self.device)
        
        # Devrait retourner un nouveau générateur
        self.assertIsInstance(generator, OriginalGenerator)
    
    @patch('torch.load')
    def test_load_generator_with_corrupted_checkpoint(self, mock_load):
        """Test avec un checkpoint corrompu"""
        # Simuler une erreur de chargement
        mock_load.side_effect = Exception("Corrupted file")
        
        # Le chargement devrait échouer proprement
        with self.assertRaises(Exception):
            self.model_manager.load_generator('corrupted.pth', self.device)
    
    @patch('torch.load')
    def test_load_discriminator_with_checkpoint(self, mock_load):
        """Test de chargement d'un discriminateur depuis un checkpoint"""
        # Préparer le mock du checkpoint
        mock_checkpoint = {
            'discriminator_state_dict': self.model_manager.create_discriminator(self.device).state_dict(),
            'epoch': 100
        }
        mock_load.return_value = mock_checkpoint
        
        # Charger le discriminateur
        discriminator = self.model_manager.load_discriminator('fake_checkpoint.pth', self.device)
        
        # Vérifications
        self.assertIsInstance(discriminator, nn.Module)
        mock_load.assert_called_once()
    
    def test_generator_parameter_count(self):
        """Test du nombre de paramètres du générateur"""
        generator = OriginalGenerator()
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in generator.parameters())
        trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        
        # Le générateur devrait avoir des millions de paramètres
        self.assertGreater(total_params, 1_000_000)
        self.assertEqual(total_params, trainable_params)  # Tous devraient être entraînables
    
    def test_different_image_sizes(self):
        """Test avec différentes tailles d'images"""
        sizes = [256, 512]
        
        for size in sizes:
            with self.subTest(size=size):
                generator = self.model_manager.create_generator(self.device, image_size=size)
                
                # Test forward pass
                input_tensor = torch.randn(1, 3, size, size)
                with torch.no_grad():
                    output = generator(input_tensor)
                
                # Vérifier la taille de sortie
                self.assertEqual(output.shape, (1, 3, size, size))
    
    def test_model_eval_mode(self):
        """Test que les modèles peuvent passer en mode évaluation"""
        generator = self.model_manager.create_generator(self.device)
        discriminator = self.model_manager.create_discriminator(self.device)
        
        # Passer en mode eval
        generator.eval()
        discriminator.eval()
        
        # Vérifier le mode
        self.assertFalse(generator.training)
        self.assertFalse(discriminator.training)
    
    def test_save_and_load_consistency(self):
        """Test de sauvegarde et chargement pour vérifier la cohérence"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'test_checkpoint.pth'
            
            # Créer et sauvegarder un modèle
            original_generator = self.model_manager.create_generator(self.device)
            
            # Créer un checkpoint factice
            checkpoint = {
                'generator_state_dict': original_generator.state_dict(),
                'epoch': 10,
                'g_losses': [0.5],
                'd_losses': [0.6]
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Charger le modèle
            loaded_generator = self.model_manager.load_generator(str(checkpoint_path), self.device)
            
            # Vérifier que les poids sont identiques
            for (name1, param1), (name2, param2) in zip(
                original_generator.named_parameters(),
                loaded_generator.named_parameters()
            ):
                self.assertEqual(name1, name2)
                self.assertTrue(torch.allclose(param1, param2))


if __name__ == '__main__':
    unittest.main()