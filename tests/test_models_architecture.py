#!/usr/bin/env python3
"""
Tests unitaires pour les architectures de modèles
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from core.models.model_manager import (
    OriginalGenerator, ModelManager, ModelLoader, 
    GeneratorLoader
)


class TestGeneratorArchitecture(unittest.TestCase):
    """Tests pour l'architecture du générateur"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.image_size = 256
        self.input_channels = 3
        self.output_channels = 3
    
    def test_generator_structure(self):
        """Test de la structure du générateur U-Net"""
        generator = OriginalGenerator(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            ngf=64
        )
        
        # Vérifier la présence des couches encoder
        for i in range(1, 9):
            self.assertTrue(hasattr(generator, f'down{i}'))
        
        # Vérifier la présence des couches decoder
        for i in range(1, 9):
            self.assertTrue(hasattr(generator, f'up{i}'))
    
    def test_generator_forward_shape(self):
        """Test des dimensions de sortie du générateur"""
        generator = OriginalGenerator()
        generator.eval()
        
        # Différentes tailles d'entrée (256 minimum: 8 divisions par 2)
        sizes = [256, 512]
        
        for size in sizes:
            with self.subTest(size=size):
                input_tensor = torch.randn(1, 3, size, size)
                
                with torch.no_grad():
                    output = generator(input_tensor)
                
                # La sortie doit avoir la même taille que l'entrée
                self.assertEqual(output.shape, (1, 3, size, size))
    
    def test_generator_output_range(self):
        """Test de la plage de valeurs de sortie"""
        generator = OriginalGenerator()
        generator.eval()
        
        input_tensor = torch.randn(2, 3, 256, 256)
        
        with torch.no_grad():
            output = generator(input_tensor)
        
        # Tanh activation → valeurs entre -1 et 1
        self.assertGreaterEqual(output.min().item(), -1.0)
        self.assertLessEqual(output.max().item(), 1.0)
    
    def test_encoder_layers(self):
        """Test détaillé des couches de l'encodeur"""
        generator = OriginalGenerator(ngf=64)
        
        # Test de chaque couche down
        x = torch.randn(1, 3, 256, 256)
        
        # Down1: 3 → 64
        d1 = generator.down1(x)
        self.assertEqual(d1.shape[1], 64)
        self.assertEqual(d1.shape[2], 128)  # Taille divisée par 2
        
        # Down2: 64 → 128
        d2 = generator.down2(d1)
        self.assertEqual(d2.shape[1], 128)
        self.assertEqual(d2.shape[2], 64)
        
        # Down3: 128 → 256
        d3 = generator.down3(d2)
        self.assertEqual(d3.shape[1], 256)
        self.assertEqual(d3.shape[2], 32)
    
    def test_skip_connections(self):
        """Test des skip connections dans le U-Net"""
        generator = OriginalGenerator()
        
        # Tracer le forward pass
        x = torch.randn(1, 3, 256, 256)
        
        # Encoder
        d1 = generator.down1(x)
        d2 = generator.down2(d1)
        d3 = generator.down3(d2)
        d4 = generator.down4(d3)
        d5 = generator.down5(d4)
        d6 = generator.down6(d5)
        d7 = generator.down7(d6)
        d8 = generator.down8(d7)
        
        # Decoder avec skip connections
        u1 = generator.up1(d8)
        self.assertEqual(u1.shape[1], 512)  # Sortie up1
        
        # Concatenation avec d7
        u2_input = torch.cat([u1, d7], 1)
        self.assertEqual(u2_input.shape[1], 1024)  # 512 + 512
    
    def test_batch_processing(self):
        """Test du traitement par batch"""
        generator = OriginalGenerator()
        generator.eval()
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                input_batch = torch.randn(batch_size, 3, 256, 256)
                
                with torch.no_grad():
                    output_batch = generator(input_batch)
                
                self.assertEqual(output_batch.shape[0], batch_size)
                self.assertEqual(output_batch.shape[1:], (3, 256, 256))
    
    def test_gradient_flow(self):
        """Test du flux de gradients"""
        generator = OriginalGenerator()
        generator.train()
        
        input_tensor = torch.randn(1, 3, 256, 256, requires_grad=True)
        output = generator(input_tensor)
        
        # Créer une loss fictive
        loss = output.mean()
        loss.backward()
        
        # Vérifier que les gradients se propagent
        self.assertIsNotNone(input_tensor.grad)
        
        # Vérifier que tous les paramètres ont des gradients
        for name, param in generator.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Pas de gradient pour {name}")


class TestDiscriminatorArchitecture(unittest.TestCase):
    """Tests pour l'architecture du discriminateur"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.device = torch.device('cpu')
        self.model_manager = ModelManager()
    
    def test_discriminator_creation(self):
        """Test de création du discriminateur"""
        discriminator = self.model_manager.create_discriminator(self.device)
        
        self.assertIsInstance(discriminator, nn.Module)
        
        # Vérifier que les couches sont dans un Sequential
        self.assertIsInstance(discriminator.model, nn.Sequential)
    
    def test_discriminator_forward(self):
        """Test du forward pass du discriminateur"""
        discriminator = self.model_manager.create_discriminator(self.device)
        discriminator.eval()
        
        # Input: concaténation de l'image conditionnelle et générée
        input_tensor = torch.randn(2, 6, 256, 256)  # 3 + 3 canaux
        
        with torch.no_grad():
            output = discriminator(input_tensor)
        
        # Vérifier la forme de sortie (PatchGAN)
        self.assertEqual(len(output.shape), 4)
        self.assertEqual(output.shape[1], 1)  # 1 canal de sortie
        
        # La taille de patch dépend de l'architecture
        self.assertGreater(output.shape[2], 1)  # Pas une sortie scalaire
        self.assertGreater(output.shape[3], 1)
    
    def test_discriminator_layers(self):
        """Test des couches du discriminateur"""
        discriminator = self.model_manager.create_discriminator(self.device)
        
        # Vérifier le nombre de couches
        layers = list(discriminator.model.children())
        
        # Compter les Conv2d
        conv_layers = [l for l in layers if isinstance(l, nn.Conv2d)]
        self.assertGreaterEqual(len(conv_layers), 4)
        
        # Vérifier la première couche (6 canaux d'entrée)
        first_conv = conv_layers[0]
        self.assertEqual(first_conv.in_channels, 6)
        
        # Vérifier la dernière couche (1 canal de sortie)
        last_conv = conv_layers[-1]
        self.assertEqual(last_conv.out_channels, 1)
    
    def test_discriminator_receptive_field(self):
        """Test du champ réceptif du discriminateur"""
        discriminator = self.model_manager.create_discriminator(self.device)
        
        # Différentes tailles d'entrée
        sizes = [64, 128, 256, 512]
        
        for size in sizes:
            with self.subTest(size=size):
                input_tensor = torch.randn(1, 6, size, size)
                
                with torch.no_grad():
                    output = discriminator(input_tensor)
                
                # La sortie devrait être plus petite que l'entrée
                self.assertLess(output.shape[2], size)
                self.assertLess(output.shape[3], size)


class TestModelManager(unittest.TestCase):
    """Tests pour le gestionnaire de modèles"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.device = torch.device('cpu')
        self.manager = ModelManager()
    
    def test_manager_initialization(self):
        """Test de l'initialisation du manager"""
        manager = ModelManager()
        self.assertIsInstance(manager, ModelManager)
    
    def test_create_generator_method(self):
        """Test de la méthode create_generator"""
        generator = self.manager.create_generator(self.device)
        
        self.assertIsInstance(generator, OriginalGenerator)
        
        # Vérifier que le modèle est sur le bon device
        param_device = next(generator.parameters()).device
        self.assertEqual(param_device.type, self.device.type)
    
    def test_create_discriminator_method(self):
        """Test de la méthode create_discriminator"""
        discriminator = self.manager.create_discriminator(self.device)
        
        self.assertIsInstance(discriminator, nn.Module)
        
        # Vérifier le device
        param_device = next(discriminator.parameters()).device
        self.assertEqual(param_device.type, self.device.type)
    
    def test_load_generator_new(self):
        """Test du chargement d'un nouveau générateur"""
        generator = self.manager.load_generator(None, self.device)
        
        self.assertIsInstance(generator, OriginalGenerator)
        
        # Devrait être en mode eval
        self.assertFalse(generator.training)
    
    @patch('torch.load')
    def test_load_generator_from_checkpoint(self, mock_load):
        """Test du chargement depuis checkpoint"""
        # Mock du checkpoint
        state_dict = OriginalGenerator().state_dict()
        mock_checkpoint = {
            'generator_state_dict': state_dict,
            'epoch': 50,
            'g_losses': [0.5, 0.4, 0.3]
        }
        mock_load.return_value = mock_checkpoint
        
        # Charger
        generator = self.manager.load_generator('fake_path.pth', self.device)
        
        self.assertIsInstance(generator, OriginalGenerator)
        mock_load.assert_called_once()
    
    def test_model_compatibility(self):
        """Test de compatibilité entre générateur et discriminateur"""
        generator = self.manager.create_generator(self.device)
        discriminator = self.manager.create_discriminator(self.device)
        
        # Test avec des données
        condition = torch.randn(1, 3, 256, 256)
        
        # Génération
        with torch.no_grad():
            fake = generator(condition)
        
        # Discrimination
        disc_input = torch.cat([condition, fake], dim=1)
        with torch.no_grad():
            prediction = discriminator(disc_input)
        
        # Vérifier que tout fonctionne
        self.assertIsNotNone(prediction)


class TestModelHelpers(unittest.TestCase):
    """Tests pour les classes helper"""
    
    def test_model_loader_interface(self):
        """Test de l'interface ModelLoader"""
        # Ne devrait pas pouvoir instancier directement
        with self.assertRaises(TypeError):
            ModelLoader()
    
    def test_generator_loader(self):
        """Test du GeneratorLoader"""
        loader = GeneratorLoader(OriginalGenerator)
        
        # Test de chargement sans checkpoint
        device = torch.device('cpu')
        generator = loader.load(None, device)
        
        self.assertIsInstance(generator, OriginalGenerator)
        self.assertFalse(generator.training)  # Devrait être en mode eval
    
    @patch('torch.load')
    def test_generator_loader_with_checkpoint(self, mock_load):
        """Test du GeneratorLoader avec checkpoint"""
        loader = GeneratorLoader(OriginalGenerator)
        device = torch.device('cpu')
        
        # Mock checkpoint
        mock_checkpoint = {
            'generator_state_dict': OriginalGenerator().state_dict()
        }
        mock_load.return_value = mock_checkpoint
        
        generator = loader.load('checkpoint.pth', device)
        
        self.assertIsInstance(generator, OriginalGenerator)
        mock_load.assert_called_once()
    

class TestModelPerformance(unittest.TestCase):
    """Tests de performance des modèles"""
    
    def test_inference_speed(self):
        """Test de la vitesse d'inférence"""
        import time
        
        generator = OriginalGenerator()
        generator.eval()
        
        input_tensor = torch.randn(1, 3, 256, 256)
        
        # Warmup
        with torch.no_grad():
            _ = generator(input_tensor)
        
        # Mesure du temps
        num_iterations = 10
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = generator(input_tensor)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        # L'inférence devrait être rapide sur CPU (< 1s par image)
        self.assertLess(avg_time, 1.0)
    
    def test_memory_usage(self):
        """Test de l'utilisation mémoire"""
        generator = OriginalGenerator()
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in generator.parameters())
        
        # Calculer la taille en MB (float32 = 4 bytes)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        # Le modèle ne devrait pas être trop gros
        self.assertLess(model_size_mb, 500)  # Moins de 500MB
        
        # Log pour information
        print(f"\nModèle: {total_params:,} paramètres ({model_size_mb:.2f} MB)")
    
    def test_different_precisions(self):
        """Test avec différentes précisions"""
        generator = OriginalGenerator()
        input_tensor = torch.randn(1, 3, 256, 256)
        
        # Float32 (par défaut)
        output_fp32 = generator(input_tensor)
        self.assertEqual(output_fp32.dtype, torch.float32)
        
        # Float16 (si supporté)
        if torch.cuda.is_available():
            generator_fp16 = generator.half()
            input_fp16 = input_tensor.half()
            
            with torch.no_grad():
                output_fp16 = generator_fp16(input_fp16)
            
            self.assertEqual(output_fp16.dtype, torch.float16)


class TestModelRobustness(unittest.TestCase):
    """Tests de robustesse des modèles"""
    
    def test_edge_cases(self):
        """Test des cas limites"""
        generator = OriginalGenerator()
        generator.eval()
        
        # Entrée trop petite pour les 8 niveaux de downsampling
        small_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            with self.assertRaises(RuntimeError):
                generator(small_input)
        
        # Entrée avec valeurs extrêmes
        extreme_input = torch.ones(1, 3, 256, 256) * 10
        with torch.no_grad():
            extreme_output = generator(extreme_input)
        # Tanh devrait limiter la sortie
        self.assertLessEqual(extreme_output.max().item(), 1.0)
        self.assertGreaterEqual(extreme_output.min().item(), -1.0)
    
    def test_nan_handling(self):
        """Test de gestion des NaN"""
        generator = OriginalGenerator()
        generator.eval()
        
        # Entrée avec NaN
        nan_input = torch.randn(1, 3, 256, 256)
        nan_input[0, 0, 0, 0] = float('nan')
        
        with torch.no_grad():
            output = generator(nan_input)
        
        # Le modèle devrait propager les NaN
        self.assertTrue(torch.isnan(output).any())
    
    def test_model_determinism(self):
        """Test du déterminisme du modèle"""
        torch.manual_seed(42)
        generator1 = OriginalGenerator()
        
        torch.manual_seed(42)
        generator2 = OriginalGenerator()
        
        # Les poids devraient être identiques
        for (n1, p1), (n2, p2) in zip(
            generator1.named_parameters(),
            generator2.named_parameters()
        ):
            self.assertTrue(torch.allclose(p1, p2), f"Différence dans {n1}")


if __name__ == '__main__':
    unittest.main()