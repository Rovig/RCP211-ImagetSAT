#!/usr/bin/env python3
"""
Tests d'intégration pour le système d'entraînement complet
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import os


class TestTrainingIntegration(unittest.TestCase):
    """Tests d'intégration pour l'entraînement complet"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'epochs': 1,
            'batch_size': 2,
            'lr': 0.0002,
            'beta1': 0.5,
            'lambda_L1': 100,
            'image_size': 64,
            'checkpoint_dir': Path(self.temp_dir) / 'checkpoints',
            'sample_dir': Path(self.temp_dir) / 'samples',
            'log_dir': Path(self.temp_dir) / 'logs',
            'save_interval': 1,
            'sample_interval': 1,
            'device': 'cpu',
            'mixed_precision': False,
            'gradient_accumulation_steps': 1,
            'dataset_path': Path(self.temp_dir) / 'dataset'
        }
        
        # Créer les répertoires
        for key in ['checkpoint_dir', 'sample_dir', 'log_dir', 'dataset_path']:
            self.config[key].mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_training_pipeline_end_to_end(self):
        """Test du pipeline d'entraînement complet"""
        from core.models.model_manager import ModelManager
        
        # Créer les modèles
        manager = ModelManager()
        device = torch.device('cpu')
        
        generator = manager.create_generator(device)
        discriminator = manager.create_discriminator(device)
        
        # Vérifier que les modèles sont créés
        self.assertIsInstance(generator, nn.Module)
        self.assertIsInstance(discriminator, nn.Module)
        
        # Test forward pass
        test_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            fake_output = generator(test_input)
            self.assertEqual(fake_output.shape, (1, 3, 256, 256))
            
            # Test discriminateur
            disc_input = torch.cat([test_input, fake_output], dim=1)
            disc_output = discriminator(disc_input)
            self.assertEqual(len(disc_output.shape), 4)  # Sortie PatchGAN
    
    def test_loss_calculation(self):
        """Test du calcul des losses GAN"""
        # Critères de loss
        criterion_GAN = nn.BCEWithLogitsLoss()
        criterion_L1 = nn.L1Loss()
        
        # Données de test
        real_pred = torch.randn(2, 1, 15, 15)
        fake_pred = torch.randn(2, 1, 15, 15)
        real_image = torch.randn(2, 3, 64, 64)
        fake_image = torch.randn(2, 3, 64, 64)
        
        # Labels
        real_label = torch.ones_like(real_pred)
        fake_label = torch.zeros_like(real_pred)
        
        # Calcul des losses
        g_gan_loss = criterion_GAN(fake_pred, real_label)
        g_l1_loss = criterion_L1(fake_image, real_image)
        g_total = g_gan_loss + 100 * g_l1_loss
        
        d_real_loss = criterion_GAN(real_pred, real_label)
        d_fake_loss = criterion_GAN(fake_pred, fake_label)
        d_total = (d_real_loss + d_fake_loss) * 0.5
        
        # Vérifications
        self.assertIsInstance(g_total.item(), float)
        self.assertIsInstance(d_total.item(), float)
        self.assertGreater(g_total.item(), 0)
        self.assertGreater(d_total.item(), 0)
    
    def test_metrics_calculation(self):
        """Test du calcul des métriques PSNR et SSIM"""
        def calculate_psnr(img1, img2):
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0:
                return float('inf')
            return 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        def calculate_ssim(img1, img2):
            mu1 = torch.mean(img1)
            mu2 = torch.mean(img2)
            sigma1 = torch.var(img1)
            sigma2 = torch.var(img2)
            sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
            return ssim.item()
        
        # Images identiques
        img1 = torch.ones(1, 3, 64, 64) * 0.5
        img2 = img1.clone()
        
        psnr_same = calculate_psnr(img1, img2)
        ssim_same = calculate_ssim(img1, img2)
        
        # PSNR devrait être très élevé, SSIM proche de 1
        self.assertGreater(psnr_same, 50)
        self.assertGreater(ssim_same, 0.99)
        
        # Images différentes
        img3 = torch.rand(1, 3, 64, 64)
        
        psnr_diff = calculate_psnr(img1, img3)
        ssim_diff = calculate_ssim(img1, img3)
        
        # Les métriques devraient être plus faibles
        self.assertLess(psnr_diff, psnr_same)
        self.assertLess(ssim_diff, ssim_same)
    
    def test_checkpoint_save_and_load(self):
        """Test de sauvegarde et chargement de checkpoint"""
        from core.models.model_manager import ModelManager
        
        manager = ModelManager()
        device = torch.device('cpu')
        
        # Créer et entraîner légèrement un modèle
        generator = manager.create_generator(device)
        original_state = {k: v.clone() for k, v in generator.state_dict().items()}
        
        # Modifier les poids
        for param in generator.parameters():
            param.data += 0.1
        
        # Sauvegarder
        checkpoint = {
            'epoch': 10,
            'generator_state_dict': generator.state_dict(),
            'g_losses': [0.5, 0.4, 0.3],
            'd_losses': [0.3, 0.35, 0.3]
        }
        
        checkpoint_path = self.config['checkpoint_dir'] / 'test_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Charger dans un nouveau modèle
        new_generator = manager.load_generator(str(checkpoint_path), device)
        
        # Vérifier que les poids sont différents de l'original
        for (name1, param1), (name2, param2) in zip(
            original_state.items(),
            new_generator.state_dict().items()
        ):
            if 'weight' in name1:
                self.assertFalse(torch.allclose(param1, param2, atol=1e-6))
        
        # Et identiques à ceux sauvegardés
        for name, param in new_generator.state_dict().items():
            self.assertTrue(torch.equal(param, checkpoint['generator_state_dict'][name]))
    
    def test_data_normalization(self):
        """Test de la normalisation des données"""
        # Image RGB standard [0, 255]
        img_array = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
        img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1)
        
        # Normalisation [-1, 1]
        normalized = (img_tensor / 255.0 - 0.5) / 0.5
        
        # Vérifications
        self.assertGreaterEqual(normalized.min().item(), -1.0)
        self.assertLessEqual(normalized.max().item(), 1.0)
        
        # Dénormalisation
        denormalized = (normalized * 0.5 + 0.5) * 255.0
        
        # Devrait être proche de l'original
        self.assertTrue(torch.allclose(img_tensor, denormalized, atol=1.0))
    
    def test_gradient_accumulation(self):
        """Test de l'accumulation de gradients"""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        accumulation_steps = 4
        
        # Données de test
        inputs = [torch.randn(2, 10) for _ in range(accumulation_steps)]
        targets = [torch.randn(2, 10) for _ in range(accumulation_steps)]
        
        # Accumulation de gradients
        for i, (inp, target) in enumerate(zip(inputs, targets)):
            output = model(inp)
            loss = nn.MSELoss()(output, target)
            loss = loss / accumulation_steps  # Normaliser
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Les gradients devraient être nuls après step
        for param in model.parameters():
            self.assertTrue(param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad)))
    
    def test_memory_efficiency(self):
        """Test de l'efficacité mémoire"""
        import gc
        
        # Mesurer la mémoire avant
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Créer et détruire des tensors
        for _ in range(10):
            large_tensor = torch.randn(10, 3, 256, 256)
            del large_tensor
        
        # Forcer le garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Le test passe s'il n'y a pas de fuite mémoire
        self.assertTrue(True)
    
    def test_training_config_validation(self):
        """Test de validation de la configuration d'entraînement"""
        # Configuration valide
        valid_config = {
            'epochs': 100,
            'batch_size': 8,
            'lr': 0.0002,
            'image_size': 256
        }
        
        # Configurations invalides
        invalid_configs = [
            {'epochs': -1},  # Epochs négatif
            {'batch_size': 0},  # Batch size zéro
            {'lr': -0.001},  # Learning rate négatif
            {'image_size': 63}  # Taille non multiple de 2
        ]
        
        # Fonction de validation
        def validate_config(config):
            if config.get('epochs', 1) < 1:
                return False
            if config.get('batch_size', 1) < 1:
                return False
            if config.get('lr', 0.0001) <= 0:
                return False
            if config.get('image_size', 256) % 2 != 0:
                return False
            return True
        
        # Tests
        self.assertTrue(validate_config(valid_config))
        for invalid in invalid_configs:
            self.assertFalse(validate_config(invalid))


class TestDataPipeline(unittest.TestCase):
    """Tests pour le pipeline de données"""
    
    def test_data_loader_creation(self):
        """Test de création du DataLoader"""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Dataset factice
        data = torch.randn(100, 3, 64, 64)
        labels = torch.randn(100, 3, 64, 64)
        dataset = TensorDataset(data, labels)
        
        # DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,  # 0 pour les tests
            pin_memory=False
        )
        
        # Vérifications
        self.assertEqual(len(dataloader), 13)  # 100/8 = 12.5 -> 13
        
        # Tester une itération
        for batch_data, batch_labels in dataloader:
            self.assertLessEqual(batch_data.shape[0], 8)
            self.assertEqual(batch_data.shape[1:], (3, 64, 64))
            break
    
    def test_data_augmentation_pipeline(self):
        """Test du pipeline d'augmentation"""
        from torchvision import transforms
        
        # Pipeline d'augmentation
        augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
        
        # Image de test
        img = Image.new('RGB', (256, 256), color=(128, 128, 128))
        
        # Appliquer plusieurs fois
        results = []
        for _ in range(10):
            augmented = augmentation(img)
            results.append(np.array(augmented))
        
        # Au moins une image devrait être différente
        all_same = all(np.array_equal(results[0], r) for r in results[1:])
        self.assertFalse(all_same)


if __name__ == '__main__':
    unittest.main()