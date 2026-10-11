#!/usr/bin/env python3
"""
Tests unitaires pour le module improved_training
"""

import unittest
import math
import torch
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

from core.training.improved_training import (
    ImprovedInriaDataset, ImprovedTrainer, ImprovedGenerator, ImprovedDiscriminator,
    calculate_psnr, calculate_ssim
)


class TestImprovedInriaDataset(unittest.TestCase):
    """Tests pour le dataset Inria amélioré"""
    
    NUM_FILES = 5
    TILE_SIZE = 64
    
    def setUp(self):
        """Configuration avant chaque test"""
        # Créer une structure de répertoires temporaire
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = Path(self.temp_dir) / "train" / "images"
        self.gt_dir = Path(self.temp_dir) / "train" / "gt"
        self.images_dir.mkdir(parents=True)
        self.gt_dir.mkdir(parents=True)
        
        half = self.TILE_SIZE // 2
        for i in range(self.NUM_FILES):
            # Image RGB: une couleur par quadrant
            img = np.zeros((self.TILE_SIZE, self.TILE_SIZE, 3), dtype=np.uint8)
            img[:half, :half] = (250, 0, 0)
            img[:half, half:] = (0, 250, 0)
            img[half:, :half] = (0, 0, 250)
            img[half:, half:] = (250, 250, 250)
            Image.fromarray(img).save(self.images_dir / f"test_{i}.tif")
            
            # GT: bâtiment uniquement dans le quadrant inférieur droit
            gt = np.zeros((self.TILE_SIZE, self.TILE_SIZE), dtype=np.uint8)
            gt[half:, half:] = 255
            Image.fromarray(gt).save(self.gt_dir / f"test_{i}.tif")
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _dataset(self, **kwargs):
        kwargs.setdefault('image_size', 32)
        return ImprovedInriaDataset(self.images_dir, self.gt_dir, **kwargs)
    
    def test_train_val_split(self):
        """Test du découpage train/validation"""
        train = self._dataset(mode='train')
        val = self._dataset(mode='val')
        
        self.assertEqual(len(train.image_files), 4)
        self.assertEqual(len(val.image_files), 1)
        self.assertEqual(len(train.image_files), len(train.gt_files))
        
        # Aucun fichier en commun, tous les fichiers utilisés
        self.assertFalse(set(train.image_files) & set(val.image_files))
        self.assertEqual(len(set(train.image_files) | set(val.image_files)), self.NUM_FILES)
        
        # Chaque image reste associée à son masque
        for image_file, gt_file in zip(train.image_files, train.gt_files):
            self.assertEqual(image_file.name, gt_file.name)
    
    def test_dataset_length(self):
        """Test de la taille du dataset selon le mode et le cropping"""
        # 4 patches par image, à l'entraînement comme en validation
        self.assertEqual(len(self._dataset(mode='train')), 16)
        self.assertEqual(len(self._dataset(mode='val')), 4)
        self.assertEqual(len(self._dataset(mode='train', crop_patches=False)), 4)
        self.assertEqual(len(self._dataset(mode='test')), self.NUM_FILES * 4)
        self.assertEqual(len(self._dataset(mode='train', val_split=0)), self.NUM_FILES * 4)
    
    def test_dataset_getitem(self):
        """Test de récupération d'un élément"""
        dataset = self._dataset(mode='train')
        
        gt, img = dataset[0]
        
        self.assertIsInstance(gt, torch.Tensor)
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(gt.shape, (3, 32, 32))
        self.assertEqual(img.shape, (3, 32, 32))
        
        # Vérifier la normalisation [-1, 1]
        self.assertTrue(gt.min() >= -1)
        self.assertTrue(gt.max() <= 1)
        self.assertTrue(img.min() >= -1)
        self.assertTrue(img.max() <= 1)
    
    def test_different_image_sizes(self):
        """Test avec différentes tailles d'images"""
        for size in [32, 64]:
            with self.subTest(size=size):
                dataset = self._dataset(mode='val', image_size=size)
                
                gt, img = dataset[0]
                self.assertEqual(gt.shape, (3, size, size))
                self.assertEqual(img.shape, (3, size, size))
    
    def test_validation_is_deterministic(self):
        """Test que la validation renvoie toujours les mêmes patches"""
        dataset = self._dataset(mode='val')
        
        for idx in range(len(dataset)):
            gt1, img1 = dataset[idx]
            gt2, img2 = dataset[idx]
            self.assertTrue(torch.equal(gt1, gt2))
            self.assertTrue(torch.equal(img1, img2))
    
    def test_validation_patches_are_the_four_corners(self):
        """Test que les 4 patches de validation sont les 4 quadrants de l'image"""
        dataset = self._dataset(mode='val')
        
        # Couleur attendue de chaque quadrant, normalisée dans [-1, 1]
        expected_colors = [(250, 0, 0), (0, 250, 0), (0, 0, 250), (250, 250, 250)]
        
        for patch_idx, color in enumerate(expected_colors):
            gt, img = dataset[patch_idx]
            expected = torch.tensor(color).view(3, 1, 1) / 255.0 * 2 - 1
            self.assertTrue(torch.allclose(img, expected.expand_as(img), atol=1e-4))
            
            # Le bâtiment n'occupe que le quadrant inférieur droit
            expected_gt = 1.0 if patch_idx == 3 else -1.0
            self.assertTrue(torch.all(gt == expected_gt))
    
    def test_binary_map_values(self):
        """Test que la carte d'entrée ne contient que du noir et du blanc"""
        dataset = self._dataset(mode='val', crop_patches=False)
        
        gt, _ = dataset[0]
        
        self.assertEqual(set(gt.unique().tolist()), {-1.0, 1.0})
        # Les 3 canaux sont identiques
        self.assertTrue(torch.equal(gt[0], gt[1]))
        self.assertTrue(torch.equal(gt[0], gt[2]))
    
    def test_create_binary_map(self):
        """Test du seuillage du masque"""
        dataset = self._dataset(mode='val')
        mask = Image.fromarray(np.array([[0, 100, 127], [128, 200, 255]], dtype=np.uint8))
        
        binary = dataset._create_binary_map(mask)
        
        self.assertEqual(binary.mode, 'RGB')
        self.assertEqual(binary.size, mask.size)
        expected = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
        for channel in range(3):
            np.testing.assert_array_equal(np.array(binary)[:, :, channel], expected)
    
    def test_random_crop_keeps_image_and_mask_aligned(self):
        """Test que l'image et le masque sont croppés au même endroit"""
        dataset = self._dataset(mode='train')
        image = Image.open(dataset.image_files[0]).convert('RGB')
        gt = Image.open(dataset.gt_files[0]).convert('L')
        
        for patch_idx in range(4):
            cropped_image, cropped_gt = dataset._random_crop(image, gt, patch_idx)
            
            half = self.TILE_SIZE // 2
            self.assertEqual(cropped_image.size, (half, half))
            self.assertEqual(cropped_gt.size, (half, half))
            
            # Dans les images de test, bâtiment <=> pixel blanc
            white = np.all(np.array(cropped_image) == 250, axis=-1)
            np.testing.assert_array_equal(white, np.array(cropped_gt) == 255)
    
    def test_empty_directory(self):
        """Test avec un répertoire sans images"""
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        
        with self.assertRaises(ValueError):
            ImprovedInriaDataset(empty_dir, empty_dir, mode='test')
    
    def test_mismatched_image_and_gt_counts(self):
        """Test avec un nombre d'images et de masques différent"""
        (self.gt_dir / "test_0.tif").unlink()
        
        with self.assertRaises(AssertionError):
            self._dataset(mode='test')


class TestMetrics(unittest.TestCase):
    """Tests pour les métriques PSNR et SSIM"""
    
    def test_psnr_identical_images(self):
        """Test du PSNR sur deux images identiques"""
        img = torch.rand(1, 3, 64, 64)
        
        self.assertEqual(calculate_psnr(img, img), float('inf'))
    
    def test_psnr_known_value(self):
        """Test du PSNR sur une erreur connue"""
        img1 = torch.ones(1, 3, 64, 64) * 0.5
        img2 = torch.zeros(1, 3, 64, 64)
        
        # MSE = 0.25 -> PSNR = 20 * log10(1 / 0.5)
        psnr = calculate_psnr(img1, img2)
        self.assertAlmostEqual(psnr.item(), 20 * math.log10(2), places=4)
    
    def test_psnr_decreases_with_noise(self):
        """Test que le PSNR diminue quand le bruit augmente"""
        torch.manual_seed(0)
        img = torch.rand(1, 3, 64, 64)
        noise = torch.randn_like(img)
        
        psnr_low_noise = calculate_psnr(img, img + 0.01 * noise)
        psnr_high_noise = calculate_psnr(img, img + 0.1 * noise)
        
        self.assertGreater(psnr_low_noise, psnr_high_noise)
    
    def test_ssim_identical_images(self):
        """Test du SSIM sur deux images identiques"""
        torch.manual_seed(0)
        img = torch.rand(1, 3, 64, 64)
        
        self.assertAlmostEqual(calculate_ssim(img, img).item(), 1.0, places=4)
    
    def test_ssim_different_images(self):
        """Test du SSIM sur deux images sans rapport"""
        torch.manual_seed(0)
        img1 = torch.rand(1, 3, 64, 64)
        img2 = torch.rand(1, 3, 64, 64)
        
        ssim = calculate_ssim(img1, img2).item()
        self.assertLess(ssim, 0.5)
        self.assertGreaterEqual(ssim, -1.0)


class TestImprovedTrainer(unittest.TestCase):
    """Tests pour l'entraîneur amélioré"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        torch.manual_seed(0)
        self.device = torch.device('cpu')
        # Modèles étroits (ngf/ndf réduits) pour des tests rapides
        self.generator = ImprovedGenerator(ngf=8)
        self.discriminator = ImprovedDiscriminator(ndf=8)
        self.trainer = ImprovedTrainer(self.generator, self.discriminator, self.device)
        
        self.real_A = torch.rand(2, 3, 256, 256) * 2 - 1
        self.real_B = torch.rand(2, 3, 256, 256) * 2 - 1
    
    def test_trainer_initialization(self):
        """Test de l'initialisation du trainer"""
        trainer = ImprovedTrainer(self.generator, self.discriminator, self.device,
                                  lr_g=0.001, lr_d=0.002, lambda_l1=50)
        
        self.assertIs(trainer.generator, self.generator)
        self.assertIs(trainer.discriminator, self.discriminator)
        self.assertEqual(trainer.device.type, 'cpu')
        self.assertEqual(trainer.lambda_l1, 50)
        self.assertEqual(trainer.optimizer_G.param_groups[0]['lr'], 0.001)
        self.assertEqual(trainer.optimizer_D.param_groups[0]['lr'], 0.002)
    
    def test_train_step(self):
        """Test d'une étape d'entraînement"""
        g_before = [p.clone() for p in self.generator.parameters()]
        d_before = [p.clone() for p in self.discriminator.parameters()]
        
        losses = self.trainer.train_step(self.real_A, self.real_B)
        
        self.assertEqual(set(losses.keys()), {
            'loss_G', 'loss_G_GAN', 'loss_G_L1',
            'loss_D', 'loss_D_real', 'loss_D_fake'
        })
        for value in losses.values():
            self.assertTrue(math.isfinite(value))
        
        # Cohérence des losses composées
        self.assertAlmostEqual(losses['loss_G'], losses['loss_G_GAN'] + losses['loss_G_L1'], places=3)
        self.assertAlmostEqual(losses['loss_D'], (losses['loss_D_real'] + losses['loss_D_fake']) / 2, places=5)
        
        # Les deux réseaux ont été mis à jour
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(g_before, self.generator.parameters())))
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(d_before, self.discriminator.parameters())))
    
    def test_validate(self):
        """Test de la validation"""
        val_loader = [(self.real_A, self.real_B), (self.real_B, self.real_A)]
        
        metrics = self.trainer.validate(val_loader)
        
        self.assertEqual(set(metrics.keys()), {'val_loss_G', 'val_loss_D', 'val_psnr', 'val_ssim'})
        for value in metrics.values():
            self.assertTrue(math.isfinite(value))
        self.assertGreater(metrics['val_psnr'], 0)
        
        # Les modèles repassent en mode entraînement
        self.assertTrue(self.generator.training)
        self.assertTrue(self.discriminator.training)
    
    def test_validate_does_not_update_weights(self):
        """Test que la validation ne modifie pas les poids"""
        g_before = [p.clone() for p in self.generator.parameters()]
        
        self.trainer.validate([(self.real_A, self.real_B)])
        
        for before, param in zip(g_before, self.generator.parameters()):
            self.assertTrue(torch.equal(before, param))
    
    def test_save_sample_images(self):
        """Test de sauvegarde d'échantillons"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = Path(tmpdir) / "samples"
            
            self.trainer.save_sample_images(self.real_A, self.real_B, epoch=5, save_dir=str(sample_dir))
            
            sample_file = sample_dir / "epoch_005.png"
            self.assertTrue(sample_file.exists())
            with Image.open(sample_file) as image:
                self.assertEqual(image.mode, 'RGB')
    
    def test_learning_rate_scheduling(self):
        """Test du scheduling du learning rate"""
        initial_lr = self.trainer.optimizer_G.param_groups[0]['lr']
        
        self.trainer.optimizer_G.step()
        self.trainer.optimizer_D.step()
        for _ in range(49):
            self.trainer.step_schedulers()
        self.assertAlmostEqual(self.trainer.optimizer_G.param_groups[0]['lr'], initial_lr)
        
        # Le LR est divisé par 2 toutes les 50 époques
        self.trainer.step_schedulers()
        self.assertAlmostEqual(self.trainer.optimizer_G.param_groups[0]['lr'], initial_lr * 0.5)
        self.assertAlmostEqual(self.trainer.optimizer_D.param_groups[0]['lr'], initial_lr * 0.5)
    
    def test_scheduler_state_roundtrip(self):
        """Test que l'état des schedulers survit à une sauvegarde/reprise"""
        self.trainer.optimizer_G.step()
        self.trainer.optimizer_D.step()
        for _ in range(60):
            self.trainer.step_schedulers()
        
        other = ImprovedTrainer(ImprovedGenerator(ngf=8), ImprovedDiscriminator(ndf=8), self.device)
        other.optimizer_G.load_state_dict(self.trainer.optimizer_G.state_dict())
        other.scheduler_G.load_state_dict(self.trainer.scheduler_G.state_dict())
        
        self.assertEqual(other.scheduler_G.last_epoch, 60)
        self.assertEqual(other.scheduler_G.get_last_lr(), self.trainer.scheduler_G.get_last_lr())


if __name__ == '__main__':
    unittest.main()
