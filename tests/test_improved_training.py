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
    TrainingState, build_checkpoint, restore_checkpoint,
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
        
        psnr = calculate_psnr(img, img)
        
        # Un tensor, comme dans le cas général (la validation appelle .item())
        self.assertIsInstance(psnr, torch.Tensor)
        self.assertEqual(psnr.item(), float('inf'))
    
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
    
    def test_validate_with_empty_loader(self):
        """Test que la validation sans données est une erreur explicite"""
        with self.assertRaises(ValueError):
            self.trainer.validate([])
        
        self.assertTrue(self.generator.training)
    
    def test_train_step_with_single_sample(self):
        """Test d'une étape d'entraînement avec un batch de taille 1"""
        losses = self.trainer.train_step(self.real_A[:1], self.real_B[:1])
        
        for value in losses.values():
            self.assertTrue(math.isfinite(value))
    
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


def val_metrics(psnr, ssim=0.5, loss=10.0):
    """Métriques de validation au format renvoyé par ImprovedTrainer.validate"""
    return {'val_loss_G': loss, 'val_loss_D': 0.5, 'val_psnr': psnr, 'val_ssim': ssim}


class TestTrainingState(unittest.TestCase):
    """Tests pour le suivi du meilleur modèle et l'early stopping"""
    
    def test_initial_state(self):
        """Test de l'état initial"""
        state = TrainingState(patience=3)
        
        self.assertEqual(state.history, [])
        self.assertEqual(state.best_epoch, 0)
        self.assertEqual(state.epochs_no_improve, 0)
        self.assertFalse(state.should_stop)
    
    def test_improvement_becomes_best_model(self):
        """Test qu'un meilleur PSNR devient le meilleur modèle"""
        state = TrainingState(patience=3)
        
        self.assertTrue(state.update(5, val_metrics(psnr=12.0, ssim=0.3, loss=40.0)))
        
        self.assertEqual(state.best_epoch, 5)
        self.assertEqual(state.best_val_psnr, 12.0)
        self.assertEqual(state.best_val_ssim, 0.3)
        self.assertEqual(state.best_val_loss, 40.0)
        
        self.assertTrue(state.update(10, val_metrics(psnr=14.0)))
        self.assertEqual(state.best_epoch, 10)
    
    def test_no_improvement_keeps_best_model(self):
        """Test qu'un PSNR égal ou inférieur ne remplace pas le meilleur modèle"""
        state = TrainingState(patience=3)
        state.update(5, val_metrics(psnr=12.0))
        
        self.assertFalse(state.update(10, val_metrics(psnr=12.0)))
        self.assertFalse(state.update(15, val_metrics(psnr=11.0)))
        
        self.assertEqual(state.best_epoch, 5)
        self.assertEqual(state.best_val_psnr, 12.0)
        self.assertEqual(state.epochs_no_improve, 2)
    
    def test_early_stopping_after_patience_validations(self):
        """Test de l'arrêt après `patience` validations sans amélioration"""
        state = TrainingState(patience=3)
        state.update(5, val_metrics(psnr=12.0))
        
        for epoch in (10, 15):
            state.update(epoch, val_metrics(psnr=11.0))
            self.assertFalse(state.should_stop)
        
        state.update(20, val_metrics(psnr=11.0))
        self.assertTrue(state.should_stop)
    
    def test_improvement_resets_patience(self):
        """Test qu'une amélioration remet le compteur à zéro"""
        state = TrainingState(patience=2)
        state.update(5, val_metrics(psnr=12.0))
        state.update(10, val_metrics(psnr=11.0))
        
        state.update(15, val_metrics(psnr=13.0))
        
        self.assertEqual(state.epochs_no_improve, 0)
        self.assertFalse(state.should_stop)


class TestCheckpointResume(unittest.TestCase):
    """Tests pour la sauvegarde et la reprise de l'entraînement"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        torch.manual_seed(0)
        self.device = torch.device('cpu')
        self.trainer = self._new_trainer()
        self.real_A = torch.rand(2, 3, 256, 256) * 2 - 1
        self.real_B = torch.rand(2, 3, 256, 256) * 2 - 1
    
    def _new_trainer(self):
        return ImprovedTrainer(ImprovedGenerator(ngf=8), ImprovedDiscriminator(ndf=8), self.device)
    
    def _train_epochs(self, trainer, num_epochs):
        for _ in range(num_epochs):
            trainer.train_step(self.real_A, self.real_B)
            trainer.step_schedulers()
    
    def _roundtrip(self, checkpoint):
        """Sauvegarde puis recharge comme le fait le script d'entraînement"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pth"
            torch.save(checkpoint, path)
            return torch.load(path, map_location=self.device)
    
    def test_resume_restores_everything(self):
        """Test que la reprise restaure modèles, optimizers, schedulers et état"""
        self._train_epochs(self.trainer, 60)
        state = TrainingState(patience=5)
        state.history = [{'epoch': e + 1, 'loss_G': 1.0} for e in range(60)]
        state.update(55, val_metrics(psnr=12.0, ssim=0.3, loss=40.0))
        state.update(60, val_metrics(psnr=11.0))
        
        checkpoint = self._roundtrip(build_checkpoint(self.trainer, state, 60))
        
        resumed = self._new_trainer()
        resumed_state = TrainingState(patience=5)
        start_epoch = restore_checkpoint(checkpoint, resumed, resumed_state)
        
        self.assertEqual(start_epoch, 60)
        
        # Poids
        for name, param in self.trainer.generator.state_dict().items():
            self.assertTrue(torch.equal(param, resumed.generator.state_dict()[name]))
        for name, param in self.trainer.discriminator.state_dict().items():
            self.assertTrue(torch.equal(param, resumed.discriminator.state_dict()[name]))
        
        # Learning rate (divisé par 2 après 50 époques) et position des schedulers
        self.assertAlmostEqual(resumed.optimizer_G.param_groups[0]['lr'], 0.0001)
        self.assertAlmostEqual(resumed.optimizer_D.param_groups[0]['lr'], 0.0001)
        self.assertEqual(resumed.scheduler_G.last_epoch, 60)
        self.assertEqual(resumed.scheduler_D.last_epoch, 60)
        
        # Historique, meilleur modèle et compteur d'early stopping
        self.assertEqual(resumed_state.history, state.history)
        self.assertEqual(resumed_state.best_epoch, 55)
        self.assertEqual(resumed_state.best_val_psnr, 12.0)
        self.assertEqual(resumed_state.best_val_ssim, 0.3)
        self.assertEqual(resumed_state.best_val_loss, 40.0)
        self.assertEqual(resumed_state.epochs_no_improve, 1)
    
    def test_resumed_training_matches_uninterrupted_training(self):
        """Test que le LR suit le même calendrier avec ou sans interruption"""
        self._train_epochs(self.trainer, 40)
        checkpoint = self._roundtrip(build_checkpoint(self.trainer, TrainingState(patience=5), 40))
        
        resumed = self._new_trainer()
        restore_checkpoint(checkpoint, resumed, TrainingState(patience=5))
        
        self._train_epochs(self.trainer, 15)
        self._train_epochs(resumed, 15)
        
        self.assertAlmostEqual(self.trainer.optimizer_G.param_groups[0]['lr'], 0.0001)
        self.assertEqual(resumed.optimizer_G.param_groups[0]['lr'],
                         self.trainer.optimizer_G.param_groups[0]['lr'])
    
    def _old_format_checkpoint(self, trainer, epoch):
        """Checkpoint au format d'avant: sauvé avant le step des schedulers, sans leur état"""
        return {
            'epoch': epoch,
            'generator_state_dict': trainer.generator.state_dict(),
            'discriminator_state_dict': trainer.discriminator.state_dict(),
            'optimizer_G_state_dict': trainer.optimizer_G.state_dict(),
            'optimizer_D_state_dict': trainer.optimizer_D.state_dict(),
            'losses': {'loss_G': 1.0}
        }
    
    def test_resume_from_old_format_checkpoint(self):
        """Test de reprise depuis un checkpoint sans état de scheduler"""
        self._train_epochs(self.trainer, 24)
        self.trainer.train_step(self.real_A, self.real_B)
        checkpoint = self._roundtrip(self._old_format_checkpoint(self.trainer, 25))
        
        resumed = self._new_trainer()
        state = TrainingState(patience=5)
        start_epoch = restore_checkpoint(checkpoint, resumed, state)
        
        self.assertEqual(start_epoch, 25)
        self.assertEqual(resumed.scheduler_G.last_epoch, 25)
        self.assertAlmostEqual(resumed.optimizer_G.param_groups[0]['lr'], 0.0002)
        self.assertEqual(state.history, [])
        self.assertEqual(state.best_epoch, 0)
        
        # La division par 2 arrive bien à l'époque 50, pas 50 époques après la reprise
        self._train_epochs(resumed, 25)
        self.assertAlmostEqual(resumed.optimizer_G.param_groups[0]['lr'], 0.0001)
    
    def test_resume_from_old_format_checkpoint_at_decay_epoch(self):
        """Test de reprise d'un ancien checkpoint sauvé juste avant une division du LR"""
        self._train_epochs(self.trainer, 49)
        self.trainer.train_step(self.real_A, self.real_B)
        checkpoint = self._roundtrip(self._old_format_checkpoint(self.trainer, 50))
        
        resumed = self._new_trainer()
        restore_checkpoint(checkpoint, resumed, TrainingState(patience=5))
        
        self.assertEqual(resumed.scheduler_G.last_epoch, 50)
        self.assertAlmostEqual(resumed.optimizer_G.param_groups[0]['lr'], 0.0001)
        self.assertAlmostEqual(resumed.optimizer_D.param_groups[0]['lr'], 0.0001)
    
    def test_mismatched_architecture_is_an_error(self):
        """Test qu'un checkpoint d'une autre architecture est refusé"""
        checkpoint = build_checkpoint(self.trainer, TrainingState(patience=5), 10)
        
        other = ImprovedTrainer(ImprovedGenerator(ngf=8, image_size=512),
                                ImprovedDiscriminator(ndf=8, image_size=512), self.device)
        
        with self.assertRaises(RuntimeError):
            restore_checkpoint(checkpoint, other, TrainingState(patience=5))


if __name__ == '__main__':
    unittest.main()
