#!/usr/bin/env python3
"""
Tests pour les utilitaires d'entraînement
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from unittest.mock import Mock, patch, mock_open


class TestTrainingUtils(unittest.TestCase):
    """Tests pour les fonctions utilitaires d'entraînement"""
    
    def test_set_random_seed(self):
        """Test de la fixation de la graine aléatoire"""
        def set_random_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Fixer la graine
        set_random_seed(42)
        
        # Générer des nombres aléatoires
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)
        
        # Fixer à nouveau la même graine
        set_random_seed(42)
        
        # Générer à nouveau
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Les résultats devraient être identiques
        self.assertTrue(torch.equal(torch_rand1, torch_rand2))
        np.testing.assert_array_equal(np_rand1, np_rand2)
    
    def test_count_model_parameters(self):
        """Test du comptage de paramètres"""
        def count_parameters(model):
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total, trainable
        
        # Modèle simple
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        total, trainable = count_parameters(model)
        
        # Calcul manuel : (10*20 + 20) + (20*10 + 10) = 220 + 210 = 430
        self.assertEqual(total, 430)
        self.assertEqual(trainable, 430)
        
        # Geler certains paramètres
        for param in model[0].parameters():
            param.requires_grad = False
        
        _, trainable_after = count_parameters(model)
        self.assertEqual(trainable_after, 210)  # Seulement la deuxième couche
    
    def test_save_model_info(self):
        """Test de sauvegarde des informations du modèle"""
        model_info = {
            'architecture': 'Pix2Pix',
            'generator': {
                'type': 'UNet',
                'layers': 8,
                'filters': 64
            },
            'discriminator': {
                'type': 'PatchGAN',
                'layers': 4,
                'patch_size': 70
            },
            'training': {
                'epochs': 200,
                'batch_size': 8,
                'learning_rate': 0.0002
            }
        }
        
        # Test JSON
        json_str = json.dumps(model_info, indent=2)
        loaded_json = json.loads(json_str)
        self.assertEqual(model_info, loaded_json)
    
    def test_learning_rate_scheduler(self):
        """Test des schedulers de learning rate"""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # StepLR
        scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        initial_lr = optimizer.param_groups[0]['lr']
        
        for epoch in range(10):
            scheduler_step.step()
        
        final_lr = optimizer.param_groups[0]['lr']
        self.assertLess(final_lr, initial_lr)
        self.assertAlmostEqual(final_lr, initial_lr * 0.01, places=6)  # 0.1^2
        
        # CosineAnnealingLR
        optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10)
        
        lrs = []
        for epoch in range(10):
            lrs.append(optimizer2.param_groups[0]['lr'])
            scheduler_cosine.step()
        
        # Le LR devrait diminuer puis augmenter
        self.assertLess(min(lrs), initial_lr)
    
    def test_gradient_clipping(self):
        """Test du clipping de gradients"""
        model = nn.Linear(10, 10)
        
        # Créer des gradients artificiellement grands
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 100
        
        # Appliquer le clipping (clip_grad_norm_ renvoie la norme AVANT clipping)
        max_norm = 1.0
        total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        self.assertGreater(total_norm_before, max_norm)
        
        # La norme devrait être limitée
        total_norm_after = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
        self.assertLessEqual(total_norm_after, max_norm + 1e-6)
    
    def test_model_freezing(self):
        """Test du gel/dégel de modèles"""
        def freeze_model(model):
            for param in model.parameters():
                param.requires_grad = False
        
        def unfreeze_model(model):
            for param in model.parameters():
                param.requires_grad = True
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        # Geler le modèle
        freeze_model(model)
        for param in model.parameters():
            self.assertFalse(param.requires_grad)
        
        # Dégeler
        unfreeze_model(model)
        for param in model.parameters():
            self.assertTrue(param.requires_grad)
    
    def test_training_stats_tracking(self):
        """Test du suivi des statistiques d'entraînement"""
        class TrainingStats:
            def __init__(self):
                self.losses = {'g_loss': [], 'd_loss': []}
                self.metrics = {'psnr': [], 'ssim': []}
                self.best_metrics = {'psnr': 0, 'ssim': 0}
            
            def update(self, g_loss, d_loss, psnr=None, ssim=None):
                self.losses['g_loss'].append(g_loss)
                self.losses['d_loss'].append(d_loss)
                
                if psnr is not None:
                    self.metrics['psnr'].append(psnr)
                    self.best_metrics['psnr'] = max(self.best_metrics['psnr'], psnr)
                
                if ssim is not None:
                    self.metrics['ssim'].append(ssim)
                    self.best_metrics['ssim'] = max(self.best_metrics['ssim'], ssim)
            
            def get_average(self, key, last_n=10):
                values = self.losses.get(key, self.metrics.get(key, []))
                if not values:
                    return 0
                return np.mean(values[-last_n:])
        
        # Test
        stats = TrainingStats()
        
        # Simuler quelques époques
        for i in range(20):
            g_loss = 0.5 - i * 0.01
            d_loss = 0.3 + np.sin(i) * 0.1
            psnr = 20 + i * 0.5
            ssim = 0.7 + i * 0.01
            
            stats.update(g_loss, d_loss, psnr, ssim)
        
        # Vérifications
        self.assertEqual(len(stats.losses['g_loss']), 20)
        self.assertAlmostEqual(stats.best_metrics['psnr'], 29.5, places=1)
        self.assertAlmostEqual(stats.best_metrics['ssim'], 0.89, places=2)
        
        # Moyenne des 10 derniers
        avg_g_loss = stats.get_average('g_loss', last_n=10)
        self.assertLess(avg_g_loss, 0.5)  # Devrait avoir diminué
    
    def test_checkpoint_manager(self):
        """Test du gestionnaire de checkpoints"""
        class CheckpointManager:
            def __init__(self, save_dir, max_checkpoints=5):
                self.save_dir = Path(save_dir)
                self.max_checkpoints = max_checkpoints
                self.save_dir.mkdir(exist_ok=True)
            
            def save(self, state, epoch, is_best=False):
                # Sauvegarde normale
                checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
                torch.save(state, checkpoint_path)
                
                # Sauvegarde du meilleur
                if is_best:
                    best_path = self.save_dir / 'best_model.pth'
                    torch.save(state, best_path)
                
                # Nettoyer les anciens checkpoints
                self._cleanup()
                
                return checkpoint_path
            
            def _cleanup(self):
                checkpoints = sorted(self.save_dir.glob('checkpoint_epoch_*.pth'))
                if len(checkpoints) > self.max_checkpoints:
                    for ckpt in checkpoints[:-self.max_checkpoints]:
                        ckpt.unlink()
            
            def load_latest(self):
                checkpoints = sorted(self.save_dir.glob('checkpoint_epoch_*.pth'))
                if checkpoints:
                    return torch.load(checkpoints[-1], map_location='cpu')
                return None
        
        # Test avec un répertoire temporaire
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_checkpoints=3)
            
            # Sauvegarder plusieurs checkpoints
            for epoch in range(5):
                state = {'epoch': epoch, 'loss': 1.0 / (epoch + 1)}
                manager.save(state, epoch, is_best=(epoch == 3))
            
            # Vérifier qu'il n'y a que 3 checkpoints
            checkpoints = list(Path(tmpdir).glob('checkpoint_epoch_*.pth'))
            self.assertEqual(len(checkpoints), 3)
            
            # Vérifier que le meilleur est sauvegardé
            best_path = Path(tmpdir) / 'best_model.pth'
            self.assertTrue(best_path.exists())
            
            # Charger le dernier
            latest = manager.load_latest()
            self.assertEqual(latest['epoch'], 4)
    
    def test_mixed_precision_training(self):
        """Test de l'entraînement en précision mixte"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA non disponible")
        
        model = nn.Linear(10, 10).cuda()
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.cuda.amp.GradScaler()
        
        # Données de test
        inputs = torch.randn(32, 10).cuda()
        targets = torch.randn(32, 10).cuda()
        
        # Entraînement avec AMP
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
        
        # Backward avec scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Le test passe si pas d'erreur
        self.assertTrue(True)
    
    def test_distributed_config(self):
        """Test de configuration pour entraînement distribué"""
        distributed_config = {
            'backend': 'nccl',
            'world_size': 4,
            'rank': 0,
            'local_rank': 0,
            'master_addr': 'localhost',
            'master_port': '12355'
        }
        
        # Validation de la configuration
        def validate_distributed_config(config):
            required_keys = ['backend', 'world_size', 'rank']
            for key in required_keys:
                if key not in config:
                    return False
            
            if config['rank'] >= config['world_size']:
                return False
            
            if config['backend'] not in ['nccl', 'gloo', 'mpi']:
                return False
            
            return True
        
        self.assertTrue(validate_distributed_config(distributed_config))
        
        # Configuration invalide
        invalid_config = distributed_config.copy()
        invalid_config['rank'] = 5  # Rank > world_size
        self.assertFalse(validate_distributed_config(invalid_config))


if __name__ == '__main__':
    unittest.main()