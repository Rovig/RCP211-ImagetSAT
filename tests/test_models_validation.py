#!/usr/bin/env python3
"""
Tests de validation et métriques pour les modèles
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch
import warnings

from core.models.model_manager import OriginalGenerator, ModelManager


class TestModelValidation(unittest.TestCase):
    """Tests de validation des modèles"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.device = torch.device('cpu')
        self.manager = ModelManager()
        self.generator = OriginalGenerator()
        self.generator.eval()
    
    def test_output_validity(self):
        """Test de validité des sorties"""
        test_inputs = [
            torch.randn(1, 3, 256, 256),  # Normal
            torch.zeros(1, 3, 256, 256),  # Tous zéros
            torch.ones(1, 3, 256, 256),   # Tous uns
            torch.randn(1, 3, 256, 256) * 10,  # Grandes valeurs
        ]
        
        for i, input_tensor in enumerate(test_inputs):
            with self.subTest(test_case=i):
                with torch.no_grad():
                    output = self.generator(input_tensor)
                
                # Vérifier pas de NaN
                self.assertFalse(torch.isnan(output).any(), 
                               f"NaN détecté dans le test {i}")
                
                # Vérifier pas d'inf
                self.assertFalse(torch.isinf(output).any(), 
                               f"Inf détecté dans le test {i}")
                
                # Vérifier la plage [-1, 1] (tanh)
                self.assertGreaterEqual(output.min().item(), -1.0,
                                      f"Valeur < -1 dans le test {i}")
                self.assertLessEqual(output.max().item(), 1.0,
                                   f"Valeur > 1 dans le test {i}")
    
    def test_consistency(self):
        """Test de cohérence des sorties"""
        input_tensor = torch.randn(1, 3, 256, 256)
        
        # Générer plusieurs fois
        outputs = []
        for _ in range(5):
            with torch.no_grad():
                output = self.generator(input_tensor)
            outputs.append(output)
        
        # Toutes les sorties devraient être identiques (mode eval)
        for i in range(1, len(outputs)):
            self.assertTrue(torch.allclose(outputs[0], outputs[i], atol=1e-6),
                          f"Sortie {i} différente de la première")
    
    def test_input_sensitivity(self):
        """Test de sensibilité aux entrées"""
        base_input = torch.randn(1, 3, 256, 256)
        
        # Petite perturbation
        epsilon = 0.01
        perturbed_input = base_input + epsilon * torch.randn_like(base_input)
        
        with torch.no_grad():
            base_output = self.generator(base_input)
            perturbed_output = self.generator(perturbed_input)
        
        # La différence de sortie devrait être proportionnelle
        output_diff = torch.abs(perturbed_output - base_output).mean()
        
        # La sortie ne devrait pas être trop sensible
        self.assertLess(output_diff.item(), 1.0,
                       "Le modèle est trop sensible aux petites perturbations")
        
        # Mais devrait quand même réagir
        self.assertGreater(output_diff.item(), 0.0,
                          "Le modèle ne réagit pas aux perturbations")


class TestModelMetrics(unittest.TestCase):
    """Tests des métriques de performance"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.generator = OriginalGenerator()
        self.generator.eval()
    
    def test_psnr_metric(self):
        """Test de la métrique PSNR"""
        from core.training.improved_training import calculate_psnr
        
        # Test avec images identiques
        img = torch.rand(1, 3, 256, 256)
        psnr_same = calculate_psnr(img, img)
        self.assertEqual(psnr_same, float('inf'))
        
        # Test avec bruit (sigma=0.05 -> environ 26 dB)
        noise = torch.randn_like(img) * 0.05
        img_noisy = img + noise
        psnr_noisy = calculate_psnr(img, img_noisy)
        
        self.assertGreater(psnr_noisy, 20)  # PSNR > 20 dB est acceptable
        self.assertLess(psnr_noisy, 100)     # Pas infini
    
    def test_ssim_metric(self):
        """Test de la métrique SSIM"""
        def calculate_ssim(img1, img2, window_size=11):
            """Calcul simplifié du SSIM"""
            # Constants
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            # Moyennes
            mu1 = F.avg_pool2d(img1, window_size, 1, window_size//2)
            mu2 = F.avg_pool2d(img2, window_size, 1, window_size//2)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            # Variances
            sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, window_size//2) - mu1_sq
            sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, window_size//2) - mu2_sq
            sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size//2) - mu1_mu2
            
            # SSIM
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                      ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return ssim_map.mean()
        
        # Test avec images identiques
        img = torch.rand(1, 3, 256, 256)
        ssim_same = calculate_ssim(img, img)
        self.assertGreater(ssim_same, 0.99)
        
        # Test avec images différentes
        img2 = torch.rand(1, 3, 256, 256)
        ssim_diff = calculate_ssim(img, img2)
        self.assertLess(ssim_diff, 0.5)
    
    def test_perceptual_loss(self):
        """Test de la loss perceptuelle"""
        # Simuler un extracteur de features simple
        class SimpleFeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
                self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
                self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
            
            def forward(self, x):
                features = []
                x = F.relu(self.conv1(x))
                features.append(x)
                x = F.relu(self.conv2(x))
                features.append(x)
                x = F.relu(self.conv3(x))
                features.append(x)
                return features
        
        extractor = SimpleFeatureExtractor()
        extractor.eval()
        
        # Images de test
        img1 = torch.rand(1, 3, 256, 256)
        img2 = torch.rand(1, 3, 256, 256)
        
        # Extraire les features
        with torch.no_grad():
            features1 = extractor(img1)
            features2 = extractor(img2)
        
        # Calculer la loss perceptuelle
        perceptual_loss = 0
        for f1, f2 in zip(features1, features2):
            perceptual_loss += F.l1_loss(f1, f2)
        
        self.assertGreater(perceptual_loss.item(), 0)


class TestModelQualityAssurance(unittest.TestCase):
    """Tests d'assurance qualité des modèles"""
    
    def test_weight_distribution(self):
        """Test de la distribution des poids"""
        generator = OriginalGenerator()
        
        weight_stats = {}
        
        for name, param in generator.named_parameters():
            # Poids de convolution uniquement (ceux de BatchNorm sont initialisés à 1)
            if 'weight' in name and param.dim() > 1:
                weight_stats[name] = {
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'min': param.min().item(),
                    'max': param.max().item(),
                    'zeros': (param == 0).sum().item(),
                    'total': param.numel()
                }
        
        # Vérifications
        for name, stats in weight_stats.items():
            # Les poids ne devraient pas être tous à zéro
            self.assertLess(stats['zeros'], stats['total'] * 0.1,
                          f"Trop de zéros dans {name}")
            
            # Distribution raisonnable
            self.assertLess(abs(stats['mean']), 0.1,
                          f"Moyenne trop élevée dans {name}")
            self.assertGreater(stats['std'], 0.001,
                             f"Variance trop faible dans {name}")
            self.assertLess(stats['std'], 1.0,
                          f"Variance trop élevée dans {name}")
    
    def test_gradient_health(self):
        """Test de la santé des gradients"""
        generator = OriginalGenerator()
        generator.train()
        
        # Forward pass
        input_tensor = torch.randn(2, 3, 256, 256, requires_grad=True)
        output = generator(input_tensor)
        
        # Loss fictive
        loss = output.mean()
        loss.backward()
        
        gradient_stats = {}
        
        for name, param in generator.named_parameters():
            if param.grad is not None:
                grad = param.grad
                gradient_stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.abs().max().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item()
                }
        
        # Vérifications
        for name, stats in gradient_stats.items():
            self.assertFalse(stats['has_nan'], f"NaN dans gradients de {name}")
            self.assertFalse(stats['has_inf'], f"Inf dans gradients de {name}")
            self.assertLess(stats['max'], 10.0, 
                          f"Gradient trop grand dans {name}")
    
    def test_activation_ranges(self):
        """Test des plages d'activation"""
        generator = OriginalGenerator()
        generator.eval()
        
        activation_ranges = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                activation_ranges[name] = {
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'mean': output.mean().item(),
                    'std': output.std().item()
                }
            return hook
        
        # Ajouter des hooks
        for name, module in generator.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Tanh)):
                h = module.register_forward_hook(hook_fn(name))
                hooks.append(h)
        
        # Forward pass
        input_tensor = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            _ = generator(input_tensor)
        
        # Analyser les activations
        for name, stats in activation_ranges.items():
            if 'relu' in name.lower():
                self.assertGreaterEqual(stats['min'], 0,
                                      f"ReLU négative dans {name}")
            elif 'tanh' in name.lower():
                self.assertGreaterEqual(stats['min'], -1,
                                      f"Tanh < -1 dans {name}")
                self.assertLessEqual(stats['max'], 1,
                                   f"Tanh > 1 dans {name}")
        
        # Nettoyer les hooks
        for h in hooks:
            h.remove()


class TestModelBenchmarking(unittest.TestCase):
    """Tests de benchmarking des modèles"""
    
    def test_inference_benchmark(self):
        """Benchmark d'inférence"""
        import time
        
        generator = OriginalGenerator()
        generator.eval()
        
        # Différentes configurations
        configs = [
            {'batch_size': 1, 'image_size': 256},
            {'batch_size': 4, 'image_size': 256},
            {'batch_size': 8, 'image_size': 256},
            {'batch_size': 1, 'image_size': 512},
        ]
        
        results = []
        
        for config in configs:
            input_tensor = torch.randn(
                config['batch_size'], 3, 
                config['image_size'], config['image_size']
            )
            
            # Warmup
            with torch.no_grad():
                _ = generator(input_tensor)
            
            # Benchmark
            num_runs = 10
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = generator(input_tensor)
            
            total_time = time.time() - start_time
            avg_time = total_time / num_runs
            throughput = config['batch_size'] / avg_time
            
            results.append({
                'config': config,
                'avg_time': avg_time,
                'throughput': throughput
            })
        
        # Afficher les résultats
        print("\nBenchmark d'inférence:")
        for result in results:
            config = result['config']
            print(f"  Batch {config['batch_size']}, Size {config['image_size']}: "
                  f"{result['avg_time']:.4f}s, "
                  f"{result['throughput']:.2f} images/s")
    
    def test_memory_footprint(self):
        """Test de l'empreinte mémoire"""
        generator = OriginalGenerator()
        
        # Calculer la taille du modèle
        param_memory = 0
        buffer_memory = 0
        
        for param in generator.parameters():
            param_memory += param.numel() * param.element_size()
        
        for buffer in generator.buffers():
            buffer_memory += buffer.numel() * buffer.element_size()
        
        total_memory_mb = (param_memory + buffer_memory) / (1024 * 1024)
        
        print(f"\nEmpreinte mémoire du modèle:")
        print(f"  Paramètres: {param_memory / (1024 * 1024):.2f} MB")
        print(f"  Buffers: {buffer_memory / (1024 * 1024):.2f} MB")
        print(f"  Total: {total_memory_mb:.2f} MB")
        
        # Le modèle ne devrait pas être trop gros
        self.assertLess(total_memory_mb, 1000)  # Moins de 1GB
    
    def test_optimization_impact(self):
        """Test de l'impact des optimisations"""
        generator = OriginalGenerator()
        generator.eval()
        
        input_tensor = torch.randn(1, 3, 256, 256)
        
        # Test sans optimisation
        import time
        
        # Mode normal
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = generator(input_tensor)
        normal_time = time.time() - start
        
        # Avec torch.jit.optimize_for_inference (si disponible)
        try:
            if hasattr(torch.jit, 'optimize_for_inference'):
                traced = torch.jit.trace(generator, input_tensor)
                optimized = torch.jit.optimize_for_inference(traced)
                
                start = time.time()
                with torch.no_grad():
                    for _ in range(10):
                        _ = optimized(input_tensor)
                optimized_time = time.time() - start
                
                speedup = normal_time / optimized_time
                print(f"\nSpeedup avec optimisation: {speedup:.2f}x")
        except:
            print("\nOptimisation JIT non disponible")


class TestModelDiagnostics(unittest.TestCase):
    """Tests de diagnostic des modèles"""
    
    def test_layer_connectivity(self):
        """Test de la connectivité des couches"""
        generator = OriginalGenerator()
        
        # Vérifier que toutes les couches sont utilisées
        used_modules = set()
        
        def register_hook(module):
            def hook(m, i, o):
                used_modules.add(m)
            return module.register_forward_hook(hook)
        
        hooks = []
        for module in generator.modules():
            if module != generator:
                hooks.append(register_hook(module))
        
        # Forward pass
        input_tensor = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            _ = generator(input_tensor)
        
        # Nettoyer les hooks
        for h in hooks:
            h.remove()
        
        # Vérifier que toutes les couches sont utilisées
        total_modules = len(list(generator.modules())) - 1  # -1 pour le module principal
        used_count = len(used_modules)
        
        self.assertEqual(used_count, total_modules,
                        f"Seulement {used_count}/{total_modules} modules utilisés")
    
    def test_numerical_stability(self):
        """Test de stabilité numérique"""
        generator = OriginalGenerator()
        generator.eval()
        
        # Test avec différentes échelles d'entrée
        scales = [0.001, 0.1, 1.0, 10.0, 100.0]
        
        for scale in scales:
            with self.subTest(scale=scale):
                input_tensor = torch.randn(1, 3, 256, 256) * scale
                
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    
                    with torch.no_grad():
                        output = generator(input_tensor)
                    
                    # Vérifier qu'il n'y a pas d'avertissements numériques
                    numerical_warnings = [warning for warning in w 
                                        if 'overflow' in str(warning.message) or
                                           'underflow' in str(warning.message)]
                    
                    self.assertEqual(len(numerical_warnings), 0,
                                   f"Avertissements numériques avec scale={scale}")
                
                # Vérifier la sortie
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())


if __name__ == '__main__':
    unittest.main()