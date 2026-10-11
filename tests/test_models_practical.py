#!/usr/bin/env python3
"""
Tests pratiques pour les modèles (utilisation réelle, intégration, etc.)
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import json
import time
from unittest.mock import Mock, patch, MagicMock

from core.models.model_manager import OriginalGenerator, ModelManager


class TestModelPracticalUsage(unittest.TestCase):
    """Tests d'utilisation pratique des modèles"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.device = torch.device('cpu')
        self.manager = ModelManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Nettoyage après test"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_inference_pipeline(self):
        """Test du pipeline d'inférence complet"""
        # 1. Charger le modèle
        generator = self.manager.load_generator(None, self.device)
        
        # 2. Préparer l'entrée
        input_image = Image.new('RGB', (256, 256), color=(128, 128, 128))
        input_array = np.array(input_image)
        input_tensor = torch.from_numpy(input_array).float().permute(2, 0, 1).unsqueeze(0)
        input_tensor = (input_tensor / 255.0 - 0.5) / 0.5  # Normalisation
        
        # 3. Inférence
        with torch.no_grad():
            output_tensor = generator(input_tensor)
        
        # 4. Post-traitement
        output_tensor = (output_tensor * 0.5 + 0.5) * 255.0  # Dénormalisation
        output_tensor = torch.clamp(output_tensor, 0, 255)
        output_array = output_tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        output_image = Image.fromarray(output_array)
        
        # Vérifications
        self.assertEqual(output_image.size, (256, 256))
        self.assertEqual(output_image.mode, 'RGB')
    
    def test_batch_inference(self):
        """Test d'inférence par batch"""
        generator = self.manager.create_generator(self.device)
        generator.eval()
        
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Créer un batch
                batch = torch.randn(batch_size, 3, 256, 256)
                
                # Inférence
                start_time = time.time()
                with torch.no_grad():
                    outputs = generator(batch)
                inference_time = time.time() - start_time
                
                # Vérifications
                self.assertEqual(outputs.shape[0], batch_size)
                
                # Temps par image
                time_per_image = inference_time / batch_size
                print(f"Batch {batch_size}: {time_per_image:.4f}s par image")
    
    def test_model_state_persistence(self):
        """Test de persistance de l'état du modèle"""
        # Créer et modifier un modèle
        generator1 = self.manager.create_generator(self.device)
        
        # Modifier quelques poids
        with torch.no_grad():
            for param in generator1.parameters():
                param.add_(0.1)
        
        # Sauvegarder
        checkpoint_path = Path(self.temp_dir) / 'test_checkpoint.pth'
        torch.save({
            'generator_state_dict': generator1.state_dict(),
            'metadata': {
                'version': '1.0',
                'date': '2024-01-01'
            }
        }, checkpoint_path)
        
        # Charger dans un nouveau modèle
        generator2 = self.manager.load_generator(str(checkpoint_path), self.device)
        
        # Comparer les poids
        for (n1, p1), (n2, p2) in zip(
            generator1.named_parameters(),
            generator2.named_parameters()
        ):
            self.assertTrue(torch.allclose(p1, p2, atol=1e-6))
    
    def test_model_versioning(self):
        """Test de versioning de modèles"""
        versions = []
        
        # Créer plusieurs versions
        for version in range(3):
            generator = self.manager.create_generator(self.device)
            
            # Modifier légèrement chaque version
            with torch.no_grad():
                for param in generator.parameters():
                    param.mul_(1.0 + version * 0.1)
            
            # Sauvegarder avec métadonnées
            checkpoint = {
                'model_state_dict': generator.state_dict(),
                'version': f'v{version + 1}.0',
                'training_epochs': version * 100,
                'performance_metrics': {
                    'psnr': 20 + version * 2,
                    'ssim': 0.7 + version * 0.1
                }
            }
            versions.append(checkpoint)
        
        # Sauvegarder toutes les versions
        for i, checkpoint in enumerate(versions):
            path = Path(self.temp_dir) / f'model_v{i + 1}.pth'
            torch.save(checkpoint, path)
        
        # Charger et vérifier une version spécifique
        v2_path = Path(self.temp_dir) / 'model_v2.pth'
        loaded_checkpoint = torch.load(v2_path, map_location=self.device)
        
        self.assertEqual(loaded_checkpoint['version'], 'v2.0')
        self.assertEqual(loaded_checkpoint['training_epochs'], 100)
    
    def test_model_compatibility_check(self):
        """Test de vérification de compatibilité"""
        # Créer un checkpoint avec configuration
        generator = OriginalGenerator(input_channels=3, output_channels=3, ngf=64)
        checkpoint = {
            'model_state_dict': generator.state_dict(),
            'model_config': {
                'input_channels': 3,
                'output_channels': 3,
                'ngf': 64
            },
            'model_architecture': 'OriginalGenerator',
            'framework_version': torch.__version__
        }
        
        # Fonction de vérification de compatibilité
        def check_compatibility(checkpoint, target_config):
            # Vérifier l'architecture
            if checkpoint.get('model_architecture') != 'OriginalGenerator':
                return False, "Architecture incompatible"
            
            # Vérifier la configuration
            ckpt_config = checkpoint.get('model_config', {})
            for key, value in target_config.items():
                if ckpt_config.get(key) != value:
                    return False, f"Configuration mismatch: {key}"
            
            return True, "Compatible"
        
        # Test de compatibilité
        target_config = {'input_channels': 3, 'output_channels': 3, 'ngf': 64}
        is_compatible, message = check_compatibility(checkpoint, target_config)
        self.assertTrue(is_compatible)
        
        # Test d'incompatibilité
        wrong_config = {'input_channels': 1, 'output_channels': 3, 'ngf': 64}
        is_compatible, message = check_compatibility(checkpoint, wrong_config)
        self.assertFalse(is_compatible)


class TestModelInferenceOptimization(unittest.TestCase):
    """Tests d'optimisation de l'inférence"""
    
    def test_inference_caching(self):
        """Test de mise en cache pour l'inférence"""
        class CachedGenerator:
            def __init__(self, model):
                self.model = model
                self.cache = {}
            
            def generate(self, input_tensor, use_cache=True):
                # Créer une clé de cache basée sur la forme et les stats
                cache_key = (
                    input_tensor.shape,
                    float(input_tensor.mean()),
                    float(input_tensor.std())
                )
                
                if use_cache and cache_key in self.cache:
                    return self.cache[cache_key]
                
                # Génération normale
                with torch.no_grad():
                    output = self.model(input_tensor)
                
                if use_cache:
                    self.cache[cache_key] = output
                
                return output
        
        # Test
        generator = OriginalGenerator()
        generator.eval()
        cached_gen = CachedGenerator(generator)
        
        # Première génération
        input1 = torch.randn(1, 3, 256, 256)
        start_time = time.time()
        output1 = cached_gen.generate(input1)
        first_time = time.time() - start_time
        
        # Deuxième génération (depuis cache)
        start_time = time.time()
        output2 = cached_gen.generate(input1)
        cached_time = time.time() - start_time
        
        # Le cache devrait être plus rapide
        self.assertLess(cached_time, first_time)
        self.assertTrue(torch.equal(output1, output2))
    
    def test_dynamic_batching(self):
        """Test du batching dynamique"""
        class DynamicBatcher:
            def __init__(self, model, max_batch_size=8, timeout=0.1):
                self.model = model
                self.max_batch_size = max_batch_size
                self.timeout = timeout
                self.queue = []
            
            def add_request(self, input_tensor):
                self.queue.append(input_tensor)
                
                if len(self.queue) >= self.max_batch_size:
                    return self._process_batch()
                
                return None
            
            def _process_batch(self):
                if not self.queue:
                    return []
                
                # Créer un batch
                batch = torch.cat(self.queue, dim=0)
                self.queue = []
                
                # Traiter le batch
                with torch.no_grad():
                    outputs = self.model(batch)
                
                # Séparer les résultats
                return list(outputs.split(1, dim=0))
        
        # Test
        generator = OriginalGenerator()
        generator.eval()
        batcher = DynamicBatcher(generator, max_batch_size=4)
        
        # Ajouter des requêtes
        inputs = [torch.randn(1, 3, 256, 256) for _ in range(4)]
        
        results = None
        for inp in inputs:
            results = batcher.add_request(inp)
        
        # Après 4 requêtes, on devrait avoir des résultats
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 4)
    
    def test_mixed_precision_inference(self):
        """Test d'inférence en précision mixte"""
        generator = OriginalGenerator()
        generator.eval()
        
        # Input en float32
        input_fp32 = torch.randn(1, 3, 256, 256)
        
        # Inférence normale
        with torch.no_grad():
            output_fp32 = generator(input_fp32)
        
        # Essayer en half precision si possible
        if torch.cuda.is_available():
            generator_fp16 = generator.half().cuda()
            input_fp16 = input_fp32.half().cuda()
            
            with torch.no_grad():
                output_fp16 = generator_fp16(input_fp16)
            
            # Comparer (avec tolérance plus grande pour fp16)
            output_fp16_32 = output_fp16.float().cpu()
            
            # Les résultats devraient être similaires
            max_diff = torch.abs(output_fp32 - output_fp16_32).max()
            self.assertLess(max_diff, 0.1)


class TestModelRealWorldScenarios(unittest.TestCase):
    """Tests de scénarios réels"""
    
    def test_streaming_inference(self):
        """Test d'inférence en streaming"""
        class StreamingGenerator:
            def __init__(self, model, chunk_size=64):
                self.model = model
                self.chunk_size = chunk_size
            
            def generate_streaming(self, input_tensor):
                """Génère l'image par morceaux (simulation)"""
                h, w = input_tensor.shape[-2:]
                output = torch.zeros_like(input_tensor)
                
                # Traiter par morceaux
                for i in range(0, h, self.chunk_size):
                    for j in range(0, w, self.chunk_size):
                        # Extraire le morceau
                        chunk_h = min(self.chunk_size, h - i)
                        chunk_w = min(self.chunk_size, w - j)
                        
                        chunk_input = input_tensor[:, :, i:i+chunk_h, j:j+chunk_w]
                        
                        # Générer (simulation - en réalité nécessiterait padding)
                        with torch.no_grad():
                            # Pour ce test, on simule juste
                            chunk_output = chunk_input * 0.9
                        
                        output[:, :, i:i+chunk_h, j:j+chunk_w] = chunk_output
                        
                        yield (i, j, chunk_output)
                
                return output
        
        # Test
        generator = OriginalGenerator()
        streaming_gen = StreamingGenerator(generator)
        
        input_tensor = torch.randn(1, 3, 256, 256)
        
        chunks_processed = 0
        for i, j, chunk in streaming_gen.generate_streaming(input_tensor):
            chunks_processed += 1
            self.assertIsInstance(chunk, torch.Tensor)
        
        self.assertGreater(chunks_processed, 0)
    
    def test_multi_resolution_inference(self):
        """Test d'inférence multi-résolution"""
        generator = OriginalGenerator()
        generator.eval()
        
        resolutions = [(256, 256), (512, 512)]
        
        inference_times = {}
        
        for resolution in resolutions:
            input_tensor = torch.randn(1, 3, *resolution)
            
            start_time = time.time()
            with torch.no_grad():
                output = generator(input_tensor)
            inference_time = time.time() - start_time
            
            inference_times[resolution] = inference_time
            
            # Vérifier la sortie
            self.assertEqual(output.shape[-2:], resolution)
        
        # Le temps devrait augmenter avec la résolution
        print("\nTemps d'inférence par résolution:")
        for res, t in inference_times.items():
            print(f"  {res}: {t:.4f}s")
    
    def test_error_recovery(self):
        """Test de récupération d'erreurs"""
        class RobustGenerator:
            def __init__(self, model):
                self.model = model
                self.fallback_output = None
            
            def generate_safe(self, input_tensor):
                try:
                    # Validation de l'entrée
                    if not isinstance(input_tensor, torch.Tensor):
                        raise ValueError("Input must be a tensor")
                    
                    if input_tensor.dim() != 4:
                        raise ValueError(f"Expected 4D tensor, got {input_tensor.dim()}D")
                    
                    if input_tensor.shape[1] != 3:
                        raise ValueError(f"Expected 3 channels, got {input_tensor.shape[1]}")
                    
                    # Génération normale
                    with torch.no_grad():
                        output = self.model(input_tensor)
                    
                    # Sauvegarder comme fallback
                    self.fallback_output = output.clone()
                    
                    return output, None
                    
                except Exception as e:
                    # En cas d'erreur, retourner le fallback ou une image noire
                    if self.fallback_output is not None:
                        return self.fallback_output, f"Error: {e}, using fallback"
                    else:
                        fallback = torch.zeros_like(input_tensor)
                        return fallback, f"Error: {e}, using zeros"
        
        # Test
        generator = OriginalGenerator()
        generator.eval()
        robust_gen = RobustGenerator(generator)
        
        # Test normal
        good_input = torch.randn(1, 3, 256, 256)
        output, error = robust_gen.generate_safe(good_input)
        self.assertIsNone(error)
        
        # Test avec mauvaise entrée
        bad_input = torch.randn(1, 4, 256, 256)  # 4 canaux au lieu de 3
        output, error = robust_gen.generate_safe(bad_input)
        self.assertIsNotNone(error)
        self.assertIn("Expected 3 channels", error)
    
    def test_model_serving_preparation(self):
        """Test de préparation pour le serving"""
        # Configuration de serving
        serving_config = {
            'model_name': 'satellite_generator',
            'version': '1.0',
            'input_shape': [1, 3, 256, 256],
            'output_shape': [1, 3, 256, 256],
            'preprocessing': {
                'normalization': 'standard',
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5]
            },
            'postprocessing': {
                'denormalization': True,
                'clipping': [0, 1]
            },
            'optimization': {
                'precision': 'fp32',
                'max_batch_size': 8,
                'dynamic_batching': True
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Sauvegarder la configuration
            config_path = Path(temp_dir) / 'serving_config.json'
            with open(config_path, 'w') as f:
                json.dump(serving_config, f, indent=2)
            
            # Vérifier que la configuration est valide
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
        
        self.assertEqual(loaded_config['model_name'], 'satellite_generator')
        self.assertEqual(loaded_config['input_shape'], [1, 3, 256, 256])
        
        # Préparer le modèle pour le serving
        generator = OriginalGenerator()
        generator.eval()
        
        # Optimisations pour le serving
        # 1. Désactiver le gradient
        for param in generator.parameters():
            param.requires_grad = False
        
        # 2. Compiler si possible (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                compiled_model = torch.compile(generator)
                self.assertIsNotNone(compiled_model)
            except:
                pass  # La compilation peut échouer sur certaines plateformes


if __name__ == '__main__':
    unittest.main()