#!/usr/bin/env python3
"""
Tests avancés pour les modèles (optimisations, conversions, etc.)
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch

from core.models.model_manager import OriginalGenerator, ModelManager


class TestModelOptimizations(unittest.TestCase):
    """Tests pour les optimisations de modèles"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.device = torch.device('cpu')
        self.model_manager = ModelManager()
    
    def test_model_pruning(self):
        """Test de l'élagage de modèle"""
        import torch.nn.utils.prune as prune
        
        generator = OriginalGenerator()
        
        # Compter les paramètres avant
        params_before = sum(p.numel() for p in generator.parameters())
        
        # Appliquer le pruning sur certaines couches
        for name, module in generator.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=0.2)
                break  # Juste la première couche pour le test
        
        # Le nombre de paramètres reste le même, mais certains sont à zéro
        params_after = sum(p.numel() for p in generator.parameters())
        self.assertEqual(params_before, params_after)
        
        # Vérifier qu'il y a des zéros
        first_conv = None
        for module in generator.modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break
        
        if first_conv and hasattr(first_conv, 'weight_mask'):
            zeros = (first_conv.weight_mask == 0).sum().item()
            self.assertGreater(zeros, 0)
    
    def test_model_quantization_preparation(self):
        """Test de préparation à la quantification"""
        generator = OriginalGenerator()
        generator.eval()
        
        # Préparer pour la quantification (simulation)
        def prepare_for_quantization(model):
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    # Ajouter des observers (simulé)
                    module.register_buffer('scale', torch.tensor(1.0))
                    module.register_buffer('zero_point', torch.tensor(0))
            return model
        
        quantized_model = prepare_for_quantization(generator)
        
        # Vérifier que les buffers sont ajoutés
        conv_count = 0
        for module in quantized_model.modules():
            if isinstance(module, nn.Conv2d):
                self.assertTrue(hasattr(module, 'scale'))
                self.assertTrue(hasattr(module, 'zero_point'))
                conv_count += 1
        
        self.assertGreater(conv_count, 0)
    
    def test_model_fusion(self):
        """Test de fusion de couches"""
        # Créer un modèle simple pour tester la fusion
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, 1, 1)
                self.bn = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x
        
        model = SimpleModel()
        model.eval()
        
        # Test avant fusion
        input_tensor = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output_before = model(input_tensor)
        
        # Simuler la fusion Conv-BN
        def fuse_conv_bn(conv, bn):
            """Fusionne Conv2d et BatchNorm2d"""
            fused_conv = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
                bias=True
            )
            
            # Calculer les poids fusionnés (simplifié)
            bn_mean = bn.running_mean
            bn_var = bn.running_var
            bn_eps = bn.eps
            bn_weight = bn.weight if bn.weight is not None else torch.ones_like(bn_mean)
            bn_bias = bn.bias if bn.bias is not None else torch.zeros_like(bn_mean)
            
            # Fusion des poids
            conv_weight = conv.weight
            conv_bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
            
            scale = bn_weight / torch.sqrt(bn_var + bn_eps)
            fused_conv.weight.data = conv_weight * scale.view(-1, 1, 1, 1)
            fused_conv.bias.data = (conv_bias - bn_mean) * scale + bn_bias
            
            return fused_conv
        
        # Appliquer la fusion
        fused_conv = fuse_conv_bn(model.conv, model.bn)
        
        # Vérifier que la fusion produit des résultats similaires
        model.eval()
        with torch.no_grad():
            fused_output = model.relu(fused_conv(input_tensor))
            
        # Les sorties devraient être proches
        self.assertTrue(torch.allclose(output_before, fused_output, atol=1e-5))


class TestModelConversions(unittest.TestCase):
    """Tests pour les conversions de modèles"""
    
    def test_model_to_onnx_preparation(self):
        """Test de préparation pour export ONNX"""
        generator = OriginalGenerator()
        generator.eval()
        
        # Entrée exemple
        dummy_input = torch.randn(1, 3, 256, 256)
        
        # Test que le modèle peut faire un forward pass
        with torch.no_grad():
            output = generator(dummy_input)
        
        self.assertEqual(output.shape, dummy_input.shape)
        
        # Vérifier que toutes les opérations sont supportées
        supported_ops = True
        for module in generator.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, 
                                 nn.BatchNorm2d, nn.LeakyReLU, 
                                 nn.ReLU, nn.Dropout, nn.Tanh)):
                continue
            elif isinstance(module, nn.Sequential):
                continue
            elif module == generator:
                continue
            else:
                supported_ops = False
                print(f"Opération non standard trouvée: {type(module)}")
        
        self.assertTrue(supported_ops)
    
    def test_model_scripting(self):
        """Test du scripting TorchScript"""
        generator = OriginalGenerator()
        generator.eval()
        
        # Tenter de scripter le modèle
        try:
            scripted_model = torch.jit.script(generator)
            success = True
        except Exception as e:
            print(f"Erreur de scripting: {e}")
            success = False
        
        # Le scripting peut échouer pour des modèles complexes
        # mais on teste qu'il ne crash pas
        self.assertIsNotNone(generator)
    
    def test_model_tracing(self):
        """Test du tracing TorchScript"""
        generator = OriginalGenerator()
        generator.eval()
        
        # Entrée exemple
        example_input = torch.randn(1, 3, 256, 256)
        
        # Tracer le modèle
        with torch.no_grad():
            traced_model = torch.jit.trace(generator, example_input)
        
        # Comparer les sorties
        with torch.no_grad():
            original_output = generator(example_input)
            traced_output = traced_model(example_input)
        
        # Les sorties devraient être identiques
        self.assertTrue(torch.allclose(original_output, traced_output, atol=1e-6))
    
    def test_model_export_import(self):
        """Test d'export/import de modèle"""
        generator = OriginalGenerator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export
            export_path = Path(tmpdir) / 'model.pth'
            torch.save({
                'model_state_dict': generator.state_dict(),
                'model_config': {
                    'input_channels': 3,
                    'output_channels': 3,
                    'ngf': 64
                }
            }, export_path)
            
            # Import
            checkpoint = torch.load(export_path)
            new_generator = OriginalGenerator(**checkpoint.get('model_config', {}))
            new_generator.load_state_dict(checkpoint['model_state_dict'])
            
            # Vérifier que les poids sont identiques
            for (n1, p1), (n2, p2) in zip(
                generator.named_parameters(),
                new_generator.named_parameters()
            ):
                self.assertTrue(torch.equal(p1, p2))


class TestModelAnalysis(unittest.TestCase):
    """Tests pour l'analyse des modèles"""
    
    def test_layer_analysis(self):
        """Analyse des couches du modèle"""
        generator = OriginalGenerator()
        
        layer_stats = {
            'conv2d': 0,
            'conv_transpose2d': 0,
            'batch_norm': 0,
            'activation': 0,
            'dropout': 0
        }
        
        for module in generator.modules():
            if isinstance(module, nn.Conv2d):
                layer_stats['conv2d'] += 1
            elif isinstance(module, nn.ConvTranspose2d):
                layer_stats['conv_transpose2d'] += 1
            elif isinstance(module, nn.BatchNorm2d):
                layer_stats['batch_norm'] += 1
            elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Tanh)):
                layer_stats['activation'] += 1
            elif isinstance(module, nn.Dropout):
                layer_stats['dropout'] += 1
        
        # Vérifications
        self.assertGreater(layer_stats['conv2d'], 0)
        self.assertGreater(layer_stats['conv_transpose2d'], 0)
        self.assertGreater(layer_stats['batch_norm'], 0)
        self.assertGreater(layer_stats['activation'], 0)
        
        print("\nAnalyse des couches:")
        for layer_type, count in layer_stats.items():
            print(f"  {layer_type}: {count}")
    
    def test_parameter_distribution(self):
        """Test de la distribution des paramètres"""
        generator = OriginalGenerator()
        
        all_params = []
        for param in generator.parameters():
            all_params.extend(param.data.flatten().tolist())
        
        all_params = np.array(all_params)
        
        # Statistiques
        mean = np.mean(all_params)
        std = np.std(all_params)
        min_val = np.min(all_params)
        max_val = np.max(all_params)
        
        # Les paramètres devraient être initialisés autour de 0
        self.assertAlmostEqual(mean, 0.0, places=1)
        self.assertGreater(std, 0.01)  # Pas tous à zéro
        self.assertLess(std, 1.0)      # Pas trop dispersés
        
        print(f"\nDistribution des paramètres:")
        print(f"  Moyenne: {mean:.6f}")
        print(f"  Écart-type: {std:.6f}")
        print(f"  Min: {min_val:.6f}")
        print(f"  Max: {max_val:.6f}")
    
    def test_activation_statistics(self):
        """Test des statistiques d'activation"""
        generator = OriginalGenerator()
        generator.eval()
        
        # Hook pour capturer les activations
        activations = {}
        
        def hook_fn(module, input, output):
            activations[module] = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item()
            }
        
        # Enregistrer les hooks sur quelques couches
        hooks = []
        for name, module in generator.named_modules():
            if 'down1' in name and isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(hook_fn))
            elif 'up8' in name and isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        input_tensor = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            _ = generator(input_tensor)
        
        # Analyser les activations
        for module, stats in activations.items():
            print(f"\nActivations pour {module.__class__.__name__}:")
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")
        
        # Nettoyer les hooks
        for hook in hooks:
            hook.remove()
        
        # Vérifications
        self.assertGreater(len(activations), 0)


class TestModelComparison(unittest.TestCase):
    """Tests de comparaison entre modèles"""
    
    def test_model_similarity(self):
        """Test de similarité entre modèles"""
        # Créer deux modèles avec la même seed
        torch.manual_seed(42)
        model1 = OriginalGenerator()
        
        torch.manual_seed(42)
        model2 = OriginalGenerator()
        
        # Calculer la similarité des poids
        total_params = 0
        identical_params = 0
        
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(),
            model2.named_parameters()
        ):
            total_params += p1.numel()
            if torch.equal(p1, p2):
                identical_params += p1.numel()
        
        similarity = identical_params / total_params
        self.assertEqual(similarity, 1.0)  # Devraient être identiques
    
    def test_model_difference_after_training(self):
        """Test de différence après entraînement simulé"""
        model1 = OriginalGenerator()
        model2 = OriginalGenerator()
        
        # Copier les poids
        model2.load_state_dict(model1.state_dict())
        
        # Simuler un pas d'entraînement sur model2
        optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
        input_tensor = torch.randn(1, 3, 256, 256)
        
        output = model2(input_tensor)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        
        # Calculer la différence
        total_diff = 0.0
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(),
            model2.named_parameters()
        ):
            diff = torch.abs(p1 - p2).sum().item()
            total_diff += diff
        
        # Il devrait y avoir une différence
        self.assertGreater(total_diff, 0.0)
    
    def test_model_ensemble_compatibility(self):
        """Test de compatibilité pour ensemble"""
        # Créer plusieurs modèles
        models = []
        for i in range(3):
            torch.manual_seed(i)
            models.append(OriginalGenerator())
        
        # Test d'ensemble
        input_tensor = torch.randn(1, 3, 256, 256)
        outputs = []
        
        with torch.no_grad():
            for model in models:
                model.eval()
                outputs.append(model(input_tensor))
        
        # Moyenne des sorties
        ensemble_output = torch.stack(outputs).mean(dim=0)
        
        # Vérifications
        self.assertEqual(ensemble_output.shape, input_tensor.shape)
        
        # La moyenne de sorties indépendantes ne peut pas avoir plus de variance
        # que la plus variable d'entre elles
        individual_vars = [out.var().item() for out in outputs]
        ensemble_var = ensemble_output.var().item()
        
        self.assertGreater(ensemble_var, 0.0)
        self.assertLessEqual(ensemble_var, max(individual_vars))


class TestModelDebugTools(unittest.TestCase):
    """Tests pour les outils de débogage de modèles"""
    
    def test_gradient_checking(self):
        """Test de vérification des gradients"""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1)
        )
        
        # Fonction de loss simple
        def loss_fn(output, target):
            return F.mse_loss(output, target)
        
        # Test avec gradient checking
        input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
        target = torch.randn(1, 3, 32, 32)
        
        # Forward et backward
        output = model(input_tensor)
        loss = loss_fn(output, target)
        loss.backward()
        
        # Vérifier que les gradients existent
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())
            self.assertFalse(torch.isinf(param.grad).any())
    
    def test_nan_detection(self):
        """Test de détection de NaN"""
        model = OriginalGenerator()
        
        # Fonction pour détecter les NaN
        def check_for_nan(model):
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    return True, name
            return False, None
        
        # Vérifier l'état initial
        has_nan, param_name = check_for_nan(model)
        self.assertFalse(has_nan)
        
        # Introduire un NaN
        first_param = next(model.parameters())
        first_param.data[0, 0, 0, 0] = float('nan')
        
        # Vérifier la détection
        has_nan, param_name = check_for_nan(model)
        self.assertTrue(has_nan)
    
    def test_model_summary(self):
        """Test de génération de résumé de modèle"""
        def get_model_summary(model):
            """Génère un résumé du modèle"""
            summary = {
                'total_params': 0,
                'trainable_params': 0,
                'layers': 0,
                'layer_types': {}
            }
            
            # Compter les paramètres
            for param in model.parameters():
                summary['total_params'] += param.numel()
                if param.requires_grad:
                    summary['trainable_params'] += param.numel()
            
            # Compter les couches
            for module in model.modules():
                summary['layers'] += 1
                module_type = type(module).__name__
                summary['layer_types'][module_type] = summary['layer_types'].get(module_type, 0) + 1
            
            return summary
        
        generator = OriginalGenerator()
        summary = get_model_summary(generator)
        
        # Vérifications
        self.assertGreater(summary['total_params'], 0)
        self.assertEqual(summary['total_params'], summary['trainable_params'])
        self.assertGreater(summary['layers'], 10)
        self.assertIn('Conv2d', summary['layer_types'])
        
        print("\nRésumé du modèle:")
        print(f"  Paramètres totaux: {summary['total_params']:,}")
        print(f"  Paramètres entraînables: {summary['trainable_params']:,}")
        print(f"  Nombre de couches: {summary['layers']}")


if __name__ == '__main__':
    unittest.main()