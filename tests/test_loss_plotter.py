#!/usr/bin/env python3
"""
Tests unitaires pour le module loss_plotter
"""

import unittest
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.training.loss_plotter import LossPlotter


def make_losses(num_epochs):
    """Crée un historique de losses au format écrit par l'entraînement"""
    return [
        {
            'epoch': i + 1,
            'loss_G': 10.0 / (i + 1),
            'loss_D': 0.5,
            'loss_G_GAN': 1.0,
            'loss_G_L1': 9.0 / (i + 1),
            'loss_D_real': 0.3,
            'loss_D_fake': 0.2
        }
        for i in range(num_epochs)
    ]


class TestLossPlotter(unittest.TestCase):
    """Tests pour le générateur de graphiques de losses"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "training_losses.json"
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        import shutil
        plt.close('all')
        shutil.rmtree(self.temp_dir)
    
    def _write_log(self, data):
        with open(self.log_file, 'w') as f:
            json.dump(data, f)
    
    def test_load_losses(self):
        """Test du chargement d'un fichier de losses valide"""
        self._write_log(make_losses(5))
        plotter = LossPlotter(str(self.log_file))
        
        self.assertTrue(plotter.load_losses())
        self.assertEqual(len(plotter.losses_data), 5)
        self.assertEqual(plotter.losses_data[0]['epoch'], 1)
    
    def test_load_losses_missing_file(self):
        """Test avec un fichier de logs inexistant"""
        plotter = LossPlotter(str(Path(self.temp_dir) / "absent.json"))
        
        self.assertFalse(plotter.load_losses())
        self.assertIsNone(plotter.losses_data)
    
    def test_load_losses_invalid_json(self):
        """Test avec un fichier JSON invalide"""
        self.log_file.write_text("ceci n'est pas du json")
        plotter = LossPlotter(str(self.log_file))
        
        self.assertFalse(plotter.load_losses())
    
    def test_create_sample_data(self):
        """Test de la génération de données d'exemple"""
        plotter = LossPlotter()
        
        self.assertTrue(plotter.create_sample_data(num_epochs=30))
        self.assertEqual(len(plotter.losses_data), 30)
        
        expected_keys = {'epoch', 'loss_G', 'loss_D', 'loss_G_GAN',
                         'loss_G_L1', 'loss_D_real', 'loss_D_fake'}
        for entry in plotter.losses_data:
            self.assertEqual(set(entry.keys()), expected_keys)
        self.assertEqual([d['epoch'] for d in plotter.losses_data], list(range(1, 31)))
    
    def test_plot_main_losses(self):
        """Test de la génération du graphique principal"""
        self._write_log(make_losses(30))
        plotter = LossPlotter(str(self.log_file))
        plotter.load_losses()
        
        save_path = Path(self.temp_dir) / "plots" / "losses.png"
        
        self.assertTrue(plotter.plot_main_losses(str(save_path)))
        self.assertTrue(save_path.exists())
        self.assertGreater(save_path.stat().st_size, 0)
    
    def test_plot_main_losses_few_epochs(self):
        """Test avec trop peu d'époques pour la moyenne mobile"""
        self._write_log(make_losses(2))
        plotter = LossPlotter(str(self.log_file))
        plotter.load_losses()
        
        save_path = Path(self.temp_dir) / "plots" / "losses.png"
        
        self.assertTrue(plotter.plot_main_losses(str(save_path)))
        self.assertTrue(save_path.exists())
    
    def test_plot_main_losses_without_data(self):
        """Test du graphique principal sans données chargées"""
        plotter = LossPlotter(str(self.log_file))
        save_path = Path(self.temp_dir) / "losses.png"
        
        self.assertFalse(plotter.plot_main_losses(str(save_path)))
        self.assertFalse(save_path.exists())
    
    def test_plot_detailed_discriminator(self):
        """Test de la génération du graphique du discriminateur"""
        self._write_log(make_losses(10))
        plotter = LossPlotter(str(self.log_file))
        plotter.load_losses()
        
        save_path = Path(self.temp_dir) / "discriminator.png"
        
        self.assertTrue(plotter.plot_detailed_discriminator(str(save_path)))
        self.assertTrue(save_path.exists())
    
    def test_plot_detailed_discriminator_without_details(self):
        """Test sans le détail real/fake du discriminateur"""
        losses = [{'epoch': 1, 'loss_G': 1.0, 'loss_D': 0.5}]
        self._write_log(losses)
        plotter = LossPlotter(str(self.log_file))
        plotter.load_losses()
        
        save_path = Path(self.temp_dir) / "discriminator.png"
        
        self.assertFalse(plotter.plot_detailed_discriminator(str(save_path)))
        self.assertFalse(save_path.exists())
    
    def test_validation_metrics_are_ignored(self):
        """Test avec des époques contenant des métriques de validation"""
        losses = make_losses(10)
        losses[4].update({'val_loss_G': 5.0, 'val_loss_D': 0.4,
                          'val_psnr': 15.0, 'val_ssim': 0.3})
        self._write_log(losses)
        plotter = LossPlotter(str(self.log_file))
        plotter.load_losses()
        
        save_path = Path(self.temp_dir) / "plots" / "losses.png"
        
        self.assertTrue(plotter.plot_main_losses(str(save_path)))
    
    def test_print_stats_without_data(self):
        """Test que print_stats ne lève pas d'erreur sans données"""
        plotter = LossPlotter(str(self.log_file))
        plotter.print_stats()


if __name__ == '__main__':
    unittest.main()
