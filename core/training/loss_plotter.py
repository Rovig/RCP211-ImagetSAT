#!/usr/bin/env python3
"""
Script pour générer des graphiques d'évolution des losses du GAN
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

class LossPlotter:
    """Générateur de graphiques d'évolution des losses"""
    
    def __init__(self, log_file="logs/training_losses.json"):
        self.log_file = log_file
        self.losses_data = None
    
    def load_losses(self):
        """Charge les données de losses depuis le fichier JSON"""
        if not os.path.exists(self.log_file):
            print(f"⚠️  Fichier de logs non trouvé: {self.log_file}")
            return False
        
        try:
            with open(self.log_file, 'r') as f:
                self.losses_data = json.load(f)
            print(f"✅ Données chargées: {len(self.losses_data)} époques")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def create_sample_data(self, num_epochs=200):
        """Crée des données d'exemple pour démonstration"""
        print(f"📊 Génération de données d'exemple pour {num_epochs} époques...")
        
        # Simulation d'évolution réaliste des losses
        epochs = np.arange(1, num_epochs + 1)
        
        # Loss du générateur (commence haut, décroît avec oscillations)
        loss_G = 3.0 + 2.0 * np.exp(-epochs/50) + 0.3 * np.sin(epochs/10) + 0.1 * np.random.normal(0, 1, len(epochs))
        loss_G = np.maximum(loss_G, 0.5)  # Minimum réaliste
        
        # Loss du discriminateur (oscille autour de 0.5-1.0)
        loss_D = 0.7 + 0.3 * np.sin(epochs/15 + 1) + 0.2 * np.exp(-epochs/80) + 0.1 * np.random.normal(0, 1, len(epochs))
        loss_D = np.maximum(loss_D, 0.1)  # Minimum réaliste
        
        # Loss L1 (décroît progressivement)
        loss_L1 = 20.0 * np.exp(-epochs/40) + 5.0 + 0.5 * np.random.normal(0, 1, len(epochs))
        loss_L1 = np.maximum(loss_L1, 2.0)  # Minimum réaliste
        
        self.losses_data = []
        for i, epoch in enumerate(epochs):
            self.losses_data.append({
                'epoch': int(epoch),
                'loss_G': float(loss_G[i]),
                'loss_D': float(loss_D[i]),
                'loss_G_GAN': float(loss_G[i] - loss_L1[i]/100),
                'loss_G_L1': float(loss_L1[i]),
                'loss_D_real': float(loss_D[i] * 0.6),
                'loss_D_fake': float(loss_D[i] * 0.4)
            })
        
        return True
    
    def plot_main_losses(self, save_path="logs/losses_evolution.png"):
        """Génère le graphique principal des losses G et D"""
        if not self.losses_data:
            print("❌ Aucune donnée disponible")
            return False
        
        epochs = [d['epoch'] for d in self.losses_data]
        loss_G = [d['loss_G'] for d in self.losses_data]
        loss_D = [d['loss_D'] for d in self.losses_data]
        
        plt.figure(figsize=(12, 8))
        
        # Graphique principal
        plt.subplot(2, 2, 1)
        plt.plot(epochs, loss_G, 'b-', linewidth=2, label='Generator Loss', alpha=0.8)
        plt.plot(epochs, loss_D, 'r-', linewidth=2, label='Discriminator Loss', alpha=0.8)
        plt.xlabel('Époque')
        plt.ylabel('Perte')
        plt.title('Évolution des Losses Générateur vs Discriminateur', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Moyennes mobiles (lissage)
        window = min(10, len(epochs)//10)
        if window > 1:
            def moving_average(data, window):
                return np.convolve(data, np.ones(window), 'valid') / window
            
            if len(loss_G) >= window:
                loss_G_smooth = moving_average(loss_G, window)
                loss_D_smooth = moving_average(loss_D, window)
                epochs_smooth = epochs[window-1:]
                
                plt.subplot(2, 2, 2)
                plt.plot(epochs_smooth, loss_G_smooth, 'b-', linewidth=3, label='Generator (lissé)', alpha=0.9)
                plt.plot(epochs_smooth, loss_D_smooth, 'r-', linewidth=3, label='Discriminator (lissé)', alpha=0.9)
                plt.xlabel('Époque')
                plt.ylabel('Perte (moyenne mobile)')
                plt.title(f'Tendances Lissées (fenêtre={window})', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        # Détail des composantes du générateur
        if 'loss_G_GAN' in self.losses_data[0] and 'loss_G_L1' in self.losses_data[0]:
            loss_G_GAN = [d['loss_G_GAN'] for d in self.losses_data]
            loss_G_L1 = [d['loss_G_L1'] for d in self.losses_data]
            
            plt.subplot(2, 2, 3)
            plt.plot(epochs, loss_G_GAN, 'g-', linewidth=2, label='GAN Loss', alpha=0.8)
            plt.plot(epochs, loss_G_L1, 'm-', linewidth=2, label='L1 Loss', alpha=0.8)
            plt.xlabel('Époque')
            plt.ylabel('Perte')
            plt.title('Composantes de la Loss Générateur', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Échelle log car L1 est beaucoup plus grande
        
        # Ratio G/D pour équilibre
        ratio_GD = [g/d if d > 0 else 0 for g, d in zip(loss_G, loss_D)]
        plt.subplot(2, 2, 4)
        plt.plot(epochs, ratio_GD, 'purple', linewidth=2, alpha=0.8)
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Équilibre parfait')
        plt.xlabel('Époque')
        plt.ylabel('Ratio Loss G / Loss D')
        plt.title('Équilibre Générateur/Discriminateur', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Graphique sauvegardé: {save_path}")
        
        # Afficher quelques statistiques
        self.print_stats()
        
        return True
    
    def plot_detailed_discriminator(self, save_path="logs/discriminator_details.png"):
        """Graphique détaillé du discriminateur"""
        if not self.losses_data or 'loss_D_real' not in self.losses_data[0]:
            return False
        
        epochs = [d['epoch'] for d in self.losses_data]
        loss_D_real = [d['loss_D_real'] for d in self.losses_data]
        loss_D_fake = [d['loss_D_fake'] for d in self.losses_data]
        loss_D_total = [d['loss_D'] for d in self.losses_data]
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, loss_D_real, 'g-', linewidth=2, label='Loss sur vraies images', alpha=0.8)
        plt.plot(epochs, loss_D_fake, 'orange', linewidth=2, label='Loss sur images générées', alpha=0.8)
        plt.plot(epochs, loss_D_total, 'r--', linewidth=2, label='Loss totale discriminateur', alpha=0.8)
        
        plt.xlabel('Époque')
        plt.ylabel('Perte')
        plt.title('Détail des Losses du Discriminateur', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Graphique discriminateur sauvegardé: {save_path}")
        
        return True
    
    def print_stats(self):
        """Affiche des statistiques sur l'entraînement"""
        if not self.losses_data:
            return
        
        loss_G = [d['loss_G'] for d in self.losses_data]
        loss_D = [d['loss_D'] for d in self.losses_data]
        
        print("\n📈 STATISTIQUES D'ENTRAÎNEMENT:")
        print(f"   Nombre d'époques: {len(self.losses_data)}")
        print(f"   Loss G - Min: {min(loss_G):.3f}, Max: {max(loss_G):.3f}, Finale: {loss_G[-1]:.3f}")
        print(f"   Loss D - Min: {min(loss_D):.3f}, Max: {max(loss_D):.3f}, Finale: {loss_D[-1]:.3f}")
        print(f"   Ratio G/D final: {loss_G[-1]/loss_D[-1]:.2f}")
        
        # Convergence
        if len(loss_G) >= 20:
            recent_G = np.mean(loss_G[-20:])
            early_G = np.mean(loss_G[:20])
            improvement_G = ((early_G - recent_G) / early_G) * 100
            print(f"   Amélioration G: {improvement_G:.1f}% (20 dernières vs 20 premières époques)")

def main():
    parser = argparse.ArgumentParser(description='Générateur de graphiques de losses')
    parser.add_argument('--log_file', type=str, default='logs/training_losses.json',
                       help='Fichier JSON contenant les losses')
    parser.add_argument('--output_dir', type=str, default='logs',
                       help='Répertoire de sortie pour les graphiques')
    parser.add_argument('--sample_data', action='store_true',
                       help='Générer des données d\'exemple')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Nombre d\'époques pour les données d\'exemple')
    
    args = parser.parse_args()
    
    plotter = LossPlotter(args.log_file)
    
    # Charger ou générer les données
    if args.sample_data:
        plotter.create_sample_data(args.epochs)
    else:
        if not plotter.load_losses():
            print("🔄 Génération de données d'exemple...")
            plotter.create_sample_data(args.epochs)
    
    # Générer les graphiques
    os.makedirs(args.output_dir, exist_ok=True)
    
    main_plot = os.path.join(args.output_dir, "losses_evolution.png")
    detail_plot = os.path.join(args.output_dir, "discriminator_details.png")
    
    success = plotter.plot_main_losses(main_plot)
    if success:
        plotter.plot_detailed_discriminator(detail_plot)
        print(f"\n✅ Graphiques générés dans: {args.output_dir}/")
    else:
        print("❌ Erreur lors de la génération des graphiques")

if __name__ == "__main__":
    main()