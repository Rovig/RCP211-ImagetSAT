#!/usr/bin/env python3
"""
Entraînement amélioré pour de meilleurs résultats avec le dataset Inria
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import random
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from torch.optim.lr_scheduler import StepLR
import torchvision.utils as vutils
import json
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from core.models.networks import ImprovedGenerator, ImprovedDiscriminator

class ImprovedInriaDataset(Dataset):
    """Dataset amélioré avec augmentation de données et meilleure résolution."""
    
    def __init__(self, images_dir, gt_dir, image_size=512, mode='train', crop_patches=True, val_split=0.2, random_seed=42):
        self.images_dir = Path(images_dir)
        self.gt_dir = Path(gt_dir)
        self.image_size = image_size
        self.mode = mode
        self.crop_patches = crop_patches
        
        # Lister tous les fichiers
        all_image_files = sorted(list(self.images_dir.glob('*.tif')))
        all_gt_files = sorted(list(self.gt_dir.glob('*.tif')))
        
        # Split train/val si on est en mode train ou val
        if mode in ['train', 'val'] and val_split > 0:
            # Créer des indices pour le split
            indices = list(range(len(all_image_files)))
            train_indices, val_indices = train_test_split(
                indices, test_size=val_split, random_state=random_seed
            )
            
            if mode == 'train':
                self.image_files = [all_image_files[i] for i in train_indices]
                self.gt_files = [all_gt_files[i] for i in train_indices]
            else:  # mode == 'val'
                self.image_files = [all_image_files[i] for i in val_indices]
                self.gt_files = [all_gt_files[i] for i in val_indices]
        else:
            # Mode test ou pas de split
            self.image_files = all_image_files
            self.gt_files = all_gt_files
        
        print(f"Dataset init - Mode: {self.mode}")
        print(f"Dataset init - Images dir: {self.images_dir}")
        print(f"Dataset init - GT dir: {self.gt_dir}")
        print(f"Dataset init - Found {len(self.image_files)} {mode} files")
        print(f"Dataset init - Total files available: {len(all_image_files)}")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {self.images_dir}")
        if len(self.gt_files) == 0:
            raise ValueError(f"No GT files found in {self.gt_dir}")
        
        assert len(self.image_files) == len(self.gt_files), \
            f"Nombre d'images ({len(self.image_files)}) et de GT ({len(self.gt_files)}) différent"
        
        # Transformations d'augmentation uniquement pour l'entraînement
        # Pas d'augmentation pour la validation
        if mode == 'train':
            self.transform_image = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.transform_gt = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.transform_gt = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def __len__(self):
        # Si on fait du patch cropping, on a 4 patches par image
        multiplier = 4 if self.crop_patches else 1
        return len(self.image_files) * multiplier
    
    def __getitem__(self, idx):
        # Index de l'image de base
        base_idx = idx % len(self.image_files)
        patch_idx = idx // len(self.image_files)
        
        # Charger l'image complète
        image = Image.open(self.image_files[base_idx]).convert('RGB')
        gt = Image.open(self.gt_files[base_idx]).convert('L')
        
        # Cropping aléatoire ou centré
        if self.crop_patches:
            image, gt = self._random_crop(image, gt, patch_idx)
        
        # Redimensionner
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        gt = gt.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        
        # Créer la carte binaire RGB (sur le patch réduit, pas sur l'image complète)
        gt_rgb = self._create_binary_map(gt)
        
        # Augmentations aléatoires uniquement pour l'entraînement
        if self.mode == 'train' and random.random() > 0.5:
            # Flip horizontal
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                gt_rgb = gt_rgb.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Rotation légère
            if random.random() > 0.7:
                angle = random.uniform(-10, 10)
                image = image.rotate(angle, fillcolor=(0, 0, 0))
                gt_rgb = gt_rgb.rotate(angle, fillcolor=(0, 0, 0))
        
        # Appliquer les transformations
        image = self.transform_image(image)
        gt_rgb = self.transform_gt(gt_rgb)
        
        return gt_rgb, image  # Input: carte, Target: satellite
    
    def _create_binary_map(self, gt):
        """Crée une carte binaire RGB à partir du masque."""
        # Bâtiments en blanc, reste en noir
        binary = (np.array(gt) > 127).astype(np.uint8) * 255
        return Image.fromarray(np.stack([binary] * 3, axis=-1))
    
    def _random_crop(self, image, gt_rgb, patch_idx):
        """Crop aléatoire pour augmenter la diversité."""
        w, h = image.size
        crop_size = min(w, h) // 2  # Crop de la moitié de la taille
        
        if patch_idx == 0:  # Coin supérieur gauche
            x, y = 0, 0
        elif patch_idx == 1:  # Coin supérieur droit
            x, y = w - crop_size, 0
        elif patch_idx == 2:  # Coin inférieur gauche
            x, y = 0, h - crop_size
        elif self.mode == 'train':  # Aléatoire
            x = random.randint(0, max(0, w - crop_size))
            y = random.randint(0, max(0, h - crop_size))
        else:  # Coin inférieur droit (patches déterministes hors entraînement)
            x, y = w - crop_size, h - crop_size
        
        # Crop
        image = image.crop((x, y, x + crop_size, y + crop_size))
        gt_rgb = gt_rgb.crop((x, y, x + crop_size, y + crop_size))
        
        return image, gt_rgb


def calculate_psnr(img1, img2, max_val=1.0):
    """Calcul du PSNR entre deux images."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def calculate_ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """Calcul du SSIM entre deux images."""
    # Créer une fenêtre gaussienne
    def gaussian_window(size, sigma=1.5):
        gauss = torch.Tensor([np.exp(-(x - size//2)**2/float(2*sigma**2)) for x in range(size)])
        return gauss/gauss.sum()
    
    def create_window(window_size, channel):
        _1D_window = gaussian_window(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    # Moyennes
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Variances et covariance
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # SSIM
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class ImprovedTrainer:
    """Trainer amélioré avec de meilleures métriques et monitoring."""
    
    def __init__(self, generator, discriminator, device, lr_g=0.0002, lr_d=0.0002, 
                 lambda_l1=100, use_wandb=False):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.lambda_l1 = lambda_l1
        self.use_wandb = use_wandb
        
        # Optimizers séparés avec différents learning rates
        self.optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        # Schedulers pour réduire le learning rate
        self.scheduler_G = StepLR(self.optimizer_G, step_size=50, gamma=0.5)
        self.scheduler_D = StepLR(self.optimizer_D, step_size=50, gamma=0.5)
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()  # LSGan au lieu de BCE
        self.criterion_L1 = nn.L1Loss()
    
    def train_step(self, real_A, real_B):
        batch_size = real_A.size(0)
        
        # Test forward pass pour déterminer la taille de sortie
        with torch.no_grad():
            test_out = self.discriminator(real_A[:1], real_B[:1])
            out_h, out_w = test_out.shape[2], test_out.shape[3]
        
        # Labels pour LSGAN (plus stable) - taille adaptative
        real_label = torch.ones(batch_size, 1, out_h, out_w).to(self.device)
        fake_label = torch.zeros(batch_size, 1, out_h, out_w).to(self.device)
        
        # ================== Train Generator ==================
        self.optimizer_G.zero_grad()
        
        fake_B = self.generator(real_A)
        pred_fake = self.discriminator(real_A, fake_B)
        
        # Losses du générateur
        loss_G_GAN = self.criterion_GAN(pred_fake, real_label)
        loss_G_L1 = self.criterion_L1(fake_B, real_B) * self.lambda_l1
        
        # Loss perceptuelle (optionnelle)
        loss_G = loss_G_GAN + loss_G_L1
        
        loss_G.backward()
        self.optimizer_G.step()
        
        # ================== Train Discriminator ==================
        self.optimizer_D.zero_grad()
        
        # Real pairs
        pred_real = self.discriminator(real_A, real_B)
        loss_D_real = self.criterion_GAN(pred_real, real_label)
        
        # Fake pairs (détacher du graph du générateur)
        pred_fake = self.discriminator(real_A, fake_B.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, fake_label)
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.optimizer_D.step()
        
        return {
            'loss_G': loss_G.item(),
            'loss_G_GAN': loss_G_GAN.item(),
            'loss_G_L1': loss_G_L1.item(),
            'loss_D': loss_D.item(),
            'loss_D_real': loss_D_real.item(),
            'loss_D_fake': loss_D_fake.item()
        }
    
    def step_schedulers(self):
        """Met à jour les learning rates."""
        self.scheduler_G.step()
        self.scheduler_D.step()
    
    def save_sample_images(self, real_A, real_B, epoch, save_dir="samples"):
        """Sauvegarde des échantillons pour monitoring."""
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            fake_B = self.generator(real_A[:4])  # Prendre 4 échantillons
            
            # Dénormaliser
            real_A_denorm = (real_A[:4] + 1) / 2.0
            real_B_denorm = (real_B[:4] + 1) / 2.0
            fake_B_denorm = (fake_B + 1) / 2.0
            
            # Créer une grille
            comparison = torch.cat([real_A_denorm, fake_B_denorm, real_B_denorm], dim=0)
            vutils.save_image(comparison, f"{save_dir}/epoch_{epoch:03d}.png", 
                            nrow=4, padding=2, normalize=False)
    
    def validate(self, val_loader):
        """Valide le modèle et retourne les métriques."""
        self.generator.eval()
        self.discriminator.eval()
        
        total_loss_g = 0
        total_loss_d = 0
        total_psnr = 0
        total_ssim = 0
        num_batches = 0
        
        with torch.no_grad():
            for real_A, real_B in val_loader:
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)
                batch_size = real_A.size(0)
                
                # Générer les images
                fake_B = self.generator(real_A)
                
                # Calculer les losses
                pred_fake = self.discriminator(real_A, fake_B)
                pred_real = self.discriminator(real_A, real_B)
                
                # Taille de sortie du discriminateur
                out_h, out_w = pred_fake.shape[2], pred_fake.shape[3]
                real_label = torch.ones(batch_size, 1, out_h, out_w).to(self.device)
                fake_label = torch.zeros(batch_size, 1, out_h, out_w).to(self.device)
                
                loss_g_gan = self.criterion_GAN(pred_fake, real_label)
                loss_g_l1 = self.criterion_L1(fake_B, real_B) * self.lambda_l1
                loss_g = loss_g_gan + loss_g_l1
                
                loss_d_real = self.criterion_GAN(pred_real, real_label)
                loss_d_fake = self.criterion_GAN(pred_fake, fake_label)
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                
                # Calculer PSNR et SSIM
                # Dénormaliser pour les métriques
                fake_B_denorm = (fake_B + 1) / 2.0
                real_B_denorm = (real_B + 1) / 2.0
                
                psnr = calculate_psnr(fake_B_denorm, real_B_denorm)
                ssim = calculate_ssim(fake_B_denorm, real_B_denorm)
                
                total_loss_g += loss_g.item()
                total_loss_d += loss_d.item()
                total_psnr += psnr.item()
                total_ssim += ssim.item()
                num_batches += 1
        
        self.generator.train()
        self.discriminator.train()
        
        return {
            'val_loss_G': total_loss_g / num_batches,
            'val_loss_D': total_loss_d / num_batches,
            'val_psnr': total_psnr / num_batches,
            'val_ssim': total_ssim / num_batches
        }


def main():
    parser = argparse.ArgumentParser(description='Entraînement amélioré Pix2Pix')
    # Chemin par défaut relatif au script
    script_dir = Path(__file__).parent.parent.parent  # Remonte jusqu'à la racine du projet
    default_data_dir = script_dir / 'dataset'
    
    parser.add_argument('--data_dir', type=str, default=str(default_data_dir),
                       help='Chemin vers le dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Taille de batch (réduit si mémoire limitée)')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Nombre d\'époques')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Taille des images (512 recommandé)')
    parser.add_argument('--lr_g', type=float, default=0.0002,
                       help='Learning rate générateur')
    parser.add_argument('--lr_d', type=float, default=0.0002,
                       help='Learning rate discriminateur')
    parser.add_argument('--lambda_l1', type=float, default=100,
                       help='Poids de la loss L1')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Utiliser Weights & Biases pour le monitoring')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint à charger')
    parser.add_argument('--save_every', type=int, default=25,
                       help='Sauvegarder tous les N epochs')
    parser.add_argument('--patience', type=int, default=20,
                       help='Patience pour early stopping')
    parser.add_argument('--val_every', type=int, default=5,
                       help='Valider tous les N epochs')
    
    args = parser.parse_args()
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    
    # Initialiser wandb si demandé
    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project='pix2pix-satellite', config=args)
    elif args.use_wandb:
        print("Wandb non disponible, monitoring désactivé")
    
    # Dataset amélioré
    images_path = Path(args.data_dir) / 'train' / 'images'
    gt_path = Path(args.data_dir) / 'train' / 'gt'
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Data directory: {args.data_dir}")
    print(f"Images path: {images_path} (exists: {images_path.exists()})")
    print(f"GT path: {gt_path} (exists: {gt_path.exists()})")
    
    if images_path.exists():
        image_files = list(images_path.glob('*.tif'))
        print(f"Found {len(image_files)} image files")
    
    if gt_path.exists():
        gt_files = list(gt_path.glob('*.tif'))
        print(f"Found {len(gt_files)} GT files")
    
    # Dataset d'entraînement
    train_dataset = ImprovedInriaDataset(
        images_path,
        gt_path,
        image_size=args.image_size,
        mode='train',
        crop_patches=True,
        val_split=0.2  # 80% train, 20% validation
    )
    
    # Dataset de validation
    val_dataset = ImprovedInriaDataset(
        images_path,
        gt_path,
        image_size=args.image_size,
        mode='val',
        crop_patches=True,  # Même échelle que l'entraînement (4 patches fixes par image)
        val_split=0.2
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train dataset size: {len(train_dataset)} échantillons")
    print(f"Val dataset size: {len(val_dataset)} échantillons")
    print(f"Batches par epoch (train): {len(train_loader)}")
    print(f"Batches par epoch (val): {len(val_loader)}")
    
    # Modèles améliorés
    generator = ImprovedGenerator(image_size=args.image_size)
    discriminator = ImprovedDiscriminator(image_size=args.image_size)
    
    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Trainer
    trainer = ImprovedTrainer(
        generator, discriminator, device,
        lr_g=args.lr_g, lr_d=args.lr_d,
        lambda_l1=args.lambda_l1,
        use_wandb=use_wandb
    )
    
    # Liste pour stocker les losses par époque
    epoch_losses_history = []
    
    # Pour early stopping et best model
    best_val_loss = float('inf')
    best_val_psnr = 0
    best_val_ssim = 0
    best_epoch = 0
    epochs_no_improve = 0
    
    # Charger checkpoint si spécifié
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        trainer.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        trainer.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch']
        
        if 'scheduler_G_state_dict' in checkpoint:
            trainer.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            trainer.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        else:
            # Ancien checkpoint, sauvé avant le step de l'époque: le LR courant vient
            # de l'optimizer, on recale le compteur puis on rejoue ce step
            trainer.scheduler_G.last_epoch = start_epoch - 1
            trainer.scheduler_D.last_epoch = start_epoch - 1
            trainer.step_schedulers()
        
        epoch_losses_history = checkpoint.get('history', [])
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        best_val_psnr = checkpoint.get('best_val_psnr', best_val_psnr)
        best_val_ssim = checkpoint.get('best_val_ssim', best_val_ssim)
        best_epoch = checkpoint.get('best_epoch', best_epoch)
        epochs_no_improve = checkpoint.get('epochs_no_improve', epochs_no_improve)
        print(f"Checkpoint chargé: epoch {start_epoch}")
    
    def make_checkpoint(epoch):
        """État complet nécessaire pour reprendre l'entraînement."""
        return {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': trainer.optimizer_G.state_dict(),
            'optimizer_D_state_dict': trainer.optimizer_D.state_dict(),
            'scheduler_G_state_dict': trainer.scheduler_G.state_dict(),
            'scheduler_D_state_dict': trainer.scheduler_D.state_dict(),
            'history': epoch_losses_history,
            'best_val_loss': best_val_loss,
            'best_val_psnr': best_val_psnr,
            'best_val_ssim': best_val_ssim,
            'best_epoch': best_epoch,
            'epochs_no_improve': epochs_no_improve
        }
    
    # Entraînement
    print("Début de l'entraînement amélioré...")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    for epoch in range(start_epoch, args.num_epochs):
        epoch_losses = {
            'loss_G': 0, 'loss_G_GAN': 0, 'loss_G_L1': 0,
            'loss_D': 0, 'loss_D_real': 0, 'loss_D_fake': 0
        }
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for i, (real_A, real_B) in enumerate(pbar):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            losses = trainer.train_step(real_A, real_B)
            
            # Accumuler les losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            
            # Mise à jour de la barre de progression
            pbar.set_postfix({
                'G': f"{losses['loss_G']:.3f}",
                'D': f"{losses['loss_D']:.3f}",
                'L1': f"{losses['loss_G_L1']:.1f}"
            })
            
            # Log vers wandb
            if use_wandb and i % 50 == 0:
                wandb.log({
                    'batch_loss_G': losses['loss_G'],
                    'batch_loss_D': losses['loss_D'],
                    'batch_loss_L1': losses['loss_G_L1']
                })
        
        # Statistiques de l'époque
        num_batches = len(train_loader)
        avg_losses = {k: v/num_batches for k, v in epoch_losses.items()}
        
        # Validation tous les N epochs
        val_metrics = None
        save_best = False
        early_stop = False
        if (epoch + 1) % args.val_every == 0:
            print("\nValidation en cours...")
            val_metrics = trainer.validate(val_loader)
            print(f"  Val Loss G: {val_metrics['val_loss_G']:.4f}")
            print(f"  Val Loss D: {val_metrics['val_loss_D']:.4f}")
            print(f"  Val PSNR: {val_metrics['val_psnr']:.2f} dB")
            print(f"  Val SSIM: {val_metrics['val_ssim']:.4f}")
            
            # Vérifier si c'est le meilleur modèle
            if val_metrics['val_psnr'] > best_val_psnr:
                best_val_psnr = val_metrics['val_psnr']
                best_val_ssim = val_metrics['val_ssim']
                best_val_loss = val_metrics['val_loss_G']
                best_epoch = epoch + 1
                epochs_no_improve = 0
                
                save_best = True
            else:
                epochs_no_improve += 1
                
            # Early stopping (la boucle s'arrête en fin d'époque, après les sauvegardes)
            if epochs_no_improve >= args.patience:
                print(f"\n\u26a0️  Early stopping: pas d'amélioration depuis {args.patience} validations")
                print(f"Meilleur modèle: epoch {best_epoch} (PSNR: {best_val_psnr:.2f}, SSIM: {best_val_ssim:.4f})")
                early_stop = True
        
        # Sauvegarder les losses de cette époque
        epoch_loss_data = {
            'epoch': epoch + 1,
            'loss_G': avg_losses['loss_G'],
            'loss_D': avg_losses['loss_D'],
            'loss_G_GAN': avg_losses['loss_G_GAN'],
            'loss_G_L1': avg_losses['loss_G_L1'],
            'loss_D_real': avg_losses['loss_D_real'],
            'loss_D_fake': avg_losses['loss_D_fake']
        }
        
        # Ajouter les métriques de validation si disponibles
        if val_metrics:
            epoch_loss_data.update(val_metrics)
            
        epoch_losses_history.append(epoch_loss_data)
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs}:")
        print(f"  Generator Loss: {avg_losses['loss_G']:.4f}")
        print(f"  Discriminator Loss: {avg_losses['loss_D']:.4f}")
        print(f"  L1 Loss: {avg_losses['loss_G_L1']:.4f}")
        print(f"  LR G: {trainer.scheduler_G.get_last_lr()[0]:.6f}")
        
        if best_epoch > 0:
            print(f"  Best model: epoch {best_epoch} (PSNR: {best_val_psnr:.2f} dB, SSIM: {best_val_ssim:.4f})")
        
        # Sauvegarder des échantillons
        if (epoch + 1) % 5 == 0:
            trainer.save_sample_images(real_A, real_B, epoch + 1)
        
        # Mettre à jour les learning rates
        trainer.step_schedulers()
        
        # Sauvegarder le meilleur modèle
        if save_best:
            best_checkpoint = make_checkpoint(epoch + 1)
            best_checkpoint.update({
                'val_psnr': best_val_psnr,
                'val_ssim': best_val_ssim,
                'val_loss': best_val_loss
            })
            torch.save(best_checkpoint, 'checkpoints/best_model.pth')
            print(f"\u2705 Nouveau meilleur modèle sauvegardé (PSNR: {best_val_psnr:.2f})")
        
        # Sauvegarder checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint = make_checkpoint(epoch + 1)
            checkpoint['losses'] = avg_losses
            torch.save(checkpoint, f'checkpoints/improved_checkpoint_epoch_{epoch+1}.pth')
            print(f"Checkpoint sauvé: improved_checkpoint_epoch_{epoch+1}.pth")
        
        # Log vers wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'epoch_loss_G': avg_losses['loss_G'],
                'epoch_loss_D': avg_losses['loss_D'],
                'epoch_loss_L1': avg_losses['loss_G_L1'],
                'lr_G': trainer.scheduler_G.get_last_lr()[0],
                'lr_D': trainer.scheduler_D.get_last_lr()[0]
            })
        
        # Sauvegarder les losses dans un fichier JSON (toutes les 10 époques)
        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1 or early_stop:
            with open('logs/training_losses.json', 'w') as f:
                json.dump(epoch_losses_history, f, indent=2)
            print("Losses sauvegardées dans logs/training_losses.json")
        
        if early_stop:
            break
    
    print("\nEntraînement terminé!")
    if best_epoch > 0:
        print(f"\u2705 Meilleur modèle: epoch {best_epoch}")
        print(f"   - PSNR: {best_val_psnr:.2f} dB")
        print(f"   - SSIM: {best_val_ssim:.4f}")
        print(f"   - Val Loss: {best_val_loss:.4f}")
        print("   Sauvegardé dans: checkpoints/best_model.pth")
    
    # Générer les graphiques de losses à la fin de l'entraînement
    print("Génération des graphiques de losses...")
    try:
        from core.training.loss_plotter import LossPlotter
        plotter = LossPlotter('logs/training_losses.json')
        if plotter.load_losses():
            plotter.plot_main_losses('logs/final_losses_evolution.png')
            plotter.plot_detailed_discriminator('logs/final_discriminator_details.png')
            print("✅ Graphiques de losses générés dans logs/")
        else:
            print("⚠️  Impossible de charger les données de losses")
    except Exception as e:
        print(f"⚠️  Erreur lors de la génération des graphiques: {e}")
        print("Vous pouvez les générer manuellement avec: python -m core.training.loss_plotter")

if __name__ == '__main__':
    main()