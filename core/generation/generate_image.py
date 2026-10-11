#!/usr/bin/env python3
"""
Script pour générer des images satellites à partir de cartes binaires
Version refactorisée avec principes SOLID et KISS
"""

import argparse
from pathlib import Path

from core.generation.satellite_generator import SimpleSatelliteGenerator
from utils.unified_upscaler import SimpleImageUpscaler
from utils.device import get_device
from utils.logger import get_logger

logger = get_logger(__name__)

def main_simple():
    """Version simplifiée avec KISS principle"""
    parser = argparse.ArgumentParser(description='Générer des images satellites - Version KISS')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Chemin vers le checkpoint du modèle entraîné')
    parser.add_argument('--input_map', type=str, default=None,
                       help='Chemin vers la carte binaire d\'entrée')
    parser.add_argument('--output', type=str, default='generated_satellite.png',
                       help='Nom du fichier de sortie')
    parser.add_argument('--environment', type=str, default='urban',
                       choices=['urban', 'maritime', 'nature', 'default'],
                       help='Type d\'environnement')
    parser.add_argument('--upscale', type=int, default=None,
                       help='Facteur d\'agrandissement (2, 3, 4, etc.)')
    parser.add_argument('--upscale_method', type=str, default='bilinear',
                       choices=['bilinear', 'bicubic', 'nearest', 'lanczos', 'torch'],
                       help='Méthode d\'upscaling')
    
    args = parser.parse_args()
    
    logger.info(f"Génération d'image satellite (environnement: {args.environment})")
    logger.info(f"Device: {get_device()}")
    
    # Utilisation simple en une ligne
    generator = SimpleSatelliteGenerator(args.checkpoint)
    output_file = generator.generate(args.input_map, args.environment, args.output)
    
    logger.info(f"Génération terminée! Fichier: {output_file}")
    
    # Upscaling optionnel
    if args.upscale:
        logger.info(f"Upscaling avec facteur {args.upscale} (méthode: {args.upscale_method})")
        upscaler = SimpleImageUpscaler()
        
        # Création du nom de fichier upscalé
        output_path = Path(output_file)
        upscaled_output = str(output_path.with_name(
            f"{output_path.stem}_upscaled_{args.upscale}x{output_path.suffix}"
        ))
        
        upscaled_file = upscaler.upscale_image(
            output_file, 
            upscaled_output, 
            args.upscale, 
            args.upscale_method
        )
        
        if upscaled_file:
            logger.info(f"✅ Upscaling terminé! Fichier: {upscaled_file}")
        else:
            logger.error("❌ Échec de l'upscaling")

if __name__ == '__main__':
    main_simple()