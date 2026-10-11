#!/usr/bin/env python3
"""
Script pour générer des images satellites pour tous les types d'environnements
Version corrigée utilisant les modules disponibles
"""

import argparse
from pathlib import Path

from core.models.model_manager import OriginalGenerator, GeneratorLoader
from core.generation.environment_styles import EnvironmentStyler
from utils.device import get_device
import torch
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path: str, image_size: int = 256) -> torch.Tensor:
    """Prétraite l'image pour le modèle"""
    image = Image.open(image_path).convert('RGB')
    
    # Redimensionner
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    # Convertir en tensor et normaliser
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return transform(image).unsqueeze(0)

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convertit un tensor en image PIL"""
    # Dénormaliser
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convertir en PIL
    transform = transforms.ToPILImage()
    return transform(tensor.squeeze(0))

def main():
    parser = argparse.ArgumentParser(description='Générer des images satellites pour tous les environnements')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle entraîné')
    parser.add_argument('--input', type=str, required=True,
                       help='Chemin vers la carte binaire d\'entrée')
    parser.add_argument('--output', type=str, default='output',
                       help='Répertoire de sortie')
    parser.add_argument('--scale', type=int, default=1,
                       help='Facteur d\'agrandissement de l\'image de sortie')
    parser.add_argument('--method', type=str, default='lanczos',
                       choices=['lanczos', 'bilinear', 'bicubic', 'nearest'],
                       help='Méthode de redimensionnement')
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Charger le modèle
    device = get_device()
    print(f"Device: {device}")
    
    loader = GeneratorLoader(OriginalGenerator)
    generator = loader.load(args.checkpoint, device)
    generator.eval()
    
    # Prétraiter l'image d'entrée
    print(f"Traitement de l'image: {args.input}")
    input_tensor = preprocess_image(args.input, generator.image_size).to(device)
    
    # Générer l'image de base
    with torch.no_grad():
        generated = generator(input_tensor)
    
    base_image = tensor_to_image(generated)
    print(f"Taille de l'image générée: {base_image.size}")
    
    # Styler pour chaque environnement
    styler = EnvironmentStyler()
    environments = ['urban', 'maritime', 'nature', 'default']
    
    input_name = Path(args.input).stem
    
    for env in environments:
        print(f"Génération pour environnement: {env}")
        
        if env == 'default':
            styled_image = base_image
        else:
            styled_image = styler.apply_style(base_image, env)
        
        # Appliquer le redimensionnement si nécessaire
        if args.scale > 1:
            # Mapper les noms de méthodes
            method_map = {
                'lanczos': Image.Resampling.LANCZOS,
                'bilinear': Image.Resampling.BILINEAR,
                'bicubic': Image.Resampling.BICUBIC,
                'nearest': Image.Resampling.NEAREST
            }
            
            # Redimensionner
            current_size = styled_image.size
            new_size = (current_size[0] * args.scale, current_size[1] * args.scale)
            styled_image = styled_image.resize(new_size, method_map[args.method])
            print(f"   Redimensionné de {current_size} vers {new_size} avec méthode {args.method}")
        
        # Sauvegarder avec suffixe de méthode si redimensionné
        if args.scale > 1:
            output_path = output_dir / f"{input_name}_{env}_{args.method}_x{args.scale}.png"
        else:
            output_path = output_dir / f"{input_name}_{env}.png"
        styled_image.save(output_path, dpi=(150, 150))
        print(f"✅ Sauvegardé: {output_path}")
    
    print(f"\n🎉 Génération terminée ! Images sauvegardées dans {output_dir}")

if __name__ == '__main__':
    main()