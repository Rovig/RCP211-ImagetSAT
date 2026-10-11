# RCP211-ImagetSAT

Génération d'images satellites à partir de cartes binaires de bâtiments, avec un
GAN conditionnel de type Pix2Pix (générateur U-Net, discriminateur PatchGAN)
entraîné sur le dataset Inria Aerial Image Labeling.

## Installation

Python 3.10 ou plus récent.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

`pip install -e .` installe les dépendances et rend les paquets `core` et `utils`
importables depuis n'importe quel répertoire. Pour reproduire
exactement l'environnement de test, installer d'abord les versions figées :

```bash
pip install -r requirements.txt
pip install -e .
```

Le suivi Weights & Biases est optionnel : `pip install -e ".[wandb]"`.

Sans installation, les commandes ci-dessous fonctionnent aussi tant qu'elles sont
lancées depuis la racine du projet.

## Données

Les tuiles et leurs masques doivent porter le même nom de fichier :

```
dataset/
└── train/
    ├── images/   # tuiles satellites (*.tif)
    └── gt/       # masques de bâtiments (*.tif)
```

20 % des tuiles sont réservées à la validation. Chaque tuile fournit 4 patches
(la moitié de la tuile en largeur et en hauteur), redimensionnés à `--image_size`.

## Entraînement

```bash
python -m core.training.improved_training --image_size 256 --batch_size 16
```

| Option | Défaut | Rôle |
|---|---|---|
| `--data_dir` | `dataset/` | Répertoire du dataset |
| `--image_size` | 512 | Taille des images ; choisit aussi l'architecture (256 ou 512) |
| `--batch_size` | 16 | Taille de batch |
| `--num_epochs` | 200 | Nombre d'époques |
| `--lr_g`, `--lr_d` | 0.0002 | Learning rates du générateur et du discriminateur |
| `--lambda_l1` | 100 | Poids de la loss L1 |
| `--val_every` | 5 | Valider toutes les N époques |
| `--patience` | 20 | Early stopping après N validations sans amélioration du PSNR |
| `--num_workers` | 4 | Nombre de workers pour le chargement des données |
| `--save_every` | 25 | Sauvegarder un checkpoint toutes les N époques |
| `--checkpoint` | — | Reprendre l'entraînement depuis un checkpoint |
| `--use_wandb` | — | Activer le suivi Weights & Biases |

Le calcul utilise le GPU s'il y en a un (CUDA, ou MPS sur Mac Apple Silicon),
sinon le CPU.

Les sorties sont écrites dans le répertoire courant :

- `checkpoints/best_model.pth` : meilleur modèle selon le PSNR de validation
- `checkpoints/improved_checkpoint_epoch_N.pth` : checkpoints périodiques
- `samples/epoch_NNN.png` : carte / image générée / image réelle
- `logs/training_losses.json` et les graphiques de losses

Un checkpoint pèse plusieurs centaines de Mo.

Pour reprendre un entraînement, utiliser la même valeur de `--image_size` que
celle du checkpoint :

```bash
python -m core.training.improved_training --image_size 256 \
    --checkpoint checkpoints/improved_checkpoint_epoch_50.pth
```

Les graphiques de losses peuvent être régénérés avec :

```bash
python -m core.training.loss_plotter --log_file logs/training_losses.json --output_dir logs
```

## Génération

Une image, avec un style d'environnement (`urban`, `maritime`, `nature`, `default`) :

```bash
python -m core.generation.generate_image \
    --checkpoint checkpoints/best_model.pth \
    --input_map input/low_complexity_shematic_map.png \
    --environment urban \
    --output generated_satellite.png
```

Options d'agrandissement : `--upscale 2 --upscale_method bicubic`.

Les quatre styles d'un coup, dans un répertoire :

```bash
python -m core.generation.generate_multiple_environments \
    --checkpoint checkpoints/best_model.pth \
    --input input/low_complexity_shematic_map.png \
    --output output
```

L'architecture (256 ou 512) est détectée à partir du checkpoint, et la carte
d'entrée est redimensionnée en conséquence. Un checkpoint introuvable ou
incompatible, ou une carte d'entrée introuvable, provoque une erreur. Sans
`--input_map`, `generate_image` génère une carte aléatoire.

## Tests

```bash
python run_tests.py
```

Un seul module : `python -m unittest tests.test_improved_training`.

## Structure

```
core/models/        Architectures (networks.py) et chargement des modèles
core/training/      Entraînement et graphiques de losses
core/generation/    Génération d'images et styles d'environnement
utils/              Logger et upscaler
tests/              Tests unitaires
input/              Cartes d'exemple
```
