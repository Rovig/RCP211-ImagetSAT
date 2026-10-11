"""
Sélection du device PyTorch
"""

import torch


def get_device() -> torch.device:
    """Retourne le meilleur device disponible: CUDA, puis MPS (Apple Silicon), puis CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
