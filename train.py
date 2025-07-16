"""Train VAE on MNIST dataset."""
import torch
import yaml
import argparse
import os
import random
import numpy as np
from torch.utils.data import DataLoader, random_split

from src.dataset.mnist_loader import get_mnist_dataloader, MNISTDataset
from src.models.vae import VAE
from src.training.trainer import VAETrainer

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_data_loaders(config: dict):
    full_dataset = MNISTDataset(data_dir=config['data']['data_dir'], train=True)
    
    train_size = int(config['data']['train_val_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='Train VAE on MNIST')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.device:
        config['device'] = args.device
    if args.resume:
        config['checkpoint']['resume'] = args.resume
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    set_seed(config['seed'])
    
    train_loader, val_loader = get_data_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    model = VAE(
        latent_dim=config['model']['latent_dim'],
        encoder_channels=config['model']['encoder_channels'],
        decoder_channels=config['model']['decoder_channels']
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    trainer.train()

if __name__ == "__main__":
    main()