"""Variational Autoencoder model for MNIST."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 20, channels: List[int] = [32, 64, 128]):
        super().__init__()
        self.latent_dim = latent_dim
        
        in_channels = 1
        layers = []
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        dummy_input = torch.zeros(1, 1, 28, 28)
        dummy_output = self.conv_layers(dummy_input)
        self.flatten_dim = dummy_output.view(1, -1).size(1)
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 20, channels: List[int] = [128, 64, 32]):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        
        self.fc = nn.Linear(latent_dim, channels[0] * 7 * 7)
        
        layers = []
        layers.extend([
            nn.ConvTranspose2d(channels[0], channels[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[1], channels[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        ])
        
        self.deconv_layers = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), self.channels[0], 7, 7)
        x = self.deconv_layers(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20, 
                 encoder_channels: List[int] = [32, 64, 128],
                 decoder_channels: List[int] = [128, 64, 32]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, encoder_channels)
        self.decoder = Decoder(latent_dim, decoder_channels)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples