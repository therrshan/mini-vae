"""Loss functions for VAE training."""
import torch
import torch.nn.functional as F
from typing import Tuple

def reconstruction_loss(recon_x: torch.Tensor, x: torch.Tensor, loss_type: str = "bce") -> torch.Tensor:
    if loss_type == "bce":
        return F.binary_cross_entropy(recon_x, x, reduction='none').sum(dim=[1, 2, 3])
    elif loss_type == "mse":
        return F.mse_loss(recon_x, x, reduction='none').sum(dim=[1, 2, 3])
    else:
        raise ValueError(f"Unknown reconstruction loss type: {loss_type}")

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def vae_loss(recon_x: torch.Tensor, 
             x: torch.Tensor, 
             mu: torch.Tensor, 
             logvar: torch.Tensor,
             beta: float = 1.0,
             loss_type: str = "bce") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = reconstruction_loss(recon_x, x, loss_type)
    kl_loss = kl_divergence(mu, logvar)
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss.mean(), recon_loss.mean(), kl_loss.mean()

class VAELoss:
    def __init__(self, beta: float = 1.0, loss_type: str = "bce"):
        self.beta = beta
        self.loss_type = loss_type
        self.current_beta = beta
    
    def update_beta(self, beta: float):
        self.current_beta = beta
    
    def __call__(self, recon_x: torch.Tensor, 
                 x: torch.Tensor, 
                 mu: torch.Tensor, 
                 logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return vae_loss(recon_x, x, mu, logvar, self.current_beta, self.loss_type)

def get_beta_schedule(schedule_type: str, 
                     num_epochs: int,
                     beta_start: float = 0.0,
                     beta_end: float = 1.0,
                     warmup_epochs: int = 10) -> list:
    if schedule_type == "constant":
        return [beta_end] * num_epochs
    elif schedule_type == "linear":
        if warmup_epochs >= num_epochs:
            return [beta_end] * num_epochs
        warmup = torch.linspace(beta_start, beta_end, warmup_epochs).tolist()
        constant = [beta_end] * (num_epochs - warmup_epochs)
        return warmup + constant
    elif schedule_type == "cyclical":
        cycle_length = num_epochs // 4
        schedule = []
        for epoch in range(num_epochs):
            cycle_pos = (epoch % cycle_length) / cycle_length
            if epoch // cycle_length % 2 == 0:
                beta = beta_start + (beta_end - beta_start) * cycle_pos
            else:
                beta = beta_end - (beta_end - beta_start) * cycle_pos
            schedule.append(beta)
        return schedule
    else:
        raise ValueError(f"Unknown beta schedule type: {schedule_type}")