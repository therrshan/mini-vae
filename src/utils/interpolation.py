"""Interpolation utilities for VAE."""
import torch
import numpy as np
from typing import List, Tuple, Optional

def linear_interpolate(start: torch.Tensor, end: torch.Tensor, num_steps: int) -> torch.Tensor:
    alphas = torch.linspace(0, 1, num_steps).to(start.device)
    interpolations = []
    
    for alpha in alphas:
        interpolation = (1 - alpha) * start + alpha * end
        interpolations.append(interpolation)
    
    return torch.stack(interpolations)

def spherical_interpolate(start: torch.Tensor, end: torch.Tensor, num_steps: int) -> torch.Tensor:
    start_norm = start / torch.norm(start, dim=-1, keepdim=True)
    end_norm = end / torch.norm(end, dim=-1, keepdim=True)
    
    omega = torch.acos(torch.clamp(torch.sum(start_norm * end_norm, dim=-1), -1, 1))
    alphas = torch.linspace(0, 1, num_steps).to(start.device)
    
    interpolations = []
    for alpha in alphas:
        if omega.abs() < 1e-6:
            interpolation = (1 - alpha) * start + alpha * end
        else:
            interpolation = (torch.sin((1 - alpha) * omega) / torch.sin(omega)).unsqueeze(-1) * start + \
                          (torch.sin(alpha * omega) / torch.sin(omega)).unsqueeze(-1) * end
        interpolations.append(interpolation)
    
    return torch.stack(interpolations)

def create_interpolation_path(model: torch.nn.Module,
                            start_image: torch.Tensor,
                            end_image: torch.Tensor,
                            num_steps: int = 60,
                            method: str = 'linear',
                            device: torch.device = torch.device('cpu')) -> torch.Tensor:
    model.eval()
    start_image = start_image.unsqueeze(0).to(device)
    end_image = end_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        start_z = model.encode(start_image)
        end_z = model.encode(end_image)
        
        if method == 'linear':
            z_path = linear_interpolate(start_z, end_z, num_steps)
        elif method == 'spherical':
            z_path = spherical_interpolate(start_z, end_z, num_steps)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        interpolated_images = []
        for z in z_path:
            img = model.decode(z)
            interpolated_images.append(img)
        
        return torch.cat(interpolated_images, dim=0)

def create_grid_interpolation(model: torch.nn.Module,
                            images: List[torch.Tensor],
                            num_steps: int = 30,
                            device: torch.device = torch.device('cpu')) -> List[torch.Tensor]:
    paths = []
    num_images = len(images)
    
    for i in range(num_images):
        next_idx = (i + 1) % num_images
        path = create_interpolation_path(
            model, images[i], images[next_idx], 
            num_steps=num_steps, device=device
        )
        paths.append(path)
    
    return paths

def create_random_walk(model: torch.nn.Module,
                      start_image: torch.Tensor,
                      num_steps: int = 100,
                      step_size: float = 0.1,
                      device: torch.device = torch.device('cpu')) -> torch.Tensor:
    model.eval()
    start_image = start_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        current_z = model.encode(start_image)
        images = []
        
        for _ in range(num_steps):
            noise = torch.randn_like(current_z) * step_size
            current_z = current_z + noise
            img = model.decode(current_z)
            images.append(img)
        
        return torch.cat(images, dim=0)