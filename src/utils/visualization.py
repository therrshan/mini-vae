"""Visualization utilities for VAE."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from typing import List, Optional, Tuple
import os

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3:
        tensor = tensor[0]
    
    img = tensor.numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def save_reconstruction_grid(original: torch.Tensor, 
                           reconstructed: torch.Tensor,
                           save_path: str,
                           num_images: int = 8):
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i in range(num_images):
        axes[0, i].imshow(tensor_to_image(original[i]), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Original', fontsize=10)
        
        axes[1, i].imshow(tensor_to_image(reconstructed[i]), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_interpolation_gif(images: torch.Tensor,
                           save_path: str,
                           fps: int = 10,
                           loop: bool = True):
    frames = []
    
    for i in range(images.size(0)):
        img = tensor_to_image(images[i])
        img_pil = Image.fromarray(img, mode='L')
        img_pil = img_pil.resize((128, 128), Image.Resampling.NEAREST)
        frames.append(np.array(img_pil))
    
    if loop:
        frames = frames + frames[-2:0:-1]
    
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Saved GIF to {save_path}")

def create_multi_interpolation_gif(paths: List[torch.Tensor],
                                 save_path: str,
                                 fps: int = 10,
                                 grid_size: Tuple[int, int] = (2, 2)):
    num_frames = paths[0].size(0)
    num_paths = len(paths)
    rows, cols = grid_size
    
    if num_paths > rows * cols:
        raise ValueError(f"Grid size {grid_size} too small for {num_paths} paths")
    
    frames = []
    
    for frame_idx in range(num_frames):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten() if rows * cols > 1 else [axes]
        
        for path_idx in range(num_paths):
            img = tensor_to_image(paths[path_idx][frame_idx])
            axes[path_idx].imshow(img, cmap='gray')
            axes[path_idx].axis('off')
        
        for idx in range(num_paths, rows * cols):
            axes[idx].axis('off')
        
        plt.tight_layout(pad=0.1)
        
        temp_path = f'/tmp/frame_{frame_idx:04d}.png'
        plt.savefig(temp_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        frame = np.array(Image.open(temp_path))
        frames.append(frame)
        os.remove(temp_path)
    
    frames = frames + frames[-2:0:-1]
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Saved multi-interpolation GIF to {save_path}")

def plot_latent_space(model: torch.nn.Module,
                     data_loader: torch.utils.data.DataLoader,
                     save_path: str,
                     device: torch.device = torch.device('cpu'),
                     num_batches: int = 10):
    if model.latent_dim != 2:
        print(f"Latent space visualization only works for 2D latent space, got {model.latent_dim}D")
        return
    
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for i, (images, _, label) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            images = images.to(device)
            mu, _ = model.encoder(images)
            latents.append(mu.cpu())
            labels.append(label)
    
    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space Visualization')
    plt.grid(True, alpha=0.3)
    
    for i in range(10):
        mask = labels == i
        if mask.any():
            center = latents[mask].mean(axis=0)
            plt.text(center[0], center[1], str(i), fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved latent space plot to {save_path}")

def create_latent_traversal(model: torch.nn.Module,
                          base_image: torch.Tensor,
                          dimension: int,
                          range_values: Tuple[float, float] = (-3, 3),
                          num_steps: int = 11,
                          save_path: str = None,
                          device: torch.device = torch.device('cpu')):
    model.eval()
    base_image = base_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        base_z = model.encode(base_image)
        
        values = torch.linspace(range_values[0], range_values[1], num_steps)
        images = []
        
        for val in values:
            z = base_z.clone()
            z[0, dimension] = val
            img = model.decode(z)
            images.append(img)
        
        images = torch.cat(images, dim=0)
    
    if save_path:
        fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))
        for i in range(num_steps):
            axes[i].imshow(tensor_to_image(images[i]), cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'{values[i]:.1f}', fontsize=10)
        
        plt.suptitle(f'Latent Dimension {dimension} Traversal', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return images