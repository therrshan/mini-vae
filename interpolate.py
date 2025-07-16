"""Generate interpolation GIFs from trained VAE model."""
import torch
import yaml
import argparse
import os
from tqdm import tqdm

from src.data.mnist_loader import MNISTDataset, get_interpolation_pairs
from src.models.vae import VAE
from src.utils.interpolation import create_interpolation_path, create_grid_interpolation
from src.utils.visualization import create_interpolation_gif, create_multi_interpolation_gif, save_reconstruction_grid

def load_model(checkpoint_path: str, config: dict, device: torch.device):
    model = VAE(
        latent_dim=config['model']['latent_dim'],
        encoder_channels=config['model']['encoder_channels'],
        decoder_channels=config['model']['decoder_channels']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['best_val_loss']:.4f}")
    return model

def generate_single_interpolation(model, dataset, start_digit, end_digit, num_steps, save_dir, device):
    pairs = get_interpolation_pairs(dataset, start_digit, end_digit, num_pairs=1)
    if not pairs:
        print(f"Could not find digits {start_digit} and {end_digit}")
        return
    
    start_img, end_img = pairs[0]
    
    interpolated = create_interpolation_path(
        model, start_img, end_img, 
        num_steps=num_steps, 
        method='linear',
        device=device
    )
    
    save_path = os.path.join(save_dir, f'interpolation_{start_digit}_to_{end_digit}.gif')
    create_interpolation_gif(interpolated, save_path, fps=10)

def generate_all_interpolations(model, dataset, config, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    
    digit_pairs = config['interpolation']['digit_pairs']
    num_steps = config['interpolation']['num_steps']
    fps = config['interpolation']['fps']
    
    all_paths = []
    
    for start_digit, end_digit in tqdm(digit_pairs, desc="Generating interpolations"):
        pairs = get_interpolation_pairs(dataset, start_digit, end_digit, num_pairs=1)
        if not pairs:
            continue
        
        start_img, end_img = pairs[0]
        
        interpolated = create_interpolation_path(
            model, start_img, end_img,
            num_steps=num_steps,
            method='linear',
            device=device
        )
        
        save_path = os.path.join(save_dir, f'interpolation_{start_digit}_to_{end_digit}.gif')
        create_interpolation_gif(interpolated, save_path, fps=fps)
        
        all_paths.append(interpolated)
    
    if len(all_paths) >= 4:
        grid_size = (2, 2) if len(all_paths) >= 4 else (1, len(all_paths))
        multi_save_path = os.path.join(save_dir, 'multi_interpolation_grid.gif')
        create_multi_interpolation_gif(all_paths[:4], multi_save_path, fps=fps, grid_size=grid_size)

def generate_reconstructions(model, dataset, save_dir, device, num_samples=16):
    os.makedirs(save_dir, exist_ok=True)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    images, _, _ = next(iter(loader))
    images = images.to(device)
    
    with torch.no_grad():
        recon, _, _ = model(images)
    
    save_path = os.path.join(save_dir, 'reconstructions.png')
    save_reconstruction_grid(images, recon, save_path, num_images=min(8, num_samples))

def main():
    parser = argparse.ArgumentParser(description='Generate VAE interpolations')
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations',
                       help='Directory to save outputs')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--single', nargs=2, type=int, default=None,
                       help='Generate single interpolation between two digits')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device if args.device else config['device'])
    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    model = load_model(args.checkpoint, config, device)
    
    test_dataset = MNISTDataset(train=False)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating reconstructions...")
    generate_reconstructions(model, test_dataset, output_dir, device)
    
    if args.single:
        print(f"Generating single interpolation: {args.single[0]} → {args.single[1]}")
        generate_single_interpolation(
            model, test_dataset, 
            args.single[0], args.single[1],
            config['interpolation']['num_steps'],
            output_dir, device
        )
    else:
        print("Generating all interpolations...")
        generate_all_interpolations(model, test_dataset, config, output_dir, device)
    
    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()