"""VAE training logic."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import json

from src.models.vae import VAE
from src.training.losses import VAELoss, get_beta_schedule

class VAETrainer:
    def __init__(self, 
                 model: VAE,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: torch.device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.loss_fn = VAELoss(
            beta=config['training']['beta'],
            loss_type=config['training']['reconstruction_loss']
        )
        
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        self.beta_schedule = get_beta_schedule(
            config['training']['beta_schedule'],
            config['training']['num_epochs'],
            config['training']['beta_start'],
            config['training']['beta_end'],
            config['training']['beta_warmup_epochs']
        )
        
        self.writer = None
        if config['logging']['tensorboard']:
            self.writer = SummaryWriter(config['logging']['log_dir'])
        
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
        os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
        os.makedirs(config['visualization']['output_dir'], exist_ok=True)
    
    def _get_optimizer(self) -> optim.Optimizer:
        opt_config = self.config['optimizer']
        if opt_config['type'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                betas=opt_config['betas'],
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
    
    def _get_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        sched_config = self.config['scheduler']
        if sched_config['type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=sched_config['min_lr']
            )
        elif sched_config['type'] == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler type: {sched_config['type']}")
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        self.loss_fn.update_beta(self.beta_schedule[self.current_epoch])
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        for batch_idx, (data, _, _) in enumerate(pbar):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            
            loss, recon_loss, kl_loss = self.loss_fn(recon_batch, data, mu, logvar)
            
            loss.backward()
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'beta': f'{self.loss_fn.current_beta:.3f}'
            })
            
            if self.writer and batch_idx % self.config['logging']['log_interval'] == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                self.writer.add_scalar('train/recon_loss', recon_loss.item(), global_step)
                self.writer.add_scalar('train/kl_loss', kl_loss.item(), global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_recon = total_recon_loss / len(self.train_loader)
        avg_kl = total_kl_loss / len(self.train_loader)
        
        return {'loss': avg_loss, 'recon_loss': avg_recon, 'kl_loss': avg_kl}
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        with torch.no_grad():
            for data, _, _ in self.val_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                
                loss, recon_loss, kl_loss = self.loss_fn(recon_batch, data, mu, logvar)
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_recon = total_recon_loss / len(self.val_loader)
        avg_kl = total_kl_loss / len(self.val_loader)
        
        return {'loss': avg_loss, 'recon_loss': avg_recon, 'kl_loss': avg_kl}
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        filepath = os.path.join(self.config['checkpoint']['save_dir'], filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.config['checkpoint']['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
    
    def train(self):
        if self.config['checkpoint']['resume']:
            self.load_checkpoint(self.config['checkpoint']['resume'])
            print(f"Resumed from epoch {self.current_epoch}")
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            if self.scheduler:
                self.scheduler.step()
            
            print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Recon: {train_metrics['recon_loss']:.4f}, "
                  f"KL: {train_metrics['kl_loss']:.4f}")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Recon: {val_metrics['recon_loss']:.4f}, "
                  f"KL: {val_metrics['kl_loss']:.4f}")
            
            if self.writer:
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('val/recon_loss', val_metrics['recon_loss'], epoch)
                self.writer.add_scalar('val/kl_loss', val_metrics['kl_loss'], epoch)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('train/beta', self.loss_fn.current_beta, epoch)
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                if self.config['checkpoint']['save_best']:
                    self.save_checkpoint('checkpoint.pth', is_best=True)
                    print(f"Saved best model with val loss: {self.best_val_loss:.4f}")
            
            if (epoch + 1) % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        if self.config['checkpoint']['save_last']:
            self.save_checkpoint('final_model.pth')
        
        if self.writer:
            self.writer.close()
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")