"""MNIST DataLoader for VAE with interpolation utilities."""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional, List
import os

class MNISTDataset(Dataset):
    def __init__(self, data_dir: str = "data", train: bool = True, transform=None):
        self.data_dir = data_dir
        self.train = train
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        
        self.dataset = datasets.MNIST(
            root=self.data_dir,
            train=self.train,
            download=True,
            transform=self.transform
        )
        
        self.digit_to_text = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
            5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
        }
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        text = self.digit_to_text[label]
        return image, text, label
    
    def get_samples_by_label(self, label: int, num_samples: int = 1) -> List[torch.Tensor]:
        samples = []
        for i in range(len(self.dataset)):
            img, lbl = self.dataset[i]
            if lbl == label:
                samples.append(img)
                if len(samples) >= num_samples:
                    break
        return samples

def get_mnist_dataloader(data_dir: str = "data", batch_size: int = 64,
                        train: bool = True, num_workers: int = 4) -> DataLoader:
    dataset = MNISTDataset(data_dir=data_dir, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )

def get_interpolation_pairs(dataset: MNISTDataset, 
                          start_digit: int, 
                          end_digit: int, 
                          num_pairs: int = 1) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    start_samples = dataset.get_samples_by_label(start_digit, num_pairs)
    end_samples = dataset.get_samples_by_label(end_digit, num_pairs)
    
    pairs = []
    for i in range(min(len(start_samples), len(end_samples), num_pairs)):
        pairs.append((start_samples[i], end_samples[i]))
    
    return pairs

if __name__ == "__main__":
    train_loader = get_mnist_dataloader(batch_size=128, train=True)
    test_loader = get_mnist_dataloader(batch_size=128, train=False)
    
    test_dataset = MNISTDataset(train=False)
    
    pairs = get_interpolation_pairs(test_dataset, start_digit=3, end_digit=8)
    print(f"Got {len(pairs)} interpolation pairs")
    
    for batch in train_loader:
        images, texts, labels = batch
        print(f"Image shape: {images.shape}")
        print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
        break