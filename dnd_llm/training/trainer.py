"""
Training implementation for the Drag-and-Drop LLM system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List
import logging
from tqdm import tqdm

from .checkpoint import CheckpointCollector
from .datasets import PromptCheckpointPairDataset

logger = logging.getLogger(__name__)


class DnDTrainer:
    """
    Training implementation following exact paper methodology
    Section 2.5: Training and Inference
    """
    def __init__(self, model, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer settings from Table 7
        self.optimizer = optim.AdamW(
            self.model.parameter_generator.parameters(),
            lr=3e-5,  # Learning rate from paper
            weight_decay=0.1,  # Weight decay from paper
            betas=(0.9, 0.999)
        )
        
        # MSE loss as mentioned in Section 2.5
        self.criterion = nn.MSELoss()
        
        # Training hyperparameters from paper
        self.max_grad_norm = 1.0
        self.noise_aug_amplitude = 1e-4
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch following paper training procedure"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (prompt_batches, target_checkpoints) in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # Process each prompt batch in the training batch
            batch_loss = 0.0
            for prompts, target_checkpoint in zip(prompt_batches, target_checkpoints):
                # Generate parameters from prompts
                generated_params = self.model(prompts)
                
                # Convert target checkpoint to parameter vector
                target_vector = self._checkpoint_to_vector(target_checkpoint)
                target_vector = target_vector.to(self.device)
                
                # Convert generated parameters to vector
                generated_vector = self._params_dict_to_vector(generated_params)
                
                # Add noise augmentation as mentioned in Table 7
                noise = torch.randn_like(target_vector) * self.noise_aug_amplitude
                target_vector_noisy = target_vector + noise
                
                # Calculate MSE loss as in Section 2.5
                loss = self.criterion(generated_vector, target_vector_noisy)
                batch_loss += loss
            
            # Average loss over batch
            batch_loss = batch_loss / len(prompt_batches)
            batch_loss.backward()
            
            # Gradient clipping as mentioned in paper
            torch.nn.utils.clip_grad_norm_(
                self.model.parameter_generator.parameters(), 
                max_norm=self.max_grad_norm
            )
            
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            pbar.set_postfix({'loss': batch_loss.item()})
            
            # Log every 100 batches as mentioned in paper experiments
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {batch_loss.item():.6f}")
        
        return total_loss / len(dataloader)
    
    def _checkpoint_to_vector(self, checkpoint: Dict) -> torch.Tensor:
        """Convert checkpoint parameters to flattened vector for MSE loss"""
        param_list = []
        for layer_name in sorted(checkpoint.keys()):
            layer_params = checkpoint[layer_name]
            param_list.append(layer_params['lora_A'].flatten())
            param_list.append(layer_params['lora_B'].flatten())
        return torch.cat(param_list)
    
    def _params_dict_to_vector(self, params_dict: Dict) -> torch.Tensor:
        """Convert generated parameters dict to flattened vector"""
        param_list = []
        for layer_name in sorted(params_dict.keys()):
            layer_params = params_dict[layer_name]
            param_list.append(layer_params['lora_A'].flatten())
            param_list.append(layer_params['lora_B'].flatten())
        return torch.cat(param_list)
    
    def train(self, datasets: Dict[str, List[str]], num_epochs: int = 5000, batch_size: int = 128):
        """
        Main training loop following paper training schedule
        Training steps: 5000 (from Table 7)
        """
        logger.info("Starting checkpoint collection phase...")
        
        # Collect checkpoints for each dataset
        collector = CheckpointCollector(self.model, self.device)
        all_checkpoints = {}
        
        for dataset_name, prompts in datasets.items():
            logger.info(f"Collecting checkpoints for {dataset_name}")
            checkpoints = collector.collect_checkpoints(prompts)
            all_checkpoints[dataset_name] = checkpoints
        
        # Create prompt-checkpoint pairs dataset
        logger.info("Creating prompt-checkpoint pairs...")
        pair_dataset = PromptCheckpointPairDataset(datasets, all_checkpoints)
        
        # Create dataloader
        dataloader = DataLoader(
            pair_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        # Training loop
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader, epoch)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
            
            # Save checkpoint every 1000 epochs as mentioned in paper
            if (epoch + 1) % 1000 == 0:
                self.save_checkpoint(f"dnd_checkpoint_epoch_{epoch+1}.pth")
        
        logger.info("Training completed!")
    
    def _collate_fn(self, batch):
        """Custom collate function for prompt-checkpoint pairs"""
        prompt_batches = []
        target_checkpoints = []
        
        for prompts, checkpoint in batch:
            prompt_batches.append(prompts)
            target_checkpoints.append(checkpoint)
        
        return prompt_batches, target_checkpoints
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Checkpoint loaded from {path}") 