#!/usr/bin/env python3
"""
Training script for Drag-and-Drop LLM system.
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
import secrets

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dnd_llm import (
    DragAndDropLLM, 
    DnDTrainer, 
    DatasetManager,
    Config,
    setup_logging
)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    secrets.SystemRandom().seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Train Drag-and-Drop LLM')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.device:
        config.device = args.device
    
    # Validate configuration
    config.validate()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(config.output_dir, "training.log")
    setup_logging(level=config.log_level, log_file=log_file)
    
    # Set random seed
    set_seed(config.seed)
    
    # Initialize model
    print("Initializing Drag-and-Drop LLM system...")
    model = DragAndDropLLM(
        foundation_model=config.model.foundation_model,
        text_encoder=config.model.text_encoder,
        lora_rank=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha
    )
    
    # Initialize trainer
    trainer = DnDTrainer(model, device=config.device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Load datasets
    print("Loading datasets...")
    common_sense_datasets = DatasetManager.load_common_sense_datasets()
    coding_datasets = DatasetManager.load_coding_datasets()
    math_datasets = DatasetManager.load_math_datasets()
    
    # Combine all datasets for training
    all_datasets = {**common_sense_datasets, **coding_datasets, **math_datasets}
    
    # Save configuration
    config.save_yaml(os.path.join(config.output_dir, "config.yaml"))
    
    # Start training
    print("Starting training...")
    trainer.train(
        datasets=all_datasets,
        num_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size
    )
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, "final_model.pth")
    trainer.save_checkpoint(final_model_path)
    
    print(f"Training completed! Model saved to: {final_model_path}")


if __name__ == "__main__":
    main() 
