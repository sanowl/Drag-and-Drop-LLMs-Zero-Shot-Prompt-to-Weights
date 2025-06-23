#!/usr/bin/env python3
"""
Inference script for Drag-and-Drop LLM system.
Generate model weights from prompts.
"""

import argparse
import sys
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dnd_llm import DragAndDropLLM, Config, setup_logging


def main():
    parser = argparse.ArgumentParser(description='Run DnD-LLM inference')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--prompts', type=str, nargs='+', required=True,
                        help='Text prompts for parameter generation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for generated parameters')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    if args.device:
        config.device = args.device
    
    # Setup logging
    setup_logging(level=config.log_level)
    
    # Initialize model
    print("Loading Drag-and-Drop LLM system...")
    model = DragAndDropLLM(
        foundation_model=config.model.foundation_model,
        text_encoder=config.model.text_encoder,
        lora_rank=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate parameters from prompts
    print(f"Generating parameters from {len(args.prompts)} prompts...")
    with torch.no_grad():
        generated_params = model(args.prompts)
    
    print("Parameter generation completed!")
    print(f"Generated parameters for {len(generated_params)} layers")
    
    # Save parameters if output specified
    if args.output:
        torch.save(generated_params, args.output)
        print(f"Parameters saved to: {args.output}")
    
    # Apply parameters to model (for demonstration)
    model.apply_parameters(generated_params)
    print("Parameters applied to model successfully!")


if __name__ == "__main__":
    main() 