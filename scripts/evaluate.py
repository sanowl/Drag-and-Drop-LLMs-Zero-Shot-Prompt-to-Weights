#!/usr/bin/env python3
"""
Evaluation script for Drag-and-Drop LLM system.
"""

import argparse
import os
import sys
import torch
import json
import random
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dnd_llm import (
    DragAndDropLLM,
    DnDEvaluator, 
    DatasetManager,
    Config,
    setup_logging
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Drag-and-Drop LLM')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--task', type=str, choices=['common_sense', 'coding', 'math', 'all'],
                        default='all', help='Task type to evaluate')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated list of datasets to evaluate')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file for results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Override device if specified
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
    
    # Initialize evaluator
    evaluator = DnDEvaluator(model, device=config.device)
    
    # Load datasets
    common_sense_datasets = DatasetManager.load_common_sense_datasets()
    coding_datasets = DatasetManager.load_coding_datasets()
    math_datasets = DatasetManager.load_math_datasets()
    
    results = {}
    
    # Determine which datasets to evaluate
    if args.datasets:
        eval_datasets = args.datasets.split(',')
    else:
        eval_datasets = config.evaluation.datasets
    
    # Common sense reasoning evaluation
    if args.task in ['common_sense', 'all']:
        print("Evaluating common sense reasoning...")
        for dataset_name in eval_datasets:
            if dataset_name in common_sense_datasets:
                # Sample test prompts
                test_prompts = random.sample(
                    common_sense_datasets[dataset_name], 
                    min(100, len(common_sense_datasets[dataset_name]))
                )
                
                result = evaluator.evaluate_common_sense(test_prompts, dataset_name)
                results[f"common_sense_{dataset_name}"] = result
    
    # Coding evaluation
    if args.task in ['coding', 'all']:
        print("Evaluating coding tasks...")
        coding_test_prompts = random.sample(
            coding_datasets['Evol-Instruct-68K-V1'], 164  # HumanEval size
        )
        coding_result = evaluator.evaluate_coding(coding_test_prompts, "HumanEval")
        results["coding_HumanEval"] = coding_result
    
    # Math evaluation  
    if args.task in ['math', 'all']:
        print("Evaluating math tasks...")
        math_test_prompts = random.sample(math_datasets['Competition-Math'], 100)
        
        gsm8k_result = evaluator.evaluate_math(math_test_prompts, "gsm8K")
        math_result = evaluator.evaluate_math(math_test_prompts, "MATH")
        
        results["math_gsm8K"] = gsm8k_result
        results["math_MATH"] = math_result
    
    # Cross-domain evaluation
    if args.task == 'all':
        print("Evaluating cross-domain transfer...")
        science_prompts = [f"Science question {i}" for i in range(100)]
        cross_domain_result = evaluator.evaluate_cross_domain(
            "common_sense", "science", science_prompts
        )
        results["cross_domain"] = cross_domain_result
    
    # Print results summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    
    for task_name, result in results.items():
        print(f"\n{task_name}:")
        if 'accuracy' in result:
            print(f"  Accuracy: {result['accuracy']:.2f}%")
        if 'pass@1' in result:
            print(f"  pass@1: {result['pass@1']:.2f}%")
            print(f"  pass@5: {result['pass@5']:.2f}%")
            print(f"  pass@10: {result['pass@10']:.2f}%")
    
    # Save results to file
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(
            os.path.dirname(args.checkpoint), 
            "evaluation_results.json"
        )
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main() 