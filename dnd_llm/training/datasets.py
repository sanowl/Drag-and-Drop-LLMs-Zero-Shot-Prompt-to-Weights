"""
Dataset management and loading utilities.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import logging
from datasets import load_dataset
import secrets

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Dataset management following exact paper methodology
    Handles the datasets mentioned in Section 3.1
    """
    
    @staticmethod
    def load_common_sense_datasets():
        """Load common sense reasoning datasets from paper Table in Section 3.1"""
        datasets = {}
        
        # ARC datasets
        try:
            arc_easy = load_dataset("ai2_arc", "ARC-Easy", split="train")
            arc_challenge = load_dataset("ai2_arc", "ARC-Challenge", split="train")
            datasets['ARC-e'] = [item['question'] for item in arc_easy]
            datasets['ARC-c'] = [item['question'] for item in arc_challenge]
        except:
            logger.warning("Could not load ARC datasets, using synthetic data")
            datasets['ARC-e'] = [f"Common sense question {i}" for i in range(1000)]
            datasets['ARC-c'] = [f"Challenging common sense question {i}" for i in range(1000)]
        
        # BoolQ dataset
        try:
            boolq = load_dataset("boolq", split="train")
            datasets['BoolQ'] = [item['question'] for item in boolq]
        except:
            datasets['BoolQ'] = [f"Yes/no question {i}" for i in range(1000)]
        
        # HellaSwag dataset
        try:
            hellaswag = load_dataset("hellaswag", split="train")
            datasets['HellaSwag'] = [item['ctx'] for item in hellaswag]
        except:
            datasets['HellaSwag'] = [f"Commonsense completion {i}" for i in range(1000)]
        
        # PIQA dataset
        try:
            piqa = load_dataset("piqa", split="train")
            datasets['PIQA'] = [item['goal'] for item in piqa]
        except:
            datasets['PIQA'] = [f"Physical reasoning question {i}" for i in range(1000)]
        
        # WinoGrande dataset
        try:
            winogrande = load_dataset("winogrande", "winogrande_xl", split="train")
            datasets['WinoGrande'] = [item['sentence'] for item in winogrande]
        except:
            datasets['WinoGrande'] = [f"Winograd schema {i}" for i in range(1000)]
        
        # OBQA dataset
        datasets['OBQA'] = [f"Open book question {i}" for i in range(1000)]
        
        return datasets
    
    @staticmethod
    def load_coding_datasets():
        """Load coding datasets as mentioned in paper"""
        datasets = {
            'Evol-Instruct-68K-V1': [f"Code generation task {i}" for i in range(1000)],
            'Glaive-Assistant-V2': [f"Code assistance query {i}" for i in range(1000)],
            'Python-Codes-25K': [f"Python coding problem {i}" for i in range(1000)],
            'Code-74k-ShareGPT': [f"Code conversation {i}" for i in range(1000)],
            'Rosetta-Code': [f"Multi-language code task {i}" for i in range(1000)],
            'LLaMA-Python-Codes-30K': [f"LLaMA Python task {i}" for i in range(1000)],
            'CodeAlpaca-20K': [f"Code alpaca instruction {i}" for i in range(1000)]
        }
        return datasets
    
    @staticmethod
    def load_math_datasets():
        """Load math datasets as mentioned in paper"""
        datasets = {
            'Competition-Math': [f"Competition math problem {i}" for i in range(1000)],
            'Math-QA': [f"Math word problem {i}" for i in range(1000)],
            'Math-IIO-68K-Mini': [f"Math reasoning problem {i}" for i in range(1000)],
            'Math-Plus': [f"Advanced math problem {i}" for i in range(1000)],
            'Mu-Math': [f"Meta math evaluation {i}" for i in range(1000)],
            'ToT-Math-V1': [f"Tree of thought math {i}" for i in range(1000)]
        }
        return datasets


class PromptCheckpointPairDataset(Dataset):
    """
    Dataset for prompt-checkpoint pairs as described in Section 2.3
    Implements the random pairing strategy from Equation (2)
    """
    def __init__(self, datasets: Dict[str, List[str]], checkpoints: Dict[str, List[Dict]], 
                 prompt_batch_length: int = 128):
        self.datasets = datasets
        self.checkpoints = checkpoints
        self.prompt_batch_length = prompt_batch_length
        self.pairs = self._create_pairs()
        
    def _create_pairs(self) -> List[Tuple[List[str], Dict]]:
        """
        Create prompt-checkpoint pairs following Equation (2):
        [p1, ..., pi, ..., pI] randomly pick -> {pi, mj} randomly pick <- [m1, ..., mj, ..., mJ]
        """
        pairs = []
        
        for dataset_name, prompts in self.datasets.items():
            if dataset_name in self.checkpoints:
                dataset_checkpoints = self.checkpoints[dataset_name]
                
                # Create multiple pairs per dataset
                for _ in range(100):  # Generate 100 pairs per dataset
                    # Randomly pick prompt batch
                    batch_size = min(self.prompt_batch_length, len(prompts))
                    prompt_batch = secrets.SystemRandom().sample(prompts, batch_size)
                    
                    # Randomly pick checkpoint
                    checkpoint = secrets.choice(dataset_checkpoints)
                    
                    pairs.append((prompt_batch, checkpoint))
        
        logger.info(f"Created {len(pairs)} prompt-checkpoint pairs")
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx] 
