"""
Configuration management utilities.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    foundation_model: str = "Qwen/Qwen2.5-0.5B"
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    prompt_batch_length: int = 128


@dataclass 
class TrainingConfig:
    """Training configuration parameters"""
    num_epochs: int = 5000
    batch_size: int = 128
    learning_rate: float = 3e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    noise_aug_amplitude: float = 1e-4
    save_every: int = 1000


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters"""
    datasets: list = None
    batch_size: int = 64
    metrics: list = None
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["ARC-e", "OBQA", "PIQA"]
        if self.metrics is None:
            self.metrics = ["accuracy", "pass@k", "bleu"]


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./outputs"
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        return cls(
            model=model_config,
            training=training_config,
            evaluation=evaluation_config,
            device=config_dict.get('device', 'cuda'),
            seed=config_dict.get('seed', 42),
            output_dir=config_dict.get('output_dir', './outputs'),
            log_level=config_dict.get('log_level', 'INFO')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'device': self.device,
            'seed': self.seed,
            'output_dir': self.output_dir,
            'log_level': self.log_level
        }
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {yaml_path}")
    
    def validate(self):
        """Validate configuration parameters"""
        if self.model.lora_rank <= 0:
            raise ValueError("LoRA rank must be positive")
        
        if self.model.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.device not in ['cuda', 'cpu', 'mps']:
            logger.warning(f"Unknown device type: {self.device}")
        
        logger.info("Configuration validation passed") 