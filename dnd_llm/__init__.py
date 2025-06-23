"""
Drag-and-Drop LLMs: Zero-Shot Prompt to Weights

A system for generating neural network weights directly from text prompts
using cascaded hyper-convolutional decoders.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .models.main_model import DragAndDropLLM
from .models.encoders import SentenceBERTEncoder
from .models.lora import QwenLoRALayer
from .models.decoders import CascadedHyperConvolutionalDecoder, HyperConvolutionalBlock
from .training.trainer import DnDTrainer
from .training.checkpoint import CheckpointCollector
from .training.datasets import DatasetManager, PromptCheckpointPairDataset
from .evaluation.evaluator import DnDEvaluator
from .utils.config import Config
from .utils.logging import setup_logging

__all__ = [
    "DragAndDropLLM",
    "SentenceBERTEncoder",
    "QwenLoRALayer", 
    "CascadedHyperConvolutionalDecoder",
    "HyperConvolutionalBlock",
    "DnDTrainer",
    "CheckpointCollector",
    "DatasetManager",
    "PromptCheckpointPairDataset",
    "DnDEvaluator",
    "Config",
    "setup_logging"
] 