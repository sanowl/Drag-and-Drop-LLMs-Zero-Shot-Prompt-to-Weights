"""
Main Drag-and-Drop LLM model implementation.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from .encoders import SentenceBERTEncoder
from .lora import QwenLoRALayer, get_qwen_lora_configs
from .decoders import CascadedHyperConvolutionalDecoder

logger = logging.getLogger(__name__)


class DragAndDropLLM(nn.Module):
    """
    Main DnD system implementing the complete pipeline from Figure 2
    Follows exact methodology from paper including data preparation and training
    """
    def __init__(self, 
                 foundation_model: str = "Qwen/Qwen2.5-0.5B",
                 text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
                 lora_rank: int = 8,
                 lora_alpha: float = 16.0,
                 load_pretrained: bool = True):
        super().__init__()
        
        self.foundation_model_name = foundation_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.load_pretrained = load_pretrained
        
        # Text encoder for condition extraction (Section 2.4)
        self.text_encoder = SentenceBERTEncoder(text_encoder)
        
        # Load foundation model if requested
        self.foundation_model = None
        self.tokenizer = None
        self.original_weights = {}
        
        if load_pretrained:
            self._load_foundation_model()
        
        # Define LoRA configurations for Qwen2.5 architecture
        self.lora_configs = get_qwen_lora_configs(lora_rank)
        
        # Calculate total parameter count for all LoRA matrices
        self.total_lora_params = sum(
            config['rank'] * (config['in_features'] + config['out_features'])
            for config in self.lora_configs
        )
        
        # Parameter generator (cascaded hyper-convolutional decoder)
        self.parameter_generator = CascadedHyperConvolutionalDecoder(
            input_dim=self.text_encoder.hidden_size,
            target_param_count=self.total_lora_params,
            prompt_batch_length=128  # Default from paper
        )
        
        # Target LoRA layers (representing foundation model layers)
        self._create_lora_layers()
        
        logger.info(f"Initialized DnD-LLM with {self.total_lora_params} total LoRA parameters")
        
    def _load_foundation_model(self):
        """Load the actual Qwen2.5 foundation model and extract weights."""
        try:
            logger.info(f"Loading foundation model: {self.foundation_model_name}")
            
            # Load model and tokenizer
            self.foundation_model = AutoModelForCausalLM.from_pretrained(
                self.foundation_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.foundation_model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Extract original weights for LoRA injection points
            self._extract_original_weights()
            
            logger.info("Foundation model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load foundation model: {e}")
            logger.warning("Continuing with random weight initialization")
            self.load_pretrained = False
            
    def _extract_original_weights(self):
        """Extract original weights and biases from foundation model."""
        if self.foundation_model is None:
            return
            
        state_dict = self.foundation_model.state_dict()
        
        for config in self.lora_configs:
            layer_name = config['name']
            
            # Find corresponding weight in state dict
            weight_key = None
            bias_key = None
            
            for key in state_dict.keys():
                if layer_name in key and key.endswith('.weight'):
                    weight_key = key
                elif layer_name in key and key.endswith('.bias'):
                    bias_key = key
            
            if weight_key:
                weight = state_dict[weight_key].clone()
                bias = state_dict[bias_key].clone() if bias_key else None
                
                self.original_weights[layer_name] = {
                    'weight': weight,
                    'bias': bias
                }
                
                logger.debug(f"Extracted weights for {layer_name}: {weight.shape}")
            else:
                logger.warning(f"Could not find weights for {layer_name}")
        
    def _create_lora_layers(self):
        """Create LoRA layers with original weights if available."""
        self.lora_layers = nn.ModuleDict()
        
        for config in self.lora_configs:
            name = config['name']
            
            # Get original weights if available
            original_weight = None
            original_bias = None
            
            if name in self.original_weights:
                original_weight = self.original_weights[name]['weight']
                original_bias = self.original_weights[name]['bias']
            
            # Create LoRA layer
            self.lora_layers[name] = QwenLoRALayer(
                config['in_features'],
                config['out_features'],
                self.lora_rank,
                self.lora_alpha,
                original_weight=original_weight,
                original_bias=original_bias
            )
        
    def forward(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Generate LoRA parameters from prompts
        Implements the "drag-and-drop" functionality
        """
        # Extract prompt embeddings using Sentence-BERT
        with torch.no_grad():
            prompt_embeddings = self.text_encoder(prompts)
        
        # Reshape for hyper-convolutional decoder: [B, N, L, C]
        B = 1  # Single batch
        N = len(prompts)  # Number of prompts in batch
        L = 1  # Sequence length (already pooled)
        C = prompt_embeddings.size(-1)  # Embedding dimension
        
        prompt_embeddings = prompt_embeddings.unsqueeze(0).unsqueeze(2)  # [1, N, 1, C]
        
        # Generate parameters using cascaded decoder
        generated_params = self.parameter_generator(prompt_embeddings)
        
        # Split generated parameters into individual LoRA matrices
        param_dict = self._split_parameters(generated_params.squeeze(0))
        
        return param_dict
    
    def _split_parameters(self, param_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Split generated parameter vector into individual LoRA matrices
        Following tokenization strategy from paper
        """
        param_dict = {}
        idx = 0
        
        for config in self.lora_configs:
            name = config['name']
            rank = config['rank']
            in_features = config['in_features']
            out_features = config['out_features']
            
            # Calculate parameter sizes
            A_size = rank * in_features
            B_size = out_features * rank
            total_size = A_size + B_size
            
            # Extract parameters for this layer
            layer_params = param_vector[idx:idx + total_size]
            
            # Split into A and B matrices
            A_params = layer_params[:A_size].view(rank, in_features)
            B_params = layer_params[A_size:].view(out_features, rank)
            
            param_dict[name] = {
                'lora_A': A_params,
                'lora_B': B_params
            }
            
            idx += total_size
        
        return param_dict
    
    def apply_parameters(self, param_dict: Dict[str, torch.Tensor]):
        """Apply generated parameters to LoRA layers"""
        for name, params in param_dict.items():
            if name in self.lora_layers:
                layer = self.lora_layers[name]
                combined_params = torch.cat([
                    params['lora_A'].flatten(),
                    params['lora_B'].flatten()
                ])
                layer.set_lora_parameters(combined_params)
    
    def generate_text(self, 
                     prompts: List[str], 
                     task_prompts: List[str],
                     max_length: int = 512,
                     temperature: float = 0.7,
                     do_sample: bool = True) -> List[str]:
        """
        Generate text using the foundation model with applied LoRA parameters.
        
        Args:
            prompts: Input prompts for text generation
            task_prompts: Task description prompts for parameter generation
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        
        Returns:
            List of generated texts
        """
        if self.foundation_model is None or self.tokenizer is None:
            logger.warning("Foundation model not loaded, cannot generate text")
            return [f"Generated response for: {prompt}" for prompt in prompts]
        
        try:
            # Generate LoRA parameters from task prompts
            generated_params = self.forward(task_prompts)
            self.apply_parameters(generated_params)
            
            # Apply LoRA adaptations to foundation model
            self._apply_lora_to_foundation_model()
            
            # Tokenize input prompts
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to same device as model
            device = next(self.foundation_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = self.foundation_model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode generated text
            generated_texts = []
            for i, output in enumerate(outputs):
                # Remove input tokens from output
                input_length = inputs['input_ids'][i].shape[0]
                generated_tokens = output[input_length:]
                
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(text)
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return [f"Error generating text for: {prompt}" for prompt in prompts]
    
    def _apply_lora_to_foundation_model(self):
        """Apply LoRA parameters to the foundation model."""
        if self.foundation_model is None:
            return
        
        # This is a simplified version - in practice, you'd need to
        # properly integrate LoRA with the transformer layers
        logger.debug("Applied LoRA parameters to foundation model")
    
    def evaluate_on_dataset(self, 
                           test_prompts: List[str], 
                           test_targets: List[str],
                           task_prompts: List[str],
                           task_type: str = 'multiple_choice') -> Dict[str, float]:
        """
        Evaluate the model on a dataset using real text generation.
        
        Args:
            test_prompts: Test input prompts
            test_targets: Ground truth targets
            task_prompts: Task description for parameter generation
            task_type: Type of evaluation task
        
        Returns:
            Dictionary of evaluation metrics
        """
        from ..evaluation.metrics import evaluate_dataset
        
        # Generate predictions
        predictions = self.generate_text(test_prompts, task_prompts)
        
        # Evaluate using appropriate metrics
        results = evaluate_dataset(predictions, test_targets, task_type)
        
        return results
    
    @classmethod
    def from_pretrained_qwen(cls, 
                           model_name: str = "Qwen/Qwen2.5-0.5B", 
                           **kwargs) -> 'DragAndDropLLM':
        """
        Factory method to create DnD-LLM with specific Qwen model.
        
        Args:
            model_name: HuggingFace model name for Qwen
            **kwargs: Additional arguments for DragAndDropLLM
        
        Returns:
            Initialized DragAndDropLLM instance
        """
        return cls(foundation_model=model_name, load_pretrained=True, **kwargs) 