"""
Utility functions for the Drag-and-Drop LLM models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count the number of parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Get the memory size of a model in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'parameters_mb': param_size / 1024 / 1024,
        'buffers_mb': buffer_size / 1024 / 1024,
        'total_mb': size_mb
    }


def freeze_model(model: nn.Module) -> None:
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module) -> None:
    """Unfreeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = True


def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """Freeze specific layers by name."""
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False


def get_layer_names(model: nn.Module) -> List[str]:
    """Get all layer names in a model."""
    return [name for name, _ in model.named_modules()]


def initialize_weights(model: nn.Module, init_type: str = 'xavier_uniform') -> None:
    """Initialize model weights with specified initialization."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif init_type == 'normal':
                nn.init.normal_(module.weight, 0, 0.01)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def get_device(model: nn.Module) -> torch.device:
    """Get the device of a model."""
    return next(model.parameters()).device


def move_to_device(data: Any, device: torch.device) -> Any:
    """Move data to specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        return data


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """Clip gradients and return the gradient norm."""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def get_gradient_norm(model: nn.Module) -> float:
    """Get the gradient norm of a model."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1.0 / 2)


def save_model_state(model: nn.Module, filepath: str, 
                     optimizer: Optional[torch.optim.Optimizer] = None,
                     epoch: Optional[int] = None,
                     loss: Optional[float] = None) -> None:
    """Save model state with optional optimizer and training info."""
    state = {
        'model_state_dict': model.state_dict(),
        'timestamp': torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    }
    
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        state['epoch'] = epoch
    if loss is not None:
        state['loss'] = loss
    
    torch.save(state, filepath)
    logger.info(f"Model state saved to {filepath}")


def load_model_state(model: nn.Module, filepath: str,
                     optimizer: Optional[torch.optim.Optimizer] = None,
                     device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Load model state with optional optimizer."""
    if device is None:
        device = get_device(model)
    
    state = torch.load(filepath, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    
    logger.info(f"Model state loaded from {filepath}")
    return state


def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, bool]:
    """Compare if two models have the same architecture and parameters."""
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    
    # Check if they have the same keys
    keys_match = set(state1.keys()) == set(state2.keys())
    
    # Check if parameters match
    params_match = True
    if keys_match:
        for key in state1.keys():
            if not torch.equal(state1[key], state2[key]):
                params_match = False
                break
    
    return {
        'architecture_match': keys_match,
        'parameters_match': params_match and keys_match,
        'identical': keys_match and params_match
    }


def print_model_summary(model: nn.Module) -> None:
    """Print a summary of the model."""
    param_info = count_parameters(model)
    size_info = get_model_size(model)
    device = get_device(model)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Total parameters: {param_info['total']:,}")
    print(f"Trainable parameters: {param_info['trainable']:,}")
    print(f"Non-trainable parameters: {param_info['non_trainable']:,}")
    print(f"Model size: {size_info['total_mb']:.2f} MB")
    print("=" * 60)


def validate_lora_parameters(lora_params: Dict[str, Dict[str, torch.Tensor]]) -> bool:
    """Validate LoRA parameter structure."""
    required_keys = ['lora_A', 'lora_B']
    
    for layer_name, params in lora_params.items():
        if not isinstance(params, dict):
            logger.error(f"Layer {layer_name} parameters must be a dictionary")
            return False
        
        for key in required_keys:
            if key not in params:
                logger.error(f"Missing {key} in layer {layer_name}")
                return False
            
            if not isinstance(params[key], torch.Tensor):
                logger.error(f"{key} in layer {layer_name} must be a tensor")
                return False
    
    return True


def merge_lora_parameters(base_params: Dict[str, torch.Tensor],
                         lora_params: Dict[str, Dict[str, torch.Tensor]],
                         alpha: float = 16.0) -> Dict[str, torch.Tensor]:
    """Merge LoRA parameters with base parameters."""
    merged_params = base_params.copy()
    
    for layer_name, lora_layer_params in lora_params.items():
        if layer_name in merged_params:
            lora_A = lora_layer_params['lora_A']
            lora_B = lora_layer_params['lora_B']
            
            # LoRA update: W = W_base + alpha * B @ A
            lora_update = alpha * torch.mm(lora_B, lora_A)
            merged_params[layer_name] = merged_params[layer_name] + lora_update
    
    return merged_params 