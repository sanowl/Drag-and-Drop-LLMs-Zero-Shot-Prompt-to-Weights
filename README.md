# Drag-and-Drop LLMs: Zero-Shot Prompt to Weights

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular implementation of the Drag-and-Drop LLM system that enables zero-shot adaptation of Large Language Models through prompt-to-weights generation using cascaded hyper-convolutional decoders.

## Overview

This system implements a novel approach to LLM adaptation that generates LoRA (Low-Rank Adaptation) parameters directly from text prompts, eliminating the need for traditional fine-tuning. The core innovation is a cascaded hyper-convolutional decoder that transforms Sentence-BERT embeddings into model-specific LoRA weights.

### Key Features

- **Zero-shot adaptation**: Generate model weights directly from prompts
- **Modular architecture**: Clean separation of models, training, and evaluation
- **Cascaded hyper-convolutional decoder**: Novel parameter generation architecture
- **LoRA integration**: Efficient low-rank adaptation for foundation models
- **Easy-to-use scripts**: Simple command-line interface for training and evaluation
- **Configurable**: YAML-based configuration system

## Installation

```bash
git clone https://github.com/sanowl/Drag-and-Drop-LLMs-Zero-Shot-Prompt-to-Weights
cd Drag-and-Drop-LLMs-Zero-Shot-Prompt-to-Weights
pip install -r requirements.txt
```

### Verify Installation

Run the installation test to ensure all components are working:

```bash
python test_installation.py
```

This will verify:
- ✓ All model components can be imported
- ✓ Main model can be instantiated  
- ✓ Individual components work correctly
- ✓ Dataset loading functions properly

### Troubleshooting

If you encounter issues with missing models folder:

1. **Ensure complete clone**: 
   ```bash
   git status
   git ls-files dnd_llm/models/
   ```

2. **Check required model files**:
   ```
   dnd_llm/models/
   ├── __init__.py           # Package initialization
   ├── main_model.py         # Main DragAndDropLLM class  
   ├── encoders.py          # SentenceBERT text encoder
   ├── lora.py              # LoRA layer implementations
   └── decoders.py          # Hyper-convolutional decoders
   ```

3. **Test imports**:
   ```bash
   python -c "from dnd_llm import DragAndDropLLM; print('Success!')"
   ```

## Quick Start

```python
from dnd_llm import DragAndDropLLM, DnDTrainer, DatasetManager

# Initialize the model
model = DragAndDropLLM(
    foundation_model="Qwen/Qwen2.5-0.5B",
    text_encoder="sentence-transformers/all-MiniLM-L6-v2",
    lora_rank=8,
    lora_alpha=16.0
)

# Load datasets
datasets = DatasetManager.load_common_sense_datasets()

# Train the system
trainer = DnDTrainer(model, device='cuda')
trainer.train(datasets, num_epochs=5000, batch_size=128)

# Generate weights from prompts
test_prompts = ["Solve common sense reasoning problems"]
generated_params = model(test_prompts)
```

## Architecture

The system consists of several key components:

1. **SentenceBERTEncoder**: Extracts semantic embeddings from text prompts
2. **CascadedHyperConvolutionalDecoder**: Transforms embeddings to parameter space
3. **QwenLoRALayer**: LoRA implementation for efficient adaptation
4. **DragAndDropLLM**: Main system orchestrating the complete pipeline

### Training

```bash
# Train with default configuration
python scripts/train.py --config configs/default.yaml

# Train with custom output directory
python scripts/train.py --config configs/default.yaml --output-dir ./my_experiment

# Resume training from checkpoint
python scripts/train.py --config configs/default.yaml --resume ./outputs/checkpoint.pth
```

### Evaluation

```bash
# Evaluate on all tasks
python scripts/evaluate.py --checkpoint ./outputs/final_model.pth --task all

# Evaluate specific tasks
python scripts/evaluate.py --checkpoint ./outputs/final_model.pth --task common_sense
python scripts/evaluate.py --checkpoint ./outputs/final_model.pth --task coding
python scripts/evaluate.py --checkpoint ./outputs/final_model.pth --task math

# Evaluate specific datasets
python scripts/evaluate.py --checkpoint ./outputs/final_model.pth --datasets ARC-e,PIQA
```

### Inference

```bash
# Generate weights from prompts
python scripts/inference.py \
    --checkpoint ./outputs/final_model.pth \
    --prompts "Solve common sense problems" "Generate Python code" \
    --output generated_weights.pth
```

## Usage Examples

### Basic Usage

```python
from dnd_llm import DragAndDropLLM

# Initialize the model
model = DragAndDropLLM(
    foundation_model="Qwen/Qwen2.5-0.5B",
    text_encoder="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate weights from prompts
prompts = ["Solve math problems", "Generate Python code"]
generated_params = model(prompts)

# Apply generated parameters
model.apply_parameters(generated_params)
```

## Project Structure

```
Drag-and-Drop-LLMs-Zero-Shot-Prompt-to-Weights/
├── dnd_llm/                    # Main package
│   ├── __init__.py
│   ├── models/                 # Model components
│   │   ├── __init__.py
│   │   ├── encoders.py         # Text encoders
│   │   ├── lora.py            # LoRA implementations
│   │   ├── decoders.py        # Hyper-convolutional decoders
│   │   └── main_model.py      # Main DnD-LLM model
│   ├── training/              # Training components
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training logic
│   │   ├── checkpoint.py      # Checkpoint collection
│   │   └── datasets.py        # Dataset management
│   ├── evaluation/            # Evaluation components
│   │   ├── __init__.py
│   │   ├── evaluator.py       # Evaluation logic
│   │   └── metrics.py         # Evaluation metrics
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   └── logging.py         # Logging utilities
│   └── data/                  # Data handling
│       ├── __init__.py
│       └── loaders.py         # Data loaders
├── scripts/                   # Execution scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── inference.py          # Inference script
├── configs/                   # Configuration files
│   ├── default.yaml          # Default configuration
│   └── experiments/          # Experiment configs
├── tests/                    # Unit tests
├── docs/                     # Documentation
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## Configuration

The system uses YAML configuration files for easy customization. See `configs/default.yaml` for the complete configuration options.

### Key Configuration Sections

- **Model**: Foundation model settings, LoRA parameters, text encoder
- **Training**: Learning rate, batch size, epochs, optimization settings  
- **Evaluation**: Datasets to evaluate on, metrics to compute
- **System**: Device selection, output directories, logging levels

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

```bibtex
@article{dnd_llm2024,
  title={Drag-and-Drop LLMs: Zero-Shot Prompt to Weights},
  author={Research Team},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Features

### Core Components

- **SentenceBERTEncoder**: Extracts semantic embeddings from text prompts
- **CascadedHyperConvolutionalDecoder**: Transforms embeddings to parameter space
- **QwenLoRALayer**: LoRA implementation for efficient adaptation
- **DragAndDropLLM**: Main system orchestrating the complete pipeline

### Supported Tasks

- **Common Sense Reasoning**: ARC-e, OBQA, PIQA, HellaSwag, BoolQ, WinoGrande
- **Code Generation**: HumanEval and other coding benchmarks
- **Mathematical Reasoning**: gsm8K, MATH, and other math datasets
- **Cross-domain Transfer**: Adaptation across different task domains

## Acknowledgments

- Built on top of PyTorch and Transformers
- Inspired by LoRA and hypernetwork research
- Thanks to the open-source AI community 