#!/usr/bin/env python3
"""
Installation test script for Drag-and-Drop LLM system.
Run this script to verify that all components are properly installed and working.
"""

import sys
import torch
import traceback

def test_imports():
    """Test that all components can be imported."""
    print("Testing imports...")
    
    try:
        from dnd_llm import DragAndDropLLM
        print("✓ DragAndDropLLM imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DragAndDropLLM: {e}")
        return False
    
    try:
        from dnd_llm.models import (
            SentenceBERTEncoder, 
            QwenLoRALayer,
            CascadedHyperConvolutionalDecoder,
            HyperConvolutionalBlock
        )
        print("✓ All model components imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import model components: {e}")
        return False
    
    try:
        from dnd_llm.training import DnDTrainer, DatasetManager
        print("✓ Training components imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import training components: {e}")
        return False
    
    try:
        from dnd_llm.evaluation import DnDEvaluator
        print("✓ Evaluation components imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluation components: {e}")
        return False
    
    return True

def test_model_instantiation():
    """Test that the main model can be instantiated."""
    print("\nTesting model instantiation...")
    
    try:
        from dnd_llm import DragAndDropLLM
        
        # Test with minimal configuration
        model = DragAndDropLLM(
            foundation_model="Qwen/Qwen2.5-0.5B",
            lora_rank=8,
            lora_alpha=16.0,
            load_pretrained=False  # Don't load pretrained for testing
        )
        print("✓ DragAndDropLLM instantiated successfully")
        
        # Test basic forward pass
        test_prompts = ["This is a test prompt"]
        with torch.no_grad():
            generated_params = model(test_prompts)
        print("✓ Model forward pass completed successfully")
        print(f"✓ Generated parameters for {len(generated_params)} layers")
        
        return True
        
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual model components."""
    print("\nTesting individual components...")
    
    try:
        from dnd_llm.models import SentenceBERTEncoder
        encoder = SentenceBERTEncoder()
        test_texts = ["Test sentence"]
        embeddings = encoder(test_texts)
        print(f"✓ SentenceBERTEncoder: output shape {embeddings.shape}")
    except Exception as e:
        print(f"✗ SentenceBERTEncoder failed: {e}")
        return False
    
    try:
        from dnd_llm.models import CascadedHyperConvolutionalDecoder
        decoder = CascadedHyperConvolutionalDecoder(
            input_dim=384,
            output_dims=[100, 100]
        )
        test_input = torch.randn(1, 384)
        output = decoder(test_input)
        print(f"✓ CascadedHyperConvolutionalDecoder: output keys {list(output.keys())}")
    except Exception as e:
        print(f"✗ CascadedHyperConvolutionalDecoder failed: {e}")
        return False
    
    try:
        from dnd_llm.models import QwenLoRALayer
        lora_layer = QwenLoRALayer(
            input_dim=1024,
            output_dim=1024,
            rank=8,
            alpha=16.0
        )
        test_input = torch.randn(1, 10, 1024)
        output = lora_layer(test_input)
        print(f"✓ QwenLoRALayer: output shape {output.shape}")
    except Exception as e:
        print(f"✗ QwenLoRALayer failed: {e}")
        return False
    
    return True

def test_dataset_loading():
    """Test dataset loading functionality."""
    print("\nTesting dataset loading...")
    
    try:
        from dnd_llm.training import DatasetManager
        
        # Test loading a small sample
        common_sense_data = DatasetManager.load_common_sense_datasets(max_samples=10)
        print(f"✓ Loaded {len(common_sense_data)} common sense datasets")
        
        coding_data = DatasetManager.load_coding_datasets(max_samples=10) 
        print(f"✓ Loaded {len(coding_data)} coding datasets")
        
        math_data = DatasetManager.load_math_datasets(max_samples=10)
        print(f"✓ Loaded {len(math_data)} math datasets")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("DRAG-AND-DROP LLM INSTALLATION TEST")
    print("=" * 60)
    
    # System info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print("-" * 60)
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Model Instantiation", test_model_instantiation),
        ("Individual Components", test_individual_components),
        ("Dataset Loading", test_dataset_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:25} {status}")
        if success:
            passed += 1
    
    print("-" * 60)
    print(f"Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Installation is working correctly.")
        return 0
    else:
        print(f"\n❌ {len(results) - passed} tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 