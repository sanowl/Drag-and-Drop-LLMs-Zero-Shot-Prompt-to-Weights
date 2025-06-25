"""
Evaluation metrics for Drag-and-Drop LLM system.
Implements metrics mentioned in the paper for comprehensive evaluation.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Union
import re
import ast
import subprocess
import tempfile
import os
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def compute_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Compute exact match accuracy for classification tasks."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    correct = 0
    for pred, target in zip(predictions, targets):
        if isinstance(pred, str) and isinstance(target, str):
            # Normalize both strings for comparison
            pred_clean = pred.strip().lower()
            target_clean = target.strip().lower()
            if pred_clean == target_clean:
                correct += 1
        elif pred == target:
            correct += 1
    
    return correct / len(predictions) if len(predictions) > 0 else 0.0


def extract_answer_choice(text: str) -> str:
    """Extract answer choice (A, B, C, D) from model output."""
    # Look for patterns like "Answer: A", "The answer is B", etc.
    patterns = [
        r'[Aa]nswer\s*:?\s*([A-D])',
        r'[Tt]he\s+answer\s+is\s+([A-D])',
        r'\b([A-D])\b(?=\s*(?:\.|$))',
        r'^\s*([A-D])\s*[:\.]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.strip())
        if match:
            return match.group(1).upper()
    
    # Fallback: look for first occurrence of A, B, C, or D
    for char in text.upper():
        if char in 'ABCD':
            return char
    
    return 'A'  # Default fallback


def compute_multiple_choice_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Compute accuracy for multiple choice questions."""
    extracted_preds = [extract_answer_choice(pred) for pred in predictions]
    extracted_targets = [extract_answer_choice(target) for target in targets]
    return compute_accuracy(extracted_preds, extracted_targets)


def extract_code_from_text(text: str) -> str:
    """Extract Python code from model output."""
    # Look for code blocks
    code_patterns = [
        r'```python\n(.*?)\n```',
        r'```\n(.*?)\n```',
        r'```(.*?)```',
    ]
    
    for pattern in code_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # If no code blocks, look for function definitions
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.strip().startswith('def ') or line.strip().startswith('class '):
            in_code = True
        if in_code:
            code_lines.append(line)
            # Stop at empty line or non-indented line after function
            if line.strip() == '' and code_lines[-2:] and not code_lines[-2].startswith(' '):
                break
    
    return '\n'.join(code_lines).strip() if code_lines else text.strip()


def execute_code_safely(code: str, test_cases: List[Dict]) -> Dict[str, Any]:
    """Safely execute code with test cases."""
    results = {
        'passed': 0,
        'total': len(test_cases),
        'errors': [],
        'success': False
    }
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # Execute each test case
        for i, test_case in enumerate(test_cases):
            try:
                # Prepare test code
                test_code = f"""
{code}

# Test case {i+1}
inputs = {test_case.get('inputs', [])}
expected = {test_case.get('expected')}

try:
    if hasattr(locals(), 'solution'):
        result = solution(*inputs)
    else:
        # Try to find the main function
        import inspect
        funcs = [obj for name, obj in locals().items() 
                if inspect.isfunction(obj) and not name.startswith('_')]
        if funcs:
            result = funcs[0](*inputs)
        else:
            result = None
    
    if result == expected:
        print("PASS")
    else:
        print(f"FAIL: got {{result}}, expected {{expected}}")
except Exception as e:
    print(f"ERROR: {{e}}")
"""
                
                # Execute with timeout
                proc = subprocess.run(
                    ['python', '-c', test_code],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if 'PASS' in proc.stdout:
                    results['passed'] += 1
                elif 'FAIL' in proc.stdout or 'ERROR' in proc.stdout:
                    results['errors'].append(f"Test {i+1}: {proc.stdout.strip()}")
                
            except subprocess.TimeoutExpired:
                results['errors'].append(f"Test {i+1}: Timeout")
            except Exception as e:
                results['errors'].append(f"Test {i+1}: {str(e)}")
        
        # Clean up
        os.unlink(temp_file)
        results['success'] = True
        
    except Exception as e:
        results['errors'].append(f"Code execution failed: {str(e)}")
    
    return results


def compute_pass_at_k(predictions: List[str], test_cases: List[List[Dict]], k: int = 1) -> float:
    """
    Compute pass@k metric for code generation.
    
    Args:
        predictions: List of generated code solutions
        test_cases: List of test cases for each problem
        k: Number of attempts to consider
    """
    if len(predictions) != len(test_cases):
        raise ValueError("Predictions and test_cases must have same length")
    
    passed_problems = 0
    total_problems = len(predictions)
    
    for i, (pred, tests) in enumerate(zip(predictions, test_cases)):
        code = extract_code_from_text(pred)
        if not code:
            continue
            
        # Execute code with test cases
        results = execute_code_safely(code, tests)
        
        # Check if all test cases passed
        if results['passed'] == results['total'] and results['total'] > 0:
            passed_problems += 1
    
    return passed_problems / total_problems if total_problems > 0 else 0.0


def extract_numerical_answer(text: str) -> Union[float, None]:
    """Extract numerical answer from math problem solution."""
    # Look for patterns like "Answer: 42", "The answer is 3.14", etc.
    patterns = [
        r'[Aa]nswer\s*:?\s*([+-]?\d*\.?\d+)',
        r'[Tt]he\s+answer\s+is\s+([+-]?\d*\.?\d+)',
        r'=\s*([+-]?\d*\.?\d+)\s*$',
        r'([+-]?\d*\.?\d+)\s*$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.strip())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # Look for any number in the text
    numbers = re.findall(r'([+-]?\d*\.?\d+)', text)
    if numbers:
        try:
            return float(numbers[-1])  # Take the last number
        except ValueError:
            pass
    
    return None


def compute_math_accuracy(predictions: List[str], targets: List[Union[str, float]]) -> float:
    """Compute accuracy for math problems."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    correct = 0
    for pred, target in zip(predictions, targets):
        pred_num = extract_numerical_answer(pred)
        
        if isinstance(target, str):
            target_num = extract_numerical_answer(target)
        else:
            target_num = float(target)
        
        if pred_num is not None and target_num is not None:
            # Allow small floating point differences
            if abs(pred_num - target_num) < 1e-6:
                correct += 1
    
    return correct / len(predictions) if len(predictions) > 0 else 0.0


def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score for text generation."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import word_tokenize
        
        smoothing = SmoothingFunction().method1
        scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())
            
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
        
    except ImportError:
        logger.warning("NLTK not available, using simple BLEU approximation")
        # Simple approximation without NLTK
        return _simple_bleu_score(predictions, references)


def _simple_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Simple BLEU approximation without external dependencies."""
    scores = []
    
    for pred, ref in zip(predictions, references):
        pred_words = pred.lower().split()
        ref_words = ref.lower().split()
        
        if not pred_words:
            scores.append(0.0)
            continue
        
        # Count matching words
        pred_counter = Counter(pred_words)
        ref_counter = Counter(ref_words)
        
        matches = sum((pred_counter & ref_counter).values())
        precision = matches / len(pred_words) if pred_words else 0.0
        
        # Simple length penalty
        bp = min(1.0, len(pred_words) / len(ref_words)) if ref_words else 0.0
        
        scores.append(bp * precision)
    
    return np.mean(scores) if scores else 0.0


def evaluate_dataset(predictions: List[str], targets: List[str], 
                    task_type: str, **kwargs) -> Dict[str, float]:
    """
    Evaluate predictions on a dataset based on task type.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        task_type: Type of task ('multiple_choice', 'code', 'math', 'text')
        **kwargs: Additional arguments (e.g., test_cases for code tasks)
    
    Returns:
        Dictionary of metric scores
    """
    results = {}
    
    if task_type == 'multiple_choice':
        results['accuracy'] = compute_multiple_choice_accuracy(predictions, targets)
    
    elif task_type == 'code':
        test_cases = kwargs.get('test_cases', [])
        if test_cases:
            for k in [1, 5, 10]:
                results[f'pass@{k}'] = compute_pass_at_k(predictions, test_cases, k)
        else:
            # Fallback to exact match if no test cases
            results['accuracy'] = compute_accuracy(predictions, targets)
    
    elif task_type == 'math':
        results['accuracy'] = compute_math_accuracy(predictions, targets)
    
    elif task_type == 'text':
        results['bleu'] = compute_bleu_score(predictions, targets)
        results['accuracy'] = compute_accuracy(predictions, targets)
    
    else:
        # Default to accuracy
        results['accuracy'] = compute_accuracy(predictions, targets)
    
    return results 