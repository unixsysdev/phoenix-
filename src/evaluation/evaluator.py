"""
Evaluation scripts for multi-head LoRA model.
"""

import torch
import argparse
import time
import ast
import subprocess
import tempfile
import os
import json
import sys
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.generator import CodeGenerator
from src.utils.length_predictor import LengthPredictor


@dataclass
class EvaluationResult:
    """Result of code evaluation."""
    success: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    output: Optional[str] = None
    syntax_valid: bool = False
    imports_valid: bool = False
    runtime_valid: bool = False


class CodeEvaluator:
    """Evaluator for generated code."""
    
    def __init__(self, tokenizer: AutoTokenizer, timeout: int = 10):
        """
        Initialize code evaluator.
        
        Args:
            tokenizer: Tokenizer for code processing
            timeout: Timeout for code execution in seconds
        """
        self.tokenizer = tokenizer
        self.timeout = timeout
    
    def check_syntax(self, code: str) -> bool:
        """
        Check if code has valid Python syntax.
        
        Args:
            code: Code to check
            
        Returns:
            True if syntax is valid
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def check_imports(self, code: str) -> bool:
        """
        Check if all imports in code are valid.
        
        Args:
            code: Code to check
            
        Returns:
            True if imports are valid
        """
        try:
            # Extract imports
            tree = ast.parse(code)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Try to import each module
            for module in set(imports):
                try:
                    __import__(module)
                except ImportError:
                    return False
            
            return True
        except:
            return False
    
    def execute_code(self, code: str, test_cases: Optional[List[Dict]] = None) -> EvaluationResult:
        """
        Execute code and return result.
        
        Args:
            code: Code to execute
            test_cases: Optional test cases to run
            
        Returns:
            EvaluationResult
        """
        result = EvaluationResult(success=False)
        
        # Check syntax
        result.syntax_valid = self.check_syntax(code)
        if not result.syntax_valid:
            result.error_message = "Syntax error"
            return result
        
        # Check imports
        result.imports_valid = self.check_imports(code)
        if not result.imports_valid:
            result.error_message = "Import error"
            return result
        
        # Execute code
        try:
            import time
            start_time = time.time()
            
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run the code
            try:
                process = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                result.output = process.stdout
                result.error_message = process.stderr if process.stderr else None
                result.success = process.returncode == 0
                result.runtime_valid = result.success
                
            except subprocess.TimeoutExpired:
                result.error_message = "Execution timeout"
                result.success = False
            finally:
                # Clean up
                os.unlink(temp_file)
            
            end_time = time.time()
            result.execution_time = end_time - start_time
            
        except Exception as e:
            result.error_message = str(e)
            result.success = False
        
        return result
    
    def evaluate_function(self, code: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate a function against test cases.
        
        Args:
            code: Function code
            test_cases: List of test cases
            
        Returns:
            Dictionary with evaluation results
        """
        results = []
        passed = 0
        
        for i, test_case in enumerate(test_cases):
            # Create test script
            test_script = f"{code}\n\n"
            test_script += f"# Test case {i+1}\n"
            test_script += f"try:\n"
            test_script += f"    result = {test_case['call']}\n"
            test_script += f"    expected = {test_case['expected']}\n"
            test_script += f"    assert result == expected, f'Expected {{expected}}, got {{result}}'\n"
            test_script += f"    print('PASS')\n"
            test_script += f"except Exception as e:\n"
            test_script += f"    print(f'FAIL: {{e}}')\n"
            
            # Execute test
            result = self.execute_code(test_script)
            
            test_passed = result.success and result.output and "PASS" in result.output
            if test_passed:
                passed += 1
            
            results.append({
                "test_case": test_case,
                "passed": test_passed,
                "result": result
            })
        
        return {
            "total_tests": len(test_cases),
            "passed": passed,
            "pass_rate": passed / len(test_cases) if test_cases else 0,
            "results": results
        }


class ModelEvaluator:
    """Evaluator for the multi-head model."""
    
    def __init__(self, model_path: str, adapter_path: str, config_path: str = None):
        """
        Initialize model evaluator.
        
        Args:
            model_path: Path to base model
            adapter_path: Path to LoRA adapters
            config_path: Path to LoRA configuration
        """
        self.generator = CodeGenerator(model_path, config_path)
        self.generator.load_adapters(adapter_path)
        self.code_evaluator = CodeEvaluator(self.generator.tokenizer)
        self.length_predictor = LengthPredictor(self.generator.tokenizer)
    
    def evaluate_ar_generation(self, prompts: List[str], max_new_tokens: int = 128) -> Dict[str, Any]:
        """
        Evaluate AR generation.
        
        Args:
            prompts: List of prompts
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with evaluation results
        """
        results = []
        
        for prompt in prompts:
            # Generate code
            generated = self.generator.generate_scaffold(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.2
            )
            
            # Evaluate
            eval_result = self.code_evaluator.execute_code(generated)
            
            results.append({
                "prompt": prompt,
                "generated": generated,
                "evaluation": eval_result
            })
        
        # Calculate metrics
        syntax_valid = sum(1 for r in results if r["evaluation"].syntax_valid)
        runtime_valid = sum(1 for r in results if r["evaluation"].runtime_valid)
        
        return {
            "total_prompts": len(prompts),
            "syntax_valid": syntax_valid,
            "syntax_valid_rate": syntax_valid / len(prompts),
            "runtime_valid": runtime_valid,
            "runtime_valid_rate": runtime_valid / len(prompts),
            "results": results
        }
    
    def evaluate_diffusion_generation(self, prompts: List[str], steps: int = 50) -> Dict[str, Any]:
        """
        Evaluate diffusion generation.
        
        Args:
            prompts: List of prompts with mask tokens
            steps: Number of diffusion steps
            
        Returns:
            Dictionary with evaluation results
        """
        results = []
        
        for prompt in prompts:
            # Generate code
            generated = self.generator.generate_infill(
                prompt,
                steps=steps,
                temperature=0.7
            )
            
            # Evaluate
            eval_result = self.code_evaluator.execute_code(generated)
            
            results.append({
                "prompt": prompt,
                "generated": generated,
                "evaluation": eval_result
            })
        
        # Calculate metrics
        syntax_valid = sum(1 for r in results if r["evaluation"].syntax_valid)
        runtime_valid = sum(1 for r in results if r["evaluation"].runtime_valid)
        
        return {
            "total_prompts": len(prompts),
            "syntax_valid": syntax_valid,
            "syntax_valid_rate": syntax_valid / len(prompts),
            "runtime_valid": runtime_valid,
            "runtime_valid_rate": runtime_valid / len(prompts),
            "results": results
        }
    
    def evaluate_length_prediction(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Evaluate length prediction.
        
        Args:
            prompts: List of prompts
            
        Returns:
            Dictionary with evaluation results
        """
        results = []
        
        for prompt in prompts:
            # Predict length
            predicted_length = self.generator.predict_length(prompt)
            
            # Get actual length (this would need ground truth)
            # For now, we'll just analyze the prediction
            results.append({
                "prompt": prompt,
                "predicted_length": predicted_length,
                "predicted_bucket": self.length_predictor.length_to_bucket(predicted_length)
            })
        
        # Analyze predictions
        predicted_lengths = [r["predicted_length"] for r in results]
        
        return {
            "total_prompts": len(prompts),
            "mean_predicted_length": sum(predicted_lengths) / len(predicted_lengths),
            "min_predicted_length": min(predicted_lengths),
            "max_predicted_length": max(predicted_lengths),
            "results": results
        }
    
    def evaluate_function_generation(self, signatures: List[Dict], steps: int = 50) -> Dict[str, Any]:
        """
        Evaluate complete function generation.
        
        Args:
            signatures: List of function signatures with test cases
            steps: Number of diffusion steps
            
        Returns:
            Dictionary with evaluation results
        """
        results = []
        
        for sig_data in signatures:
            signature = sig_data["signature"]
            docstring = sig_data.get("docstring", "")
            test_cases = sig_data.get("test_cases", [])
            
            # Generate function
            generated = self.generator.generate_function(
                signature,
                docstring,
                steps=steps,
                temperature=0.7
            )
            
            # Evaluate
            eval_result = {}
            
            # Basic checks
            eval_result["syntax_valid"] = self.code_evaluator.check_syntax(generated)
            
            # Test cases
            if test_cases and eval_result["syntax_valid"]:
                test_result = self.code_evaluator.evaluate_function(generated, test_cases)
                eval_result["test_result"] = test_result
            
            results.append({
                "signature": signature,
                "docstring": docstring,
                "generated": generated,
                "evaluation": eval_result
            })
        
        # Calculate metrics
        syntax_valid = sum(1 for r in results if r["evaluation"].get("syntax_valid", False))
        test_passed = sum(1 for r in results if r["evaluation"].get("test_result", {}).get("pass_rate", 0) == 1.0)
        
        metrics = {
            "total_functions": len(signatures),
            "syntax_valid": syntax_valid,
            "syntax_valid_rate": syntax_valid / len(signatures),
            "test_passed": test_passed,
            "test_pass_rate": test_passed / len(signatures) if signatures else 0,
            "results": results
        }
        
        # Calculate average test pass rate
        pass_rates = [r["evaluation"].get("test_result", {}).get("pass_rate", 0) for r in results]
        if pass_rates:
            metrics["avg_test_pass_rate"] = sum(pass_rates) / len(pass_rates)
        
        return metrics
    
    def run_comprehensive_evaluation(self, test_data: Dict[str, List]) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on all heads.
        
        Args:
            test_data: Dictionary with test data for each head
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        results = {}
        
        # AR generation
        if "ar_prompts" in test_data:
            results["ar_generation"] = self.evaluate_ar_generation(test_data["ar_prompts"])
        
        # Diffusion generation
        if "diffusion_prompts" in test_data:
            results["diffusion_generation"] = self.evaluate_diffusion_generation(test_data["diffusion_prompts"])
        
        # Length prediction
        if "length_prompts" in test_data:
            results["length_prediction"] = self.evaluate_length_prediction(test_data["length_prompts"])
        
        # Function generation
        if "function_signatures" in test_data:
            results["function_generation"] = self.evaluate_function_generation(test_data["function_signatures"])
        
        # Overall summary
        summary = {
            "evaluation_timestamp": time.time(),
            "model_path": getattr(self.generator, "model_path", None),
            "adapter_path": getattr(self.generator, "adapter_path", None),
            "results": results
        }
        
        return summary


def create_test_data() -> Dict[str, List]:
    """Create sample test data for evaluation."""
    return {
        "ar_prompts": [
            "def calculate_sum(numbers):",
            "class DataProcessor:",
            "import numpy as np\ndef process_data(data):",
            "def quicksort(arr):",
            "def binary_search(arr, target):"
        ],
        "diffusion_prompts": [
            "def calculate_sum(numbers):\n    <|mask|>\n    return result",
            "class DataProcessor:\n    <|mask|>\n    def process(self):",
            "def quicksort(arr):\n    <|mask|>\n    return sorted_arr",
            "def binary_search(arr, target):\n    <|mask|>\n    return index"
        ],
        "length_prompts": [
            "def calculate_sum(numbers):",
            "class DataProcessor:",
            "def quicksort(arr):",
            "def binary_search(arr, target):",
            "def process_large_dataset(data):"
        ],
        "function_signatures": [
            {
                "signature": "def calculate_sum(numbers):",
                "docstring": "Calculate the sum of a list of numbers",
                "test_cases": [
                    {"call": "calculate_sum([1, 2, 3])", "expected": 6},
                    {"call": "calculate_sum([])", "expected": 0},
                    {"call": "calculate_sum([-1, 1])", "expected": 0}
                ]
            },
            {
                "signature": "def quicksort(arr):",
                "docstring": "Sort an array using quicksort algorithm",
                "test_cases": [
                    {"call": "quicksort([3, 1, 4, 1, 5])", "expected": [1, 1, 3, 4, 5]},
                    {"call": "quicksort([])", "expected": []},
                    {"call": "quicksort([1])", "expected": [1]}
                ]
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-head LoRA model")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
                       help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True,
                       help="Path to LoRA adapters")
    parser.add_argument("--config_path", type=str, default="configs/lora_configs.yaml",
                       help="Path to LoRA configuration")
    parser.add_argument("--test_data_file", type=str,
                       help="Path to test data file")
    parser.add_argument("--output_file", type=str,
                       help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Load or create test data
    if args.test_data_file and os.path.exists(args.test_data_file):
        with open(args.test_data_file, 'r') as f:
            test_data = json.load(f)
    else:
        test_data = create_test_data()
        print("Using default test data")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.adapter_path, args.config_path)
    
    # Run evaluation
    print("Running comprehensive evaluation...")
    results = evaluator.run_comprehensive_evaluation(test_data)
    
    # Print summary
    print("\nEvaluation Results:")
    print("=" * 50)
    
    for head, result in results.items():
        print(f"\n{head.upper()}:")
        if "syntax_valid_rate" in result:
            print(f"  Syntax valid rate: {result['syntax_valid_rate']:.2%}")
        if "runtime_valid_rate" in result:
            print(f"  Runtime valid rate: {result['runtime_valid_rate']:.2%}")
        if "test_pass_rate" in result:
            print(f"  Test pass rate: {result['test_pass_rate']:.2%}")
        if "avg_test_pass_rate" in result:
            print(f"  Avg test pass rate: {result['avg_test_pass_rate']:.2%}")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
