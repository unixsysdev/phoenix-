"""
Basic usage examples for the multi-head LoRA model.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.generator import CodeGenerator


def example_ar_generation():
    """Example of AR (autoregressive) generation."""
    print("=" * 50)
    print("AR Generation Example")
    print("=" * 50)
    
    # Initialize generator
    generator = CodeGenerator(
        model_path="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        config_path="configs/lora_configs.yaml"
    )
    
    # Load adapters (replace with your actual path)
    # generator.load_adapters("logs/checkpoint-40000")
    
    # Generate scaffold
    prompt = "def calculate_factorial(n):"
    scaffold = generator.generate_scaffold(
        prompt,
        max_new_tokens=64,
        temperature=0.2
    )
    
    print(f"Prompt: {prompt}")
    print(f"Generated scaffold:\n{scaffold}")
    print()


def example_diffusion_infill():
    """Example of diffusion infilling."""
    print("=" * 50)
    print("Diffusion Infill Example")
    print("=" * 50)
    
    # Initialize generator
    generator = CodeGenerator(
        model_path="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        config_path="configs/lora_configs.yaml"
    )
    
    # Load adapters (replace with your actual path)
    # generator.load_adapters("logs/checkpoint-40000")
    
    # Generate infill
    prompt = "def quicksort(arr):\n    '''Sort array using quicksort algorithm'''\n    <|mask|>\n    return sorted_arr"
    infill = generator.generate_infill(
        prompt,
        steps=50,
        temperature=0.7
    )
    
    print(f"Prompt with mask:\n{prompt}")
    print(f"Generated infill:\n{infill}")
    print()


def example_function_generation():
    """Example of complete function generation."""
    print("=" * 50)
    print("Function Generation Example")
    print("=" * 50)
    
    # Initialize generator
    generator = CodeGenerator(
        model_path="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        config_path="configs/lora_configs.yaml"
    )
    
    # Load adapters (replace with your actual path)
    # generator.load_adapters("logs/checkpoint-40000")
    
    # Generate function
    signature = "def binary_search(arr, target):"
    docstring = "Binary search implementation"
    
    function_code = generator.generate_function(
        signature,
        docstring,
        steps=50,
        temperature=0.7
    )
    
    print(f"Signature: {signature}")
    print(f"Docstring: {docstring}")
    print(f"Generated function:\n{function_code}")
    print()


def example_multiple_candidates():
    """Example of generating multiple candidates."""
    print("=" * 50)
    print("Multiple Candidates Example")
    print("=" * 50)
    
    # Initialize generator
    generator = CodeGenerator(
        model_path="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        config_path="configs/lora_configs.yaml"
    )
    
    # Load adapters (replace with your actual path)
    # generator.load_adapters("logs/checkpoint-40000")
    
    # Generate multiple candidates
    signature = "def process_data(data):"
    docstring = "Process input data and return results"
    
    candidates = generator.generate_multiple_candidates(
        signature,
        docstring,
        num_candidates=3,
        steps=50,
        temperature=0.7
    )
    
    print(f"Signature: {signature}")
    print(f"Docstring: {docstring}")
    print(f"Generated {len(candidates)} candidates:")
    
    for i, candidate in enumerate(candidates):
        print(f"\n--- Candidate {i+1} ---")
        print(candidate)
        
        # Evaluate candidate
        metrics = generator.evaluate_candidate(candidate, signature)
        print("Metrics:", ", ".join(f"{k}={v}" for k, v in metrics.items()))
    
    print()


def example_length_prediction():
    """Example of length prediction."""
    print("=" * 50)
    print("Length Prediction Example")
    print("=" * 50)
    
    # Initialize generator
    generator = CodeGenerator(
        model_path="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        config_path="configs/lora_configs.yaml"
    )
    
    # Load adapters (replace with your actual path)
    # generator.load_adapters("logs/checkpoint-40000")
    
    # Test different prompts
    prompts = [
        "def simple_function():",
        "def complex_algorithm(arr, target, options=None):",
        "class DataProcessor:",
        "def process_large_dataset(data, chunk_size=1000):"
    ]
    
    for prompt in prompts:
        predicted_length = generator.predict_length(prompt)
        print(f"Prompt: {prompt}")
        print(f"Predicted length: {predicted_length} tokens")
        print()


def main():
    """Run all examples."""
    print("Qwen3 Coder 30B MoE - Multi-head LoRA Examples")
    print("=" * 60)
    print()
    
    print("Note: These examples require trained LoRA adapters.")
    print("Uncomment the generator.load_adapters() lines with your adapter path.")
    print()
    
    # Run examples
    example_ar_generation()
    example_diffusion_infill()
    example_function_generation()
    example_multiple_candidates()
    example_length_prediction()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()