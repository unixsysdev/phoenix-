"""
Script to prepare training data for multi-head LoRA training.
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any
from transformers import AutoTokenizer


def load_finecode_samples(data_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Load samples from FineCode dataset."""
    samples = []
    
    if os.path.isfile(data_path):
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = data.get('samples', [])
        elif data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
    elif os.path.isdir(data_path):
        for filename in os.listdir(data_path):
            if filename.endswith('.json') or filename.endswith('.jsonl'):
                file_path = os.path.join(data_path, filename)
                if filename.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            samples.extend(file_data)
                        else:
                            samples.extend(file_data.get('samples', []))
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                samples.append(json.loads(line))
    
    # Limit samples if specified
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
    
    return samples


def create_synthetic_samples(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Create synthetic code samples for testing."""
    samples = []
    
    # Function templates
    function_templates = [
        {
            "prompt": "def {name}({params}):",
            "code": "    '''{docstring}'''\n    {body}\n    return {return_val}",
            "params": ["arr", "items", "data", "lst", "values"],
            "bodies": [
                "return sum({param})",
                "return max({param}) if {param} else None",
                "return sorted({param})",
                "return [x for x in {param} if x > 0]",
                "return {param}[::-1]"
            ],
            "return_vals": ["result", "max_val", "sorted_list", "filtered", "reversed"]
        },
        {
            "prompt": "class {name}:",
            "code": "    '''{docstring}'''\n    def __init__(self{init_params}):\n{init_body}\n    def {method}({method_params}):\n{method_body}",
            "params": ["", ", value", ", data=None", ", items=[]"],
            "bodies": [
                "        self.value = value",
                "        self.data = data or {}",
                "        self.items = items",
                "        self.config = {}"
            ],
            "methods": ["process", "transform", "calculate", "validate"],
            "method_bodies": [
                "        return self.value * 2",
                "        return {param}.upper() if isinstance({param}, str) else {param}",
                "        return len(self.items)",
                "        return self.data.get(key, default)"
            ]
        }
    ]
    
    # Generate samples
    for i in range(num_samples):
        template = random.choice(function_templates)
        
        if "prompt" in template and "def" in template["prompt"]:
            # Function sample
            name = random.choice([
                "calculate_sum", "find_maximum", "sort_array", "filter_data", 
                "transform_values", "validate_input", "process_items", "compute_result"
            ])
            params = random.choice(template["params"])
            body = random.choice(template["bodies"]).format(param=params)
            return_val = random.choice(template["return_vals"])
            
            prompt = template["prompt"].format(name=name, params=params)
            code = template["code"].format(
                docstring=f"Function to {name.replace('_', ' ')}",
                body=body,
                return_val=return_val
            )
        else:
            # Class sample
            name = f"{''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(8)).capitalize()}Class"
            init_params = random.choice(template["params"])
            init_body = random.choice(template["bodies"])
            method = random.choice(template["methods"])
            method_body = random.choice(template["method_bodies"])
            
            prompt = template["prompt"].format(name=name)
            code = template["code"].format(
                docstring=f"Class for {name.lower()}",
                init_params=init_params,
                init_body=init_body,
                method=method,
                method_params="self, data",
                method_body=method_body
            )
        
        samples.append({
            "prompt": prompt,
            "code": code,
            "language": "python"
        })
    
    return samples


def process_samples(samples: List[Dict[str, Any]], tokenizer: AutoTokenizer, 
                   max_length: int = 2048) -> List[Dict[str, Any]]:
    """Process samples to ensure they fit within max_length."""
    processed_samples = []
    
    for sample in samples:
        # Extract code and prompt
        code = sample.get("code", "")
        prompt = sample.get("prompt", "")
        
        # Tokenize and check length
        full_text = f"{prompt}\n{code}" if prompt else code
        tokens = tokenizer.encode(full_text, add_special_tokens=True)
        
        if len(tokens) <= max_length:
            processed_samples.append(sample)
    
    return processed_samples


def split_data(samples: List[Dict[str, Any]], train_ratio: float = 0.9) -> tuple:
    """Split data into train and validation sets."""
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    return samples[:split_idx], samples[split_idx:]


def save_samples(samples: List[Dict[str, Any]], output_path: str):
    """Save samples to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for multi-head LoRA training")
    parser.add_argument("--input_path", type=str, help="Path to input data (file or directory)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--synthetic_samples", type=int, default=1000, help="Number of synthetic samples to generate")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
                       help="Model name for tokenizer")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train/validation split ratio")
    
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Load or create samples
    all_samples = []
    
    if args.input_path:
        print(f"Loading samples from {args.input_path}...")
        real_samples = load_finecode_samples(args.input_path, args.max_samples)
        print(f"Loaded {len(real_samples)} real samples")
        all_samples.extend(real_samples)
    
    # Generate synthetic samples
    print(f"Generating {args.synthetic_samples} synthetic samples...")
    synthetic_samples = create_synthetic_samples(args.synthetic_samples)
    print(f"Generated {len(synthetic_samples)} synthetic samples")
    all_samples.extend(synthetic_samples)
    
    # Process samples
    print(f"Processing samples...")
    processed_samples = process_samples(all_samples, tokenizer, args.max_length)
    print(f"Processed {len(processed_samples)} samples (max_length={args.max_length})")
    
    # Split data
    train_samples, val_samples = split_data(processed_samples, args.train_ratio)
    print(f"Split into {len(train_samples)} train and {len(val_samples)} validation samples")
    
    # Save data
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.output_dir, "train_samples.json")
    val_path = os.path.join(args.output_dir, "val_samples.json")
    
    save_samples(train_samples, train_path)
    save_samples(val_samples, val_path)
    
    print(f"Saved training data to {train_path}")
    print(f"Saved validation data to {val_path}")
    
    # Save metadata
    metadata = {
        "total_samples": len(processed_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "max_length": args.max_length,
        "model_name": args.model_name,
        "synthetic_samples": len(synthetic_samples)
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()