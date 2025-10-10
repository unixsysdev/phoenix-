"""
Dataset classes for multi-head training.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Any, Tuple
import json
import random
import os
from ..model.diffusion_utils import apply_length_bucket


class CodeDataset(Dataset):
    """Dataset for code samples with multi-task support."""
    
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 2048, 
                 mode: str = "train", length_buckets: int = 20, max_code_length: int = 1000):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.length_buckets = length_buckets
        self.max_code_length = max_code_length
        
        # Load data
        self.samples = self._load_data()
        
        # Filter by length
        self.samples = [s for s in self.samples if len(s.get("code", "")) > 10]
        
        print(f"Loaded {len(self.samples)} samples for {mode} mode")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file or directory."""
        if os.path.isfile(self.data_path):
            if self.data_path.endswith('.json'):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif self.data_path.endswith('.jsonl'):
                samples = []
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
                return samples
        elif os.path.isdir(self.data_path):
            samples = []
            for filename in os.listdir(self.data_path):
                if filename.endswith('.json') or filename.endswith('.jsonl'):
                    file_path = os.path.join(self.data_path, filename)
                    if filename.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            samples.extend(json.load(f))
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    samples.append(json.loads(line))
            return samples
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Extract code and metadata
        code = sample.get("code", "")
        prompt = sample.get("prompt", "")
        language = sample.get("language", "python")
        
        # Create different training samples for each head
        result = {}
        
        # AR training sample (prompt -> code)
        if prompt and code:
            ar_text = f"{prompt}\n{code}"
            ar_encoding = self.tokenizer(
                ar_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # For AR, we want to predict the code part given the prompt
            prompt_encoding = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Create labels (prompt tokens are masked, code tokens are predicted)
            prompt_length = prompt_encoding["attention_mask"].sum().item()
            labels = ar_encoding["input_ids"].clone()
            labels[:, :prompt_length] = -100  # Mask prompt tokens
            
            result.update({
                "ar_input_ids": ar_encoding["input_ids"].squeeze(0),
                "ar_attention_mask": ar_encoding["attention_mask"].squeeze(0),
                "ar_labels": labels.squeeze(0)
            })
        
        # Diffusion training sample (masked code -> full code)
        if code:
            diff_encoding = self.tokenizer(
                code,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            result.update({
                "diff_input_ids": diff_encoding["input_ids"].squeeze(0),
                "diff_attention_mask": diff_encoding["attention_mask"].squeeze(0),
                "diff_labels": diff_encoding["input_ids"].squeeze(0)
            })
        
        # Length prediction sample (prompt -> length bucket)
        if prompt and code:
            length_encoding = self.tokenizer(
                prompt,
                max_length=self.max_length // 2,  # Shorter for length prediction
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Calculate length bucket for the code
            code_tokens = self.tokenizer.encode(code, add_special_tokens=False)
            code_length = min(len(code_tokens), self.max_code_length)
            length_bucket = apply_length_bucket(code_length, self.length_buckets, self.max_code_length)
            
            result.update({
                "length_input_ids": length_encoding["input_ids"].squeeze(0),
                "length_attention_mask": length_encoding["attention_mask"].squeeze(0),
                "length_labels": torch.tensor(length_bucket, dtype=torch.long)
            })
        
        return result


class SyntheticCodeDataset(Dataset):
    """Generate synthetic code samples for testing."""
    
    def __init__(self, tokenizer: AutoTokenizer, num_samples: int = 1000, 
                 max_length: int = 512, length_buckets: int = 20):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        self.length_buckets = length_buckets
        
        # Code templates
        self.templates = [
            {
                "prompt": "def {function_name}({params}):",
                "code": "    '''{docstring}'''\n    {implementation}\n    return {return_value}",
                "language": "python"
            },
            {
                "prompt": "class {class_name}:",
                "code": "    '''{docstring}'''\n    def __init__(self{init_params}):\n{init_body}\n    def {method_name}({method_params}):\n{method_body}",
                "language": "python"
            },
            {
                "prompt": "def {function_name}({params}) -> {return_type}:",
                "code": "    '''{docstring}'''\n    {implementation}\n    return {return_value}",
                "language": "python"
            }
        ]
        
        # Function names and implementations
        self.function_names = ["calculate_sum", "find_max", "sort_list", "filter_items", "transform_data"]
        self.implementations = [
            "result = sum(items)",
            "max_val = max(items) if items else None",
            "sorted_items = sorted(items)",
            "filtered = [item for item in items if condition(item)]",
            "transformed = [transform(item) for item in items]"
        ]
        
        # Generate samples
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate synthetic code samples."""
        samples = []
        
        for i in range(self.num_samples):
            template = random.choice(self.templates)
            function_name = random.choice(self.function_names)
            implementation = random.choice(self.implementations)
            
            # Fill template
            prompt = template["prompt"].format(
                function_name=function_name,
                params="items",
                class_name=f"TestClass{i % 10}",
                return_type="List[int]",
                init_params=", self",
                method_name="process",
                method_params="self, data"
            )
            
            code = template["code"].format(
                docstring=f"Generated function {function_name}",
                implementation=implementation,
                return_value="result",
                init_body="        self.data = data",
                method_body="        return self.process_data(data)"
            )
            
            samples.append({
                "prompt": prompt,
                "code": code,
                "language": template["language"]
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Create different training samples for each head
        result = {}
        
        # AR training sample
        ar_text = f"{sample['prompt']}\n{sample['code']}"
        ar_encoding = self.tokenizer(
            ar_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        prompt_encoding = self.tokenizer(
            sample['prompt'],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels
        prompt_length = prompt_encoding["attention_mask"].sum().item()
        labels = ar_encoding["input_ids"].clone()
        labels[:, :prompt_length] = -100
        
        result.update({
            "ar_input_ids": ar_encoding["input_ids"].squeeze(0),
            "ar_attention_mask": ar_encoding["attention_mask"].squeeze(0),
            "ar_labels": labels.squeeze(0)
        })
        
        # Diffusion training sample
        diff_encoding = self.tokenizer(
            sample['code'],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        result.update({
            "diff_input_ids": diff_encoding["input_ids"].squeeze(0),
            "diff_attention_mask": diff_encoding["attention_mask"].squeeze(0),
            "diff_labels": diff_encoding["input_ids"].squeeze(0)
        })
        
        # Length prediction sample
        length_encoding = self.tokenizer(
            sample['prompt'],
            max_length=self.max_length // 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        code_tokens = self.tokenizer.encode(sample['code'], add_special_tokens=False)
        code_length = min(len(code_tokens), 500)
        length_bucket = apply_length_bucket(code_length, self.length_buckets, 500)
        
        result.update({
            "length_input_ids": length_encoding["input_ids"].squeeze(0),
            "length_attention_mask": length_encoding["attention_mask"].squeeze(0),
            "length_labels": torch.tensor(length_bucket, dtype=torch.long)
        })
        
        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching samples."""
    result = {}
    
    # Get all keys from the first batch item
    keys = batch[0].keys()
    
    for key in keys:
        if key in batch[0]:
            # Stack tensors for this key
            result[key] = torch.stack([item[key] for item in batch if key in item])
    
    return result


def create_dataloader(dataset: Dataset, batch_size: int = 8, shuffle: bool = True, 
                     num_workers: int = 4) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
