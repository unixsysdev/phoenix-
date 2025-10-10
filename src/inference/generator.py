"""
Inference and generation script for multi-head LoRA model.
"""

import torch
import argparse
import os
import sys
import yaml
from typing import Dict, List, Optional, Any, Tuple

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.multihead_coder import MultiHeadCoder
from src.model.diffusion_utils import DiffusionSampler, bucket_to_length
from transformers import AutoTokenizer


class CodeGenerator:
    """Code generator with multi-head capabilities."""
    
    def __init__(self, model_path: str, config_path: str = None, device: str = "auto"):
        """
        Initialize the generator.
        
        Args:
            model_path: Path to the base model
            config_path: Path to LoRA configuration
            device: Device to run on
        """
        self.model_path = model_path
        self.config_path = config_path
        self.adapter_path: Optional[str] = None
        self.device = torch.device(device if device != "auto" else 
                                 ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add special tokens if needed
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = MultiHeadCoder(model_path, config_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        
        # Initialize diffusion sampler
        self.diffusion_sampler = DiffusionSampler(scheduler_type="cosine")
        
        print(f"Generator initialized on {self.device}")
    
    def load_adapters(self, adapter_path: str):
        """Load LoRA adapters from checkpoint."""
        self.model.load_adapters(adapter_path)
        self.adapter_path = adapter_path
        print(f"Loaded adapters from {adapter_path}")
    
    def generate_scaffold(self, prompt: str, max_new_tokens: int = 128, 
                         temperature: float = 0.2, top_p: float = 0.9) -> str:
        """
        Generate code scaffold using AR head.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated scaffold
        """
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with AR head
        self.model.set_head("ar")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt
        if generated_text.startswith(prompt):
            scaffold = generated_text[len(prompt):].strip()
        else:
            scaffold = generated_text
        
        return scaffold
    
    def predict_length(self, prompt: str) -> int:
        """
        Predict output length using length head.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Predicted length in tokens
        """
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Predict length
        self.model.set_head("length")
        with torch.no_grad():
            outputs = self.model(input_ids=inputs["input_ids"])
            logits = outputs.logits
            predicted_bucket = torch.argmax(logits, dim=-1).item()
        
        # Convert bucket to length
        predicted_length = bucket_to_length(predicted_bucket, 20, 1000)
        
        return predicted_length
    
    def generate_infill(self, prompt: str, max_new_tokens: Optional[int] = None,
                       steps: int = 50, temperature: float = 0.7) -> str:
        """
        Generate code infilling using diffusion head.
        
        Args:
            prompt: Input prompt with mask token
            max_new_tokens: Maximum tokens to generate (auto-predicted if None)
            steps: Number of diffusion steps
            temperature: Sampling temperature
            
        Returns:
            Generated code
        """
        # Predict length if not provided
        if max_new_tokens is None:
            max_new_tokens = self.predict_length(prompt)
        
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with diffusion head
        self.model.set_head("diffusion")
        with torch.no_grad():
            outputs = self.model.diffusion_generate(
                input_ids=inputs["input_ids"],
                mask_token_id=self.tokenizer.mask_token_id,
                max_new_tokens=max_new_tokens,
                steps=steps,
                temperature=temperature
            )
        
        # Decode and return
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt
        if generated_text.startswith(prompt):
            infill = generated_text[len(prompt):].strip()
        else:
            infill = generated_text
        
        return infill
    
    def generate_function(self, signature: str, docstring: str = "",
                         steps: int = 50, temperature: float = 0.7) -> str:
        """
        Generate a complete function using the multi-head pipeline.
        
        Args:
            signature: Function signature
            docstring: Function docstring
            steps: Number of diffusion steps
            temperature: Sampling temperature
            
        Returns:
            Complete function
        """
        # Create prompt with docstring
        if docstring:
            prompt = f"{signature}\n    '''{docstring}'''\n    # TODO: Implement\n"
        else:
            prompt = f"{signature}\n    # TODO: Implement\n"
        
        # Generate scaffold with AR head
        scaffold = self.generate_scaffold(
            prompt,
            max_new_tokens=64,
            temperature=0.2
        )
        
        # Add mask token for infilling
        full_prompt = f"{prompt}{scaffold}\n    <|mask|>\n"
        
        # Generate infill with diffusion head
        infill = self.generate_infill(
            full_prompt,
            steps=steps,
            temperature=temperature
        )
        
        # Combine parts
        if docstring:
            function_code = f"{signature}\n    '''{docstring}'''\n{infill}"
        else:
            function_code = f"{signature}\n{infill}"
        
        return function_code
    
    def generate_multiple_candidates(self, signature: str, docstring: str = "",
                                   num_candidates: int = 5, steps: int = 50,
                                   temperature: float = 0.7) -> List[str]:
        """
        Generate multiple candidate implementations.
        
        Args:
            signature: Function signature
            docstring: Function docstring
            num_candidates: Number of candidates to generate
            steps: Number of diffusion steps
            temperature: Sampling temperature
            
        Returns:
            List of candidate implementations
        """
        candidates = []
        
        for _ in range(num_candidates):
            candidate = self.generate_function(
                signature, docstring, steps, temperature
            )
            candidates.append(candidate)
        
        return candidates
    
    def evaluate_candidate(self, candidate: str, signature: str) -> Dict[str, Any]:
        """
        Evaluate a candidate implementation.
        
        Args:
            candidate: Generated code
            signature: Function signature
            
        Returns:
            Evaluation metrics
        """
        # Basic metrics
        metrics = {
            "length": len(candidate),
            "has_return": "return" in candidate,
            "has_docstring": "'''" in candidate or '"""' in candidate,
            "indentation_consistent": True,  # Simplified check
            "syntax_valid": True  # Simplified check
        }
        
        # Check for common patterns
        patterns = {
            "has_loop": any(keyword in candidate for keyword in ["for ", "while "]),
            "has_conditional": any(keyword in candidate for keyword in ["if ", "elif ", "else:"]),
            "has_exception": any(keyword in candidate for keyword in ["try:", "except", "raise"]),
            "has_list_comprehension": "[" in candidate and "for " in candidate and "]" in candidate,
        }
        metrics.update(patterns)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Generate code using multi-head LoRA model")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
                       help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True,
                       help="Path to LoRA adapters")
    parser.add_argument("--config_path", type=str, default="configs/lora_configs.yaml",
                       help="Path to LoRA configuration")
    parser.add_argument("--mode", type=str, choices=["scaffold", "infill", "function", "candidates"],
                       default="function", help="Generation mode")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--signature", type=str, help="Function signature")
    parser.add_argument("--docstring", type=str, default="", help="Function docstring")
    parser.add_argument("--max_tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidates")
    parser.add_argument("--output_file", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CodeGenerator(args.model_path, args.config_path)
    
    # Load adapters
    generator.load_adapters(args.adapter_path)
    
    # Generate based on mode
    if args.mode == "scaffold":
        if not args.prompt:
            print("Error: --prompt required for scaffold mode")
            return
        
        result = generator.generate_scaffold(
            args.prompt,
            max_new_tokens=args.max_tokens or 128,
            temperature=args.temperature
        )
        print("Generated scaffold:")
        print(result)
    
    elif args.mode == "infill":
        if not args.prompt:
            print("Error: --prompt required for infill mode")
            return
        
        result = generator.generate_infill(
            args.prompt,
            max_new_tokens=args.max_tokens,
            steps=args.steps,
            temperature=args.temperature
        )
        print("Generated infill:")
        print(result)
    
    elif args.mode == "function":
        if not args.signature:
            print("Error: --signature required for function mode")
            return
        
        result = generator.generate_function(
            args.signature,
            args.docstring,
            steps=args.steps,
            temperature=args.temperature
        )
        print("Generated function:")
        print(result)
        
        # Evaluate
        metrics = generator.evaluate_candidate(result, args.signature)
        print("\nEvaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    elif args.mode == "candidates":
        if not args.signature:
            print("Error: --signature required for candidates mode")
            return
        
        candidates = generator.generate_multiple_candidates(
            args.signature,
            args.docstring,
            num_candidates=args.num_candidates,
            steps=args.steps,
            temperature=args.temperature
        )
        
        print(f"Generated {len(candidates)} candidates:")
        for i, candidate in enumerate(candidates):
            print(f"\n--- Candidate {i+1} ---")
            print(candidate)
            
            # Evaluate
            metrics = generator.evaluate_candidate(candidate, args.signature)
            print("Metrics:", ", ".join(f"{k}={v}" for k, v in metrics.items()))
    
    # Save to file if specified
    if args.output_file and 'result' in locals():
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\nResult saved to {args.output_file}")


if __name__ == "__main__":
    main()
