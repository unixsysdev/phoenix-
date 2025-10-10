"""
Utilities for masked diffusion training and inference.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import random
import math


class MaskScheduler:
    """Scheduler for masking ratios during diffusion training."""
    
    def __init__(self, scheduler_type: str = "cosine", min_ratio: float = 0.1, max_ratio: float = 0.9):
        self.scheduler_type = scheduler_type
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    
    def get_mask_ratio(self, step: int, total_steps: int) -> float:
        """Get mask ratio for current step."""
        progress = step / max(1, total_steps - 1)
        
        if self.scheduler_type == "uniform":
            return random.uniform(self.min_ratio, self.max_ratio)
        elif self.scheduler_type == "cosine":
            return self.min_ratio + (self.max_ratio - self.min_ratio) * (0.5 * (1 + math.cos(math.pi * progress)))
        elif self.scheduler_type == "linear":
            return self.min_ratio + (self.max_ratio - self.min_ratio) * progress
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")


def mask_tokens(input_ids: torch.Tensor, mask_token_id: int, mask_ratio: float, 
                eos_token_id: Optional[int] = None, pad_token_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask tokens in the input sequence.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        mask_token_id: ID of the mask token
        mask_ratio: Fraction of tokens to mask
        eos_token_id: EOS token ID (won't be masked)
        pad_token_id: PAD token ID (won't be masked)
    
    Returns:
        Tuple of (masked_input_ids, mask_positions)
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Create mask for positions that can be masked
    maskable_positions = torch.ones_like(input_ids, dtype=torch.bool)
    
    # Don't mask special tokens
    if eos_token_id is not None:
        maskable_positions &= (input_ids != eos_token_id)
    if pad_token_id is not None:
        maskable_positions &= (input_ids != pad_token_id)
    
    # Calculate number of tokens to mask
    num_maskable = maskable_positions.sum().item()
    num_to_mask = int(num_maskable * mask_ratio)
    
    # Randomly select positions to mask
    maskable_indices = maskable_positions.nonzero(as_tuple=False)
    if len(maskable_indices) > 0 and num_to_mask > 0:
        selected_indices = maskable_indices[torch.randperm(len(maskable_indices))[:num_to_mask]]
        
        # Create mask positions tensor
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        mask_positions[selected_indices[:, 0], selected_indices[:, 1]] = True
        
        # Apply mask
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_positions] = mask_token_id
        
        return masked_input_ids, mask_positions
    else:
        # No masking possible
        return input_ids, torch.zeros_like(input_ids, dtype=torch.bool)


def compute_mdm_loss(logits: torch.Tensor, target_ids: torch.Tensor, 
                    mask_positions: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Compute masked diffusion model loss.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        target_ids: Target token IDs [batch_size, seq_len]
        mask_positions: Boolean mask of masked positions [batch_size, seq_len]
        ignore_index: Index to ignore in loss computation
    
    Returns:
        Loss tensor
    """
    # Flatten for cross-entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    target_ids_flat = target_ids.view(-1)
    mask_positions_flat = mask_positions.view(-1)
    
    # Create labels with ignore_index for non-masked positions
    labels = torch.where(mask_positions_flat, target_ids_flat, torch.tensor(ignore_index, device=target_ids.device))
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits_flat, labels, ignore_index=ignore_index)
    
    return loss


def apply_length_bucket(length: int, num_buckets: int, max_length: int) -> int:
    """
    Convert length to bucket index.
    
    Args:
        length: Actual length
        num_buckets: Number of buckets
        max_length: Maximum expected length
    
    Returns:
        Bucket index
    """
    if length >= max_length:
        return num_buckets - 1
    
    bucket_size = max_length // num_buckets
    return min(length // bucket_size, num_buckets - 1)


def bucket_to_length(bucket: int, num_buckets: int, max_length: int) -> int:
    """
    Convert bucket index to approximate length.
    
    Args:
        bucket: Bucket index
        num_buckets: Number of buckets
        max_length: Maximum expected length
    
    Returns:
        Approximate length
    """
    bucket_size = max_length // num_buckets
    return (bucket + 1) * bucket_size


class DiffusionSampler:
    """Sampler for diffusion inference."""
    
    def __init__(self, scheduler_type: str = "cosine"):
        self.scheduler_type = scheduler_type
    
    def sample(self, model_fn, input_ids: torch.Tensor, mask_token_id: int, 
               max_new_tokens: int, steps: int = 50, temperature: float = 0.7,
               use_kv_cache: bool = True, **kwargs) -> torch.Tensor:
        """
        Sample using diffusion process.
        
        Args:
            model_fn: Model function that takes input_ids and returns logits
            input_ids: Input token IDs
            mask_token_id: Mask token ID
            max_new_tokens: Maximum tokens to generate
            steps: Number of diffusion steps
            temperature: Sampling temperature
            use_kv_cache: Whether to use KV cache
            **kwargs: Additional model arguments
        
        Returns:
            Generated token IDs
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initialize with masks
        masked_ids = torch.cat([
            input_ids,
            torch.full((batch_size, max_new_tokens), mask_token_id, device=device, dtype=input_ids.dtype)
        ], dim=1)
        
        # KV cache for efficiency
        past_key_values = None
        
        # Diffusion steps
        for step in range(steps):
            # Calculate mask ratio for this step
            mask_ratio = self._get_step_ratio(step, steps)
            
            # Get model predictions
            with torch.no_grad():
                if use_kv_cache and past_key_values is not None:
                    outputs = model_fn(masked_ids, past_key_values=past_key_values, **kwargs)
                else:
                    outputs = model_fn(masked_ids, **kwargs)
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Store KV cache for next step
                if use_kv_cache and hasattr(outputs, 'past_key_values'):
                    past_key_values = outputs.past_key_values
                
                # Apply temperature
                logits = logits / temperature
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                sampled_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, -1)
                
                # Update masked positions based on current mask ratio
                mask_positions = self._get_step_mask(masked_ids, mask_token_id, mask_ratio)
                masked_ids = torch.where(mask_positions, sampled_tokens, masked_ids)
        
        return masked_ids
    
    def _get_step_ratio(self, step: int, total_steps: int) -> float:
        """Get mask ratio for current step."""
        progress = step / max(1, total_steps - 1)
        
        if self.scheduler_type == "cosine":
            return 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        elif self.scheduler_type == "linear":
            return 1.0 - progress
        else:
            return max(0.1, 1.0 - progress)
    
    def _get_step_mask(self, input_ids: torch.Tensor, mask_token_id: int, mask_ratio: float) -> torch.Tensor:
        """Get mask positions for current step."""
        mask_positions = (input_ids == mask_token_id)
        
        # Calculate how many masks to keep
        num_masks = mask_positions.sum().item()
        num_to_keep = int(num_masks * mask_ratio)
        
        if num_to_keep == 0:
            return torch.zeros_like(mask_positions, dtype=torch.bool)
        
        # Randomly select masks to keep
        mask_indices = mask_positions.nonzero(as_tuple=False)
        if len(mask_indices) > 0:
            selected_indices = mask_indices[torch.randperm(len(mask_indices))[:num_to_keep]]
            
            step_mask = torch.zeros_like(mask_positions, dtype=torch.bool)
            step_mask[selected_indices[:, 0], selected_indices[:, 1]] = True
            
            return step_mask
        
        return mask_positions


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                   mask_positions: torch.Tensor, tokenizer: Any) -> Dict[str, float]:
    """
    Compute evaluation metrics for diffusion model.
    
    Args:
        predictions: Predicted token IDs
        targets: Target token IDs
        mask_positions: Mask positions
        tokenizer: Tokenizer for decoding
    
    Returns:
        Dictionary of metrics
    """
    # Extract only masked positions
    pred_tokens = predictions[mask_positions]
    target_tokens = targets[mask_positions]
    
    # Accuracy
    accuracy = (pred_tokens == target_tokens).float().mean().item()
    
    # Decode for additional metrics
    pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
    target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
    
    # Exact match
    exact_match = float(pred_text == target_text)
    
    return {
        "accuracy": accuracy,
        "exact_match": exact_match,
        "pred_length": len(pred_tokens),
        "target_length": len(target_tokens)
    }