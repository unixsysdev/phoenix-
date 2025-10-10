"""
Utilities for length prediction and bucket management.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from transformers import AutoTokenizer


class LengthPredictor:
    """Utility class for length prediction and bucket management."""
    
    def __init__(self, tokenizer: AutoTokenizer, num_buckets: int = 20, max_length: int = 1000):
        """
        Initialize length predictor.
        
        Args:
            tokenizer: Tokenizer for text processing
            num_buckets: Number of length buckets
            max_length: Maximum expected length
        """
        self.tokenizer = tokenizer
        self.num_buckets = num_buckets
        self.max_length = max_length
        self.bucket_size = max_length // num_buckets
        
        # Pre-compute bucket boundaries
        self.bucket_boundaries = [i * self.bucket_size for i in range(num_buckets + 1)]
    
    def text_to_bucket(self, text: str) -> int:
        """
        Convert text to length bucket.
        
        Args:
            text: Input text
            
        Returns:
            Bucket index
        """
        # Tokenize text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        length = len(tokens)
        
        # Convert to bucket
        return self.length_to_bucket(length)
    
    def length_to_bucket(self, length: int) -> int:
        """
        Convert length to bucket index.
        
        Args:
            length: Text length in tokens
            
        Returns:
            Bucket index
        """
        if length >= self.max_length:
            return self.num_buckets - 1
        
        bucket = length // self.bucket_size
        return min(bucket, self.num_buckets - 1)
    
    def bucket_to_length(self, bucket: int) -> int:
        """
        Convert bucket index to approximate length.
        
        Args:
            bucket: Bucket index
            
        Returns:
            Approximate length
        """
        if bucket >= self.num_buckets - 1:
            return self.max_length
        
        # Return middle of bucket
        return (bucket + 0.5) * self.bucket_size
    
    def bucket_to_range(self, bucket: int) -> Tuple[int, int]:
        """
        Convert bucket index to length range.
        
        Args:
            bucket: Bucket index
            
        Returns:
            Tuple of (min_length, max_length)
        """
        if bucket >= self.num_buckets - 1:
            return (self.bucket_boundaries[-2], self.max_length)
        
        return (self.bucket_boundaries[bucket], self.bucket_boundaries[bucket + 1])
    
    def create_length_labels(self, texts: List[str]) -> torch.Tensor:
        """
        Create length labels for a batch of texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Tensor of bucket indices
        """
        buckets = [self.text_to_bucket(text) for text in texts]
        return torch.tensor(buckets, dtype=torch.long)
    
    def analyze_length_distribution(self, texts: List[str]) -> Dict[str, any]:
        """
        Analyze length distribution of texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary with distribution statistics
        """
        lengths = [len(self.tokenizer.encode(text, add_special_tokens=False)) for text in texts]
        buckets = [self.length_to_bucket(length) for length in lengths]
        
        # Calculate statistics
        stats = {
            "total_texts": len(texts),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "mean_length": np.mean(lengths),
            "median_length": np.median(lengths),
            "std_length": np.std(lengths),
            "bucket_distribution": {},
            "length_percentiles": {}
        }
        
        # Bucket distribution
        for i in range(self.num_buckets):
            count = buckets.count(i)
            stats["bucket_distribution"][i] = {
                "count": count,
                "percentage": count / len(texts) * 100,
                "range": self.bucket_to_range(i)
            }
        
        # Length percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats["length_percentiles"][f"p{p}"] = np.percentile(lengths, p)
        
        return stats
    
    def visualize_distribution(self, texts: List[str], save_path: Optional[str] = None):
        """
        Visualize length distribution.
        
        Args:
            texts: List of texts
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Skipping visualization.")
            return
        
        # Get lengths
        lengths = [len(self.tokenizer.encode(text, add_special_tokens=False)) for text in texts]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(lengths, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Length (tokens)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Length Distribution')
        ax1.axvline(x=np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
        ax1.legend()
        
        # Bucket distribution
        buckets = [self.length_to_bucket(length) for length in lengths]
        bucket_counts = [buckets.count(i) for i in range(self.num_buckets)]
        bucket_labels = [f"{i}" for i in range(self.num_buckets)]
        
        ax2.bar(bucket_labels, bucket_counts, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Bucket')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Bucket Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def optimize_buckets(self, texts: List[str], target_buckets: Optional[int] = None) -> Dict[str, any]:
        """
        Optimize bucket boundaries based on data distribution.
        
        Args:
            texts: List of texts
            target_buckets: Target number of buckets (default: current num_buckets)
            
        Returns:
            Dictionary with optimization results
        """
        if target_buckets is None:
            target_buckets = self.num_buckets
        
        # Get lengths
        lengths = [len(self.tokenizer.encode(text, add_special_tokens=False)) for text in texts]
        lengths.sort()
        
        # Calculate optimal boundaries using quantiles
        quantiles = np.linspace(0, 1, target_buckets + 1)
        optimal_boundaries = [int(np.quantile(lengths, q)) for q in quantiles]
        
        # Ensure boundaries are unique and sorted
        optimal_boundaries = sorted(list(set(optimal_boundaries)))
        
        # Calculate bucket statistics
        bucket_stats = []
        for i in range(len(optimal_boundaries) - 1):
            min_len = optimal_boundaries[i]
            max_len = optimal_boundaries[i + 1]
            count = sum(1 for l in lengths if min_len <= l < max_len)
            bucket_stats.append({
                "bucket": i,
                "range": (min_len, max_len),
                "count": count,
                "percentage": count / len(lengths) * 100
            })
        
        return {
            "optimal_boundaries": optimal_boundaries,
            "bucket_stats": bucket_stats,
            "original_buckets": self.num_buckets,
            "optimized_buckets": len(optimal_boundaries) - 1
        }


class AdaptiveLengthPredictor(LengthPredictor):
    """Adaptive length predictor that can adjust bucket boundaries."""
    
    def __init__(self, tokenizer: AutoTokenizer, num_buckets: int = 20, max_length: int = 1000,
                 adaptive: bool = True):
        """
        Initialize adaptive length predictor.
        
        Args:
            tokenizer: Tokenizer for text processing
            num_buckets: Initial number of buckets
            max_length: Maximum expected length
            adaptive: Whether to adapt bucket boundaries
        """
        super().__init__(tokenizer, num_buckets, max_length)
        self.adaptive = adaptive
        self.adaptation_history = []
    
    def adapt_buckets(self, texts: List[str], method: str = "quantile") -> Dict[str, any]:
        """
        Adapt bucket boundaries based on data distribution.
        
        Args:
            texts: List of texts
            method: Adaptation method ("quantile", "kmeans", "uniform")
            
        Returns:
            Dictionary with adaptation results
        """
        if not self.adaptive:
            return {"error": "Adaptive mode is disabled"}
        
        # Get lengths
        lengths = [len(self.tokenizer.encode(text, add_special_tokens=False)) for text in texts]
        
        if method == "quantile":
            result = self._adapt_quantile(lengths)
        elif method == "kmeans":
            result = self._adapt_kmeans(lengths)
        elif method == "uniform":
            result = self._adapt_uniform(lengths)
        else:
            raise ValueError(f"Unknown adaptation method: {method}")
        
        # Update bucket boundaries
        self.bucket_boundaries = result["new_boundaries"]
        self.bucket_size = self.max_length // len(result["new_boundaries"])
        
        # Record adaptation
        self.adaptation_history.append({
            "method": method,
            "timestamp": len(self.adaptation_history),
            "old_boundaries": result["old_boundaries"],
            "new_boundaries": result["new_boundaries"],
            "improvement": result.get("improvement", 0)
        })
        
        return result
    
    def _adapt_quantile(self, lengths: List[int]) -> Dict[str, any]:
        """Adapt using quantile-based boundaries."""
        old_boundaries = self.bucket_boundaries.copy()
        
        # Calculate new boundaries using quantiles
        quantiles = np.linspace(0, 1, self.num_buckets + 1)
        new_boundaries = [int(np.quantile(lengths, q)) for q in quantiles]
        
        # Ensure boundaries are unique and sorted
        new_boundaries = sorted(list(set(new_boundaries)))
        
        # Calculate improvement (reduction in variance)
        old_variance = self._calculate_bucket_variance(lengths, old_boundaries)
        new_variance = self._calculate_bucket_variance(lengths, new_boundaries)
        improvement = (old_variance - new_variance) / old_variance if old_variance > 0 else 0
        
        return {
            "method": "quantile",
            "old_boundaries": old_boundaries,
            "new_boundaries": new_boundaries,
            "improvement": improvement,
            "old_variance": old_variance,
            "new_variance": new_variance
        }
    
    def _adapt_kmeans(self, lengths: List[int]) -> Dict[str, any]:
        """Adapt using k-means clustering."""
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("sklearn not available. Falling back to quantile method.")
            return self._adapt_quantile(lengths)
        
        old_boundaries = self.bucket_boundaries.copy()
        
        # Reshape for k-means
        lengths_reshaped = np.array(lengths).reshape(-1, 1)
        
        # Apply k-means
        kmeans = KMeans(n_clusters=self.num_buckets, random_state=42)
        clusters = kmeans.fit_predict(lengths_reshaped)
        
        # Calculate new boundaries
        new_boundaries = [0]
        for i in range(self.num_buckets):
            cluster_lengths = [lengths[j] for j in range(len(lengths)) if clusters[j] == i]
            if cluster_lengths:
                new_boundaries.append(max(cluster_lengths))
        
        # Ensure boundaries are sorted and unique
        new_boundaries = sorted(list(set(new_boundaries)))
        
        # Calculate improvement
        old_variance = self._calculate_bucket_variance(lengths, old_boundaries)
        new_variance = self._calculate_bucket_variance(lengths, new_boundaries)
        improvement = (old_variance - new_variance) / old_variance if old_variance > 0 else 0
        
        return {
            "method": "kmeans",
            "old_boundaries": old_boundaries,
            "new_boundaries": new_boundaries,
            "improvement": improvement,
            "old_variance": old_variance,
            "new_variance": new_variance
        }
    
    def _adapt_uniform(self, lengths: List[int]) -> Dict[str, any]:
        """Adapt using uniform distribution."""
        old_boundaries = self.bucket_boundaries.copy()
        
        # Calculate min and max
        min_len = min(lengths)
        max_len = max(lengths)
        
        # Create uniform boundaries
        new_boundaries = list(range(min_len, max_len + 1, max_len // self.num_buckets))
        new_boundaries.append(max_len)
        
        # Calculate improvement
        old_variance = self._calculate_bucket_variance(lengths, old_boundaries)
        new_variance = self._calculate_bucket_variance(lengths, new_boundaries)
        improvement = (old_variance - new_variance) / old_variance if old_variance > 0 else 0
        
        return {
            "method": "uniform",
            "old_boundaries": old_boundaries,
            "new_boundaries": new_boundaries,
            "improvement": improvement,
            "old_variance": old_variance,
            "new_variance": new_variance
        }
    
    def _calculate_bucket_variance(self, lengths: List[int], boundaries: List[int]) -> float:
        """Calculate variance within buckets."""
        bucket_variances = []
        
        for i in range(len(boundaries) - 1):
            min_len = boundaries[i]
            max_len = boundaries[i + 1]
            
            bucket_lengths = [l for l in lengths if min_len <= l < max_len]
            if bucket_lengths:
                bucket_variances.append(np.var(bucket_lengths))
        
        return np.mean(bucket_variances) if bucket_variances else 0


def create_length_predictor(tokenizer: AutoTokenizer, config: Dict[str, any]) -> LengthPredictor:
    """
    Create length predictor from configuration.
    
    Args:
        tokenizer: Tokenizer for text processing
        config: Configuration dictionary
        
    Returns:
        LengthPredictor instance
    """
    num_buckets = config.get("num_buckets", 20)
    max_length = config.get("max_length", 1000)
    adaptive = config.get("adaptive", False)
    
    if adaptive:
        return AdaptiveLengthPredictor(tokenizer, num_buckets, max_length, adaptive)
    else:
        return LengthPredictor(tokenizer, num_buckets, max_length)