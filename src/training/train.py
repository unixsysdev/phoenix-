"""
Main training script for multi-head LoRA model.
"""

import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import MultiHeadTrainer
from src.data.dataset import CodeDataset, create_dataloader


def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train multi-head LoRA model")
    parser.add_argument("--config_file", type=str, default="configs/qwen3_coder_30b_moe.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default="logs",
                       help="Directory to save outputs")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Update config with command line arguments
    config["data"]["train_path"] = os.path.join(args.data_dir, "train_samples.json")
    config["data"]["val_path"] = os.path.join(args.data_dir, "val_samples.json")
    config["training"]["output_dir"] = args.output_dir
    
    # Print configuration
    print("Training configuration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Max steps: {config['training']['max_steps']}")
    print(f"  Batch size: {config['training']['micro_batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Initialize trainer
    trainer = MultiHeadTrainer(config)
    
    # Load checkpoint if specified
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Create datasets
    print("Loading datasets...")
    length_cfg = config["training"].get("length_prediction", {"num_buckets": 20, "max_length": 1000})

    train_dataset = CodeDataset(
        config["data"]["train_path"],
        trainer.tokenizer,
        max_length=config["data"]["max_length"],
        mode="train",
        length_buckets=length_cfg.get("num_buckets", 20),
        max_code_length=length_cfg.get("max_length", 1000)
    )
    
    val_dataset = None
    if os.path.exists(config["data"]["val_path"]):
        val_dataset = CodeDataset(
            config["data"]["val_path"],
            trainer.tokenizer,
            max_length=config["data"]["max_length"],
            mode="val",
            length_buckets=length_cfg.get("num_buckets", 20),
            max_code_length=length_cfg.get("max_length", 1000)
        )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["training"]["micro_batch_size"],
        shuffle=True,
        num_workers=4 if not args.debug else 0
    )
    
    val_loader = None
    if val_dataset:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config["training"]["micro_batch_size"],
            shuffle=False,
            num_workers=4 if not args.debug else 0
        )
    
    print(f"Training dataset: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Start training
    print("Starting training...")
    trainer.train(train_loader, val_loader)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
