# Quick Start Guide

This guide will help you get started with training and using the Qwen3 Coder 30B MoE model with LoRA adapters for masked diffusion.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (128GB VRAM recommended for 30B model)
- PyTorch 2.0+
- Hugging Face Transformers

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd qwen_diffusion_training
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Training

### Tiny Workflow (fast smoke test)

Use the bundled tiny config and dataset to ensure everything is wired correctly:

```bash
bash scripts/train_tiny.sh
bash scripts/generate_tiny.sh
```

This keeps the 30B backbone frozen and trains only LoRA adapters and the length head for ~50 steps.

### 1. Prepare Data

Generate synthetic training data (or provide your own):
```bash
python scripts/prepare_data.py \
    --output_dir data \
    --synthetic_samples 5000 \
    --max_length 2048
```

### 2. Start Training

For H100 GPU (recommended):
```bash
bash scripts/train_lora_h100.sh \
    --adapter_path logs/h100_training \
    --batch_size 16 \
    --max_steps 40000
```

For other GPUs:
```bash
bash scripts/train_lora.sh \
    --config_file configs/qwen3_coder_30b_moe.yaml \
    --data_dir data \
    --output_dir logs
```

### 3. Monitor Training

Training progress is logged to:
- Console output
- Weights & Biases (if configured)
- Log files in the output directory

## Quick Inference

### 1. Generate Code

After training, generate code using the trained adapters:

```bash
python scripts/generate.py \
    --adapter_path logs/h100_training \
    --mode function \
    --signature "def quicksort(arr):" \
    --docstring "Sort array using quicksort algorithm" \
    --steps 50 \
    --temperature 0.7
```

### 2. Generate Multiple Candidates

```bash
python scripts/generate.py \
    --adapter_path logs/h100_training \
    --mode candidates \
    --signature "def binary_search(arr, target):" \
    --num_candidates 5 \
    --steps 50
```

### 3. Python API

```python
from src.inference.generator import CodeGenerator

# Initialize generator
generator = CodeGenerator(
    model_path="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    config_path="configs/lora_configs.yaml"
)

# Load trained adapters
generator.load_adapters("logs/h100_training")

# Generate function
function_code = generator.generate_function(
    signature="def calculate_sum(numbers):",
    docstring="Calculate sum of a list",
    steps=50,
    temperature=0.7
)

print(function_code)
```

## Evaluation

### 1. Run Evaluation

```bash
python scripts/evaluate.py \
    --adapter_path logs/h100_training \
    --output_file evaluation_results.json
```

### 2. Custom Evaluation

```python
from src.evaluation.evaluator import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(
    model_path="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    adapter_path="logs/h100_training"
)

# Run evaluation
results = evaluator.run_comprehensive_evaluation(test_data)
print(results)
```

## Configuration

### Training Configuration

Edit `configs/qwen3_coder_30b_moe.yaml` to adjust:
- Model parameters
- LoRA settings
- Training hyperparameters
- Diffusion settings

### LoRA Configuration

Edit `configs/lora_configs.yaml` to adjust:
- LoRA rank and alpha
- Target modules
- Task types for each head

### Optional: Bend/HVM Verification

Enable verification in the main config (`configs/qwen3_coder_30b_moe.yaml`) once Bend/HVM are installed:

```yaml
verifier:
  bend:
    enabled: true
    path: "bend"
    timeout: 30
    use_cuda: true
  hvm:
    enabled: true
    path: "hvm"
    timeout: 30
  on_policy_learning:
    enabled: true
  verification_frequency: 100
```

If Bend/HVM are not available, leave them disabled or use the tiny config (`configs/qwen3_coder_30b_moe_tiny.yaml`), which disables verification by default.

## Tips & Tricks

### Memory Optimization

For GPU memory constraints:
- Use smaller batch size
- Enable gradient checkpointing
- Use QLoRA (4-bit quantization)
- Reduce sequence length

### Quality vs Speed

- Fewer diffusion steps (25-50) = faster, lower quality
- More diffusion steps (75-100) = slower, higher quality
- Lower temperature = more deterministic output
- Higher temperature = more creative output

### Multi-head Usage

- Use AR head for scaffolding and structure
- Use Diffusion head for implementation details
- Use Length head for efficient generation
- Combine heads for best results

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use QLoRA

2. **Poor Generation Quality**
   - Increase training steps
   - Adjust learning rate
   - Check data quality

3. **Slow Inference**
   - Reduce diffusion steps
   - Use KV cache
   - Enable block-wise generation

### Getting Help

- Check the logs in the output directory
- Review the configuration files
- Run with smaller model first for testing

## Next Steps

1. Experiment with different configurations
2. Try custom datasets
3. Implement custom evaluation metrics
4. Deploy the model for production use

## Examples

See the `examples/` directory for more detailed usage examples.
