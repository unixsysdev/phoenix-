#!/bin/bash

# Setup script for Bend and HVM - High-level parallel programming language
# This script installs Bend and HVM for use as verifiers in on-policy learning

set -e

echo "🚀 Setting up Bend and HVM for parallel code verification..."

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "📦 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "✅ Rust already installed"
fi

# Install HVM2
echo "📦 Installing HVM2 (Interaction Combinator evaluator)..."
cargo install hvm

# Verify HVM installation
if command -v hvm &> /dev/null; then
    echo "✅ HVM2 installed successfully"
    hvm --version
else
    echo "❌ HVM2 installation failed"
    exit 1
fi

# Install Bend
echo "📦 Installing Bend (high-level parallel programming language)..."
cargo install bend-lang

# Verify Bend installation
if command -v bend &> /dev/null; then
    echo "✅ Bend installed successfully"
    bend --version
else
    echo "❌ Bend installation failed"
    exit 1
fi

# Check for CUDA toolkit (optional but recommended)
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA toolkit found - GPU acceleration available"
    nvcc --version
else
    echo "⚠️  CUDA toolkit not found - CPU execution only"
    echo "   For GPU acceleration, install CUDA 12.x from NVIDIA"
fi

# Check for GCC (for C backend)
if command -v gcc &> /dev/null; then
    echo "✅ GCC found - C backend available"
    gcc --version | head -1
else
    echo "⚠️  GCC not found - install for C backend"
    echo "   Ubuntu/Debian: sudo apt install gcc"
    echo "   macOS: brew install gcc"
fi

# Create test directory
mkdir -p tests/bend_hvm

# Create a simple test file for Bend
cat > tests/bend_hvm/test_parallel_sum.bend << 'EOF'
# Test file for Bend parallel execution
# Implements parallel sum to verify installation

def Sum(start, target):
  if start == target:
    return start
  else:
    half = (start + target) / 2
    left = Sum(start, half)
    right = Sum(half + 1, target)
    return left + right

def main():
  return Sum(1, 1000)
EOF

# Test Bend installation
echo "🧪 Testing Bend installation..."
if bend run-c tests/bend_hvm/test_parallel_sum.bend -s &> tests/bend_hvm/bend_test_output.txt; then
    echo "✅ Bend C backend working"
    cat tests/bend_hvm/bend_test_output.txt | grep -E "(reductions|time|interactions)"
else
    echo "❌ Bend C backend test failed"
    cat tests/bend_hvm/bend_test_output.txt
fi

# Test Bend with CUDA if available
if command -v nvcc &> /dev/null && command -v nvidia-smi &> /dev/null; then
    echo "🧪 Testing Bend CUDA backend..."
    if bend run-cu tests/bend_hvm/test_parallel_sum.bend -s &> tests/bend_hvm/bend_cuda_test_output.txt; then
        echo "✅ Bend CUDA backend working"
        cat tests/bend_hvm/bend_cuda_test_output.txt | grep -E "(reductions|time|interactions)"
    else
        echo "❌ Bend CUDA backend test failed"
        cat tests/bend_hvm/bend_cuda_test_output.txt
    fi
fi

# Create HVM test file
cat > tests/bend_hvm/test.hvm << 'EOF'
// Simple HVM test file
(main (lambda (x) (+ x 1)))
EOF

# Test HVM installation
echo "🧪 Testing HVM installation..."
if hvm run tests/bend_hvm/test.hvm &> tests/bend_hvm/hvm_test_output.txt; then
    echo "✅ HVM working"
else
    echo "❌ HVM test failed"
    cat tests/bench_hvm/hvm_test_output.txt
fi

# Create verifier configuration
mkdir -p config
cat > config/bend_hvm_config.yaml << 'EOF'
# Bend/HVM Verifier Configuration
bend:
  enabled: true
  timeout: 30
  use_cuda: true
  backend: "run-cu"  # Change to "run-c" if CUDA not available
  
hvm:
  enabled: true
  timeout: 30
  
performance:
  # Performance thresholds for rewards
  min_interactions_per_second: 500000  # 500K interactions/sec
  max_execution_time_ms: 1000  # 1 second
  
on_policy_learning:
  target_steps: 50
  reward_weights:
    correctness: 1.0
    speed: 0.5
    efficiency: 0.2
EOF

echo "📝 Created verifier configuration at config/bend_hvm_config.yaml"

# Create integration test script
cat > scripts/test_verifier_integration.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for Bend/HVM verifier integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.verifier.bend_verifier import BendVerifier, HVMVerifier, OnPolicyLearningManager

def test_bend_verifier():
    """Test Bend verifier with a simple example"""
    print("🧪 Testing Bend verifier...")
    
    verifier = BendVerifier()
    
    # Simple test code
    test_code = """
def Sum(start, target):
  if start == target:
    return start
  else:
    half = (start + target) / 2
    left = Sum(start, half)
    right = Sum(half + 1, target)
    return left + right

def main():
  return Sum(1, 100)
"""
    
    result = verifier.verify_code(test_code)
    print(f"   Correctness: {result.is_correct}")
    print(f"   Execution time: {result.execution_time:.3f}s")
    if result.performance_metrics:
        print(f"   Performance: {result.performance_metrics}")
    
    return result.is_correct

def test_on_policy_learning():
    """Test on-policy learning manager"""
    print("🧪 Testing on-policy learning manager...")
    
    bend_verifier = BendVerifier()
    manager = OnPolicyLearningManager(bend_verifier)
    
    # Test code
    test_code = """
def quicksort(arr):
  if len(arr) <= 1:
    return arr
  else:
    pivot = arr[0]
    left = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x > pivot]
    return quicksort(left) + [pivot] + quicksort(right)

def main():
  return quicksort([5, 2, 8, 1, 9, 3])
"""
    
    # Test with different step counts
    for steps in [25, 50, 100]:
        reward, metadata = manager.evaluate_and_reward(test_code, steps)
        print(f"   Steps: {steps}, Reward: {reward:.3f}")
    
    # Show statistics
    stats = manager.get_reward_statistics()
    print(f"   Statistics: {stats}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Running Bend/HVM verifier integration tests...")
    
    try:
        # Test Bend verifier
        bend_ok = test_bend_verifier()
        
        # Test on-policy learning
        learning_ok = test_on_policy_learning()
        
        if bend_ok and learning_ok:
            print("✅ All tests passed!")
            return 0
        else:
            print("❌ Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
EOF

chmod +x scripts/test_verifier_integration.py

echo "🧪 Created integration test script at scripts/test_verifier_integration.py"

# Create sample test cases for verification
mkdir -p data
cat > data/test_cases.json << 'EOF'
{
  "test_cases": [
    {
      "name": "quicksort",
      "description": "Test quicksort implementation",
      "expected": "[1, 2, 3, 5, 8, 9]"
    },
    {
      "name": "factorial",
      "description": "Test factorial function",
      "expected": "120"
    },
    {
      "name": "fibonacci",
      "description": "Test fibonacci sequence",
      "expected": "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"
    }
  ]
}
EOF

echo "📝 Created sample test cases at data/test_cases.json"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Summary:"
echo "   ✅ HVM2 installed"
echo "   ✅ Bend installed"
echo "   ✅ Test files created"
echo "   ✅ Configuration files created"
echo "   ✅ Integration test script created"
echo ""
echo "🧪 Run tests with:"
echo "   python scripts/test_verifier_integration.py"
echo ""
echo "📖 For more information:"
echo "   - Bend: https://github.com/HigherOrderCO/Bend"
echo "   - HVM: https://github.com/HigherOrderCO/HVM"
echo ""
echo "🚀 You're ready to use Bend/HVM for on-policy learning!"