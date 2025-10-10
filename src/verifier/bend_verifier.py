"""
Bend/HVM Verifier for On-Policy Learning

This module provides integration with Bend (high-level parallel programming language)
and HVM (Interaction Combinator evaluator) for verifying generated code correctness
and providing on-policy learning feedback.
"""

import os
import subprocess
import tempfile
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of code verification"""
    is_correct: bool
    execution_time: float
    output: str
    error: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None


class BendVerifier:
    """Bend verifier for parallel execution and correctness checking"""
    
    def __init__(self, 
                 bend_path: str = "bend",
                 hvm_path: str = "hvm",
                 timeout: int = 30,
                 use_cuda: bool = True):
        """
        Initialize Bend verifier
        
        Args:
            bend_path: Path to bend executable
            hvm_path: Path to hvm executable  
            timeout: Timeout for execution in seconds
            use_cuda: Whether to use CUDA backend for parallel execution
        """
        self.bend_path = bend_path
        self.hvm_path = hvm_path
        self.timeout = timeout
        self.use_cuda = use_cuda
        
        # Verify bend and hvm are available
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if bend and hvm are available"""
        try:
            result = subprocess.run([self.bend_path, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            logger.info(f"Bend version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"Bend not found or not working: {e}")
            raise RuntimeError("Bend is required for verification")
        
        try:
            result = subprocess.run([self.hvm_path, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            logger.info(f"HVM version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"HVM not found or not working: {e}")
            logger.warning("HVM verification will be disabled")
    
    def verify_code(self, code: str, test_cases: Optional[List[Dict]] = None) -> VerificationResult:
        """
        Verify generated code using Bend
        
        Args:
            code: Generated code to verify
            test_cases: Optional test cases for validation
            
        Returns:
            VerificationResult with correctness and performance metrics
        """
        start_time = time.time()
        
        try:
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.bend', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Choose backend based on availability and preference
            backend = "run-cu" if self.use_cuda else "run-c"
            
            # Execute with Bend
            result = subprocess.run(
                [self.bend_path, backend, temp_file, "-s"],  # -s for stats
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse performance metrics
            performance_metrics = self._parse_performance_stats(result.stderr)
            
            # Clean up temp file
            os.unlink(temp_file)
            
            if result.returncode == 0:
                # Code executed successfully
                is_correct = self._verify_output(result.stdout, test_cases)
                return VerificationResult(
                    is_correct=is_correct,
                    execution_time=execution_time,
                    output=result.stdout,
                    performance_metrics=performance_metrics
                )
            else:
                # Code execution failed
                return VerificationResult(
                    is_correct=False,
                    execution_time=execution_time,
                    output="",
                    error=result.stderr,
                    performance_metrics=performance_metrics
                )
                
        except subprocess.TimeoutExpired:
            return VerificationResult(
                is_correct=False,
                execution_time=self.timeout,
                output="",
                error="Execution timed out"
            )
        except Exception as e:
            return VerificationResult(
                is_correct=False,
                execution_time=time.time() - start_time,
                output="",
                error=str(e)
            )
    
    def _parse_performance_stats(self, stderr: str) -> Dict[str, float]:
        """Parse performance statistics from Bend output"""
        metrics = {}
        
        # Parse Bend performance stats
        lines = stderr.split('\n')
        for line in lines:
            if "reductions" in line.lower():
                try:
                    # Extract reduction count
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "reductions" in part.lower() and i > 0:
                            metrics["reductions"] = float(parts[i-1].replace(',', ''))
                            break
                except (ValueError, IndexError):
                    pass
            
            elif "interactions per second" in line.lower():
                try:
                    # Extract interactions per second
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.replace(',', '').replace('.', '').isdigit():
                            metrics["interactions_per_second"] = float(part.replace(',', ''))
                            break
                except (ValueError, IndexError):
                    pass
            
            elif "time" in line.lower() and "ms" in line:
                try:
                    # Extract execution time
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "ms" in part and i > 0:
                            metrics["bend_time_ms"] = float(parts[i-1])
                            break
                except (ValueError, IndexError):
                    pass
        
        return metrics
    
    def _verify_output(self, output: str, test_cases: Optional[List[Dict]]) -> bool:
        """Verify output against test cases"""
        if not test_cases:
            # If no test cases provided, assume correct if no errors
            return "error" not in output.lower() and "exception" not in output.lower()
        
        # Check each test case
        for test_case in test_cases:
            expected = test_case.get("expected", "")
            if expected and expected not in output:
                return False
        
        return True
    
    def calculate_on_policy_reward(self, 
                                 verification_result: VerificationResult,
                                 generation_steps: int,
                                 target_steps: int = 50) -> float:
        """
        Calculate reward for on-policy learning
        
        Args:
            verification_result: Result from verification
            generation_steps: Number of steps used in generation
            target_steps: Target number of steps for optimal speed
            
        Returns:
            Reward value for on-policy learning
        """
        if not verification_result.is_correct:
            # Penalize incorrect code heavily
            return -1.0
        
        # Base reward for correctness
        base_reward = 1.0
        
        # Speed bonus reward (fewer steps is better)
        if generation_steps <= target_steps:
            speed_bonus = 0.5 * (1.0 - generation_steps / target_steps)
        else:
            # Penalize using more steps than target
            speed_penalty = -0.3 * (generation_steps - target_steps) / target_steps
            speed_bonus = speed_penalty
        
        # Performance bonus based on execution metrics
        performance_bonus = 0.0
        if verification_result.performance_metrics:
            # Bonus for high interactions per second (parallel efficiency)
            ips = verification_result.performance_metrics.get("interactions_per_second", 0)
            if ips > 1000000:  # 1M interactions per second
                performance_bonus = 0.2
            elif ips > 500000:  # 500K interactions per second
                performance_bonus = 0.1
        
        total_reward = base_reward + speed_bonus + performance_bonus
        return max(-1.0, min(1.0, total_reward))  # Clamp to [-1, 1]


class HVMVerifier:
    """HVM verifier for functional correctness and interaction optimization"""
    
    def __init__(self, hvm_path: str = "hvm", timeout: int = 30):
        """
        Initialize HVM verifier
        
        Args:
            hvm_path: Path to hvm executable
            timeout: Timeout for execution in seconds
        """
        self.hvm_path = hvm_path
        self.timeout = timeout
    
    def verify_interactions(self, code: str) -> VerificationResult:
        """
        Verify code using HVM for interaction optimization
        
        Args:
            code: Code to verify with HVM
            
        Returns:
            VerificationResult with interaction metrics
        """
        start_time = time.time()
        
        try:
            # Convert to HVM format (simplified)
            hvm_code = self._convert_to_hvm(code)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.hvm', delete=False) as f:
                f.write(hvm_code)
                temp_file = f.name
            
            # Execute with HVM
            result = subprocess.run(
                [self.hvm_path, "run", temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            os.unlink(temp_file)
            
            # Parse interaction metrics
            performance_metrics = self._parse_hvm_stats(result.stderr)
            
            if result.returncode == 0:
                return VerificationResult(
                    is_correct=True,
                    execution_time=execution_time,
                    output=result.stdout,
                    performance_metrics=performance_metrics
                )
            else:
                return VerificationResult(
                    is_correct=False,
                    execution_time=execution_time,
                    output="",
                    error=result.stderr,
                    performance_metrics=performance_metrics
                )
                
        except Exception as e:
            return VerificationResult(
                is_correct=False,
                execution_time=time.time() - start_time,
                output="",
                error=str(e)
            )
    
    def _convert_to_hvm(self, code: str) -> str:
        """Convert Bend code to HVM format (simplified)"""
        # This is a placeholder for actual conversion logic
        # In practice, this would involve translating Bend syntax to HVM lambda calculus
        return f"(main {{ {code} }})"
    
    def _parse_hvm_stats(self, stderr: str) -> Dict[str, float]:
        """Parse HVM interaction statistics"""
        metrics = {}
        
        lines = stderr.split('\n')
        for line in lines:
            if "interactions" in line.lower():
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit() and i > 0 and "interactions" in parts[i-1].lower():
                            metrics["interactions"] = float(part)
                            break
                except (ValueError, IndexError):
                    pass
            
            elif "rewrites" in line.lower():
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit() and i > 0 and "rewrites" in parts[i-1].lower():
                            metrics["rewrites"] = float(part)
                            break
                except (ValueError, IndexError):
                    pass
        
        return metrics


class OnPolicyLearningManager:
    """Manager for on-policy learning with Bend/HVM verification"""
    
    def __init__(self, 
                 bend_verifier: BendVerifier,
                 hvm_verifier: Optional[HVMVerifier] = None,
                 reward_history_file: str = "reward_history.json"):
        """
        Initialize on-policy learning manager
        
        Args:
            bend_verifier: Bend verifier instance
            hvm_verifier: Optional HVM verifier instance
            reward_history_file: File to store reward history
        """
        self.bend_verifier = bend_verifier
        self.hvm_verifier = hvm_verifier
        self.reward_history_file = reward_history_file
        self.reward_history = []
        
        # Load existing history if available
        self._load_reward_history()
    
    def evaluate_and_reward(self, 
                          code: str, 
                          generation_steps: int,
                          test_cases: Optional[List[Dict]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate generated code and calculate reward
        
        Args:
            code: Generated code to evaluate
            generation_steps: Number of steps used in generation
            test_cases: Optional test cases for validation
            
        Returns:
            Tuple of (reward, metadata)
        """
        # Verify with Bend
        bend_result = self.bend_verifier.verify_code(code, test_cases)
        
        # Calculate primary reward
        reward = self.bend_verifier.calculate_on_policy_reward(
            bend_result, generation_steps
        )
        
        # Optional: Verify with HVM for additional metrics
        hvm_result = None
        if self.hvm_verifier:
            hvm_result = self.hvm_verifier.verify_interactions(code)
            # Adjust reward based on HVM metrics
            if hvm_result.performance_metrics:
                interactions = hvm_result.performance_metrics.get("interactions", 0)
                if interactions > 0:
                    # Bonus for efficient interaction patterns
                    reward += 0.1 * min(1.0, 1000000 / interactions)
        
        # Create metadata
        metadata = {
            "bend_result": bend_result,
            "hvm_result": hvm_result,
            "generation_steps": generation_steps,
            "reward": reward,
            "timestamp": time.time()
        }
        
        history_entry = {
            "bend_result": self._serialize_verification_result(bend_result),
            "hvm_result": self._serialize_verification_result(hvm_result),
            "generation_steps": generation_steps,
            "reward": reward,
            "timestamp": metadata["timestamp"]
        }
        
        # Store in history
        self.reward_history.append(history_entry)
        self._save_reward_history()
        
        return reward, metadata
    
    @staticmethod
    def _serialize_verification_result(result: Optional[VerificationResult]) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        return {
            "is_correct": result.is_correct,
            "execution_time": result.execution_time,
            "output": result.output,
            "error": result.error,
            "performance_metrics": result.performance_metrics,
        }
    
    def _load_reward_history(self):
        """Load reward history from file"""
        if os.path.exists(self.reward_history_file):
            try:
                with open(self.reward_history_file, 'r') as f:
                    self.reward_history = json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning(f"Could not load reward history from {self.reward_history_file}")
                self.reward_history = []
    
    def _save_reward_history(self):
        """Save reward history to file"""
        try:
            with open(self.reward_history_file, 'w') as f:
                json.dump(self.reward_history, f, indent=2)
        except IOError as e:
            logger.error(f"Could not save reward history: {e}")
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """Get statistics about reward history"""
        if not self.reward_history:
            return {}
        
        rewards = [entry["reward"] for entry in self.reward_history]
        steps = [entry["generation_steps"] for entry in self.reward_history]
        
        return {
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "mean_steps": sum(steps) / len(steps),
            "total_evaluations": len(rewards)
        }
