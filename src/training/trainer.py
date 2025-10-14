"""
Custom trainer implementation for multi-head diffusion LoRA training.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ..model.multihead_coder import MultiHeadCoder
from ..model.diffusion_utils import MaskScheduler, compute_mdm_loss, mask_tokens
from ..verifier.bend_verifier import (
    BendVerifier,
    HVMVerifier,
    OnPolicyLearningManager,
    VerificationResult,
)
from ..utils.length_predictor import LengthPredictor

logger = logging.getLogger(__name__)


@dataclass
class TrainingStepResult:
    loss: float
    ar_loss: Optional[float] = None
    diffusion_loss: Optional[float] = None
    length_loss: Optional[float] = None


class MultiHeadTrainer:
    """Lightweight training loop orchestrator for multi-head LoRA training."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.data_config = config.get("data", {})
        self.output_dir = self.training_config.get("output_dir", "logs")
        os.makedirs(self.output_dir, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, config.get("logging", {}).get("level", "INFO")),
            format=config.get("logging", {}).get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initialising trainer on device: %s", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["name"],
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        lora_config_path = self.training_config.get("lora_config_path")
        self.model = MultiHeadCoder(
            self.model_config["name"],
            config_path=lora_config_path,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

        self.length_predictor = LengthPredictor(
            self.tokenizer,
            num_buckets=self.training_config.get("length_prediction", {}).get(
                "num_buckets", 20
            ),
            max_length=self.training_config.get("length_prediction", {}).get(
                "max_length", 1000
            ),
        )

        self.task_weights = self.data_config.get(
            "task_weights", {"ar": 1.0, "diffusion": 0.7, "length": 0.3}
        )

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.training_config.get("learning_rate", 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.training_config.get("weight_decay", 0.01),
        )

        self.grad_accum = self.training_config.get("gradient_accumulation_steps", 1)
        max_steps = self.training_config.get("max_steps", 1000)
        warmup_steps = self.training_config.get("warmup_steps", 0)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )


        diffusion_cfg = self.training_config.get("diffusion", {})
        self.mask_scheduler = MaskScheduler(
            scheduler_type=diffusion_cfg.get("scheduler", "cosine"),
            min_ratio=diffusion_cfg.get("mask_ratio_range", [0.1, 0.9])[0],
            max_ratio=diffusion_cfg.get("mask_ratio_range", [0.1, 0.9])[1],
        )

        self.global_step = 0
        self.verification_metrics = []
        self.reward_history = []

        self.verifier_manager: Optional[OnPolicyLearningManager] = None
        self._setup_verifiers(config.get("verifier", {}))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def train(self, train_loader, val_loader=None) -> None:
        """Main training loop."""
        max_steps = self.training_config.get("max_steps", 1000)
        logging_steps = self.training_config.get("logging_steps", 100)
        eval_steps = self.training_config.get("eval_steps", 0)
        save_steps = self.training_config.get("save_steps", 0)
        max_grad_norm = self.training_config.get("max_grad_norm", 1.0)

        running_losses = []

        self.model.train()
        self.optimizer.zero_grad()
        micro_step = 0

        while self.global_step < max_steps:
            for batch in train_loader:
                self.model.train()
                batch = self._move_batch_to_device(batch)
                loss, metrics = self._compute_batch_loss(batch)

                loss_to_backprop = loss / self.grad_accum

                loss_to_backprop.backward()

                micro_step += 1

                if micro_step % self.grad_accum == 0:
                    if max_grad_norm is not None:
                        clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    micro_step = 0
                    self.global_step += 1
                    running_losses.append(loss.item())

                    if self.global_step % logging_steps == 0:
                        avg_loss = sum(running_losses[-logging_steps:]) / len(
                            running_losses[-logging_steps:]
                        )
                        logger.info(
                            "Step %d | loss %.4f | ar %.4f | diff %.4f | len %.4f",
                            self.global_step,
                            avg_loss,
                            metrics.get("ar_loss", float("nan")),
                            metrics.get("diffusion_loss", float("nan")),
                            metrics.get("length_loss", float("nan")),
                        )

                    if (
                        eval_steps
                        and val_loader is not None
                        and self.global_step % eval_steps == 0
                    ):
                        self.evaluate(val_loader)

                    if save_steps and self.global_step % save_steps == 0:
                        self.save_checkpoint(os.path.join(self.output_dir, f"step_{self.global_step}"))

                    if (
                        self.verifier_manager
                        and self.training_config.get("seed_diffusion", {})
                        .get("on_policy_learning", False)
                    ):
                        freq = self.config.get("verifier", {}).get(
                            "verification_frequency", 0
                        )
                        if freq and self.global_step % freq == 0:
                            self._perform_on_policy_verification()

                    if self.global_step >= max_steps:
                        break

            if self.global_step >= max_steps:
                break

        logger.info("Training completed at step %d", self.global_step)
        self.save_checkpoint(os.path.join(self.output_dir, "final"))

    def evaluate(self, val_loader) -> Dict[str, float]:
        """Evaluate model on validation loader."""
        self.model.eval()
        ar_losses, diff_losses, len_losses = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)
                _, metrics = self._compute_batch_loss(batch, backward=False)
                if "ar_loss" in metrics:
                    ar_losses.append(metrics["ar_loss"])
                if "diffusion_loss" in metrics:
                    diff_losses.append(metrics["diffusion_loss"])
                if "length_loss" in metrics:
                    len_losses.append(metrics["length_loss"])

        avg_metrics = {
            "eval/ar_loss": sum(ar_losses) / len(ar_losses) if ar_losses else float("nan"),
            "eval/diffusion_loss": sum(diff_losses) / len(diff_losses) if diff_losses else float("nan"),
            "eval/length_loss": sum(len_losses) / len(len_losses) if len_losses else float("nan"),
        }

        logger.info("Evaluation metrics: %s", avg_metrics)
        self.model.train()
        return avg_metrics

    def save_checkpoint(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, "model")
        os.makedirs(model_path, exist_ok=True)
        self.model.save_adapters(model_path)

        torch.save(self.optimizer.state_dict(), os.path.join(directory, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(directory, "scheduler.pt"))
        state = {
            "global_step": self.global_step,
            "task_weights": self.task_weights,
        }
        with open(os.path.join(directory, "trainer_state.json"), "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

    def load_checkpoint(self, directory: str) -> None:
        model_path = os.path.join(directory, "model")
        self.model.load_adapters(model_path)

        opt_path = os.path.join(directory, "optimizer.pt")
        if os.path.exists(opt_path):
            self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))

        sched_path = os.path.join(directory, "scheduler.pt")
        if os.path.exists(sched_path):
            self.lr_scheduler.load_state_dict(torch.load(sched_path, map_location=self.device))

        state_path = os.path.join(directory, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
            self.global_step = state.get("global_step", 0)

    # ------------------------------------------------------------------ #
    # Batch helpers
    # ------------------------------------------------------------------ #

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _compute_batch_loss(
        self,
        batch: Dict[str, torch.Tensor],
        backward: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, device=self.device)
        metrics: Dict[str, float] = {}

        if "ar_input_ids" in batch:
            ar_loss = self._compute_ar_loss(batch)
            total_loss = total_loss + self.task_weights.get("ar", 1.0) * ar_loss
            metrics["ar_loss"] = ar_loss.detach().item()

        if "diff_input_ids" in batch:
            diff_loss = self._compute_diffusion_loss(batch)
            total_loss = total_loss + self.task_weights.get("diffusion", 0.7) * diff_loss
            metrics["diffusion_loss"] = diff_loss.detach().item()

        if "length_input_ids" in batch:
            length_loss = self._compute_length_loss(batch)
            total_loss = total_loss + self.task_weights.get("length", 0.3) * length_loss
            metrics["length_loss"] = length_loss.detach().item()

        return total_loss, metrics

    def _compute_ar_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.set_head("ar")
        outputs = self.model(
            input_ids=batch["ar_input_ids"],
            attention_mask=batch["ar_attention_mask"],
            labels=batch["ar_labels"],
            head="ar",
        )
        return outputs.loss

    def _compute_diffusion_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.set_head("diffusion")
        input_ids = batch["diff_input_ids"]
        attention_mask = batch["diff_attention_mask"]
        labels = batch["diff_labels"]

        mask_ratio = random.uniform(
            self.mask_scheduler.min_ratio, self.mask_scheduler.max_ratio
        )
        masked_input_ids, mask_positions = mask_tokens(
            input_ids,
            mask_token_id=self.tokenizer.mask_token_id,
            mask_ratio=mask_ratio,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        outputs = self.model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            head="diffusion",
        )
        logits = outputs.logits
        return compute_mdm_loss(logits, labels, mask_positions)

    def _compute_length_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.set_head("length")
        outputs = self.model(
            input_ids=batch["length_input_ids"],
            attention_mask=batch["length_attention_mask"],
            head="length",
        )
        logits = outputs.logits
        labels = batch["length_labels"]
        if labels.dim() == 2 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------ #
    # Verification helpers
    # ------------------------------------------------------------------ #

    def _setup_verifiers(self, verifier_cfg: Dict[str, Any]) -> None:
        if not verifier_cfg:
            logger.info("Verifier configuration not provided – skipping setup.")
            return

        bend_cfg = verifier_cfg.get("bend", {})
        hvm_cfg = verifier_cfg.get("hvm", {})

        try:
            bend = (
                BendVerifier(
                    bend_path=bend_cfg.get("path", "bend"),
                    hvm_path=bend_cfg.get("path", "bend"),
                    timeout=bend_cfg.get("timeout", 30),
                    use_cuda=bend_cfg.get("use_cuda", True),
                )
                if bend_cfg.get("enabled", False)
                else None
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to initialise Bend verifier: %s", exc)
            bend = None

        try:
            hvm = (
                HVMVerifier(
                    hvm_path=hvm_cfg.get("path", "hvm"),
                    timeout=hvm_cfg.get("timeout", 30),
                )
                if hvm_cfg.get("enabled", False)
                else None
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to initialise HVM verifier: %s", exc)
            hvm = None

        if not bend:
            logger.info("Verification disabled – Bend verifier unavailable.")
            return

        reward_file = verifier_cfg.get("on_policy_learning", {}).get(
            "reward_history_file", os.path.join(self.output_dir, "reward_history.json")
        )
        self.verifier_manager = OnPolicyLearningManager(
            bend_verifier=bend,
            hvm_verifier=hvm,
            reward_history_file=reward_file,
        )

    def _perform_on_policy_verification(self) -> None:
        prompts = self._get_verification_prompts()
        verification_results = []

        for prompt in prompts:
            generated = self._generate_for_verification(prompt)
            if not generated:
                continue

            reward, metadata = self.verifier_manager.evaluate_and_reward(
                generated["code"],
                generation_steps=generated["steps"],
                test_cases=None,
            )
            verification_results.append(
                {
                    "prompt": prompt,
                    "reward": reward,
                    "steps": generated["steps"],
                    "metadata": self._serialize_metadata(metadata),
                }
            )

        if verification_results:
            self.verification_metrics.extend(verification_results)
            self._update_model_based_on_rewards()

    def _generate_for_verification(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            self.model.set_head("ar")
            ar_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                scaffold_tokens = self.model.generate(
                    input_ids=ar_inputs["input_ids"],
                    attention_mask=ar_inputs.get("attention_mask"),
                    max_new_tokens=64,
                    temperature=0.2,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            scaffold = self.tokenizer.decode(scaffold_tokens[0], skip_special_tokens=True)

            predicted_length = min(
                128,  # fallback length
                self.length_predictor.bucket_to_length(
                    self.length_predictor.text_to_bucket(prompt)
                ),
            )

            masked_input = self._create_masked_input(scaffold, predicted_length)
            self.model.set_head("diffusion")
            with torch.no_grad():
                diff_tokens = self.model.diffusion_generate(
                    masked_input,
                    mask_token_id=self.tokenizer.mask_token_id,
                    max_new_tokens=predicted_length,
                    steps=self.training_config.get("diffusion", {}).get("inference_steps", 50),
                )
            generated_code = self.tokenizer.decode(diff_tokens[0], skip_special_tokens=True)
            return {"code": generated_code, "steps": self.training_config.get("diffusion", {}).get("inference_steps", 50)}
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Verification generation failed: %s", exc)
            return None

    def _create_masked_input(self, scaffold: str, length: int) -> torch.Tensor:
        mask_token = self.tokenizer.mask_token or "<|mask|>"
        if "pass" in scaffold:
            masked_text = scaffold.replace("pass", mask_token * max(1, length))
        else:
            masked_text = scaffold + f"\n{mask_token * max(1, length)}"
        encoded = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
        return encoded["input_ids"]

    def _update_model_based_on_rewards(self) -> None:
        if not self.verification_metrics:
            return

        recent = self.verification_metrics[-min(20, len(self.verification_metrics)) :]
        avg_reward = sum(item["reward"] for item in recent) / len(recent)

        if avg_reward < 0.0:
            for group in self.optimizer.param_groups:
                group["lr"] *= 0.95
            logger.info(
                "Reduced learning rate due to negative verification reward: %.3f",
                avg_reward,
            )
        elif avg_reward > 0.5:
            for group in self.optimizer.param_groups:
                group["lr"] *= 1.02
            logger.info(
                "Slightly increased learning rate due to positive verification reward: %.3f",
                avg_reward,
            )

    @staticmethod
    def _get_verification_prompts() -> Tuple[str, ...]:
        return (
            "def fibonacci(n):",
            "def quicksort(arr):",
            "def binary_search(arr, target):",
        )

    @staticmethod
    def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        def _serialize_result(result: Optional[VerificationResult]) -> Optional[Dict[str, Any]]:
            if result is None:
                return None
            return {
                "is_correct": result.is_correct,
                "execution_time": result.execution_time,
                "output": result.output,
                "error": result.error,
                "performance_metrics": result.performance_metrics,
            }

        return {
            "bend_result": _serialize_result(metadata.get("bend_result")),
            "hvm_result": _serialize_result(metadata.get("hvm_result")),
            "generation_steps": metadata.get("generation_steps"),
            "reward": metadata.get("reward"),
            "timestamp": metadata.get("timestamp"),
        }
