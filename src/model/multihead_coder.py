"""
Multi-head Qwen Coder with LoRA adapters for autoregressive, diffusion, and
length-prediction objectives.
"""

from __future__ import annotations

import yaml
import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from torch import Tensor
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutput,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from .diffusion_utils import DiffusionSampler


class MultiHeadCoder(nn.Module):
    """Unified wrapper around a causal LM with multiple LoRA adapters."""

    def __init__(self, model_name: str, config_path: Optional[str] = None):
        super().__init__()
        self.model_name = model_name

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Freeze base weights – LoRA adapters and additional heads remain trainable.
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.active_head: Optional[str] = None
        self.diffusion_sampler = DiffusionSampler()

        self.configs: Dict[str, Dict[str, Any]] = {}
        self.lora_model = None
        self.length_classifier: Optional[nn.Module] = None

        if config_path:
            self._load_lora_configs(config_path)
            self._init_adapters()

    # --------------------------------------------------------------------- #
    # Initialisation helpers
    # --------------------------------------------------------------------- #

    def _load_lora_configs(self, config_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as fh:
            configs = yaml.safe_load(fh)

        required_keys = {"ar_head", "diffusion_head", "length_head"}
        missing = required_keys.difference(configs.keys())
        if missing:
            raise ValueError(f"Missing LoRA configs for heads: {sorted(missing)}")

        self.configs = configs

    def _init_adapters(self) -> None:
        """Initialise LoRA adapters and auxiliary heads."""
        if not self.configs:
            raise ValueError("LoRA configuration must be loaded before initialisation.")

        ar_cfg = self._build_lora_config(self.configs["ar_head"], TaskType.CAUSAL_LM)
        self.lora_model = get_peft_model(
            self.base_model,
            ar_cfg,
            adapter_name="ar",
        )

        diff_cfg = self._build_lora_config(
            self.configs["diffusion_head"], TaskType.CAUSAL_LM
        )
        self.lora_model.add_adapter("diffusion", diff_cfg)

        len_cfg = self._build_lora_config(
            self.configs["length_head"], TaskType.CAUSAL_LM
        )
        self.lora_model.add_adapter("length", len_cfg)

        hidden_size = self.lora_model.config.hidden_size
        num_labels = self.configs["length_head"]["num_labels"]
        self.length_classifier = nn.Linear(hidden_size, num_labels)

        self.set_head("ar")

    @staticmethod
    def _build_lora_config(config: Dict[str, Any], task_type: TaskType) -> LoraConfig:
        return LoraConfig(
            task_type=task_type,
            r=config["r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
            bias=config.get("bias", "none"),
        )

    # --------------------------------------------------------------------- #
    # Utility helpers
    # --------------------------------------------------------------------- #

    def resize_token_embeddings(self, vocab_size: int) -> None:
        """Resize embeddings on the underlying model."""
        if self.lora_model:
            self.lora_model.resize_token_embeddings(vocab_size)
        else:
            self.base_model.resize_token_embeddings(vocab_size)

    def set_head(self, head_name: str) -> None:
        """Activate the adapter corresponding to the requested head."""
        if head_name not in {"ar", "diffusion", "length"}:
            raise ValueError("head_name must be one of {'ar', 'diffusion', 'length'}")

        if not self.lora_model:
            raise RuntimeError("LoRA adapters are not initialised.")

        self.lora_model.set_adapter(head_name)
        self.active_head = head_name

    def set_active_head(self, head_name: str) -> None:
        """Alias for compatibility with the original trainer."""
        self.set_head(head_name)

    # --------------------------------------------------------------------- #
    # Forward / generation
    # --------------------------------------------------------------------- #

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        head: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        if head:
            self.set_head(head)

        if self.active_head is None:
            raise RuntimeError("Head is not set. Call set_head before forward().")

        if self.active_head == "length":
            outputs = self.lora_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None or len(hidden_states) == 0:
                raise RuntimeError(
                    "Model did not return hidden_states; cannot compute length logits."
                )
            last_hidden = hidden_states[-1]
            pooled = last_hidden[:, 0, :]
            logits = self.length_classifier(pooled)
            return SequenceClassifierOutput(logits=logits)

        return self.lora_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, input_ids: Tensor, head: str = "ar", **kwargs: Any) -> Tensor:
        self.set_head(head)

        if head == "diffusion":
            return self._diffusion_generate(input_ids, **kwargs)

        return self.lora_model.generate(input_ids=input_ids, **kwargs)

    def diffusion_generate(
        self,
        input_ids: Tensor,
        mask_token_id: int,
        max_new_tokens: int,
        steps: int = 50,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Tensor:
        """Public helper that reuses the diffusion sampler."""
        return self._diffusion_generate(
            input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            steps=steps,
            temperature=temperature,
            **kwargs,
        )

    def _diffusion_generate(
        self,
        input_ids: Tensor,
        mask_token_id: Optional[int] = None,
        max_new_tokens: int = 256,
        steps: int = 50,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Tensor:
        if mask_token_id is None:
            if getattr(self.lora_model.config, "mask_token_id", None) is not None:
                mask_token_id = self.lora_model.config.mask_token_id
            else:
                raise ValueError("mask_token_id must be provided for diffusion sampling.")

        self.set_head("diffusion")

        def model_fn(ids: Tensor, **model_kwargs: Any) -> CausalLMOutputWithPast:
            return self.lora_model(
                input_ids=ids,
                attention_mask=model_kwargs.get("attention_mask"),
                **{k: v for k, v in model_kwargs.items() if k != "attention_mask"},
            )

        return self.diffusion_sampler.sample(
            model_fn=model_fn,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            steps=steps,
            temperature=temperature,
            **kwargs,
        )

    # --------------------------------------------------------------------- #
    # Persistence helpers
    # --------------------------------------------------------------------- #

    def save_adapters(self, save_dir: str) -> None:
        if not self.lora_model:
            raise RuntimeError("LoRA adapters are not initialised.")
        self.lora_model.save_pretrained(save_dir)
        torch.save(self.length_classifier.state_dict(), f"{save_dir}/length_head.pt")

    def load_adapters(self, save_dir: str) -> None:
        self.lora_model = PeftModel.from_pretrained(self.base_model, save_dir)

        length_head_path = f"{save_dir}/length_head.pt"
        if self.length_classifier is None:
            hidden_size = self.lora_model.config.hidden_size
            num_labels = self.configs.get("length_head", {}).get("num_labels", 20)
            self.length_classifier = nn.Linear(hidden_size, num_labels)

        state_dict = torch.load(length_head_path, map_location="cpu")
        self.length_classifier.load_state_dict(state_dict)

        device = next(self.lora_model.parameters()).device
        self.length_classifier.to(device)
        self.set_head("ar")
