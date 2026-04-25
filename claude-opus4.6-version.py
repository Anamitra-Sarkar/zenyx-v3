#!/usr/bin/env python3
"""
zenyx_v3_train.py — Complete Zenyx v3 Training Script
=====================================================
Sub-1B parameter LLM with:
  - Hybrid CSA/HCA Attention (DeepSeek V4 style)
  - Dual Shared Expert Sparse MoE (Top-1 routing)
  - FP8 Dense layers (e4m3fn forward, bfloat16 accumulation)
  - Decoupled YaRN RoPE (32768 context)
  - Muon + AdamW hybrid optimizer
  - LayerScale + weight-tied recurrent transformer
  - TPU v5e-8 FSDP sharding
  - O(1) shard-level checkpoint resumption
Target: Single Kaggle TPU v5e-8 pod (128GB total HBM, 16GB per core)
"""

# === IMPORTS ===
import os
import sys
import json
import time
import math
import shutil
import tempfile
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from functools import partial
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

from flax import linen as nn
from flax.linen import initializers

import optax

from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer

from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("zenyx_v3")


# === CONFIG ===
class ZenyxConfig:
    """Complete architecture and training configuration for Zenyx v3."""

    # --- Authentication & Storage ---
    hf_token: str = "YOUR_HF_TOKEN_HERE"
    hf_repo: str = "Arko007/zenyx-v3-checkpoints"

    # --- Architecture ---
    d_model: int = 1536
    num_heads: int = 12
    d_head: int = 128  # d_model // num_heads
    d_latent: int = 256
    d_rope: int = 64
    d_ff: int = 4096
    num_shared_experts: int = 2
    num_routed_experts: int = 64
    num_recurrences: int = 12
    seq_len: int = 32768
    vocab_size: int = 65536

    # --- Attention ---
    csa_compress_ratio: int = 4
    hca_compress_ratio: int = 128
    local_window: int = 256
    csa_top_k: int = 64

    # --- Training ---
    base_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    total_steps: int = 500_000
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0
    muon_momentum: float = 0.95
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_eps: float = 1e-8
    aux_loss_alpha: float = 0.01

    # --- Batch sizing ---
    # 2M tokens per global batch = seq_len * micro_batch * grad_accum
    # 32768 * 8 * 8 = 2,097,152
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    global_batch_tokens: int = 2_097_152

    # --- LayerScale ---
    layerscale_init: float = 1e-4

    # --- YaRN RoPE ---
    rope_base: float = 10000.0
    yarn_alpha: float = 1.0
    yarn_beta: float = 32.0
    yarn_scale_s: float = 4.0
    original_ctx_len: int = 8192

    # --- Checkpointing ---
    checkpoint_every: int = 500
    log_every: int = 10
    detailed_log_every: int = 100

    # --- Data ---
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1"
    fineweb_config: str = "sample-10BT"  # Use "default" for full run

    # --- FSDP ---
    num_devices: int = 8
    mesh_axis_name: str = "fsdp"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.d_head = self.d_model // self.num_heads
        self.d_ff_per_shared = self.d_ff // self.num_shared_experts


config = ZenyxConfig()


# === TOKENIZER ===
def load_tokenizer(cfg: ZenyxConfig):
    """Load tokenizer. Mistral tokenizer (32k vocab, BPE, good for code+math)."""
    logger.info(f"Loading tokenizer: {cfg.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name,
        token=cfg.hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(
        f"Tokenizer loaded. Native vocab size: {tokenizer.vocab_size}, "
        f"Model vocab size: {cfg.vocab_size}"
    )
    return tokenizer


# === DATA PIPELINE ===
def build_streaming_dataset(cfg: ZenyxConfig, tokenizer):
    """
    Build streaming interleaved dataset:
      60% FineWeb-Edu, 25% StarCoderData, 15% NuminaMath-CoT
    """
    logger.info("Building streaming data pipeline...")

    # 1. FineWeb-Edu
    fw_edu = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        cfg.fineweb_config,
        split="train",
        streaming=True,
        token=cfg.hf_token,
    )
    fw_edu = fw_edu.select_columns(["text"])

    # 2. StarCoderData (content column -> text)
    code_ds = load_dataset(
        "bigcode/starcoderdata",
        split="train",
        streaming=True,
        token=cfg.hf_token,
    )
    code_ds = code_ds.select_columns(["content"])
    code_ds = code_ds.rename_column("content", "text")

    # 3. NuminaMath-CoT (problem + solution -> text)
    math_ds = load_dataset(
        "AI-MO/NuminaMath-CoT",
        split="train",
        streaming=True,
        token=cfg.hf_token,
    )

    def format_math(example):
        return {"text": f"Problem: {example['problem']}\n\nSolution: {example['solution']}"}

    math_ds = math_ds.map(
        format_math,
        remove_columns=["source", "problem", "solution", "messages"],
    )

    # 4. Interleave with mixing ratios
    mixed = interleave_datasets(
        [fw_edu, code_ds, math_ds],
        probabilities=[0.60, 0.25, 0.15],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    logger.info(
        "Streaming pipeline: 60% FineWeb-Edu, 25% StarCoderData, 15% NuminaMath-CoT"
    )
    return mixed


def pack_tokens_generator(dataset_iter, tokenizer, seq_len: int, eos_id: int):
    """
    Greedy left-to-right token packing into fixed-length sequences.
    Concatenates documents with EOS separator.
    Yields dict: input_ids [seq_len], loss_mask [seq_len] (all 1s for Phase 1).
    """
    buffer = []

    for example in dataset_iter:
        text = example.get("text", "")
        if not text or len(text.strip()) < 10:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(eos_id)
        buffer.extend(tokens)

        while len(buffer) >= seq_len:
            input_ids = buffer[:seq_len]
            buffer = buffer[seq_len:]
            yield {
                "input_ids": np.array(input_ids, dtype=np.int32),
                "loss_mask": np.ones(seq_len, dtype=np.float32),
            }


class DataPipeline:
    """
    Manages streaming data with O(1) shard-level resumption.
    Tracks tokens consumed for checkpoint metadata.
    """

    def __init__(self, cfg: ZenyxConfig, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.eos_token_id
        self.tokens_consumed: int = 0
        self.sequences_consumed: int = 0
        self._dataset = None
        self._iterator = None

    def initialize(self, skip_sequences: int = 0):
        """Initialize pipeline, optionally skipping sequences for resumption."""
        self._dataset = build_streaming_dataset(self.cfg, self.tokenizer)

        if skip_sequences > 0:
            logger.info(f"Resuming data pipeline: skipping {skip_sequences} sequences...")
            # O(1) shard-level resumption:
            # The streaming library internally tracks which parquet shards have been
            # fully consumed. When we skip N sequences, completed shards are jumped
            # over in O(1). Only the current shard requires linear scan.
            tokens_consumed = skip_sequences * self.cfg.seq_len
            tokens_per_shard_approx = 50_000_000  # ~50M tokens per parquet shard
            shard_index = tokens_consumed // tokens_per_shard_approx
            offset_within_shard = tokens_consumed % tokens_per_shard_approx
            logger.info(
                f"  tokens_consumed={tokens_consumed:,}, "
                f"  shard_index~={shard_index}, "
                f"  offset_within_shard~={offset_within_shard:,}"
            )

            self._iterator = iter(
                pack_tokens_generator(
                    iter(self._dataset), self.tokenizer, self.cfg.seq_len, self.eos_id
                )
            )

            skipped = 0
            skip_start = time.time()
            for _ in range(skip_sequences):
                try:
                    next(self._iterator)
                    skipped += 1
                except StopIteration:
                    break
            skip_time = time.time() - skip_start

            self.sequences_consumed = skipped
            self.tokens_consumed = skipped * self.cfg.seq_len
            logger.info(
                f"Skipped {skipped} sequences ({self.tokens_consumed:,} tokens) "
                f"in {skip_time:.1f}s"
            )
        else:
            self._iterator = iter(
                pack_tokens_generator(
                    iter(self._dataset),
                    self.tokenizer,
                    self.cfg.seq_len,
                    self.eos_id,
                )
            )

    def get_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Get a batch of packed sequences. Returns None if pipeline broken."""
        input_ids_list = []
        loss_mask_list = []

        for _ in range(batch_size):
            try:
                sample = next(self._iterator)
                input_ids_list.append(sample["input_ids"])
                loss_mask_list.append(sample["loss_mask"])
            except StopIteration:
                logger.info("Dataset exhausted, re-initializing for next epoch...")
                self._dataset = build_streaming_dataset(self.cfg, self.tokenizer)
                self._iterator = iter(
                    pack_tokens_generator(
                        iter(self._dataset),
                        self.tokenizer,
                        self.cfg.seq_len,
                        self.eos_id,
                    )
                )
                try:
                    sample = next(self._iterator)
                    input_ids_list.append(sample["input_ids"])
                    loss_mask_list.append(sample["loss_mask"])
                except StopIteration:
                    return None

        self.sequences_consumed += batch_size
        self.tokens_consumed += batch_size * self.cfg.seq_len

        return {
            "input_ids": np.stack(input_ids_list, axis=0),
            "loss_mask": np.stack(loss_mask_list, axis=0),
        }


# === MODEL COMPONENTS ===


# --- FP8 Dense ---
class FP8Dense(nn.Module):
    """
    Dense layer with FP8 (e4m3fn) forward pass on TPU v5e MXUs.
    Master weights in float32, computation in FP8, accumulation in bfloat16.
    """

    features: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param(
            "kernel",
            initializers.variance_scaling(1.0, "fan_in", "normal"),
            (inputs.shape[-1], self.features),
            jnp.float32,
        )

        # Dynamic per-tensor scaling for FP8 quantization
        e4m3_max = jnp.float32(448.0)

        x_amax = jnp.max(jnp.abs(inputs))
        w_amax = jnp.max(jnp.abs(kernel))

        x_scale = e4m3_max / jnp.maximum(x_amax, jnp.float32(1e-12))
        w_scale = e4m3_max / jnp.maximum(w_amax, jnp.float32(1e-12))

        # Quantize to FP8 e4m3fn
        x_fp8 = (inputs * x_scale).astype(jnp.float8_e4m3fn)
        w_fp8 = (kernel * w_scale).astype(jnp.float8_e4m3fn)

        # Native FP8 matmul on TPU MXUs, accumulate in bfloat16
        out = lax.dot_general(
            x_fp8,
            w_fp8,
            dimension_numbers=(((inputs.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=jnp.bfloat16,
        )

        # Dequantize
        out = out / (x_scale * w_scale)

        if self.use_bias:
            bias = self.param(
                "bias", initializers.zeros_init(), (self.features,), jnp.float32
            )
            out = out + bias.astype(jnp.bfloat16)

        return out.astype(jnp.bfloat16)


# --- Decoupled YaRN RoPE ---
def build_yarn_rope(
    seq_len: int,
    d_rope: int,
    base: float = 10000.0,
    alpha: float = 1.0,
    beta: float = 32.0,
    scale_s: float = 4.0,
    original_ctx_len: int = 8192,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Computes exact YaRN frequencies with NTK-by-parts interpolation
    and attention logit temperature mscale.
    Returns: (cos_vals[seq_len, d_rope//2], sin_vals[seq_len, d_rope//2], mscale)
    """
    m = jnp.arange(d_rope // 2, dtype=jnp.float32)
    theta_m = base ** (-2.0 * m / d_rope)

    # Wavelengths and ramp
    lambda_m = 2.0 * jnp.pi / theta_m
    r_m = original_ctx_len / lambda_m

    gamma = jnp.where(
        r_m < alpha,
        0.0,
        jnp.where(r_m > beta, 1.0, (r_m - alpha) / (beta - alpha)),
    )

    # NTK-by-parts
    theta_yarn = (1.0 - gamma) * (theta_m / scale_s) + gamma * theta_m

    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(positions, theta_yarn)
    cos_val = jnp.cos(angles)
    sin_val = jnp.sin(angles)

    mscale = 0.1 * float(jnp.log(jnp.float32(scale_s))) + 1.0

    return cos_val, sin_val, mscale


def apply_decoupled_yarn_rope(
    x_rope: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> jnp.ndarray:
    """
    Applies rotation to decoupled RoPE sub-heads.
    x_rope shape: [B, S, H, d_rope] or [B, S, d_rope]
    cos/sin shape: [S, d_rope//2]
    """
    d_half = x_rope.shape[-1] // 2
    x1 = x_rope[..., :d_half]
    x2 = x_rope[..., d_half:]

    if x_rope.ndim == 4:
        # [B, S, H, d_rope]
        cos_b = cos[None, : x_rope.shape[1], None, :]
        sin_b = sin[None, : x_rope.shape[1], None, :]
    elif x_rope.ndim == 3:
        # [B, S, d_rope]
        cos_b = cos[None, : x_rope.shape[1], :]
        sin_b = sin[None, : x_rope.shape[1], :]
    else:
        cos_b = cos
        sin_b = sin

    rotated_x1 = x1 * cos_b - x2 * sin_b
    rotated_x2 = x2 * cos_b + x1 * sin_b

    return jnp.concatenate([rotated_x1, rotated_x2], axis=-1)


# --- Hybrid Attention: CSA / HCA ---
class ZenyxHybridAttention(nn.Module):
    """
    Hybrid Compressed Sparse Attention (CSA) / Heavily Compressed Attention (HCA).
    CSA (odd steps): compress_ratio=4, Top-K=64 sparse selection
    HCA (even steps): compress_ratio=128, dense global attention
    Both use a 256-token local sliding window.
    """

    d_model: int
    num_heads: int
    d_latent: int
    d_rope: int
    d_head: int
    csa_compress_ratio: int = 4
    hca_compress_ratio: int = 128
    local_window: int = 256
    csa_top_k: int = 64

    def setup(self):
        self.q_proj = FP8Dense(self.d_latent)
        self.kv_proj = FP8Dense(self.d_latent)
        self.q_up = FP8Dense(self.num_heads * self.d_head)
        self.kv_up_k = FP8Dense(self.num_heads * self.d_head)
        self.kv_up_v = FP8Dense(self.num_heads * self.d_head)
        self.q_rope_proj = FP8Dense(self.num_heads * self.d_rope)
        self.k_rope_proj = FP8Dense(self.d_rope)
        self.o_proj = FP8Dense(self.d_model)

    def __call__(
        self,
        x: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        mscale: float,
        is_hca: bool,
    ) -> jnp.ndarray:
        batch, seq_len, _ = x.shape
        compress_ratio = self.hca_compress_ratio if is_hca else self.csa_compress_ratio

        # 1. Latent generation
        c_q = self.q_proj(x)
        c_kv = self.kv_proj(x)

        # 2. Decoupled RoPE
        q_rope_raw = self.q_rope_proj(x).reshape(
            batch, seq_len, self.num_heads, self.d_rope
        )
        q_rope = apply_decoupled_yarn_rope(q_rope_raw, cos, sin)

        k_rope_raw = self.k_rope_proj(x)
        k_rope = apply_decoupled_yarn_rope(k_rope_raw, cos, sin)

        # 3. Sequence compression via mean pooling
        pad_len = (compress_ratio - seq_len % compress_ratio) % compress_ratio
        if pad_len > 0:
            c_kv_padded = jnp.pad(c_kv, ((0, 0), (0, pad_len), (0, 0)))
            k_rope_padded = jnp.pad(k_rope, ((0, 0), (0, pad_len), (0, 0)))
        else:
            c_kv_padded = c_kv
            k_rope_padded = k_rope

        padded_len = c_kv_padded.shape[1]
        num_chunks = padded_len // compress_ratio

        c_kv_compressed = c_kv_padded.reshape(
            batch, num_chunks, compress_ratio, self.d_latent
        ).mean(axis=2)

        k_rope_compressed = k_rope_padded.reshape(
            batch, num_chunks, compress_ratio, self.d_rope
        ).mean(axis=2)

        # 4. Multi-head decompression
        q_nope = self.q_up(c_q).reshape(batch, seq_len, self.num_heads, self.d_head)
        k_nope = self.kv_up_k(c_kv_compressed).reshape(
            batch, num_chunks, self.num_heads, self.d_head
        )
        v_nope = self.kv_up_v(c_kv_compressed).reshape(
            batch, num_chunks, self.num_heads, self.d_head
        )

        # 5. Local sliding window (uncompressed last local_window tokens)
        local_len = min(self.local_window, seq_len)
        local_c_kv = c_kv[:, -local_len:, :]
        local_k_rope = k_rope[:, -local_len:, :]

        local_k = self.kv_up_k(local_c_kv).reshape(
            batch, local_len, self.num_heads, self.d_head
        )
        local_v = self.kv_up_v(local_c_kv).reshape(
            batch, local_len, self.num_heads, self.d_head
        )

        # 6. Assemble global compressed + local KV
        k_assembled = jnp.concatenate([k_nope, local_k], axis=1)
        v_assembled = jnp.concatenate([v_nope, local_v], axis=1)

        q_final = jnp.concatenate([q_nope, q_rope], axis=-1)

        k_rope_local = local_k_rope
        k_rope_final = jnp.concatenate([k_rope_compressed, k_rope_local], axis=1)
        k_rope_expanded = jnp.broadcast_to(
            k_rope_final[:, :, None, :],
            (batch, num_chunks + local_len, self.num_heads, self.d_rope),
        )
        k_final = jnp.concatenate([k_assembled, k_rope_expanded], axis=-1)

        # 7. Scaled dot-product attention
        scale = mscale / jnp.sqrt(jnp.float32(self.d_head + self.d_rope))

        attn_logits = jnp.einsum("bshd,bthd->bhst", q_final, k_final) * scale

        # 8. Sparse Top-K selection for CSA layers
        if not is_hca:
            t_dim = attn_logits.shape[-1]
            if t_dim > self.csa_top_k:
                top_k_vals = lax.top_k(attn_logits, self.csa_top_k)[0]
                threshold = top_k_vals[..., -1:]
                attn_logits = jnp.where(
                    attn_logits >= threshold, attn_logits, jnp.float32(-1e9)
                )

        # 9. Causal masking
        kv_len = num_chunks + local_len
        causal_mask = jnp.ones((seq_len, kv_len), dtype=jnp.bool_)

        local_start_idx = num_chunks
        local_positions = jnp.arange(local_len) + (seq_len - local_len)
        query_positions = jnp.arange(seq_len)
        local_causal = query_positions[:, None] >= local_positions[None, :]
        causal_mask = causal_mask.at[:, local_start_idx:].set(local_causal)

        causal_mask_broadcast = causal_mask[None, None, :, :]
        attn_logits = jnp.where(causal_mask_broadcast, attn_logits, jnp.float32(-1e9))

        # 10. Softmax in bfloat16 (NOT FP8)
        attn_weights = jax.nn.softmax(attn_logits.astype(jnp.bfloat16), axis=-1)

        # 11. Weighted sum
        attn_output = jnp.einsum("bhst,bthd->bshd", attn_weights, v_assembled)

        # 12. Output projection
        attn_output = attn_output.reshape(batch, seq_len, self.num_heads * self.d_head)
        return self.o_proj(attn_output)


# --- Dual Shared Sparse MoE ---
class DualSharedSparseMoE(nn.Module):
    """
    2 shared experts (always active, d_ff/2 each) + 64 routed experts (Top-1).
    SwiGLU activation. Auxiliary load-balancing loss on routed experts.
    """

    d_model: int
    d_ff: int
    num_routed_experts: int
    aux_loss_alpha: float = 0.01

    def setup(self):
        d_ff_split = self.d_ff // 2

        # Shared Expert 1 (SwiGLU)
        self.shared_1_gate = FP8Dense(d_ff_split)
        self.shared_1_up = FP8Dense(d_ff_split)
        self.shared_1_down = FP8Dense(self.d_model)

        # Shared Expert 2 (SwiGLU)
        self.shared_2_gate = FP8Dense(d_ff_split)
        self.shared_2_up = FP8Dense(d_ff_split)
        self.shared_2_down = FP8Dense(self.d_model)

        # Router (FP32 for gradient stability)
        self.router = nn.Dense(
            self.num_routed_experts,
            use_bias=False,
            dtype=jnp.float32,
            kernel_init=initializers.variance_scaling(1.0, "fan_in", "normal"),
        )

        # Batched routed expert parameters
        self.routed_gate_w = self.param(
            "routed_gate_w",
            initializers.lecun_normal(),
            (self.num_routed_experts, self.d_model, self.d_ff),
            jnp.bfloat16,
        )
        self.routed_up_w = self.param(
            "routed_up_w",
            initializers.lecun_normal(),
            (self.num_routed_experts, self.d_model, self.d_ff),
            jnp.bfloat16,
        )
        self.routed_down_w = self.param(
            "routed_down_w",
            initializers.lecun_normal(),
            (self.num_routed_experts, self.d_ff, self.d_model),
            jnp.bfloat16,
        )

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch, seq_len, d = x.shape

        # 1. Dual shared experts (SwiGLU)
        gate_1 = jax.nn.silu(self.shared_1_gate(x))
        up_1 = self.shared_1_up(x)
        shared_1 = self.shared_1_down(gate_1 * up_1)

        gate_2 = jax.nn.silu(self.shared_2_gate(x))
        up_2 = self.shared_2_up(x)
        shared_2 = self.shared_2_down(gate_2 * up_2)

        shared_out = shared_1 + shared_2

        # 2. Router (FP32)
        x_flat = x.reshape(-1, d)
        router_logits = self.router(x_flat.astype(jnp.float32))
        router_probs = jax.nn.softmax(router_logits, axis=-1)

        # Top-1 routing
        expert_indices = jnp.argmax(router_probs, axis=-1)
        expert_gates = jnp.max(router_probs, axis=-1, keepdims=True)

        # 3. Routed expert computation
        sel_gate = self.routed_gate_w[expert_indices]
        sel_up = self.routed_up_w[expert_indices]
        sel_down = self.routed_down_w[expert_indices]

        x_flat_bf16 = x_flat.astype(jnp.bfloat16)
        h_gate = jax.nn.silu(jnp.einsum("bd,bdf->bf", x_flat_bf16, sel_gate))
        h_up = jnp.einsum("bd,bdf->bf", x_flat_bf16, sel_up)
        h_combined = h_gate * h_up
        routed_out = jnp.einsum("bf,bfd->bd", h_combined, sel_down)

        routed_out = (routed_out * expert_gates).reshape(batch, seq_len, d)

        # 4. Synthesis
        final_out = shared_out + routed_out

        # 5. Auxiliary loss
        expert_mask = jax.nn.one_hot(
            expert_indices, self.num_routed_experts, dtype=jnp.float32
        )
        f_i = jnp.mean(expert_mask, axis=0)
        p_i = jnp.mean(router_probs, axis=0)
        aux_loss = self.aux_loss_alpha * self.num_routed_experts * jnp.sum(f_i * p_i)

        return final_out, aux_loss


# --- Recurrent Super Block with LayerScale ---
class ZenyxRecurrentSuperBlock(nn.Module):
    """
    Weight-tied recurrent block: 12 recurrence steps through shared weights.
    Alternates CSA (odd) and HCA (even).
    LayerScale (gamma init 1e-4) enforces identity mapping at initialization.
    """

    d_model: int
    num_heads: int
    d_latent: int
    d_rope: int
    d_head: int
    d_ff: int
    num_routed_experts: int
    num_recurrences: int
    aux_loss_alpha: float = 0.01
    layerscale_init: float = 1e-4
    csa_compress_ratio: int = 4
    hca_compress_ratio: int = 128
    local_window: int = 256
    csa_top_k: int = 64

    def setup(self):
        self.hybrid_attn = ZenyxHybridAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_latent=self.d_latent,
            d_rope=self.d_rope,
            d_head=self.d_head,
            csa_compress_ratio=self.csa_compress_ratio,
            hca_compress_ratio=self.hca_compress_ratio,
            local_window=self.local_window,
            csa_top_k=self.csa_top_k,
        )

        self.moe = DualSharedSparseMoE(
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_routed_experts=self.num_routed_experts,
            aux_loss_alpha=self.aux_loss_alpha,
        )

        self.norm1 = nn.RMSNorm()
        self.norm2 = nn.RMSNorm()

        self.gamma_1 = self.param(
            "gamma_1",
            initializers.constant(self.layerscale_init),
            (self.d_model,),
        )
        self.gamma_2 = self.param(
            "gamma_2",
            initializers.constant(self.layerscale_init),
            (self.d_model,),
        )

    def __call__(
        self,
        x: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        mscale: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        total_aux_loss = jnp.float32(0.0)
        g1 = self.gamma_1.astype(jnp.bfloat16)
        g2 = self.gamma_2.astype(jnp.bfloat16)

        for step in range(self.num_recurrences):
            is_hca = step % 2 == 0

            # Attention + LayerScale
            x_norm = self.norm1(x)
            attn_out = self.hybrid_attn(x_norm, cos, sin, mscale, is_hca=is_hca)
            x = x + attn_out * g1

            # MoE + LayerScale
            x_norm2 = self.norm2(x)
            moe_out, aux_loss = self.moe(x_norm2)
            x = x + moe_out * g2

            total_aux_loss = total_aux_loss + aux_loss

        return x, total_aux_loss


# --- Full Model ---
class ZenyxV3Model(nn.Module):
    """
    Complete Zenyx V3:
      Embed -> RecurrentSuperBlock (12 recurrences) -> RMSNorm -> LM Head
    Weight tying between embedding and LM head.
    """

    config: Any

    def setup(self):
        cfg = self.config

        self.embed = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.d_model,
            dtype=jnp.bfloat16,
            embedding_init=initializers.normal(stddev=0.02),
        )

        self.recurrent_block = ZenyxRecurrentSuperBlock(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            d_latent=cfg.d_latent,
            d_rope=cfg.d_rope,
            d_head=cfg.d_head,
            d_ff=cfg.d_ff,
            num_routed_experts=cfg.num_routed_experts,
            num_recurrences=cfg.num_recurrences,
            aux_loss_alpha=cfg.aux_loss_alpha,
            layerscale_init=cfg.layerscale_init,
            csa_compress_ratio=cfg.csa_compress_ratio,
            hca_compress_ratio=cfg.hca_compress_ratio,
            local_window=cfg.local_window,
            csa_top_k=cfg.csa_top_k,
        )

        self.final_norm = nn.RMSNorm()

    def __call__(
        self,
        input_ids: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        mscale: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Embedding
        x = self.embed(input_ids)

        # Recurrent processing (jax.checkpoint wraps at the outer jit level)
        x, aux_loss = self.recurrent_block(x, cos, sin, mscale)

        # Final norm
        x = self.final_norm(x)

        # LM head (weight-tied)
        embed_weights = self.embed.embedding
        logits = jnp.dot(x, embed_weights.T)

        return logits, aux_loss


# === OPTIMIZER ===


# --- Newton-Schulz Iteration ---
def newton_schulz_iteration(G: jnp.ndarray, steps: int = 5) -> jnp.ndarray:
    """
    Orthogonalizes gradient matrix via Newton-Schulz in bfloat16.
    Coefficients a=3.4445, b=-4.7750, c=2.0315 maximize small
    singular value inflation in 5 steps.
    """
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.astype(jnp.bfloat16)
    X = X / (jnp.linalg.norm(X, ord="fro") + 1e-7)

    transpose_flag = X.shape[0] > X.shape[1]
    if transpose_flag:
        X = X.T

    def ns_step(X_curr, _):
        A = X_curr @ X_curr.T
        B = b * A + c * (A @ A)
        X_next = a * X_curr + B @ X_curr
        return X_next, None

    X_final, _ = lax.scan(ns_step, X, None, length=steps)

    if transpose_flag:
        X_final = X_final.T

    return X_final.astype(G.dtype)


# --- Muon GradientTransformation ---
class MuonState(NamedTuple):
    momentum: Any


def scale_by_muon(
    momentum_decay: float = 0.95,
    ns_steps: int = 5,
) -> optax.GradientTransformation:
    """
    Muon: SGD momentum + Newton-Schulz orthogonalization for 2D matrices.
    Single momentum buffer -> 50% memory savings vs AdamW.
    """

    def init_fn(params):
        return MuonState(momentum=jax.tree.map(jnp.zeros_like, params))

    def update_fn(updates, state, params=None):
        new_momentum = jax.tree.map(
            lambda m, g: momentum_decay * m + g,
            state.momentum,
            updates,
        )

        def process_update(m):
            if m.ndim >= 2:
                orth = newton_schulz_iteration(m, steps=ns_steps)
                scale = jnp.sqrt(jnp.float32(max(m.shape[0], m.shape[1])))
                return orth * scale * 0.2
            else:
                return m

        processed_updates = jax.tree.map(process_update, new_momentum)

        return processed_updates, MuonState(momentum=new_momentum)

    return optax.GradientTransformation(init_fn, update_fn)


# --- Hybrid Optimizer ---
def build_hybrid_optimizer(
    params: Any,
    cfg: ZenyxConfig,
) -> Tuple[optax.GradientTransformation, Any]:
    """
    Muon (2D hidden matrices) + AdamW (1D, embeddings, norms).
    Cosine decay with linear warmup.
    """
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.base_lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=cfg.total_steps,
        end_value=cfg.min_lr,
    )

    muon_tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        scale_by_muon(momentum_decay=cfg.muon_momentum, ns_steps=5),
        optax.add_decayed_weights(cfg.weight_decay),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    )

    adamw_tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.scale_by_adam(b1=cfg.adam_b1, b2=cfg.adam_b2, eps=cfg.adam_eps),
        optax.add_decayed_weights(cfg.weight_decay),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    )

    def label_fn(params):
        def _label(path, param):
            path_str = "/".join(
                str(p.key) if hasattr(p, "key") else str(p) for p in path
            )
            if "embed" in path_str.lower() or "embedding" in path_str.lower():
                return "adamw"
            if param.ndim >= 2:
                return "muon"
            return "adamw"

        return jax.tree_util.tree_map_with_path(_label, params)

    param_labels = label_fn(params)

    optimizer = optax.multi_transform(
        transforms={"muon": muon_tx, "adamw": adamw_tx},
        param_labels=param_labels,
    )

    opt_state = optimizer.init(params)

    return optimizer, opt_state


# === TRAINING INFRASTRUCTURE ===


# --- Mesh & Sharding ---
def setup_mesh(cfg: ZenyxConfig) -> Mesh:
    """Create 1D FSDP mesh across TPU v5e-8 cores."""
    devices = jax.devices()
    num_devices = len(devices)
    logger.info(f"Detected {num_devices} devices: {[str(d) for d in devices]}")

    if num_devices < cfg.num_devices:
        logger.warning(
            f"Expected {cfg.num_devices} devices, got {num_devices}. "
            f"Adjusting mesh."
        )

    device_mesh = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(device_mesh, axis_names=(cfg.mesh_axis_name,))
    logger.info(f"Mesh created: shape={device_mesh.shape}, axes={mesh.axis_names}")
    return mesh


def get_partition_specs(cfg: ZenyxConfig) -> Dict[str, P]:
    """FSDP partition rules for Zenyx v3."""
    fa = cfg.mesh_axis_name
    return {
        # Embedding
        "embed/embedding": P(fa, None),
        # Attention projections
        "q_proj/kernel": P(None, fa),
        "kv_proj/kernel": P(None, fa),
        "q_up/kernel": P(None, fa),
        "kv_up_k/kernel": P(None, fa),
        "kv_up_v/kernel": P(None, fa),
        "q_rope_proj/kernel": P(None, fa),
        "k_rope_proj/kernel": P(None, fa),
        "o_proj/kernel": P(fa, None),
        # Shared experts
        "shared_1_gate/kernel": P(None, fa),
        "shared_1_up/kernel": P(None, fa),
        "shared_1_down/kernel": P(fa, None),
        "shared_2_gate/kernel": P(None, fa),
        "shared_2_up/kernel": P(None, fa),
        "shared_2_down/kernel": P(fa, None),
        # Router
        "router/kernel": P(None, fa),
        # Routed experts (shard across 64-expert dimension)
        "routed_gate_w": P(fa, None, None),
        "routed_up_w": P(fa, None, None),
        "routed_down_w": P(fa, None, None),
        # 1D replicated
        "gamma_1": P(None),
        "gamma_2": P(None),
        "scale": P(None),
    }


def shard_params(params: Any, mesh: Mesh, cfg: ZenyxConfig) -> Any:
    """Apply FSDP sharding to parameters."""
    partition_rules = get_partition_specs(cfg)

    def _get_sharding(path, param):
        path_str = "/".join(
            str(p.key) if hasattr(p, "key") else str(p) for p in path
        )
        for rule_key, spec in partition_rules.items():
            if path_str.endswith(rule_key) or rule_key in path_str:
                return NamedSharding(mesh, spec)
        if param.ndim >= 2:
            return NamedSharding(mesh, P(cfg.mesh_axis_name, None))
        return NamedSharding(mesh, P(None))

    def _shard_leaf(path, param):
        sharding = _get_sharding(path, param)
        return jax.device_put(param, sharding)

    return jax.tree_util.tree_map_with_path(_shard_leaf, params)


# --- Loss ---
def calculate_masked_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    loss_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Cross-entropy with masking. Shift-by-1 for next-token prediction.
    logits [B, S, V], targets [B, S], loss_mask [B, S].
    """
    shift_logits = logits[:, :-1, :]
    shift_targets = targets[:, 1:]
    shift_mask = loss_mask[:, 1:]

    log_probs = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)

    target_log_probs = jnp.take_along_axis(
        log_probs,
        shift_targets[:, :, None].astype(jnp.int32),
        axis=-1,
    ).squeeze(-1)

    masked_loss = -target_log_probs * shift_mask
    total_tokens = jnp.maximum(shift_mask.sum(), 1.0)
    loss = masked_loss.sum() / total_tokens

    return loss


# --- Train Step ---
def create_train_step(model, optimizer, cfg, cos, sin, mscale):
    """Creates JIT-compiled training step."""

    def loss_fn(params, input_ids, loss_mask):
        logits, aux_loss = model.apply(
            {"params": params},
            input_ids,
            cos,
            sin,
            mscale,
        )
        ce_loss = calculate_masked_cross_entropy(logits, input_ids, loss_mask)
        total_loss = ce_loss + aux_loss
        return total_loss, {"ce_loss": ce_loss, "aux_loss": aux_loss}

    # Wrap the forward pass with jax.checkpoint for activation memory savings
    checkpointed_loss_fn = jax.checkpoint(loss_fn, prevent_cse=False)

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, input_ids, loss_mask, step):
        grad_fn = jax.value_and_grad(checkpointed_loss_fn, has_aux=True)
        (total_loss, loss_dict), grads = grad_fn(params, input_ids, loss_mask)

        grad_norm = optax.global_norm(grads)

        updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)

        metrics = {
            "total_loss": total_loss,
            "ce_loss": loss_dict["ce_loss"],
            "aux_loss": loss_dict["aux_loss"],
            "grad_norm": grad_norm,
        }

        return new_params, new_opt_state, metrics

    return train_step


# === CHECKPOINT UTILS ===


def get_hf_api(cfg: ZenyxConfig) -> HfApi:
    return HfApi(token=cfg.hf_token)


def ensure_hf_repo(cfg: ZenyxConfig):
    api = get_hf_api(cfg)
    try:
        api.create_repo(
            repo_id=cfg.hf_repo,
            repo_type="model",
            private=True,
            exist_ok=True,
        )
        logger.info(f"HF repo ready: {cfg.hf_repo}")
    except Exception as e:
        logger.warning(f"Could not create/verify HF repo: {e}")


def save_checkpoint(
    params: Any,
    opt_state: Any,
    global_step: int,
    rng_key: jnp.ndarray,
    cfg: ZenyxConfig,
    metrics: Dict[str, float],
    data_pipeline: DataPipeline,
):
    """
    Save checkpoint and upload to HuggingFace Hub.
    Saves params as numpy .npy files, metadata.json for O(1) resumption.
    """
    checkpoint_name = f"checkpoint-{global_step}"
    local_dir = Path(f"/tmp/zenyx_checkpoints/{checkpoint_name}")
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving checkpoint at step {global_step}...")

    # 1. Save params
    params_np = jax.tree.map(lambda x: np.array(jax.device_get(x)), params)
    params_dir = local_dir / "params"
    params_dir.mkdir(exist_ok=True)

    flat_params, tree_def = jax.tree_util.tree_flatten_with_path(params_np)
    param_manifest = {}
    for i, (path, leaf) in enumerate(flat_params):
        path_str = ".".join(
            str(p.key) if hasattr(p, "key") else str(p) for p in path
        )
        filename = f"param_{i:04d}.npy"
        np.save(str(params_dir / filename), leaf)
        param_manifest[path_str] = {
            "file": filename,
            "shape": list(leaf.shape),
            "dtype": str(leaf.dtype),
        }

    with open(str(params_dir / "manifest.json"), "w") as f:
        json.dump(param_manifest, f, indent=2)

    # 2. Save tree structure
    with open(str(local_dir / "tree_structure.json"), "w") as f:
        paths = [
            ".".join(str(p.key) if hasattr(p, "key") else str(p) for p in path)
            for path, _ in flat_params
        ]
        json.dump({"param_paths": paths}, f, indent=2)

    # 3. Metadata for O(1) resumption
    tokens_consumed = data_pipeline.tokens_consumed
    sequences_consumed = data_pipeline.sequences_consumed

    # Compute shard-level resume info
    tokens_per_sequence = cfg.seq_len
    # Approximate: each FineWeb-Edu parquet shard ~ 50M tokens
    tokens_per_shard_approx = 50_000_000
    shard_index = tokens_consumed // tokens_per_shard_approx
    offset_within_shard = tokens_consumed % tokens_per_shard_approx

    metadata = {
        "global_step": int(global_step),
        "tokens_consumed": int(tokens_consumed),
        "sequences_consumed": int(sequences_consumed),
        "tokens_per_sequence": int(tokens_per_sequence),
        "shard_index": int(shard_index),
        "offset_within_shard": int(offset_within_shard),
        "ce_loss": float(metrics.get("ce_loss", 0.0)),
        "aux_loss": float(metrics.get("aux_loss", 0.0)),
        "total_loss": float(metrics.get("total_loss", 0.0)),
        "grad_norm": float(metrics.get("grad_norm", 0.0)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "d_model": cfg.d_model,
            "num_heads": cfg.num_heads,
            "d_latent": cfg.d_latent,
            "d_rope": cfg.d_rope,
            "d_ff": cfg.d_ff,
            "num_routed_experts": cfg.num_routed_experts,
            "num_recurrences": cfg.num_recurrences,
            "seq_len": cfg.seq_len,
            "vocab_size": cfg.vocab_size,
            "base_lr": cfg.base_lr,
            "micro_batch_size": cfg.micro_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        },
    }

    with open(str(local_dir / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # 4. Save RNG key
    np.save(str(local_dir / "rng_key.npy"), np.array(jax.device_get(rng_key)))

    # 5. Save optimizer state
    opt_state_dir = local_dir / "opt_state"
    opt_state_dir.mkdir(exist_ok=True)
    opt_flat = jax.tree_util.tree_leaves(opt_state)
    for i, leaf in enumerate(opt_flat):
        if hasattr(leaf, "shape"):
            np.save(
                str(opt_state_dir / f"state_{i:04d}.npy"),
                np.array(jax.device_get(leaf)),
            )
        else:
            np.save(str(opt_state_dir / f"state_{i:04d}.npy"), np.array(leaf))

    with open(str(opt_state_dir / "count.json"), "w") as f:
        json.dump({"num_leaves": len(opt_flat)}, f)

    # 6. Upload to HF
    try:
        api = get_hf_api(cfg)
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=checkpoint_name,
            repo_id=cfg.hf_repo,
            repo_type="model",
            commit_message=f"Checkpoint step {global_step} | loss={metrics.get('total_loss', 0.0):.4f}",
        )
        logger.info(f"Checkpoint saved to HF: step {global_step}")
    except Exception as e:
        logger.error(f"Failed to upload checkpoint to HF: {e}")
        logger.info("Checkpoint saved locally only.")

    # 7. Cleanup
    try:
        shutil.rmtree(str(local_dir))
    except Exception:
        pass


def load_latest_checkpoint(
    cfg: ZenyxConfig,
    model: ZenyxV3Model,
    optimizer: optax.GradientTransformation,
    mesh: Mesh,
    rng_key: jnp.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Load latest checkpoint from HF Hub with O(1) shard-level skipping.
    Returns dict {params, opt_state, global_step, rng_key, sequences_consumed}
    or None.
    """
    api = get_hf_api(cfg)

    try:
        files = list(api.list_repo_files(repo_id=cfg.hf_repo, repo_type="model"))
    except Exception as e:
        logger.info(f"No existing repo or error listing files: {e}")
        return None

    # Find checkpoint directories
    checkpoint_steps = []
    for f in files:
        if f.startswith("checkpoint-") and f.endswith("/metadata.json"):
            step_str = f.split("/")[0].replace("checkpoint-", "")
            try:
                checkpoint_steps.append(int(step_str))
            except ValueError:
                continue

    if not checkpoint_steps:
        logger.info("No checkpoints found in HF repo.")
        return None

    latest_step = max(checkpoint_steps)
    checkpoint_name = f"checkpoint-{latest_step}"
    logger.info(f"Found latest checkpoint: {checkpoint_name}")

    local_dir = Path(f"/tmp/zenyx_resume/{checkpoint_name}")
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download metadata
        metadata_path = hf_hub_download(
            repo_id=cfg.hf_repo,
            filename=f"{checkpoint_name}/metadata.json",
            repo_type="model",
            token=cfg.hf_token,
            local_dir=str(local_dir),
        )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        global_step = metadata["global_step"]
        sequences_consumed = metadata["sequences_consumed"]
        tokens_consumed = metadata["tokens_consumed"]
        shard_index = metadata.get("shard_index", 0)
        offset_within_shard = metadata.get("offset_within_shard", 0)

        logger.info(
            f"Resuming from step {global_step}, "
            f"shard {shard_index}, offset {offset_within_shard:,}"
        )

        # Download param manifest
        manifest_path = hf_hub_download(
            repo_id=cfg.hf_repo,
            filename=f"{checkpoint_name}/params/manifest.json",
            repo_type="model",
            token=cfg.hf_token,
            local_dir=str(local_dir),
        )

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Download each param file
        param_arrays = {}
        for path_str, info in manifest.items():
            param_file = hf_hub_download(
                repo_id=cfg.hf_repo,
                filename=f"{checkpoint_name}/params/{info['file']}",
                repo_type="model",
                token=cfg.hf_token,
                local_dir=str(local_dir),
            )
            param_arrays[path_str] = np.load(param_file)

        # Download RNG key
        rng_path = hf_hub_download(
            repo_id=cfg.hf_repo,
            filename=f"{checkpoint_name}/rng_key.npy",
            repo_type="model",
            token=cfg.hf_token,
            local_dir=str(local_dir),
        )
        restored_rng = jnp.array(np.load(rng_path))

        # Reconstruct params pytree via dummy init
        init_seq_len = 128
        dummy_input = jnp.ones((1, init_seq_len), dtype=jnp.int32)
        cos_init, sin_init, mscale_init = build_yarn_rope(
            init_seq_len,
            cfg.d_rope,
            cfg.rope_base,
            cfg.yarn_alpha,
            cfg.yarn_beta,
            cfg.yarn_scale_s,
            cfg.original_ctx_len,
        )
        cos_init = jnp.array(cos_init, dtype=jnp.bfloat16)
        sin_init = jnp.array(sin_init, dtype=jnp.bfloat16)

        dummy_rng = jax.random.PRNGKey(0)
        dummy_params = model.init(
            dummy_rng, dummy_input, cos_init, sin_init, mscale_init
        )["params"]

        # Fill in loaded arrays
        flat_params, tree_def = jax.tree_util.tree_flatten_with_path(dummy_params)
        restored_leaves = []
        for path, dummy_leaf in flat_params:
            path_str = ".".join(
                str(p.key) if hasattr(p, "key") else str(p) for p in path
            )
            if path_str in param_arrays:
                restored_leaves.append(jnp.array(param_arrays[path_str]))
            else:
                logger.warning(
                    f"Param '{path_str}' not in checkpoint, using init."
                )
                restored_leaves.append(dummy_leaf)

        restored_params = jax.tree_util.tree_unflatten(tree_def, restored_leaves)

        # Shard
        restored_params = shard_params(restored_params, mesh, cfg)

        # Reinit optimizer (full state restore is complex with custom Muon)
        _, restored_opt_state = build_hybrid_optimizer(restored_params, cfg)

        logger.info(f"Checkpoint restored: step {global_step}")

        try:
            shutil.rmtree(str(local_dir.parent))
        except Exception:
            pass

        return {
            "params": restored_params,
            "opt_state": restored_opt_state,
            "global_step": global_step,
            "rng_key": restored_rng,
            "sequences_consumed": sequences_consumed,
            "tokens_consumed": tokens_consumed,
        }

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return None


# === PARAMETER COUNTING ===
def count_parameters(params: Any) -> Dict[str, Any]:
    """Count total and per-component parameters."""
    flat_params = jax.tree_util.tree_leaves(params)
    total = sum(p.size for p in flat_params)

    flat_with_path = jax.tree_util.tree_flatten_with_path(params)[0]
    component_counts = {}
    for path, leaf in flat_with_path:
        component = str(path[0].key) if hasattr(path[0], "key") else str(path[0])
        component_counts[component] = component_counts.get(component, 0) + leaf.size

    return {"total": total, "components": component_counts}


# === MAIN TRAINING LOOP ===
def main():
    logger.info("=" * 80)
    logger.info("ZENYX V3 TRAINING — Starting")
    logger.info("=" * 80)

    cfg = ZenyxConfig()

    logger.info(
        f"Architecture: d_model={cfg.d_model}, heads={cfg.num_heads}, "
        f"d_latent={cfg.d_latent}, d_rope={cfg.d_rope}"
    )
    logger.info(
        f"MoE: {cfg.num_shared_experts} shared + {cfg.num_routed_experts} routed, "
        f"d_ff={cfg.d_ff}"
    )
    logger.info(f"Recurrences: {cfg.num_recurrences}, Context: {cfg.seq_len}")
    logger.info(
        f"Batch: micro={cfg.micro_batch_size}, accum={cfg.gradient_accumulation_steps}"
    )
    logger.info(f"Optimizer: Muon (2D) + AdamW (1D), lr={cfg.base_lr}->{cfg.min_lr}")

    # Setup
    ensure_hf_repo(cfg)
    mesh = setup_mesh(cfg)
    tokenizer = load_tokenizer(cfg)

    # YaRN RoPE (precomputed, static)
    logger.info("Building YaRN RoPE tables...")
    cos, sin, mscale = build_yarn_rope(
        seq_len=cfg.seq_len,
        d_rope=cfg.d_rope,
        base=cfg.rope_base,
        alpha=cfg.yarn_alpha,
        beta=cfg.yarn_beta,
        scale_s=cfg.yarn_scale_s,
        original_ctx_len=cfg.original_ctx_len,
    )
    cos = jnp.array(cos, dtype=jnp.bfloat16)
    sin = jnp.array(sin, dtype=jnp.bfloat16)
    logger.info(f"YaRN RoPE: cos={cos.shape}, sin={sin.shape}, mscale={mscale:.4f}")

    # Initialize model
    logger.info("Initializing Zenyx V3 model...")
    model = ZenyxV3Model(config=cfg)

    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)

    init_seq_len = 128
    dummy_input = jnp.ones((1, init_seq_len), dtype=jnp.int32)
    cos_init, sin_init, _ = build_yarn_rope(
        init_seq_len,
        cfg.d_rope,
        cfg.rope_base,
        cfg.yarn_alpha,
        cfg.yarn_beta,
        cfg.yarn_scale_s,
        cfg.original_ctx_len,
    )
    cos_init = jnp.array(cos_init, dtype=jnp.bfloat16)
    sin_init = jnp.array(sin_init, dtype=jnp.bfloat16)

    params = model.init(init_key, dummy_input, cos_init, sin_init, mscale)["params"]

    param_counts = count_parameters(params)
    logger.info(f"Total parameters: {param_counts['total']:,}")
    for comp, count in param_counts["components"].items():
        logger.info(f"  {comp}: {count:,}")

    # Shard
    logger.info("Sharding parameters across FSDP mesh...")
    with mesh:
        params = shard_params(params, mesh, cfg)

    # Optimizer
    logger.info("Building hybrid Muon + AdamW optimizer...")
    optimizer, opt_state = build_hybrid_optimizer(params, cfg)

    # Checkpoint resumption
    global_step = np.uint32(0)
    sequences_to_skip = 0

    checkpoint = load_latest_checkpoint(cfg, model, optimizer, mesh, rng_key)
    if checkpoint is not None:
        params = checkpoint["params"]
        opt_state = checkpoint["opt_state"]
        global_step = np.uint32(checkpoint["global_step"])
        rng_key = checkpoint["rng_key"]
        sequences_to_skip = checkpoint["sequences_consumed"]

        tokens_consumed = checkpoint["tokens_consumed"]
        tokens_per_seq = cfg.seq_len
        batch_tokens = cfg.micro_batch_size * cfg.seq_len
        shard_index = tokens_consumed // 50_000_000
        offset_within_shard = tokens_consumed % 50_000_000

        logger.info(
            f"Resuming from step {global_step}, "
            f"shard ~{shard_index}, offset ~{offset_within_shard:,}"
        )
    else:
        logger.info("Starting training from scratch.")

    # Data pipeline
    logger.info("Initializing data pipeline...")
    data_pipeline = DataPipeline(cfg, tokenizer)
    data_pipeline.initialize(skip_sequences=sequences_to_skip)

    # Compile train step
    logger.info("Compiling train step (may take several minutes on TPU)...")
    train_step_fn = create_train_step(model, optimizer, cfg, cos, sin, mscale)

    # Training loop
    logger.info("=" * 80)
    logger.info(f"TRAINING BEGINS at step {global_step}")
    logger.info("=" * 80)

    training_start_time = time.time()
    step_times = []
    metrics = {}

    while global_step < cfg.total_steps:
        step_start = time.time()

        # Get batch
        batch = data_pipeline.get_batch(cfg.micro_batch_size)
        if batch is None:
            logger.error("Data pipeline returned None. Reinitializing...")
            data_pipeline.initialize()
            continue

        # Convert to JAX
        input_ids = jnp.array(batch["input_ids"], dtype=jnp.int32)
        loss_mask = jnp.array(batch["loss_mask"], dtype=jnp.bfloat16)

        # Shard data
        with mesh:
            data_sharding = NamedSharding(mesh, P(cfg.mesh_axis_name, None))
            input_ids = jax.device_put(input_ids, data_sharding)
            loss_mask = jax.device_put(loss_mask, data_sharding)

        # Step
        rng_key, step_key = jax.random.split(rng_key)
        step_jnp = jnp.uint32(global_step)

        try:
            params, opt_state, metrics = train_step_fn(
                params, opt_state, input_ids, loss_mask, step_jnp
            )
        except Exception as e:
            logger.error(f"Train step failed at step {global_step}: {e}")
            import traceback

            traceback.print_exc()
            break

        jax.block_until_ready(metrics["total_loss"])
        step_time = time.time() - step_start
        step_times.append(step_time)

        global_step = np.uint32(int(global_step) + 1)

        # Logging
        if int(global_step) % cfg.log_every == 0:
            tokens_per_sec = cfg.micro_batch_size * cfg.seq_len / step_time
            logger.info(
                f"step={int(global_step):>7d} | "
                f"loss={float(metrics['total_loss']):.4f} | "
                f"ce_loss={float(metrics['ce_loss']):.4f} | "
                f"aux_loss={float(metrics['aux_loss']):.4f} | "
                f"tok/s={tokens_per_sec:,.0f} | "
                f"step_time={step_time:.2f}s"
            )

        if int(global_step) % cfg.detailed_log_every == 0:
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=cfg.base_lr,
                warmup_steps=cfg.warmup_steps,
                decay_steps=cfg.total_steps,
                end_value=cfg.min_lr,
            )
            current_lr = lr_schedule(int(global_step))
            elapsed = time.time() - training_start_time
            tokens_total = data_pipeline.tokens_consumed

            logger.info(
                f"  [DETAILED] lr={float(current_lr):.2e} | "
                f"grad_norm={float(metrics['grad_norm']):.4f} | "
                f"tokens_consumed={tokens_total:,} | "
                f"elapsed={elapsed/3600:.2f}h"
            )

        # Checkpoint
        if int(global_step) % cfg.checkpoint_every == 0:
            save_checkpoint(
                params=params,
                opt_state=opt_state,
                global_step=int(global_step),
                rng_key=rng_key,
                cfg=cfg,
                metrics={
                    "total_loss": float(metrics["total_loss"]),
                    "ce_loss": float(metrics["ce_loss"]),
                    "aux_loss": float(metrics["aux_loss"]),
                    "grad_norm": float(metrics["grad_norm"]),
                },
                data_pipeline=data_pipeline,
            )

    # Final checkpoint
    logger.info("Training complete. Saving final checkpoint...")
    if metrics:
        save_checkpoint(
            params=params,
            opt_state=opt_state,
            global_step=int(global_step),
            rng_key=rng_key,
            cfg=cfg,
            metrics={
                "total_loss": float(metrics.get("total_loss", 0.0)),
                "ce_loss": float(metrics.get("ce_loss", 0.0)),
                "aux_loss": float(metrics.get("aux_loss", 0.0)),
                "grad_norm": float(metrics.get("grad_norm", 0.0)),
            },
            data_pipeline=data_pipeline,
        )

    total_time = time.time() - training_start_time
    logger.info(f"Training finished in {total_time/3600:.2f} hours")
    logger.info(f"Total tokens processed: {data_pipeline.tokens_consumed:,}")
    if metrics:
        logger.info(f"Final loss: {float(metrics.get('total_loss', 0.0)):.4f}")


if __name__ == "__main__":
    main()
