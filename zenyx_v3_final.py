# Cell 1: Clean install
# !pip uninstall -y torch torchvision torchaudio 2>/dev/null
# !pip install -q torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Cell 2: JAX (Kept Kaggle original version for TPU!)
"""!pip install -q \
    "jax[tpu]>=0.4.20" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    --force-reinstall""" # I pasted as docstring as I dont want to harm code and also show them as unnecessary comments!

# Cell 3: (Installation required on Kaggle TPU)
# !pip install -q \
#     "flax>=0.12.2" \
#     "optax>=0.2.7" \
#     "transformers>=4.47.0" \
#     "datasets>=3.0.0" \
#     "huggingface_hub>=0.26.0"
# Same goes for Cell 3 as well, the installation commands were run in Kaggle and hence have been pasted as commands and docstrings only for help! They should be run accordingly in terminal when required >.<!

# Cell 4: (Verification of JAX devices!)

import jax
print(jax.__version__)
print(jax.devices())


# Cell 5: (The whole working code!)

#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ZENYX V3 — BASE MODEL TRAINING (v2)                      ║
║                                                                              ║
║  Architecture: DeepSeek-V4-Pro adapted (MLA + MoE + HC + MTP)                ║
║  Optimizer: Muon (optax.contrib.muon) + AdamW hybrid                         ║
║  Context: Progressive 2048 → 4096 → 8192 with YaRN extension                 ║
║  Data: 60% FineWeb-Edu | 25% StarCoder | 15% NuminaMath-CoT                  ║
║  Resume: Shard-aware skipping with per-dataset ratio accounting              ║
║  Checkpoint: Auto-upload to HF every 350 steps, full state resumable         ║
║  Tokenizer: DeepSeek-V4-Pro (129,280 vocab)                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_tpu_spmd_rng_bit_generator_unsafe=true"

import sys
import re
import math
import time
import json
import pickle
import logging
import functools
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Any, Dict

import numpy as np

# ============================================================================
# JAX / FLAX / OPTAX
# ============================================================================
import jax
import jax.numpy as jnp
from jax import random, lax
from jax.sharding import NamedSharding, PartitionSpec as P

# Patch for optax <-> JAX version mismatch
_original_jax_update = jax.config.update
def _patched_update(name, val):
    try:
        _original_jax_update(name, val)
    except AttributeError:
        if name == 'jax_pmap_shmap_merge':
            return  # silently ignore removed config
        raise
jax.config.update = _patched_update

import flax.linen as nn
from flax.linen import initializers, remat
from flax.training import train_state
import optax

# ============================================================================
# HF / DATA
# ============================================================================
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets
from huggingface_hub import HfApi, login, hf_hub_download

jax.config.update("jax_default_matmul_precision", "bfloat16")

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("zenyx_v3")

# ============================================================================
# HARDCODED CREDENTIALS & REPO
# ============================================================================
HF_TOKEN = "HF_TOKEN_PASTE" # Paste the HF_TOKEN!
HF_REPO_ID = "Arko007/Zenyx-V3-Base"
TOKENIZER_REPO = "deepseek-ai/DeepSeek-V4-Pro"


# ================================================================
# CONFIGURATION
# ================================================================
@dataclass
class ZenyxConfig:
    # --- Model Architecture ---
    vocab_size: int = 129280
    dim: int = 1536
    n_layers: int = 16
    n_heads: int = 12
    head_dim: int = 128
    rope_head_dim: int = 32
    q_lora_rank: int = 384
    o_groups: int = 4
    o_lora_rank: int = 256
    norm_eps: float = 1e-6

    # --- MoE ---
    n_routed_experts: int = 12
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    moe_inter_dim: int = 1408
    score_func: str = "softmax"
    route_scale: float = 1.0
    swiglu_limit: float = 15.0
    n_dense_layers: int = 2
    load_balance_alpha: float = 0.008

    # --- Hyper-Connections ---
    hc_mult: int = 3
    hc_sinkhorn_iters: int = 8
    hc_eps: float = 1e-6

    # --- KV Compression ---
    window_size: int = 128
    compress_ratio: int = 4

    # --- MTP ---
    n_mtp_layers: int = 1
    mtp_loss_weight: float = 0.02

    # --- RoPE / YaRN ---
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    original_seq_len: int = 0
    beta_fast: int = 32
    beta_slow: int = 1

    # --- Progressive Context Length ---
    # Phase 0: steps 0-60000 → seq_len=2048
    # Phase 1: steps 60001-80000 → seq_len=4096
    # Phase 2: steps 80001-100000 → seq_len=8192 (with YaRN)
    ctx_phase_boundaries: Tuple[int, ...] = (60000, 80000, 100000)
    ctx_phase_lengths: Tuple[int, ...] = (2048, 4096, 8192)
    # YaRN params for phase 2 (8192 extension from 4096 base)
    yarn_scale: float = 40.0
    yarn_alpha: float = 1.0
    yarn_beta: float = 32.0
    yarn_original_seq_len: int = 4096

    # --- Training ---
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    max_lr: float = 3e-4           # Moonlight recipe (Muon + MoE)
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    total_steps: int = 200000
    decay_steps: int = 180000
    weight_decay: float = 0.1
    muon_beta: float = 0.95
    muon_ns_steps: int = 7
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    grad_clip_norm: float = 1.0
    init_std: float = 0.006
    dtype: str = "bfloat16"

    # --- Data ---
    fineweb_config: str = "default"
    hf_token: str = HF_TOKEN

    # --- Data pipeline token counts per shard iteration ---
    approx_tokens_per_fineweb_doc: int = 800
    approx_tokens_per_code_doc: int = 1200
    approx_tokens_per_math_doc: int = 600
    data_mix_probs: Tuple[float, ...] = (0.60, 0.25, 0.15)

    # --- Checkpointing ---
    save_every_steps: int = 350
    log_every_steps: int = 10
    push_every_steps: int = 350
    checkpoint_dir: str = "/kaggle/working/zenyxv3checkpoints"

    def get_seq_len_for_step(self, step: int) -> int:
        """Returns the sequence length for the current training step."""
        for boundary, length in zip(self.ctx_phase_boundaries, self.ctx_phase_lengths):
            if step < boundary:
                return length
        return self.ctx_phase_lengths[-1]

    def get_batch_size_for_step(self, step: int) -> int:
        """Scale batch size inversely with seq_len to keep tokens/step constant."""
        base_tokens = self.micro_batch_size * self.ctx_phase_lengths[0]
        current_seq = self.get_seq_len_for_step(step)
        return max(1, base_tokens // current_seq)


# ================================================================
# RMS NORM
# ================================================================
class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        weight = self.param("weight", initializers.ones, (self.dim,))
        var = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
        x_normed = x.astype(jnp.float32) * jax.lax.rsqrt(var + self.eps)
        return (weight * x_normed).astype(x.dtype)


# ================================================================
# ROTARY POSITIONAL EMBEDDINGS WITH YaRN
# ================================================================
def precompute_freqs_cis(
    dim: int, max_len: int, theta: float = 10000.0,
    factor: float = 1.0, original_seq_len: int = 0,
    beta_fast: int = 32, beta_slow: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    if original_seq_len > 0 and factor > 1.0:
        def find_correction_dim(num_rotations):
            return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(theta))
        low = max(math.floor(find_correction_dim(beta_fast)), 0)
        high = min(math.ceil(find_correction_dim(beta_slow)), dim - 1)
        if low == high:
            high = low + 1
        smooth = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = np.arange(max_len, dtype=np.float32)
    freqs = np.outer(t, freqs)
    return jnp.array(np.cos(freqs), dtype=jnp.float32), jnp.array(np.sin(freqs), dtype=jnp.float32)


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    cos = cos.astype(x.dtype)
    sin = sin.astype(x.dtype)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


# ================================================================
# HYPER-CONNECTIONS (HC)
# ================================================================
class HyperConnections(nn.Module):
    dim: int
    hc_mult: int = 4
    sinkhorn_iters: int = 10
    eps: float = 1e-6

    def setup(self):
        hc = self.hc_mult
        mix_hc = (2 + hc) * hc
        hc_dim = hc * self.dim
        self.hc_fn = self.param("hc_fn", initializers.normal(0.02), (mix_hc, hc_dim))
        self.hc_base = self.param("hc_base", initializers.zeros, (mix_hc,))
        self.hc_scale = self.param("hc_scale", initializers.ones, (3,))

    def sinkhorn(self, comb):
        eps = self.eps
        comb = comb - jnp.max(comb, axis=-1, keepdims=True)
        comb = jnp.exp(comb)
        row_sum = jnp.sum(comb, axis=-1, keepdims=True)
        comb = comb / (row_sum + eps) + eps
        col_sum = jnp.sum(comb, axis=-2, keepdims=True)
        comb = comb / (col_sum + eps)

        def sinkhorn_step(comb, _):
            row_sum = jnp.sum(comb, axis=-1, keepdims=True)
            comb = comb / (row_sum + eps)
            col_sum = jnp.sum(comb, axis=-2, keepdims=True)
            comb = comb / (col_sum + eps)
            return comb, None
        comb, _ = lax.scan(sinkhorn_step, comb, None, length=self.sinkhorn_iters - 1)
        return comb

    def pre(self, x):
        shape = x.shape
        hc = self.hc_mult
        x_flat = x.reshape(*shape[:2], -1).astype(jnp.float32)
        rsqrt = jax.lax.rsqrt(jnp.mean(x_flat ** 2, axis=-1, keepdims=True) + 1e-6)
        mixes = jnp.dot(x_flat, self.hc_fn.T) * rsqrt
        pre_weights = jax.nn.sigmoid(mixes[..., :hc] * self.hc_scale[0] + self.hc_base[:hc]) + self.eps
        post_weights = 2.0 * jax.nn.sigmoid(mixes[..., hc:2*hc] * self.hc_scale[1] + self.hc_base[hc:2*hc])
        comb_input = mixes[..., 2*hc:].reshape(*shape[:2], hc, hc) * self.hc_scale[2] + self.hc_base[2*hc:].reshape(hc, hc)
        comb = self.sinkhorn(comb_input)
        y = jnp.sum(pre_weights[..., None] * x.astype(jnp.float32), axis=2)
        return y.astype(x.dtype), post_weights, comb

    def post(self, x, residual, post_weights, comb):
        expanded = post_weights[..., None] * x[..., None, :]
        combined = jnp.einsum("...ij,...jd->...id", comb, residual.astype(jnp.float32))
        return (expanded + combined).astype(x.dtype)


class HCHead(nn.Module):
    dim: int
    hc_mult: int = 4
    eps: float = 1e-6

    def setup(self):
        hc_dim = self.hc_mult * self.dim
        self.hc_fn = self.param("hc_fn", initializers.normal(0.02), (self.hc_mult, hc_dim))
        self.hc_base = self.param("hc_base", initializers.zeros, (self.hc_mult,))
        self.hc_scale = self.param("hc_scale", initializers.ones, (1,))

    def __call__(self, x):
        shape = x.shape
        x_flat = x.reshape(*shape[:2], -1).astype(jnp.float32)
        rsqrt = jax.lax.rsqrt(jnp.mean(x_flat ** 2, axis=-1, keepdims=True) + 1e-6)
        mixes = jnp.dot(x_flat, self.hc_fn.T) * rsqrt
        pre = jax.nn.sigmoid(mixes * self.hc_scale + self.hc_base) + self.eps
        y = jnp.sum(pre[..., None] * x.astype(jnp.float32), axis=2)
        return y.astype(x.dtype)


# ================================================================
# MULTI-HEAD LATENT ATTENTION (MLA)
# ================================================================
class MLA(nn.Module):
    config: ZenyxConfig
    layer_id: int

    def setup(self):
        cfg = self.config
        self.wqa = nn.Dense(cfg.q_lora_rank, use_bias=False, kernel_init=initializers.normal(cfg.init_std))
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
        self.wqb = nn.Dense(cfg.n_heads * cfg.head_dim, use_bias=False, kernel_init=initializers.normal(cfg.init_std))
        self.wk = nn.Dense(cfg.head_dim, use_bias=False, kernel_init=initializers.normal(cfg.init_std))
        self.wv = nn.Dense(cfg.head_dim, use_bias=False, kernel_init=initializers.normal(cfg.init_std))
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.v_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.woa = nn.Dense(cfg.o_groups * cfg.o_lora_rank, use_bias=False, kernel_init=initializers.normal(cfg.init_std))
        self.wob = nn.Dense(cfg.dim, use_bias=False, kernel_init=initializers.normal(cfg.init_std))

    def __call__(self, x, freqs_cos, freqs_sin, deterministic=True):
        cfg = self.config
        B, S, D = x.shape
        rd = cfg.rope_head_dim
        nd = cfg.head_dim - rd
    
        # Q path
        qr = self.q_norm(self.wqa(x))
        q = self.wqb(qr).reshape(B, S, cfg.n_heads, cfg.head_dim)
        q_var = jnp.mean(q.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
        q = (q.astype(jnp.float32) * jax.lax.rsqrt(q_var + cfg.norm_eps)).astype(x.dtype)
        q_nope, q_rope = q[..., :nd], q[..., nd:]
        q_rope = apply_rotary_emb(q_rope, freqs_cos[:S, None, :], freqs_sin[:S, None, :])
        q = jnp.concatenate([q_nope, q_rope], axis=-1)  # [B, S, H, D]
    
        # K path — single-head MQA
        k = self.k_norm(self.wk(x))
        k_nope, k_rope = k[..., :nd], k[..., nd:]
        k_rope = apply_rotary_emb(k_rope, freqs_cos[:S], freqs_sin[:S])
        k = jnp.concatenate([k_nope, k_rope], axis=-1)  # [B, S, D]
    
        v = self.v_norm(self.wv(x))  # [B, S, D]
    
        # Broadcast to [B,H,S,D] for FlashAttn
        q_t = q.transpose(0, 2, 1, 3)  # [B, H, S, D]
        k_b = jnp.broadcast_to(k[:, None, :, :], (B, cfg.n_heads, S, cfg.head_dim))
        v_b = jnp.broadcast_to(v[:, None, :, :], (B, cfg.n_heads, S, cfg.head_dim))
    
        out = jax.nn.dot_product_attention(
            query=q_t, key=k_b, value=v_b,
            is_causal=True, scale=cfg.head_dim ** -0.5,
        )  # [B, H, S, D]
    
        o = out.transpose(0, 2, 1, 3)  # [B, S, H, D]
        
        # ✅ FIXED: No inverse RoPE on output — DeepSeek MLA does NOT do this
        # (removed the 3-line o_nope/o_rope block that was injecting noise)
    
        o_compressed = self.woa(o.reshape(B, S, -1))
        return self.wob(o_compressed)
        
# ================================================================
# EXPERT / MOE / GATE / DENSE FFN
# ================================================================
class Expert(nn.Module):
    dim: int
    inter_dim: int
    swiglu_limit: float = 0.0
    init_std: float = 0.006

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.inter_dim, use_bias=False, kernel_init=initializers.normal(self.init_std), name="w1")(x).astype(jnp.float32)
        up = nn.Dense(self.inter_dim, use_bias=False, kernel_init=initializers.normal(self.init_std), name="w3")(x).astype(jnp.float32)
        if self.swiglu_limit > 0:
            gate = jnp.clip(gate, max=self.swiglu_limit)
            up = jnp.clip(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        h = jax.nn.silu(gate) * up
        return nn.Dense(self.dim, use_bias=False, kernel_init=initializers.normal(self.init_std), name="w2")(h.astype(x.dtype))


class MoEGate(nn.Module):
    config: ZenyxConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        gate_weight = self.param("weight", initializers.normal(cfg.init_std), (cfg.n_routed_experts, cfg.dim))
        gate_bias = self.param("bias", initializers.zeros, (cfg.n_routed_experts,))
        scores = jnp.dot(x.astype(jnp.float32), gate_weight.T.astype(jnp.float32))
        if cfg.score_func == "sqrtsoftplus":
            scores = jnp.sqrt(jax.nn.softplus(scores))
        elif cfg.score_func == "sigmoid":
            scores = jax.nn.sigmoid(scores)
        else:
            scores = jax.nn.softmax(scores, axis=-1)
        original_scores = scores
        biased_scores = scores + gate_bias
        topk_vals, topk_indices = jax.lax.top_k(biased_scores, cfg.n_activated_experts)
        weights = jnp.take_along_axis(original_scores, topk_indices, axis=-1)
        if cfg.score_func != "softmax":
            weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-8)
        weights = weights * cfg.route_scale
        return weights, topk_indices, original_scores

class MoELayer(nn.Module):
    """
    Mixture-of-Experts with dispatch/combine einsum pattern.

    Instead of computing all experts on all tokens (8x FLOPs overhead),
    this dispatches tokens to their top-k assigned experts via one-hot
    gather, runs each expert on only its assigned tokens, then combines
    results back. Standard TPU/JAX pattern from Switch Transformer.

    With 16 experts and top-2: each expert processes ~2/16 = 12.5% of tokens
    (plus capacity_factor padding ≈ 25% buffer).
    """
    config: ZenyxConfig

    def setup(self):
        cfg = self.config
        self.gate = MoEGate(cfg)
        self.shared_expert = Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit, cfg.init_std)
        self.routed_experts = [
            Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit, cfg.init_std)
            for _ in range(cfg.n_routed_experts)
        ]

    def __call__(self, x):
        cfg = self.config
        B, S, D = x.shape
        num_tokens = B * S
        x_flat = x.reshape(num_tokens, D)

        weights, indices, all_scores = self.gate(x_flat)

        capacity_factor = 1.25
        # Round up to multiple of 8 for TPU data-parallel sharding
        capacity = int(capacity_factor * num_tokens * cfg.n_activated_experts / cfg.n_routed_experts)
        capacity = max(capacity, 1)
        n_devices = len(jax.devices())
        capacity = math.ceil(capacity / n_devices) * n_devices  # e.g. 319 → 320

        expert_one_hot = jax.nn.one_hot(indices, cfg.n_routed_experts)
        flat_one_hot = expert_one_hot.reshape(num_tokens * cfg.n_activated_experts, cfg.n_routed_experts)
        positions = jnp.cumsum(flat_one_hot, axis=0) * flat_one_hot
        positions = positions.reshape(num_tokens, cfg.n_activated_experts, cfg.n_routed_experts)

        capacity_mask = (positions > 0) & (positions <= capacity)
        positions = (positions - 1) * capacity_mask
        dispatch_weights = weights[..., None] * expert_one_hot * capacity_mask.astype(weights.dtype)

        expert_inputs = jnp.zeros((cfg.n_routed_experts, capacity, D), dtype=x.dtype)

        # Sharding constraint: applied when mesh is active (training loop),
        # skipped gracefully during single-device init.
        try:
            expert_inputs = jax.lax.with_sharding_constraint(
                expert_inputs, P(None, 'data', None)
            )
        except RuntimeError:
            pass

        for k in range(cfg.n_activated_experts):
            exp_idx = indices[:, k]
            pos_idx = positions[:, k, :]
            token_positions = jnp.take_along_axis(
                pos_idx, exp_idx[:, None], axis=1
            ).squeeze(1).astype(jnp.int32)
            w = dispatch_weights[:, k, :]
            token_weights = jnp.take_along_axis(
                w, exp_idx[:, None], axis=1
            ).squeeze(1)
            # ✅ FIX: cast token_weights to x.dtype (bf16) before scatter.
            # weights from MoEGate are fp32. Without this cast, JAX inserts
            # an implicit fp32→bf16 conversion buffer on every .at[].add()
            # call — that's 2 dispatches × n_moe_layers short-lived fp32
            # temporaries that fragment HBM right when you're near the limit.
            token_weights = token_weights.astype(x.dtype)
            expert_inputs = expert_inputs.at[exp_idx, token_positions].add(
                x_flat * token_weights[:, None]
            )

        expert_outputs = jnp.stack([
            self.routed_experts[i](expert_inputs[i])
            for i in range(cfg.n_routed_experts)
        ], axis=0)

        routed_out = jnp.zeros((num_tokens, D), dtype=jnp.float32)
        for k in range(cfg.n_activated_experts):
            exp_idx = indices[:, k]
            pos_idx = positions[:, k, :]
            token_positions = jnp.take_along_axis(
                pos_idx, exp_idx[:, None], axis=1
            ).squeeze(1).astype(jnp.int32)
            gathered = expert_outputs[exp_idx, token_positions]
            routed_out += gathered.astype(jnp.float32)

        shared_out = self.shared_expert(x_flat)
        out = routed_out.astype(x.dtype) + shared_out
        out = out.reshape(B, S, D)

        expert_counts = jnp.sum(expert_one_hot.sum(axis=1), axis=0)
        fraction_tokens = expert_counts / (num_tokens * cfg.n_activated_experts)
        fraction_scores = jnp.mean(all_scores, axis=0)
        balance_loss = cfg.n_routed_experts * jnp.sum(fraction_tokens * fraction_scores)

        return out, balance_loss
        
class DenseFFN(nn.Module):
    config: ZenyxConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        inter_dim = cfg.moe_inter_dim * cfg.n_activated_experts
        return Expert(cfg.dim, inter_dim, cfg.swiglu_limit, cfg.init_std)(x), 0.0

# ================================================================
# TRANSFORMER BLOCK / MTP / FULL MODEL
# ================================================================
class TransformerBlock(nn.Module):
    config: ZenyxConfig
    layer_id: int

    def setup(self):
        cfg = self.config
        self.hc_attn = HyperConnections(cfg.dim, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps)
        self.hc_ffn = HyperConnections(cfg.dim, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps)
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = MLA(cfg, self.layer_id)
        self.ffn = DenseFFN(cfg) if self.layer_id < cfg.n_dense_layers else MoELayer(cfg)

    def __call__(self, x, freqs_cos, freqs_sin, deterministic=True):
        residual = x
        h, post_w, comb = self.hc_attn.pre(x)
        h = self.attn_norm(h)
        h = self.attn(h, freqs_cos, freqs_sin, deterministic=deterministic)
        x = self.hc_attn.post(h, residual, post_w, comb)
        residual = x
        h, post_w, comb = self.hc_ffn.pre(x)
        h = self.ffn_norm(h)
        h, balance_loss = self.ffn(h)
        x = self.hc_ffn.post(h, residual, post_w, comb)
        return x, balance_loss

class MTPBlock(nn.Module):
    config: ZenyxConfig
    layer_id: int

    def setup(self):
        cfg = self.config
        self.e_proj = nn.Dense(cfg.dim, use_bias=False, kernel_init=initializers.normal(cfg.init_std))
        self.h_proj = nn.Dense(cfg.dim, use_bias=False, kernel_init=initializers.normal(cfg.init_std))
        self.enorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.hnorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.block = TransformerBlock(cfg, self.layer_id)

    def __call__(self, hidden_states, embeddings, freqs_cos, freqs_sin, deterministic=True):
        e = self.enorm(embeddings)
        h = self.hnorm(hidden_states)
        combined = self.e_proj(e).astype(jnp.float32) + self.h_proj(h).astype(jnp.float32)
        combined = combined.astype(hidden_states.dtype)
        combined_hc = jnp.broadcast_to(
            combined[..., None, :],
            (*combined.shape[:-1], self.config.hc_mult, self.config.dim)
        )
        combined_hc = jnp.array(combined_hc)  # ✅ materialize — matches ZenyxV3Model pattern
        return self.block(combined_hc, freqs_cos, freqs_sin, deterministic=deterministic)

class ZenyxV3Model(nn.Module):
    config: ZenyxConfig

    def setup(self):
        cfg = self.config
        self.embed = nn.Embed(cfg.vocab_size, cfg.dim, embedding_init=initializers.normal(cfg.init_std))
        # --- remat for activation checkpointing ---
        self.layers = [remat(TransformerBlock, prevent_cse=False)(cfg, i) 
                       for i in range(cfg.n_layers)]
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.hc_head = HCHead(cfg.dim, cfg.hc_mult, cfg.hc_eps)
        # --- REMOVED: self.lm_head = nn.Dense(...) ---
        # lm_head is now tied to embed (see __call__)
        self.mtp_layers = [remat(MTPBlock, prevent_cse=False)(cfg, cfg.n_layers + i) 
                           for i in range(cfg.n_mtp_layers)]
        self.mtp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mtp_hc_head = HCHead(cfg.dim, cfg.hc_mult, cfg.hc_eps)

    def __call__(self, input_ids, freqs_cos, freqs_sin, deterministic=True):
        cfg = self.config
        B, S = input_ids.shape
        h = self.embed(input_ids)
        embed_for_mtp = h
        h = jnp.broadcast_to(h[..., None, :], (B, S, cfg.hc_mult, cfg.dim))
        h = jnp.array(h)
        total_balance_loss = 0.0
        for layer in self.layers:
            h, bl = layer(h, freqs_cos, freqs_sin, deterministic=deterministic)
            total_balance_loss += bl
        h_reduced = self.hc_head(h)
        h_normed = self.norm(h_reduced)
        # --- CHANGED: tied lm_head ---
        logits = jnp.dot(h_normed, self.embed.embedding.T)
        
        mtp_logits = None
        if cfg.n_mtp_layers > 0 and S > 1:
            shifted_embed = embed_for_mtp[:, 1:]
            mtp_h = h[:, :-1]
            mtp_h = self.hc_head(mtp_h)
            cos_mtp = freqs_cos[1:S]
            sin_mtp = freqs_sin[1:S]
            for mtp_layer in self.mtp_layers:
                mtp_h, mtp_bl = mtp_layer(mtp_h, shifted_embed, cos_mtp, sin_mtp, deterministic=deterministic)
                total_balance_loss += mtp_bl
            mtp_reduced = self.mtp_hc_head(mtp_h)
            mtp_normed = self.mtp_norm(mtp_reduced)
            # --- CHANGED: tied lm_head for MTP ---
            mtp_logits = jnp.dot(mtp_normed, self.embed.embedding.T)
        return logits, mtp_logits, total_balance_loss

# ================================================================
# LOSS
# ================================================================
def cross_entropy_loss(logits, targets):
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), targets
    ).mean()

def compute_loss(logits, mtp_logits, balance_loss, input_ids, config):
    main_loss = cross_entropy_loss(logits[:, :-1], input_ids[:, 1:])
    total_loss = main_loss
    if mtp_logits is not None and config.n_mtp_layers > 0:
        mtp_targets = input_ids[:, 2:]
        mtp_preds = mtp_logits[:, :-1]
        if mtp_targets.shape[1] > 0:
            total_loss += config.mtp_loss_weight * cross_entropy_loss(mtp_preds, mtp_targets)
    total_loss += config.load_balance_alpha * balance_loss
    return total_loss, main_loss

# ================================================================
# LR SCHEDULE
# ================================================================
def create_lr_schedule(config: ZenyxConfig):
    def schedule_fn(step):
        warmup_factor = jnp.minimum(step / config.warmup_steps, 1.0)
        decay_step = jnp.maximum(step - config.warmup_steps, 0)
        decay_ratio = jnp.minimum(decay_step / config.decay_steps, 1.0)
        cosine_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
        lr = config.min_lr + (config.max_lr - config.min_lr) * cosine_factor
        return lr * warmup_factor
    return schedule_fn


# ================================================================
# OPTIMIZER
# ================================================================
def create_optimizer(config: ZenyxConfig):
    lr_schedule = create_lr_schedule(config)
    optimizer = optax.contrib.muon(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
        beta=config.muon_beta,
        ns_steps=config.muon_ns_steps,
        nesterov=True,
        consistent_rms=0.2,
        adam_b1=config.adam_beta1,
        adam_b2=config.adam_beta2,
        adam_weight_decay=config.weight_decay,
        mu_dtype=jnp.bfloat16,   # critical: halves Muon memory
    )
    return optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optimizer,
    )

# ================================================================
# TRAIN STEP
# ================================================================
def make_compute_grads(config):
    @jax.jit
    def compute_grads(state, batch, freqs_cos, freqs_sin):
        def loss_fn(params):
            logits, mtp_logits, balance_loss = state.apply_fn(
                params, batch, freqs_cos, freqs_sin, deterministic=True
            )
            total_loss, main_loss = compute_loss(logits, mtp_logits, balance_loss, batch, config)
            return total_loss, main_loss
        (total_loss, main_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return grads, total_loss, main_loss
    return compute_grads


@jax.jit
def apply_accumulated_grads(state, grads):
    """JIT-compiled function that applies pre-averaged gradients ONCE."""
    return state.apply_gradients(grads=grads)


# ================================================================
# TRAINING STATE CREATION
# ================================================================
def create_train_state(rng, config: ZenyxConfig, mesh, max_seq_len: int = 2048):
    model = ZenyxV3Model(config)
    # ✅ Init dummy input at PHASE 0 seq_len — avoids 3.95GB logit slab at 8192
    # max_seq_len is still used to precompute freqs at full length so sharding is correct
    init_seq_len = config.ctx_phase_lengths[0]  # 2048 — safe for init
    dummy_input = jnp.ones((1, init_seq_len), dtype=jnp.int32)
    cos_f, sin_f = precompute_freqs_cis(
        config.rope_head_dim, init_seq_len, config.rope_theta
    )

    with mesh:
        with jax.default_device(jax.devices()[0]):
            params = model.init(rng, dummy_input, cos_f, sin_f)

    # Cast float32 → bfloat16
    params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if hasattr(x, 'dtype') and x.dtype == jnp.float32 else x,
        params
    )

    def make_sharding(x):
        if not hasattr(x, 'ndim'):
            return NamedSharding(mesh, P())
        # Only shard the two large matrices that benefit from it AND are safely divisible:
        # embed (129280, 1536): 129280 / 8 = 16160 ✅
        # lm_head (1536, 129280): 1536 / 8 = 192 ✅
        # Everything else → replicate across all 8 chips
        if x.ndim == 2 and x.shape[0] % len(jax.devices()) == 0 and x.shape[0] >= 1024:
            return NamedSharding(mesh, P('data', None))
        return NamedSharding(mesh, P())

    with mesh:
        params = jax.device_put(
            params, jax.tree_util.tree_map(make_sharding, params)
        )

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"Model initialized: {param_count:,} parameters ({param_count/1e9:.2f}B)")

    optimizer = create_optimizer(config)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )
    return state, model

# ================================================================
# DATA PIPELINE
# ================================================================
STARCODER_LANGS =[
    "ada","agda","alloy","antlr","applescript","assembly","augeas","awk",
    "batchfile","bluespec","c-sharp","c","clojure","cmake","coffeescript",
    "common-lisp","cpp","css","cuda","dart","dockerfile","elixir","elm",
    "emacs-lisp","erlang","f-sharp","fortran","git-commits-cleaned",
    "github-issues-filtered-structured","glsl","go","groovy","haskell",
    "html","idris","isabelle","java-server-pages","java","javascript",
    "json","julia","jupyter-scripts-dedup-filtered","jupyter-structured-clean-dedup",
    "kotlin","lean","literate-agda","literate-coffeescript","literate-haskell",
    "lua","makefile","maple","markdown","mathematica","matlab","ocaml",
    "pascal","perl","php","powershell","prolog","protocol-buffer","python",
    "r","racket","restructuredtext","rmarkdown","ruby","rust","sas",
    "scala","scheme","shell","smalltalk","solidity","sparql","sql",
    "stan","standard-ml","stata","systemverilog","tcl","tcsh","tex",
    "thrift","typescript","verilog","vhdl","visual-basic","xslt","yacc",
    "yaml","zig"
]

def build_streaming_dataset(cfg):
    logger.info(f"Building streaming dataset (fineweb-edu config='{cfg.fineweb_config}') ...")
    fw_edu = load_dataset("HuggingFaceFW/fineweb-edu", cfg.fineweb_config, split="train", streaming=True, token=cfg.hf_token).select_columns(["text"])
    code_datasets =[]
    for lang in STARCODER_LANGS:
        try:
            ds = load_dataset("bigcode/starcoderdata", data_dir=lang, split="train", streaming=True, token=cfg.hf_token).select_columns(["content"]).rename_column("content", "text")
            code_datasets.append(ds)
        except Exception as e:
            logger.warning(f"StarCoder subset '{lang}' load failed: {e}")
    code_ds = interleave_datasets(code_datasets, seed=42, stopping_strategy="all_exhausted")
    math_ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True, token=cfg.hf_token).map(lambda ex: {"text": f"Problem: {ex['problem']}\n\nSolution: {ex['solution']}"}, remove_columns=["source","problem","solution","messages"])
    mixed = interleave_datasets([fw_edu, code_ds, math_ds], probabilities=[0.60,0.25,0.15], seed=42, stopping_strategy="all_exhausted")
    logger.info(f"Pipeline Initialized: 60% FineWeb-Edu | 25% StarCoder ({len(code_datasets)} subsets) | 15% NuminaMath")
    return mixed

class StreamingTokenDataset:
    def __init__(self, dataset, tokenizer, seq_len: int, eos_id: int):
        self.dataset=dataset; self.tokenizer=tokenizer; self.seq_len=seq_len; self.eos_id=eos_id
    def __iter__(self):
        buffer=[]
        for ex in iter(self.dataset):
            text=ex.get("text","")
            if not text or len(text.strip())<10: continue
            buffer.extend(self.tokenizer.encode(text, add_special_tokens=False)+[self.eos_id])
            while len(buffer)>=self.seq_len:
                yield np.array(buffer[:self.seq_len], dtype=np.int32)
                buffer=buffer[self.seq_len:]

class DataPipeline:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.eos_token_id
        self.tokens_consumed = 0
        self.sequences_consumed = 0
        self._iterator = None

    def _compute_per_dataset_skip_docs(self, total_tokens_consumed: int) -> Tuple[int, int, int]:
        cfg = self.cfg
        prob_fw, prob_code, prob_math = cfg.data_mix_probs
        tokens_fw = total_tokens_consumed * prob_fw
        tokens_code = total_tokens_consumed * prob_code
        tokens_math = total_tokens_consumed * prob_math
        skip_fw = int(tokens_fw / cfg.approx_tokens_per_fineweb_doc)
        skip_code = int(tokens_code / cfg.approx_tokens_per_code_doc)
        skip_math = int(tokens_math / cfg.approx_tokens_per_math_doc)
        return skip_fw, skip_code, skip_math

    def initialize(self, resume_tokens: int = 0, resume_sequences: int = 0, seq_len: int = 2048):
        cfg = self.cfg
        if resume_tokens > 0:
            skip_fw, skip_code, skip_math = self._compute_per_dataset_skip_docs(resume_tokens)
            logger.info(f"Resume: skipping ~{skip_fw:,} FineWeb docs, ~{skip_code:,} StarCoder docs, ~{skip_math:,} NuminaMath docs")
            fw_edu = load_dataset("HuggingFaceFW/fineweb-edu", cfg.fineweb_config, split="train", streaming=True, token=cfg.hf_token).select_columns(["text"]).skip(skip_fw)
            code_datasets =[]
            skip_per_lang = skip_code // len(STARCODER_LANGS)
            for lang in STARCODER_LANGS:
                try:
                    ds = load_dataset("bigcode/starcoderdata", data_dir=lang, split="train", streaming=True, token=cfg.hf_token).select_columns(["content"]).rename_column("content", "text").skip(skip_per_lang)
                    code_datasets.append(ds)
                except Exception as e:
                    pass
            code_ds = interleave_datasets(code_datasets, seed=42, stopping_strategy="all_exhausted")
            math_ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True, token=cfg.hf_token).map(lambda ex: {"text": f"Problem: {ex['problem']}\n\nSolution: {ex['solution']}"}, remove_columns=["source", "problem", "solution", "messages"]).skip(skip_math)
            mixed = interleave_datasets([fw_edu, code_ds, math_ds], probabilities=[0.60, 0.25, 0.15], seed=42, stopping_strategy="all_exhausted")
            self.tokens_consumed = resume_tokens
            self.sequences_consumed = resume_sequences
        else:
            mixed = build_streaming_dataset(cfg)
            self.tokens_consumed = 0
            self.sequences_consumed = 0
        iterable_ds = StreamingTokenDataset(mixed, self.tokenizer, seq_len, self.eos_id)
        self._iterator = iter(iterable_ds)
        logger.info(f"Data pipeline ready (seq_len={seq_len}). tokens={self.tokens_consumed:,}, seqs={self.sequences_consumed:,}")

    def reinitialize_with_new_seq_len(self, new_seq_len: int):
        self.initialize(resume_tokens=self.tokens_consumed, resume_sequences=self.sequences_consumed, seq_len=new_seq_len)

    def get_batch(self, batch_size: int, seq_len: int) -> np.ndarray:
        sequences =[]
        for _ in range(batch_size):
            try:
                seq = next(self._iterator)
            except StopIteration:
                self.initialize(seq_len=seq_len)
                seq = next(self._iterator)
            sequences.append(seq)
        batch = np.stack(sequences, axis=0)
        self.sequences_consumed += batch_size
        self.tokens_consumed += batch_size * seq_len
        return batch

# ================================================================
# CHECKPOINTING
# ================================================================
def save_checkpoint(state, config, step, data_pipeline, tokenizer):
    ckpt_path = os.path.join(config.checkpoint_dir, f"step_{step}")
    os.makedirs(ckpt_path, exist_ok=True)
    params_np = jax.tree_util.tree_map(lambda x: np.array(x), state.params)
    with open(os.path.join(ckpt_path, "params.pkl"), "wb") as f:
        pickle.dump(params_np, f)
    opt_state_np = jax.tree_util.tree_map(lambda x: np.array(x) if hasattr(x, 'shape') else x, state.opt_state)
    with open(os.path.join(ckpt_path, "opt_state.pkl"), "wb") as f:
        pickle.dump(opt_state_np, f)
    cfg_dict = asdict(config)
    cfg_dict.pop("hf_token", None)  # strip token before writing to disk
    metadata = {"step": int(step), "tokens_consumed": int(data_pipeline.tokens_consumed), "sequences_consumed": int(data_pipeline.sequences_consumed), "config": cfg_dict}
    with open(os.path.join(ckpt_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return ckpt_path

def find_latest_checkpoint(config):
    ckpt_base = config.checkpoint_dir
    if not os.path.exists(ckpt_base): return None, None
    steps = []
    for d in os.listdir(ckpt_base):
        if d.startswith("step_"):
            try:
                if os.path.exists(os.path.join(ckpt_base, d, "metadata.json")):
                    steps.append(int(d.split("_")[1]))
            except: pass
    if not steps: return None, None
    return os.path.join(ckpt_base, f"step_{max(steps)}"), max(steps)

def load_checkpoint(ckpt_path, state, mesh):
    with open(os.path.join(ckpt_path, "params.pkl"), "rb") as f:
        params_np = pickle.load(f)
    # ✅ FIXED — preserve bf16 dtype on load, same cast as create_train_state
    params_jax = jax.tree_util.tree_map(
        lambda x: jnp.array(x).astype(jnp.bfloat16) if hasattr(x, 'dtype') and np.issubdtype(x.dtype, np.floating) else jnp.array(x),
        params_np
    )

    # ✅ Identical guard to create_train_state — do NOT shard MoE expert weights
    def make_sharding(x):
        if not hasattr(x, 'ndim'):
            return NamedSharding(mesh, P())
        if x.ndim == 2 and x.shape[0] % len(jax.devices()) == 0 and x.shape[0] >= 1024:
            return NamedSharding(mesh, P('data', None))
        return NamedSharding(mesh, P())

    params_jax = jax.device_put(
        params_jax, jax.tree_util.tree_map(make_sharding, params_jax)
    )

    with open(os.path.join(ckpt_path, "opt_state.pkl"), "rb") as f:
        opt_state_np = pickle.load(f)
    # ✅ FIXED — Muon set mu_dtype=jnp.bfloat16 at init, restore same on resume
    opt_state_jax = jax.tree_util.tree_map(
        lambda x: jnp.array(x).astype(jnp.bfloat16) if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating) else (jnp.array(x) if isinstance(x, np.ndarray) else x),
        opt_state_np
    )

    # ✅ Only shard actual jnp arrays — leave step counters/scalars alone
    def shard_if_array(x):
        if isinstance(x, jnp.ndarray):
            return jax.device_put(x, make_sharding(x))
        return x
    opt_state_jax = jax.tree_util.tree_map(shard_if_array, opt_state_jax)

    with open(os.path.join(ckpt_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    return state.replace(
        step=metadata["step"],
        params=params_jax,
        opt_state=opt_state_jax
    ), metadata

def push_to_hub(ckpt_path, config, step, tokenizer):
    try:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(HF_REPO_ID, private=True, exist_ok=True)
        api.upload_folder(
            folder_path=ckpt_path,
            repo_id=HF_REPO_ID,
            path_in_repo=f"checkpoints/step_{step}",
            commit_message=f"step {step}"
        )
        tokenizer.save_pretrained(os.path.join(ckpt_path, "tokenizer"))
        api.upload_folder(
            folder_path=os.path.join(ckpt_path, "tokenizer"),
            repo_id=HF_REPO_ID,
            path_in_repo="tokenizer"
        )
        import shutil
        shutil.rmtree(ckpt_path)
        logger.info(f"Uploaded and deleted local: {ckpt_path}")
    except Exception as e:
        logger.error(f"Upload failed, KEEPING local checkpoint: {e}")
        # do NOT delete — leave it for manual retry

# ================================================================
# MAIN — CORRECTED: Proper gradient accumulation + cached freqs
# ================================================================
def main():
    logger.info("=" * 72)
    logger.info("  ZENYX V3 BASE MODEL PRETRAINING v2 | Muon | Progressive Context")
    logger.info("=" * 72)

    config = ZenyxConfig()
    devices = jax.devices()
    logger.info(f"JAX devices: {len(devices)} × {devices[0].device_kind}")
    logger.info(f"Platform: {jax.default_backend()}")
    mesh = jax.sharding.Mesh(np.array(devices), axis_names=("data",))

    # Tokenizer
    logger.info(f"Loading tokenizer from {TOKENIZER_REPO} ...")
    login(token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, token=HF_TOKEN)
    # ADD THIS:
    if len(tokenizer) < config.vocab_size:
        num_pad = config.vocab_size - len(tokenizer)
        tokenizer.add_tokens([f"<pad_{i}>" for i in range(num_pad)])
        logger.info(f"Added {num_pad} placeholder tokens to match vocab_size={config.vocab_size}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}, EOS ID: {tokenizer.eos_token_id}")

    # Initialize model at max seqlen so weights never retrace on phase change
    logger.info("Initializing model ...")
    rng = random.PRNGKey(42)
    max_seq = max(config.ctx_phase_lengths)  # 8192
    state, model = create_train_state(rng, config, mesh, max_seq_len=max_seq)

    # ── Pull latest checkpoint from HuggingFace before scanning local dir ──
    try:
        logger.info("Checking HuggingFace for existing checkpoints ...")
        api = HfApi(token=HF_TOKEN)
        all_files = list(api.list_repo_files(HF_REPO_ID, token=HF_TOKEN))

        # Collect all step numbers that have a metadata.json on HF
        hf_steps = set()
        for f in all_files:
            parts = f.split("/")
            # Expected path pattern: checkpoints/step_800/metadata.json
            if (
                len(parts) == 3
                and parts[0] == "checkpoints"
                and parts[1].startswith("step")
                and parts[2] == "metadata.json"
            ):
                m = re.search(r'\d+', parts[1])
                if m:
                    hf_steps.add(int(m.group()))

        if hf_steps:
            latest_hf_step = max(hf_steps)
            logger.info(f"Found checkpoint at step {latest_hf_step} on HuggingFace. Downloading ...")
            local_ckpt_dir = os.path.join(config.checkpoint_dir, f"step_{latest_hf_step}")
            os.makedirs(local_ckpt_dir, exist_ok=True)

            import shutil
            # Download all files for that checkpoint folder
            for fname in ["params.pkl", "opt_state.pkl", "metadata.json"]:
                remote_path = f"checkpoints/step_{latest_hf_step}/{fname}"
                local_path  = os.path.join(local_ckpt_dir, fname)
                if not os.path.exists(local_path) and remote_path in all_files:
                    downloaded = hf_hub_download(
                        repo_id=HF_REPO_ID,
                        filename=remote_path,
                        token=HF_TOKEN,
                    )
                    shutil.copy2(downloaded, local_path)

            logger.info(f"Checkpoint step_{latest_hf_step} restored to {local_ckpt_dir}")
        else:
            logger.info("No checkpoints found on HuggingFace. Will start fresh.")

    except Exception as e:
        logger.warning(f"Could not restore checkpoint from HuggingFace (will start fresh): {e}")
    # ── END HF RESTORE BLOCK ───────────────────────────────────────────────

    # Check for existing checkpoint / resume
    ckpt_path, resume_step = find_latest_checkpoint(config)
    resume_tokens    = 0
    resume_sequences = 0
    start_step       = 1

    if ckpt_path is not None:
        logger.info(f"Found checkpoint at step {resume_step}: {ckpt_path}")
        state, metadata = load_checkpoint(ckpt_path, state, mesh)
        start_step       = metadata["step"] + 1
        resume_tokens    = metadata["tokens_consumed"]
        resume_sequences = metadata["sequences_consumed"]
        logger.info(f"Resuming from step {start_step}, tokens={resume_tokens:,}, sequences={resume_sequences:,}")
        logger.info(f"Optimizer state restored — LR schedule will pick up at step {start_step}")
    else:
        logger.info("No checkpoint found, starting fresh.")

    current_seqlen = config.get_seq_len_for_step(start_step)
    logger.info(f"Starting seqlen={current_seqlen} @ step {start_step}")

    logger.info("Initializing data pipeline ...")
    data_pipeline = DataPipeline(config, tokenizer)
    data_pipeline.initialize(
        resume_tokens=resume_tokens,
        resume_sequences=resume_sequences,
        seq_len=current_seqlen,
    )

    max_len = max(config.ctx_phase_lengths)
    std_cos, std_sin = precompute_freqs_cis(config.rope_head_dim, max_len, config.rope_theta)
    yarn_cos, yarn_sin = precompute_freqs_cis(
        config.rope_head_dim, max_len, config.rope_theta,
        config.yarn_scale, config.yarn_original_seq_len,
        config.beta_fast, config.beta_slow,
    )

    def get_freqs(seqlen):
        return (yarn_cos, yarn_sin) if seqlen > config.ctx_phase_lengths[1] else (std_cos, std_sin)

    cached_freqs_cos, cached_freqs_sin = get_freqs(current_seqlen)
    logger.info(
        f"RoPE tables ready: std={max_len},{config.rope_head_dim//2} "
        f"yarn={max_len},{config.rope_head_dim//2} | "
        f"Active: {'YaRN' if current_seqlen > config.ctx_phase_lengths[1] else 'standard'} "
        f"seqlen={current_seqlen}"
    )

    compute_grads_fn = make_compute_grads(config)

    logger.info(f"Training steps {start_step} → {config.total_steps}")
    logger.info(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(
        f"  Effective batch per step: "
        f"{config.get_batch_size_for_step(start_step)} × {len(devices)} × {config.gradient_accumulation_steps}"
    )
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    step_start  = time.time()
    train_start = time.time()
    data_sharding = NamedSharding(mesh, P("data"))

    for step in range(start_step, config.total_steps + 1):

        new_seqlen = config.get_seq_len_for_step(step)
        if new_seqlen != current_seqlen:
            logger.info(f"Context phase change: {current_seqlen} → {new_seqlen} at step {step}")
            current_seqlen = new_seqlen
            cached_freqs_cos, cached_freqs_sin = get_freqs(current_seqlen)
            data_pipeline.reinitialize_with_new_seq_len(current_seqlen)
            logger.info(
                f"Phase change complete: seqlen={current_seqlen}, "
                f"RoPE={'YaRN' if current_seqlen > config.ctx_phase_lengths[1] else 'standard'}, "
                f"no XLA retrace (shape unchanged)"
            )

        accumulated_grads  = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        total_loss_accum   = 0.0
        main_loss_accum    = 0.0
        batch_size         = config.get_batch_size_for_step(step) * len(devices)

        with mesh:
            for microstep in range(config.gradient_accumulation_steps):
                batch_np = data_pipeline.get_batch(batch_size, current_seqlen)
                batch    = jnp.array(batch_np)
                batch    = jax.device_put(batch, data_sharding)

                grads, total_loss, main_loss = compute_grads_fn(
                    state, batch, cached_freqs_cos, cached_freqs_sin
                )

                accumulated_grads = jax.tree_util.tree_map(jnp.add, accumulated_grads, grads)
                total_loss_accum += float(total_loss)
                main_loss_accum  += float(main_loss)

        averaged_grads = jax.tree_util.tree_map(
            lambda g: g / config.gradient_accumulation_steps, accumulated_grads
        )

        state = apply_accumulated_grads(state, averaged_grads)

        avg_total_loss = total_loss_accum / config.gradient_accumulation_steps
        avg_main_loss  = main_loss_accum  / config.gradient_accumulation_steps

        if step % config.log_every_steps == 0:
            elapsed = time.time() - step_start
            tokens_per_sec = (
                config.log_every_steps * config.gradient_accumulation_steps
                * batch_size * current_seqlen / elapsed
            )
            lr = float(create_lr_schedule(config)(step))
            logger.info(
                f"step={step:6d} | loss={avg_total_loss:.4f} "
                f"main_loss={avg_main_loss:.4f} lr={lr:.2e} "
                f"seqlen={current_seqlen} tok/s={tokens_per_sec:,.0f} "
                f"tokens={data_pipeline.tokens_consumed:,.0f}"
            )
            step_start = time.time()

        if step % config.save_every_steps == 0:
            ckpt = save_checkpoint(state, config, step, data_pipeline, tokenizer)
            import threading
            upload_thread = threading.Thread(
                target=push_to_hub, 
                args=(ckpt, config, step, tokenizer), 
                daemon=False  # wait for upload to finish
            )
            upload_thread.start()
            upload_thread.join()

    logger.info("Training complete! Saving final checkpoint ...")
    ckpt = save_checkpoint(state, config, step, data_pipeline, tokenizer)
    push_to_hub(ckpt, config, step, tokenizer)

    total_time = time.time() - train_start
    logger.info(f"Total training time: {total_time/3600:.2f}h")
    logger.info(f"Total tokens consumed: {data_pipeline.tokens_consumed:,.0f}")
    logger.info(f"Model: https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
