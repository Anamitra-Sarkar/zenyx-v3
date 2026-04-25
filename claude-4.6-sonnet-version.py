#!/usr/bin/env python3
"""
zenyx_v3_train.py
=================
Zenyx v3 — Sub-1B LLM, complete Phase-1 pre-training script.
Target hardware : Kaggle TPU v5e-8 (8 cores × 16 GB HBM = 128 GB)
Context window  : 32 768 tokens via YaRN-decoupled RoPE + CSA/HCA hybrid attention
Active params   : < 850 M  (weight-tied across 12 recurrent steps)

Datasets used (all non-gated, open-source):
  60 %  HuggingFaceFW/fineweb-edu   (sample-10BT split, CC0)
  25 %  bigcode/the-stack-smol      (permissive-licensed code)
  15 %  EleutherAI/proof-pile-2     (math / formal reasoning)

Run on Kaggle:  !python zenyx_v3_train.py
"""

# ===========================================================================
# SECTION 1 : IMPORTS
# ===========================================================================
from __future__ import annotations

import datetime
import functools
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

import flax.linen as nn
import optax
import orbax.checkpoint as ocp

from datasets import interleave_datasets, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("zenyx_v3")

# ===========================================================================
# SECTION 2 : CONFIGURATION
# ===========================================================================

# ── Authentication / storage ────────────────────────────────────────────────
HF_TOKEN        = "YOUR_HF_TOKEN_HERE"
HF_REPO         = "Arko007/zenyx-v3-checkpoints"
CKPT_LOCAL_DIR  = Path("/tmp/zenyx_v3_ckpts")
CKPT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Model architecture ──────────────────────────────────────────────────────
D_MODEL             = 1536
NUM_HEADS           = 12          # head dim = D_MODEL // NUM_HEADS = 128
D_LATENT            = 256         # MLA latent dim (6× KV-cache compression)
D_ROPE              = 64          # decoupled RoPE sub-head dim
D_FF                = 4096        # MoE intermediate (split across shared experts)
NUM_SHARED_EXPERTS  = 2
NUM_ROUTED_EXPERTS  = 64
NUM_RECURRENCES     = 12          # recurrent depth steps
SEQ_LEN             = 32768
VOCAB_SIZE          = 65536
LOCAL_WINDOW        = 256         # uncompressed sliding window (recent tokens)

# ── Optimizer ───────────────────────────────────────────────────────────────
BASE_LR             = 3e-4
MIN_LR              = 3e-5
WARMUP_STEPS        = 2_000
WEIGHT_DECAY        = 0.05
GRAD_CLIP_NORM      = 1.0
MUON_MOMENTUM       = 0.95
NS_STEPS            = 5           # Newton-Schulz iteration count
ADAM_B1, ADAM_B2    = 0.9, 0.95
ADAM_EPS            = 1e-8

# ── YaRN RoPE extension ─────────────────────────────────────────────────────
ROPE_BASE   = 10_000.0
ROPE_ALPHA  = 1.0    # interpolation boundary (low-freq)
ROPE_BETA   = 32.0   # extrapolation boundary (high-freq)
ROPE_SCALE  = 4.0    # scale_s  (original ctx = 8192, target = 32768 → ×4)

# ── MoE ─────────────────────────────────────────────────────────────────────
MOE_AUX_ALPHA = 0.01  # auxiliary load-balancing loss coefficient

# ── LayerScale ──────────────────────────────────────────────────────────────
LAYER_SCALE_INIT = 1e-4

# ── Batch / throughput ──────────────────────────────────────────────────────
NUM_DEVICES         = 8           # TPU v5e-8
PER_DEVICE_BATCH    = 1           # sequences per core per micro-step
GRAD_ACCUM_STEPS    = 8           # gradient accumulation steps
# Global batch ≈ 8 × 1 × 8 × 32 768 ≈ 2.1 M tokens per parameter update

# ── Training schedule (Phase 1 only) ────────────────────────────────────────
TOTAL_TOKENS_PHASE1 = 800_000_000_000        # 800 B tokens
GLOBAL_BATCH_SEQ    = PER_DEVICE_BATCH * NUM_DEVICES * GRAD_ACCUM_STEPS  # 64
TOKENS_PER_UPDATE   = GLOBAL_BATCH_SEQ * SEQ_LEN                         # ≈ 2.1 M
TOTAL_STEPS_PHASE1  = TOTAL_TOKENS_PHASE1 // TOKENS_PER_UPDATE

# ── Logging / checkpointing ─────────────────────────────────────────────────
SAVE_EVERY_STEPS    = 500
LOG_EVERY_STEPS     = 10
VERBOSE_EVERY_STEPS = 100

# ── Average tokens/example (used for O(1) resume skip calculation) ──────────
AVG_TOKENS_PER_EXAMPLE = 512      # conservative estimate; adjust per dataset


# ===========================================================================
# SECTION 3 : TOKENIZER
# ===========================================================================

def load_tokenizer() -> PreTrainedTokenizerBase:
    """
    Attempts to load a SentencePiece-based tokenizer from the checkpoint repo
    (place tokenizer files there once trained).  Falls back to GPT-2 for
    bootstrapping / testing on Kaggle.
    """
    try:
        tok = AutoTokenizer.from_pretrained(
            HF_REPO, token=HF_TOKEN, trust_remote_code=True
        )
        log.info("Tokenizer loaded from HF repo.")
        return tok
    except Exception:
        log.warning(
            "Tokenizer not found in HF repo — using GPT-2 as placeholder "
            "(effective vocab 50 257; replace before real runs)."
        )
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        return tok


# ===========================================================================
# SECTION 4 : DATA PIPELINE
# ===========================================================================

def build_streaming_dataset(skip_examples: int = 0):
    """
    Build interleaved streaming dataset from three non-gated corpora:
      60 %  FineWeb-Edu  (general high-quality web text)
      25 %  The Stack Smol  (permissive-license source code)
      15 %  Proof-Pile-2  (mathematical / formal reasoning)

    Args:
        skip_examples: number of interleaved examples to skip for O(1) resume.
    Returns:
        Streaming HuggingFace IterableDataset yielding {"text": str}.
    """
    fw_edu = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        trust_remote_code=True,
    ).map(lambda ex: {"text": ex.get("text", "")})

    stack = load_dataset(
        "bigcode/the-stack-smol",
        split="train",
        streaming=True,
        trust_remote_code=True,
    ).map(lambda ex: {"text": ex.get("content", "")})

    proof = load_dataset(
        "EleutherAI/proof-pile-2",
        split="train",
        streaming=True,
        trust_remote_code=True,
    ).map(lambda ex: {"text": ex.get("text", ex.get("content", ""))})

    dataset = interleave_datasets(
        [fw_edu, stack, proof],
        probabilities=[0.60, 0.25, 0.15],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    if skip_examples > 0:
        log.info(f"Skipping {skip_examples:,} examples for O(1) resume …")
        dataset = dataset.skip(skip_examples)

    return dataset


class RingTokenBuffer:
    """
    Ring-buffer that tokenises raw text and emits fixed-length chunks of
    exactly SEQ_LEN + 1 token IDs (the +1 gives us labels = input shifted
    by one position).
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, seq_len: int) -> None:
        self.tokenizer  = tokenizer
        self.seq_len    = seq_len
        self._buf: List[int] = []

    def feed(self, text: str) -> None:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids.append(self.tokenizer.eos_token_id or 0)
        self._buf.extend(ids)

    def ready(self) -> bool:
        return len(self._buf) >= self.seq_len + 1

    def pop(self) -> np.ndarray:
        """Returns numpy array of shape (seq_len + 1,)."""
        chunk     = self._buf[: self.seq_len + 1]
        self._buf = self._buf[self.seq_len + 1 :]
        return np.array(chunk, dtype=np.int32)


def batch_generator(
    dataset,
    tokenizer:  PreTrainedTokenizerBase,
    batch_size: int,
    seq_len:    int,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Yields (input_ids, labels, loss_mask) each of shape (batch_size, seq_len).
    All-ones loss_mask for Phase-1 causal LM.
    """
    buf        = RingTokenBuffer(tokenizer, seq_len)
    b_inp: List[np.ndarray] = []
    b_lbl: List[np.ndarray] = []

    for example in dataset:
        text = example.get("text", "")
        if not text.strip():
            continue
        buf.feed(text)
        while buf.ready():
            chunk = buf.pop()          # (seq_len + 1,)
            b_inp.append(chunk[:-1])   # (seq_len,)  — inputs
            b_lbl.append(chunk[1:])    # (seq_len,)  — labels
            if len(b_inp) == batch_size:
                inputs = np.stack(b_inp)                              # (B, S)
                labels = np.stack(b_lbl)
                mask   = np.ones_like(labels, dtype=np.float32)
                yield inputs, labels, mask
                b_inp.clear()
                b_lbl.clear()


# ===========================================================================
# SECTION 5 : MODEL COMPONENTS
# ===========================================================================

# ---------------------------------------------------------------------------
# 5.1  FP8 Dense Layer
# ---------------------------------------------------------------------------

class FP8Dense(nn.Module):
    """
    Linear projection that executes the matmul in FP8 (e4m3fn) on TPU MXUs.
    Master weights live in float32; activations are dynamically quantised.
    Falls back to bfloat16 on hardware / JAX versions without FP8 support.
    """

    features: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param(
            "kernel",
            nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
            (x.shape[-1], self.features),
            jnp.float32,
        )

        E4M3_MAX = 448.0  # approximate max representable value in float8_e4m3fn

        x_amax  = jnp.max(jnp.abs(x))  + 1e-12
        w_amax  = jnp.max(jnp.abs(kernel)) + 1e-12
        x_scale = E4M3_MAX / x_amax
        w_scale = E4M3_MAX / w_amax

        try:
            x_q = (x * x_scale).astype(jnp.float8_e4m3fn)       # type: ignore[attr-defined]
            w_q = (kernel * w_scale).astype(jnp.float8_e4m3fn)   # type: ignore[attr-defined]
            out = jax.lax.dot_general(
                x_q, w_q,
                dimension_numbers=(((x.ndim - 1,), (0,)), ((), ())),
                preferred_element_type=jnp.bfloat16,
            )
        except (AttributeError, TypeError):
            # Graceful fallback: bfloat16 matmul
            out = jax.lax.dot_general(
                x.astype(jnp.bfloat16),
                kernel.astype(jnp.bfloat16),
                dimension_numbers=(((x.ndim - 1,), (0,)), ((), ())),
                preferred_element_type=jnp.bfloat16,
            )
            x_scale = w_scale = jnp.ones((), dtype=jnp.float32)

        # Dequantise back to bfloat16 for the residual stream
        out = out / (x_scale * w_scale)

        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,), jnp.float32)
            out  = out + bias.astype(jnp.bfloat16)

        return out.astype(jnp.bfloat16)


# ---------------------------------------------------------------------------
# 5.2  Decoupled YaRN RoPE
# ---------------------------------------------------------------------------

def build_yarn_rope(
    seq_len:  int,
    d_rope:   int,
    base:     float = 10_000.0,
    alpha:    float = 1.0,
    beta:     float = 32.0,
    scale_s:  float = 4.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Pre-compute YaRN positional tables.

    Returns:
        cos_table : (seq_len, d_rope // 2) — bfloat16
        sin_table : (seq_len, d_rope // 2)
        mscale    : Python float — temperature multiplier for attention logits
    """
    m       = jnp.arange(d_rope // 2, dtype=jnp.float32)
    theta   = base ** (-2.0 * m / d_rope)          # (d_rope // 2,)
    lam     = 2.0 * jnp.pi / theta
    r       = 8192.0 / lam                          # ratio to original context

    # Piecewise ramp: 0 → interpolate, 1 → extrapolate
    gamma     = jnp.where(r < alpha, 0.0,
                jnp.where(r > beta,  1.0,
                          (r - alpha) / (beta - alpha)))
    theta_y   = (1.0 - gamma) * (theta / scale_s) + gamma * theta

    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles    = jnp.outer(positions, theta_y)       # (seq_len, d_rope // 2)
    cos_t     = jnp.cos(angles).astype(jnp.bfloat16)
    sin_t     = jnp.sin(angles).astype(jnp.bfloat16)
    mscale    = float(0.1 * math.log(scale_s) + 1.0)   # ≈ 1.139 for scale_s=4
    return cos_t, sin_t, mscale


def apply_rope(
    x:   jnp.ndarray,   # (..., seq, d)
    cos: jnp.ndarray,   # (seq, d // 2)
    sin: jnp.ndarray,
) -> jnp.ndarray:
    """Standard complex-multiplication rotation via real arithmetic."""
    d2          = x.shape[-1] // 2
    x1, x2      = x[..., :d2], x[..., d2:]
    rotated     = jnp.concatenate([-x2, x1], axis=-1)   # 90-degree rotation
    cos_full    = jnp.concatenate([cos, cos], axis=-1)   # (seq, d)
    sin_full    = jnp.concatenate([sin, sin], axis=-1)
    return x * cos_full + rotated * sin_full


# ---------------------------------------------------------------------------
# 5.3  Hybrid Attention  (CSA + HCA with decoupled RoPE)
# ---------------------------------------------------------------------------

class ZenyxHybridAttention(nn.Module):
    """
    Single attention module used across ALL 12 recurrent steps (weight tying).
    The Python argument `is_hca` selects between:
      • HCA (is_hca=True,  even steps) : compress_ratio = 128, dense global
      • CSA (is_hca=False, odd  steps) : compress_ratio =   4, Top-64 sparse
    Both modes prepend a local uncompressed sliding window of 256 tokens.
    """

    d_model:   int
    num_heads: int
    d_latent:  int
    d_rope:    int

    def setup(self) -> None:
        dh              = self.d_model // self.num_heads
        self.d_head     = dh
        # MLA latent projections
        self.q_proj     = FP8Dense(self.d_latent)
        self.kv_proj    = FP8Dense(self.d_latent)
        # Up-projections to multi-head space
        self.q_up       = FP8Dense(self.num_heads * dh)
        self.kv_up_k    = FP8Dense(self.num_heads * dh)
        self.kv_up_v    = FP8Dense(self.num_heads * dh)
        self.o_proj     = FP8Dense(self.d_model)
        # Decoupled RoPE sub-head projections (narrow, no absorption conflict)
        self.q_rope_p   = FP8Dense(self.num_heads * self.d_rope)
        self.k_rope_p   = FP8Dense(self.num_heads * self.d_rope)

    def __call__(
        self,
        x:       jnp.ndarray,   # (B, S, d_model)  bfloat16
        cos:     jnp.ndarray,   # (S, d_rope // 2) bfloat16
        sin:     jnp.ndarray,
        mscale:  float,
        is_hca:  bool = False,
    ) -> jnp.ndarray:
        B, S, _         = x.shape
        H, dh           = self.num_heads, self.d_head
        compress_ratio  = 128 if is_hca else 4
        num_chunks      = S // compress_ratio

        # ── 1. MLA latent projections ──────────────────────────────────────
        c_q  = self.q_proj(x)      # (B, S, d_latent)
        c_kv = self.kv_proj(x)     # (B, S, d_latent)

        # ── 2. Sequence compression (mean-pool over chunks) ────────────────
        c_kv_global = (
            c_kv.reshape(B, num_chunks, compress_ratio, self.d_latent)
            .mean(axis=2)
        )   # (B, num_chunks, d_latent)

        # ── 3. Decompression to multi-head space ──────────────────────────
        q_nope  = self.q_up(c_q).reshape(B, S, H, dh)
        k_glob  = self.kv_up_k(c_kv_global).reshape(B, num_chunks, H, dh)
        v_glob  = self.kv_up_v(c_kv_global).reshape(B, num_chunks, H, dh)

        # ── 4. Local sliding window (last LOCAL_WINDOW tokens, exact) ──────
        loc_kv  = c_kv[:, -LOCAL_WINDOW:, :]
        k_loc   = self.kv_up_k(loc_kv).reshape(B, LOCAL_WINDOW, H, dh)
        v_loc   = self.kv_up_v(loc_kv).reshape(B, LOCAL_WINDOW, H, dh)

        k_all   = jnp.concatenate([k_glob, k_loc], axis=1)   # (B, T, H, dh)
        v_all   = jnp.concatenate([v_glob, v_loc], axis=1)
        T       = num_chunks + LOCAL_WINDOW

        # ── 5. Decoupled YaRN RoPE ─────────────────────────────────────────
        q_r_raw = self.q_rope_p(c_q).reshape(B, S, H, self.d_rope)
        k_r_raw = self.k_rope_p(c_kv).reshape(B, S, H, self.d_rope)

        q_r     = apply_rope(q_r_raw, cos, sin)   # (B, S, H, d_rope)

        # Pool k_rope to match compressed + local positions
        k_r_global = (
            k_r_raw.reshape(B, num_chunks, compress_ratio, H, self.d_rope)
            .mean(axis=2)
        )   # (B, num_chunks, H, d_rope)
        k_r_local   = k_r_raw[:, -LOCAL_WINDOW:, :, :]
        k_r_all     = jnp.concatenate([k_r_global, k_r_local], axis=1)

        # Construct positional tables for the T KV positions
        cos_kv = jnp.concatenate([
            cos[compress_ratio // 2 :: compress_ratio, :][:num_chunks],
            cos[-LOCAL_WINDOW:, :],
        ], axis=0)
        sin_kv = jnp.concatenate([
            sin[compress_ratio // 2 :: compress_ratio, :][:num_chunks],
            sin[-LOCAL_WINDOW:, :],
        ], axis=0)
        k_r = apply_rope(k_r_all, cos_kv, sin_kv)

        # ── 6. Assemble full Q and K (nope || rope) ────────────────────────
        q_final = jnp.concatenate([q_nope, q_r], axis=-1)          # (B, S, H, dh+dr)
        k_final = jnp.concatenate([k_all,  k_r], axis=-1)          # (B, T, H, dh+dr)

        # ── 7. Attention logits (kept in bfloat16 to avoid FP8 underflow) ──
        scale   = mscale / math.sqrt(dh + self.d_rope)
        # b=Batch, s=query-seq, t=kv-seq, h=head, d=head-dim
        logits  = jnp.einsum("bshd,bthd->bhst", q_final, k_final) * scale

        # ── 8. Sparse top-K gate for CSA layers ───────────────────────────
        if not is_hca:
            top_k   = min(64, T)
            # Sort-based threshold: retain only the top-K positions per query
            sorted_l = jnp.sort(logits, axis=-1)
            thresh   = sorted_l[..., -top_k : -top_k + 1]   # (B, H, S, 1)
            logits   = jnp.where(logits >= thresh, logits, jnp.full_like(logits, -1e9))

        # ── 9. Softmax (float32 for numerical safety) + weighted sum ───────
        attn_w  = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
        out     = jnp.einsum("bhst,bthd->bshd", attn_w, v_all)     # (B, S, H, dh)
        out     = out.reshape(B, S, H * dh)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# 5.4  Dual Shared Sparse MoE
# ---------------------------------------------------------------------------

class DualSharedSparseMoE(nn.Module):
    """
    2 always-active shared experts (d_ff split equally between them)
    + 64 routed experts with Top-1 routing.
    Auxiliary load-balancing loss is returned alongside the output.
    """

    d_model:            int
    d_ff:               int
    num_routed_experts: int

    def setup(self) -> None:
        d_split = self.d_ff // NUM_SHARED_EXPERTS   # 2048 per shared expert
        # Shared expert 1
        self.sh1_w1 = FP8Dense(d_split)
        self.sh1_w2 = FP8Dense(self.d_model)
        # Shared expert 2
        self.sh2_w1 = FP8Dense(d_split)
        self.sh2_w2 = FP8Dense(self.d_model)
        # Token router (runs in float32 to prevent gradient underflow)
        self.router = nn.Dense(
            self.num_routed_experts, use_bias=False, dtype=jnp.float32
        )
        # Routed expert weight banks (bfloat16, indexed by expert id)
        self.routed_w1 = self.param(
            "routed_w1",
            nn.initializers.lecun_normal(),
            (self.num_routed_experts, self.d_model, self.d_ff),
            jnp.bfloat16,
        )
        self.routed_w2 = self.param(
            "routed_w2",
            nn.initializers.lecun_normal(),
            (self.num_routed_experts, self.d_ff, self.d_model),
            jnp.bfloat16,
        )

    def __call__(
        self, x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # ── Shared expert path (100 % of tokens) ──────────────────────────
        s1_out = self.sh1_w2(jax.nn.silu(self.sh1_w1(x)))
        s2_out = self.sh2_w2(jax.nn.silu(self.sh2_w1(x)))
        shared = s1_out + s2_out

        # ── Router: Top-1 selection ────────────────────────────────────────
        r_logits = self.router(x.astype(jnp.float32))          # (B, S, E)
        r_probs  = jax.nn.softmax(r_logits, axis=-1)
        idx      = jnp.argmax(r_probs, axis=-1)                # (B, S)
        gate     = jnp.max(r_probs, axis=-1, keepdims=True)    # (B, S, 1)

        # ── Routed expert computation via gather-einsum ────────────────────
        x_bf16    = x.astype(jnp.bfloat16)
        w1_sel    = self.routed_w1[idx]    # (B, S, d_model, d_ff)
        w2_sel    = self.routed_w2[idx]    # (B, S, d_ff, d_model)
        h         = jax.nn.silu(jnp.einsum("bsd,bsdf->bsf", x_bf16, w1_sel))
        routed    = jnp.einsum("bsf,bsfd->bsd", h, w2_sel)

        final_out = shared + (routed * gate.astype(jnp.bfloat16))

        # ── Auxiliary load-balancing loss (routed experts only) ────────────
        expert_mask = jax.nn.one_hot(idx, self.num_routed_experts, dtype=jnp.float32)
        f_i   = jnp.mean(expert_mask, axis=(0, 1))    # fraction per expert
        P_i   = jnp.mean(r_probs,    axis=(0, 1))    # mean routing prob
        aux   = jnp.array(MOE_AUX_ALPHA * self.num_routed_experts) * jnp.sum(f_i * P_i)
        return final_out, aux


# ---------------------------------------------------------------------------
# 5.5  Recurrent Super-Block  (weight-tied across NUM_RECURRENCES steps)
# ---------------------------------------------------------------------------

class ZenyxRecurrentSuperBlock(nn.Module):
    """
    A single (attention + MoE) block whose weights are shared across
    NUM_RECURRENCES depth steps.  Each step alternates between HCA (even)
    and CSA (odd) attention via a Python loop, which JAX traces into
    separate but weight-sharing sub-graphs — equivalent to depth-recurrence.

    LayerScale (gamma init = 1e-4) ensures identity mapping at initialisation,
    providing a stable gradient highway for the 12-step recurrence.
    """

    d_model:         int
    num_recurrences: int

    def setup(self) -> None:
        self.attn   = ZenyxHybridAttention(
            d_model=self.d_model,
            num_heads=NUM_HEADS,
            d_latent=D_LATENT,
            d_rope=D_ROPE,
        )
        self.moe    = DualSharedSparseMoE(
            d_model=self.d_model,
            d_ff=D_FF,
            num_routed_experts=NUM_ROUTED_EXPERTS,
        )
        self.norm1  = nn.RMSNorm()
        self.norm2  = nn.RMSNorm()
        # LayerScale vectors: one per residual branch
        self.gamma1 = self.param(
            "gamma1", nn.initializers.constant(LAYER_SCALE_INIT), (self.d_model,)
        )
        self.gamma2 = self.param(
            "gamma2", nn.initializers.constant(LAYER_SCALE_INIT), (self.d_model,)
        )

    def __call__(
        self,
        x:      jnp.ndarray,   # (B, S, d_model)
        cos:    jnp.ndarray,
        sin:    jnp.ndarray,
        mscale: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        total_aux = jnp.zeros((), dtype=jnp.float32)
        g1 = self.gamma1.astype(jnp.bfloat16)
        g2 = self.gamma2.astype(jnp.bfloat16)

        for step in range(self.num_recurrences):
            # Even steps → HCA (heavy global compression, dense attention)
            # Odd  steps → CSA (light compression, sparse Top-64 attention)
            is_hca = (step % 2 == 0)

            # Sub-layer 1: attention + LayerScale residual
            a_out      = self.attn(self.norm1(x), cos, sin, mscale, is_hca=is_hca)
            x          = x + a_out * g1

            # Sub-layer 2: MoE + LayerScale residual
            m_out, aux = self.moe(self.norm2(x))
            x          = x + m_out * g2
            total_aux  = total_aux + aux

        return x, total_aux


# ---------------------------------------------------------------------------
# 5.6  Full Zenyx V3 Model
# ---------------------------------------------------------------------------

class ZenyxV3Model(nn.Module):
    """
    Embedding → weight-tied recurrent block (12 steps) → RMSNorm → LM head.
    LM head reuses the embedding matrix (weight tying), halving parameter count.
    """

    vocab_size: int
    d_model:    int

    def setup(self) -> None:
        self.embed      = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        # nn.remat wraps the recurrent block for activation checkpointing,
        # dropping intermediate activations and recomputing during backward pass.
        self.recurrent  = nn.remat(ZenyxRecurrentSuperBlock)(
            d_model=self.d_model,
            num_recurrences=NUM_RECURRENCES,
        )
        self.final_norm = nn.RMSNorm()

    def __call__(
        self,
        input_ids: jnp.ndarray,   # (B, S)  int32
        cos:       jnp.ndarray,   # (S, D_ROPE // 2)
        sin:       jnp.ndarray,
        mscale:    float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x        = self.embed(input_ids).astype(jnp.bfloat16)   # (B, S, d_model)
        x, aux   = self.recurrent(x, cos, sin, mscale)
        x        = self.final_norm(x)
        # Weight-tied LM head: project back to vocab space
        emb_mat  = self.embed.embedding.astype(jnp.float32)     # (V, d_model)
        logits   = x.astype(jnp.float32) @ emb_mat.T            # (B, S, V)
        return logits, aux


# ===========================================================================
# SECTION 6 : OPTIMIZER — MUON + ADAMW HYBRID
# ===========================================================================

class MuonState(NamedTuple):
    """Single-moment state for the Muon optimizer."""
    momentum: Any


def newton_schulz_5(G: jnp.ndarray) -> jnp.ndarray:
    """
    Orthogonalise gradient matrix G via 5-step Newton-Schulz iteration.
    Operates in bfloat16 for TPU MXU efficiency.
    Coefficients (a, b, c) maximise singular-value inflation in 5 steps.
    """
    X = G.astype(jnp.bfloat16)
    X = X / (jnp.linalg.norm(X, ord="fro") + 1e-7)

    # Prefer the wider dimension as the "row" for numerical stability
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T

    a, b, c = 3.4445, -4.7750, 2.0315

    def ns_step(X_cur: jnp.ndarray, _: Any) -> Tuple[jnp.ndarray, None]:
        A = X_cur @ X_cur.T
        B = b * A + c * (A @ A)
        return a * X_cur + B @ X_cur, None

    X, _ = jax.lax.scan(ns_step, X, xs=None, length=NS_STEPS)

    if transposed:
        X = X.T
    return X.astype(G.dtype)


def scale_by_muon(
    lr:       float,
    momentum: float = 0.95,
) -> optax.GradientTransformation:
    """
    Muon: SGD-momentum followed by Newton-Schulz orthogonalisation.
    Applied exclusively to 2D weight matrices.
    Reduces optimizer state by 50 % vs AdamW (one moment buffer, not two).
    """

    def init_fn(params: Any) -> MuonState:
        return MuonState(
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params)
        )

    def update_fn(
        updates: Any, state: MuonState, params: Optional[Any] = None
    ) -> Tuple[Any, MuonState]:
        # Nesterov-style momentum accumulation
        new_mom = jax.tree_util.tree_map(
            lambda m, g: momentum * m + g,
            state.momentum, updates,
        )
        # Orthogonalise 2-D matrices; pass 1-D vectors unchanged
        orth = jax.tree_util.tree_map(
            lambda m: newton_schulz_5(m) if m.ndim >= 2 else m,
            new_mom,
        )
        # μP-compatible RMS normalisation factor (0.2 × lr matches AdamW RMS)
        scaled = jax.tree_util.tree_map(
            lambda u: (-lr * 0.2) * u if u.ndim >= 2 else (-lr) * u,
            orth,
        )
        return scaled, MuonState(momentum=new_mom)

    return optax.GradientTransformation(init_fn, update_fn)


def build_hybrid_optimizer(
    params:       Any,
    lr:           float = BASE_LR,
    min_lr:       float = MIN_LR,
    warmup_steps: int   = WARMUP_STEPS,
    total_steps:  int   = TOTAL_STEPS_PHASE1,
    wd:           float = WEIGHT_DECAY,
) -> optax.GradientTransformation:
    """
    Route 2-D weight matrices → Muon.
    Route 1-D vectors, norms, biases → AdamW.
    Both share the same cosine-decay + warmup learning-rate schedule.
    """
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, lr, warmup_steps),
            optax.cosine_decay_schedule(lr, total_steps - warmup_steps, alpha=min_lr / lr),
        ],
        boundaries=[warmup_steps],
    )

    # ── Muon branch: gradient clipping → momentum + Newton-Schulz → weight decay
    muon_tx = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM),
        scale_by_muon(lr, MUON_MOMENTUM),
        optax.add_decayed_weights(wd),
    )

    # ── AdamW branch: gradient clipping → Adam → weight decay → lr scale
    adamw_tx = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM),
        optax.scale_by_adam(b1=ADAM_B1, b2=ADAM_B2, eps=ADAM_EPS),
        optax.add_decayed_weights(wd),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    )

    # Build label pytree: 'muon' for 2-D+ tensors, 'adamw' for everything else
    labels = jax.tree_util.tree_map(
        lambda leaf: "muon" if leaf.ndim >= 2 else "adamw",
        params,
    )
    return optax.multi_transform(
        {"muon": muon_tx, "adamw": adamw_tx},
        labels,
    )


# ===========================================================================
# SECTION 7 : TRAINING INFRASTRUCTURE
# ===========================================================================

class TrainState(NamedTuple):
    """Immutable training state — fully JAX-compatible pytree."""
    params:    Any
    opt_state: Any
    step:      jnp.ndarray   # uint32 scalar
    rng:       jnp.ndarray   # PRNGKey


def setup_mesh() -> Mesh:
    """Create 1-D FSDP mesh over all TPU cores."""
    devices = mesh_utils.create_device_mesh((NUM_DEVICES,))
    return Mesh(devices, axis_names=("fsdp",))


def get_partition_specs(params: Any) -> Any:
    """
    Assign PartitionSpec for FSDP:
      • 2-D+ tensors: shard along first axis
      • 1-D vectors / scalars: replicate (P(None))
    """
    def _spec(leaf: jnp.ndarray) -> P:  # type: ignore[return]
        if leaf.ndim == 0:
            return P()
        elif leaf.ndim == 1:
            return P(None)
        else:
            return P("fsdp", *([None] * (leaf.ndim - 1)))

    return jax.tree_util.tree_map(_spec, params)


def masked_cross_entropy(
    logits: jnp.ndarray,   # (B, S, V)  float32
    labels: jnp.ndarray,   # (B, S)     int32
    mask:   jnp.ndarray,   # (B, S)     float32  (1 = active, 0 = pad/ignore)
) -> jnp.ndarray:
    """Token-level cross-entropy averaged over active positions."""
    log_p   = jax.nn.log_softmax(logits, axis=-1)
    tgt_lp  = jnp.take_along_axis(log_p, labels[..., None], axis=-1).squeeze(-1)
    loss    = -(tgt_lp * mask).sum() / (mask.sum() + 1e-9)
    return loss


def init_train_state(
    model:        ZenyxV3Model,
    rng:          jnp.ndarray,
    total_steps:  int,
) -> TrainState:
    """Initialise parameters, optimizer state, and step counter."""
    dummy_ids         = jnp.zeros((1, SEQ_LEN), dtype=jnp.int32)
    cos, sin, mscale  = build_yarn_rope(SEQ_LEN, D_ROPE, ROPE_BASE, ROPE_ALPHA, ROPE_BETA, ROPE_SCALE)

    rng, init_rng = jax.random.split(rng)
    variables     = jax.jit(model.init)(init_rng, dummy_ids, cos, sin, mscale)
    params        = variables["params"]

    optimizer  = build_hybrid_optimizer(params, total_steps=total_steps)
    opt_state  = optimizer.init(params)

    return TrainState(
        params=params,
        opt_state=opt_state,
        step=jnp.zeros((), dtype=jnp.uint32),
        rng=rng,
    )


def make_fns(
    model:       ZenyxV3Model,
    cos:         jnp.ndarray,
    sin:         jnp.ndarray,
    mscale:      float,
    total_steps: int,
):
    """
    Returns two jit-compiled functions:
      compute_grads(params, inp, lbl, msk) → (loss, aux, grads)
      apply_updates(state, avg_grads)       → new_state
    """

    # ── Loss closure (pure function of params) ───────────────────────────
    def loss_fn(params, inp, lbl, msk):
        logits, aux = model.apply({"params": params}, inp, cos, sin, mscale)
        ce           = masked_cross_entropy(logits, lbl, msk)
        total        = ce + aux
        return total, (ce, aux)

    @jax.jit
    def compute_grads(
        params: Any,
        inp:    jnp.ndarray,
        lbl:    jnp.ndarray,
        msk:    jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Any]:
        (_, (ce, aux)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, inp, lbl, msk
        )
        return ce, aux, grads

    # Build optimizer once; used inside apply_updates via closure
    # (We need a dummy params to build labels — use a placeholder then rebuild)
    _optimizer_holder: List[optax.GradientTransformation] = []

    def get_optimizer(params: Any) -> optax.GradientTransformation:
        if not _optimizer_holder:
            _optimizer_holder.append(
                build_hybrid_optimizer(params, total_steps=total_steps)
            )
        return _optimizer_holder[0]

    @jax.jit
    def apply_updates(
        state:     TrainState,
        avg_grads: Any,
    ) -> Tuple[TrainState, jnp.ndarray]:
        optimizer  = get_optimizer(state.params)
        grad_norm  = optax.global_norm(avg_grads)
        updates, new_opt = optimizer.update(avg_grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        rng, _     = jax.random.split(state.rng)
        new_state  = TrainState(
            params=new_params,
            opt_state=new_opt,
            step=state.step + jnp.ones((), dtype=jnp.uint32),
            rng=rng,
        )
        return new_state, grad_norm

    return compute_grads, apply_updates


# ===========================================================================
# SECTION 8 : CHECKPOINT UTILITIES
# ===========================================================================

_hf_api = HfApi(token=HF_TOKEN)


def save_checkpoint(
    state:                TrainState,
    global_step:          int,
    shard_index:          int,
    offset_within_shard:  int,
    loss:                 float,
) -> None:
    """
    1. Serialise TrainState with orbax PyTreeCheckpointer.
    2. Write metadata.json alongside.
    3. Upload the entire checkpoint folder to HF repo.
    """
    ckpt_dir = CKPT_LOCAL_DIR / f"checkpoint-{global_step}"
    state_dir = ckpt_dir / "state"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    # ── Orbax serialisation ─────────────────────────────────────────────
    checkpointer = ocp.PyTreeCheckpointer()
    save_payload = {
        "params":    state.params,
        "opt_state": state.opt_state,
        "step":      state.step,
        "rng":       state.rng,
    }
    checkpointer.save(str(state_dir), save_payload, force=True)

    # ── Metadata ────────────────────────────────────────────────────────
    meta = {
        "global_step":           global_step,
        "shard_index":           shard_index,
        "offset_within_shard":   offset_within_shard,
        "loss":                  float(loss),
        "timestamp":             datetime.datetime.utcnow().isoformat() + "Z",
        "tokens_per_step":       TOKENS_PER_UPDATE,
        "seq_len":               SEQ_LEN,
        "d_model":               D_MODEL,
    }
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── HuggingFace upload ───────────────────────────────────────────────
    try:
        _hf_api.create_repo(HF_REPO, repo_type="model", exist_ok=True, token=HF_TOKEN)
        _hf_api.upload_folder(
            folder_path=str(ckpt_dir),
            repo_id=HF_REPO,
            path_in_repo=f"checkpoint-{global_step}",
            repo_type="model",
            token=HF_TOKEN,
        )
        log.info(f"Checkpoint saved to HF: step {global_step}")
    except Exception as exc:
        log.error(f"HF upload failed at step {global_step}: {exc}")


def load_latest_checkpoint(
    model:       ZenyxV3Model,
    total_steps: int,
    rng:         jnp.ndarray,
) -> Tuple[Optional[TrainState], int, int, int]:
    """
    Query HF repo for the latest checkpoint.
    Download its state/ folder, restore with orbax, return:
        (state, global_step, shard_index, offset_within_shard)
    Returns (None, 0, 0, 0) if no checkpoint exists.
    """
    try:
        all_files = list(_hf_api.list_repo_files(
            HF_REPO, repo_type="model", token=HF_TOKEN
        ))
    except Exception:
        log.info("HF repo inaccessible or empty — starting from scratch.")
        return None, 0, 0, 0

    meta_files = [f for f in all_files if f.endswith("metadata.json")]
    if not meta_files:
        log.info("No checkpoints found in HF repo — starting from scratch.")
        return None, 0, 0, 0

    # Find the highest completed step
    best_step, best_meta = -1, ""
    for mf in meta_files:
        try:
            step = int(mf.split("checkpoint-")[1].split("/")[0])
            if step > best_step:
                best_step, best_meta = step, mf
        except (IndexError, ValueError):
            continue

    if best_step < 0:
        return None, 0, 0, 0

    # Download and parse metadata
    local_meta = hf_hub_download(
        repo_id=HF_REPO, filename=best_meta,
        repo_type="model", token=HF_TOKEN,
    )
    with open(local_meta) as f:
        meta = json.load(f)

    global_step         = int(meta["global_step"])
    shard_index         = int(meta["shard_index"])
    offset_within_shard = int(meta["offset_within_shard"])
    log.info(
        f"Resuming from step {global_step}, "
        f"shard {shard_index}, offset {offset_within_shard}"
    )

    # Download state files
    state_prefix = f"checkpoint-{global_step}/state"
    state_files  = [f for f in all_files if state_prefix in f]
    local_ckpt   = CKPT_LOCAL_DIR / f"checkpoint-{global_step}"
    local_ckpt.mkdir(parents=True, exist_ok=True)

    for sf in state_files:
        try:
            hf_hub_download(
                repo_id=HF_REPO, filename=sf,
                repo_type="model", token=HF_TOKEN,
                local_dir=str(local_ckpt),
            )
        except Exception as exc:
            log.warning(f"Could not download {sf}: {exc}")

    # Build a target structure for orbax to restore into
    dummy_state   = init_train_state(model, rng, total_steps)
    restore_tgt   = {
        "params":    dummy_state.params,
        "opt_state": dummy_state.opt_state,
        "step":      dummy_state.step,
        "rng":       dummy_state.rng,
    }

    checkpointer = ocp.PyTreeCheckpointer()
    try:
        restored = checkpointer.restore(
            str(local_ckpt / "state"),
            item=restore_tgt,
        )
        state = TrainState(
            params=restored["params"],
            opt_state=restored["opt_state"],
            step=restored["step"],
            rng=restored["rng"],
        )
        log.info(f"Checkpoint restored: step {global_step}")
        return state, global_step, shard_index, offset_within_shard
    except Exception as exc:
        log.error(f"Orbax restore failed ({exc}) — starting from scratch.")
        return None, 0, 0, 0


# ===========================================================================
# SECTION 9 : MAIN TRAINING LOOP
# ===========================================================================

def main() -> None:
    # ── JAX config ──────────────────────────────────────────────────────────
    jax.config.update("jax_default_matmul_precision", "bfloat16")
    log.info(f"JAX version {jax.__version__}  |  devices: {jax.devices()}")

    # ── Mesh ────────────────────────────────────────────────────────────────
    mesh = setup_mesh()
    log.info(f"FSDP mesh: {mesh}")

    # ── YaRN positional tables (computed once, never updated) ────────────
    cos_np, sin_np, mscale = build_yarn_rope(
        SEQ_LEN, D_ROPE, ROPE_BASE, ROPE_ALPHA, ROPE_BETA, ROPE_SCALE
    )
    cos_g = jnp.array(cos_np)   # (SEQ_LEN, D_ROPE // 2)
    sin_g = jnp.array(sin_np)
    log.info(f"YaRN mscale = {mscale:.4f}  |  context = {SEQ_LEN} tokens")

    # ── Model ────────────────────────────────────────────────────────────────
    model = ZenyxV3Model(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    root_rng = jax.random.PRNGKey(0)

    # ── Checkpoint resume ────────────────────────────────────────────────────
    with mesh:
        state, global_step, shard_idx, shard_offset = load_latest_checkpoint(
            model, TOTAL_STEPS_PHASE1, root_rng
        )
        if state is None:
            state = init_train_state(model, root_rng, TOTAL_STEPS_PHASE1)
            global_step = shard_idx = shard_offset = 0
            log.info("Initialised fresh model.")

        # Shard params across the FSDP mesh
        param_specs  = get_partition_specs(state.params)
        sharded_p    = jax.device_put(
            state.params, NamedSharding(mesh, param_specs)  # type: ignore[arg-type]
        )
        state = TrainState(
            params=sharded_p,
            opt_state=state.opt_state,
            step=state.step,
            rng=state.rng,
        )

    # ── Compiled training functions ─────────────────────────────────────────
    compute_grads, apply_updates = make_fns(
        model, cos_g, sin_g, mscale, TOTAL_STEPS_PHASE1
    )

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = load_tokenizer()

    # ── Dataset — O(1) resume via .skip() ───────────────────────────────────
    #
    # tokens_consumed = global_step * TOKENS_PER_UPDATE
    # examples_consumed ≈ tokens_consumed / AVG_TOKENS_PER_EXAMPLE
    # The interleaved streaming dataset's .skip(n) is O(1) on sharded Parquet
    # because the HuggingFace datasets library seeks directly to the correct shard.
    #
    tokens_consumed  = global_step * TOKENS_PER_UPDATE
    skip_examples    = tokens_consumed // AVG_TOKENS_PER_EXAMPLE
    log.info(
        f"Tokens consumed so far: {tokens_consumed:,}  |  "
        f"Skipping {skip_examples:,} dataset examples"
    )

    dataset  = build_streaming_dataset(skip_examples=skip_examples)
    micro_bs = PER_DEVICE_BATCH * NUM_DEVICES   # sequences per micro-step
    data_it  = batch_generator(dataset, tokenizer, micro_bs, SEQ_LEN)

    # ── Training state accumulators ──────────────────────────────────────────
    acc_grads: Optional[Any] = None
    acc_ce    = 0.0
    acc_aux   = 0.0
    micro_ctr = 0

    t0_log   = time.perf_counter()
    tok_log  = 0

    log.info(
        f"Training Phase 1 | steps {global_step} → {TOTAL_STEPS_PHASE1} "
        f"| {TOKENS_PER_UPDATE/1e6:.2f}M tokens/step"
    )

    # ── Main loop ────────────────────────────────────────────────────────────
    for inp_np, lbl_np, msk_np in data_it:

        if global_step >= TOTAL_STEPS_PHASE1:
            log.info("Phase 1 training complete.")
            break

        # Convert numpy → JAX (stays on host until jit dispatch)
        inp_j = jnp.array(inp_np, dtype=jnp.int32)
        lbl_j = jnp.array(lbl_np, dtype=jnp.int32)
        msk_j = jnp.array(msk_np, dtype=jnp.bfloat16)

        # ── Micro-step: compute gradients ───────────────────────────────────
        with mesh:
            ce, aux, grads = compute_grads(state.params, inp_j, lbl_j, msk_j)

        acc_ce  += float(ce)
        acc_aux += float(aux)
        tok_log += micro_bs * SEQ_LEN

        if acc_grads is None:
            acc_grads = grads
        else:
            acc_grads = jax.tree_util.tree_map(
                lambda a, b: a + b, acc_grads, grads
            )

        micro_ctr += 1

        # ── Parameter update after GRAD_ACCUM_STEPS micro-steps ─────────────
        if micro_ctr < GRAD_ACCUM_STEPS:
            continue

        # Average accumulated gradients
        avg_grads = jax.tree_util.tree_map(
            lambda g: g / GRAD_ACCUM_STEPS, acc_grads
        )

        with mesh:
            state, grad_norm = apply_updates(state, avg_grads)

        global_step += 1

        # Reset accumulators
        acc_grads = None
        avg_ce    = acc_ce  / GRAD_ACCUM_STEPS
        avg_aux   = acc_aux / GRAD_ACCUM_STEPS
        acc_ce    = acc_aux = 0.0
        micro_ctr = 0

        # ── Logging ──────────────────────────────────────────────────────────
        if global_step % LOG_EVERY_STEPS == 0:
            t_now     = time.perf_counter()
            elapsed   = max(t_now - t0_log, 1e-6)
            tok_s     = tok_log / elapsed
            step_time = elapsed / LOG_EVERY_STEPS
            print(
                f"step {global_step:>8,} | "
                f"loss {avg_ce:.4f} | aux {avg_aux:.5f} | "
                f"tok/s {tok_s:>10,.0f} | "
                f"step_time {step_time:.2f}s"
            )
            t0_log  = t_now
            tok_log = 0

        if global_step % VERBOSE_EVERY_STEPS == 0:
            # Approximate current learning rate from cosine schedule
            w  = min(global_step / max(WARMUP_STEPS, 1), 1.0)
            frac = max(0.0, (global_step - WARMUP_STEPS) / max(TOTAL_STEPS_PHASE1 - WARMUP_STEPS, 1))
            lr_now = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + math.cos(math.pi * frac))
            print(
                f"  └─ lr {lr_now:.3e} | "
                f"grad_norm {float(grad_norm):.4f} | "
                f"step {global_step}/{TOTAL_STEPS_PHASE1}"
            )

        # ── Checkpoint ───────────────────────────────────────────────────────
        if global_step % SAVE_EVERY_STEPS == 0:
            cur_tokens   = global_step * TOKENS_PER_UPDATE
            # shard_index and offset for O(1) resume calculation:
            #   shard_index         = examples consumed // examples per shard
            #   offset_within_shard = examples consumed %  examples per shard
            #
            # We approximate: 1 "shard" ≈ 1 000 000 source examples.
            EXAMPLES_PER_SHARD   = 1_000_000
            total_examples_done  = cur_tokens // AVG_TOKENS_PER_EXAMPLE
            shard_idx            = int(total_examples_done // EXAMPLES_PER_SHARD)
            shard_offset_val     = int(total_examples_done %  EXAMPLES_PER_SHARD)

            save_checkpoint(
                state        = state,
                global_step  = global_step,
                shard_index  = shard_idx,
                offset_within_shard = shard_offset_val,
                loss         = avg_ce,
            )

    # ── Final checkpoint ─────────────────────────────────────────────────────
    log.info("Saving final checkpoint …")
    final_tokens       = global_step * TOKENS_PER_UPDATE
    final_examples     = final_tokens // AVG_TOKENS_PER_EXAMPLE
    EXAMPLES_PER_SHARD = 1_000_000
    save_checkpoint(
        state        = state,
        global_step  = global_step,
        shard_index  = int(final_examples // EXAMPLES_PER_SHARD),
        offset_within_shard = int(final_examples % EXAMPLES_PER_SHARD),
        loss         = avg_ce if acc_ce == 0.0 else acc_ce / max(micro_ctr, 1),
    )
    log.info("Done.")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    main()
