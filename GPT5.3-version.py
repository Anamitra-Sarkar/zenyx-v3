
from __future__ import annotations

import json, math, os, re, time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import linen as nn
from flax import struct
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from datasets import load_dataset, interleave_datasets
import sentencepiece as spm

jax.config.update("jax_default_matmul_precision", "bfloat16")
jax.config.update("jax_enable_x64", False)

@dataclass(frozen=True)
class Config:
    hf_token: str = "YOUR_HF_TOKEN_HERE"
    hf_repo: str = "Arko007/zenyx-v3-checkpoints"
    tokenizer_repo: str = "Arko007/zenyx-v3-tokenizer"
    tokenizer_filename: str = "tokenizer.model"

    d_model: int = 1536
    num_heads: int = 12
    d_latent: int = 256
    d_rope: int = 64
    d_ff: int = 4096
    num_shared_experts: int = 2
    num_routed_experts: int = 64
    num_recurrences: int = 12
    seq_len: int = 32768
    vocab_size: int = 65536
    local_window: int = 256
    rope_base: float = 10000.0
    rope_alpha: float = 1.0
    rope_beta: float = 32.0
    rope_scale_s: float = 4.0

    base_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.05
    muon_momentum: float = 0.95
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0

    seed: int = 42
    per_device_batch_size: int = 1
    global_batch_size: int = 8
    max_steps: int = 100000
    log_every: int = 10
    log_lr_every: int = 100
    checkpoint_every: int = 500
    tokens_per_shard: int = 131072
    checkpoint_root: str = "./zenyx_v3_ckpts"

    fineweb_repo: str = "HuggingFaceFW/fineweb-edu"
    stack_repo: str = "bigcode/the-stack-v2"
    math_repo: str = "AI-MO/NuminaMath-CoT"
    interleave_probs: Tuple[float, float, float] = (0.60, 0.25, 0.15)
    checkpoint_repo_type: str = "model"

CFG = Config()

TEXT_KEYS = ("text", "content", "code", "body", "question", "problem", "prompt", "response", "solution", "messages")

def extract_text(example: Dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value:
            if isinstance(value[0], str):
                return "\n".join(value)
            if isinstance(value[0], dict):
                chunks = []
                for item in value:
                    if isinstance(item, dict):
                        for k in ("content", "text", "value", "message"):
                            v = item.get(k)
                            if isinstance(v, str) and v.strip():
                                chunks.append(v)
                                break
                if chunks:
                    return "\n".join(chunks)
    return ""

def normalize_example(example: Dict[str, Any]) -> Dict[str, str]:
    return {"text": extract_text(example)}

def load_sentencepiece_tokenizer(cfg: Config) -> spm.SentencePieceProcessor:
    candidates = [(cfg.tokenizer_repo, cfg.tokenizer_filename), (cfg.hf_repo, cfg.tokenizer_filename)]
    last_error = None
    tokenizer_path = None
    for repo_id, filename in candidates:
        try:
            tokenizer_path = hf_hub_download(repo_id=repo_id, filename=filename, token=cfg.hf_token, repo_type="model")
            break
        except Exception as exc:
            last_error = exc
    if tokenizer_path is None:
        raise RuntimeError("Could not download tokenizer.model") from last_error
    sp = spm.SentencePieceProcessor()
    if not sp.load(tokenizer_path):
        raise RuntimeError(f"Failed to load SentencePiece model: {tokenizer_path}")
    return sp

def load_streaming_sources(cfg: Config):
    fineweb = load_dataset(cfg.fineweb_repo, split="train", streaming=True).map(normalize_example)
    stack = load_dataset(cfg.stack_repo, split="train", streaming=True).map(normalize_example)
    math = load_dataset(cfg.math_repo, split="train", streaming=True).map(normalize_example)
    return interleave_datasets([fineweb, stack, math], probabilities=list(cfg.interleave_probs), seed=cfg.seed, stopping_strategy="all_exhausted")

class TokenRingBuffer:
    def __init__(self, seq_len: int, eos_id: int):
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.buffer = deque()

    def feed(self, tokens: Sequence[int]) -> None:
        self.buffer.extend(int(t) for t in tokens)

    def can_emit(self) -> bool:
        return len(self.buffer) >= self.seq_len

    def emit(self) -> np.ndarray:
        return np.asarray([self.buffer.popleft() for _ in range(self.seq_len)], dtype=np.int32)

def pack_token_stream(iterable_dataset, tokenizer: spm.SentencePieceProcessor, seq_len: int, batch_size: int, skip_tokens: int = 0):
    eos_id = tokenizer.eos_id() if tokenizer.eos_id() >= 0 else 1
    ring = TokenRingBuffer(seq_len=seq_len, eos_id=eos_id)
    skipped = 0
    for example in iterable_dataset:
        text = extract_text(example)
        if not text:
            continue
        ids = tokenizer.encode(text, out_type=int)
        if not ids:
            continue
        for tok in ids + [eos_id]:
            if skipped < skip_tokens:
                skipped += 1
                continue
            ring.buffer.append(int(tok))
            while len(ring.buffer) >= seq_len * batch_size:
                blocks = [ring.emit() for _ in range(batch_size)]
                yield {"tokens": np.stack(blocks, axis=0), "loss_mask": np.ones((batch_size, seq_len), dtype=np.float32)}

def build_training_stream(cfg: Config, tokenizer: spm.SentencePieceProcessor, shard_index: int = 0, offset_within_shard: int = 0):
    ds = load_streaming_sources(cfg)
    if shard_index > 0:
        ds = ds.skip(shard_index)
    return pack_token_stream(ds, tokenizer, cfg.seq_len, cfg.global_batch_size, skip_tokens=offset_within_shard)

FP8_E4M3 = getattr(jnp, "float8_e4m3fn", jnp.bfloat16)

class FP8Dense(nn.Module):
    features: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param("kernel", nn.initializers.variance_scaling(1.0, "fan_in", "normal"), (inputs.shape[-1], self.features), jnp.float32)
        if FP8_E4M3 == jnp.bfloat16:
            out = jnp.einsum("...d,df->...f", inputs.astype(jnp.bfloat16), kernel.astype(jnp.bfloat16))
        else:
            x_amax = jnp.max(jnp.abs(inputs))
            w_amax = jnp.max(jnp.abs(kernel))
            e4m3_max = jnp.finfo(FP8_E4M3).max
            x_scale = e4m3_max / jnp.maximum(x_amax, 1e-12)
            w_scale = e4m3_max / jnp.maximum(w_amax, 1e-12)
            x_fp8 = (inputs * x_scale).astype(FP8_E4M3)
            w_fp8 = (kernel * w_scale).astype(FP8_E4M3)
            out = jax.lax.dot_general(x_fp8, w_fp8, dimension_numbers=(((inputs.ndim - 1,), (0,)), ((), ())), preferred_element_type=jnp.bfloat16)
            out = out / (x_scale * w_scale)
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,), jnp.float32)
            out = out + bias.astype(jnp.bfloat16)
        return out.astype(jnp.bfloat16)

def build_yarn_rope(seq_len: int, d_rope: int, base: float = 10000.0, alpha: float = 1.0, beta: float = 32.0, scale_s: float = 4.0):
    m = jnp.arange(d_rope // 2)
    theta_m = base ** (-2.0 * m / d_rope)
    lambda_m = 2 * jnp.pi / theta_m
    r_m = 8192.0 / lambda_m
    gamma = jnp.where(r_m < alpha, 0.0, jnp.where(r_m > beta, 1.0, (r_m - alpha) / (beta - alpha)))
    theta_yarn = (1.0 - gamma) * (theta_m / scale_s) + gamma * theta_m
    positions = jnp.arange(seq_len)
    angles = jnp.outer(positions, theta_yarn)
    cos_val = jnp.cos(angles).astype(jnp.bfloat16)
    sin_val = jnp.sin(angles).astype(jnp.bfloat16)
    mscale = (0.1 * jnp.log(scale_s) + 1.0).astype(jnp.bfloat16)
    return cos_val, sin_val, mscale

def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_decoupled_yarn_rope(x_rope: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x_rope, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

class ZenyxHybridAttention(nn.Module):
    d_model: int
    num_heads: int
    d_latent: int
    d_rope: int
    seq_len: int
    local_window: int = 256

    def setup(self):
        self.d_head = self.d_model // self.num_heads
        self.q_proj = FP8Dense(self.d_latent)
        self.kv_proj = FP8Dense(self.d_latent)
        self.q_up = FP8Dense(self.num_heads * self.d_head)
        self.kv_up_k = FP8Dense(self.num_heads * self.d_head)
        self.kv_up_v = FP8Dense(self.num_heads * self.d_head)
        self.q_rope_proj = FP8Dense(self.num_heads * self.d_rope)
        self.k_rope_proj = FP8Dense(self.num_heads * self.d_rope)
        self.o_proj = FP8Dense(self.d_model)

    def __call__(self, x: jnp.ndarray, is_hca_layer: bool) -> jnp.ndarray:
        batch, seq_len, _ = x.shape
        compress_ratio = 128 if is_hca_layer else 4
        num_chunks = seq_len // compress_ratio
        ctx_len = num_chunks + self.local_window

        c_q = self.q_proj(x)
        c_kv = self.kv_proj(x)
        q_nope = self.q_up(c_q).reshape(batch, seq_len, self.num_heads, self.d_head)

        c_kv_compressed = c_kv.reshape(batch, num_chunks, compress_ratio, self.d_latent).mean(axis=2)
        k_nope = self.kv_up_k(c_kv_compressed).reshape(batch, num_chunks, self.num_heads, self.d_head)
        v_nope = self.kv_up_v(c_kv_compressed).reshape(batch, num_chunks, self.num_heads, self.d_head)

        local_c_kv = c_kv[:, -self.local_window :, :]
        local_k = self.kv_up_k(local_c_kv).reshape(batch, self.local_window, self.num_heads, self.d_head)
        local_v = self.kv_up_v(local_c_kv).reshape(batch, self.local_window, self.num_heads, self.d_head)

        k_assembled = jnp.concatenate([k_nope, local_k], axis=1)
        v_assembled = jnp.concatenate([v_nope, local_v], axis=1)

        q_rope = self.q_rope_proj(x).reshape(batch, seq_len, self.num_heads, self.d_rope)
        k_rope = self.k_rope_proj(x).reshape(batch, seq_len, self.num_heads, self.d_rope)
        k_rope_compressed = k_rope.reshape(batch, num_chunks, compress_ratio, self.num_heads, self.d_rope).mean(axis=2)
        k_rope_local = k_rope[:, -self.local_window :, :, :]
        k_rope_final = jnp.concatenate([k_rope_compressed, k_rope_local], axis=1)

        cos_q, sin_q, mscale_q = build_yarn_rope(seq_len, self.d_rope)
        cos_k, sin_k, mscale_k = build_yarn_rope(ctx_len, self.d_rope)
        q_rope = apply_decoupled_yarn_rope(q_rope, cos_q[None, :, None, :], sin_q[None, :, None, :]) * mscale_q
        k_rope_final = apply_decoupled_yarn_rope(k_rope_final, cos_k[None, :, None, :], sin_k[None, :, None, :]) * mscale_k

        q_final = jnp.concatenate([q_nope, q_rope], axis=-1)
        k_final = jnp.concatenate([k_assembled, k_rope_final], axis=-1)
        scale = 1.0 / jnp.sqrt(self.d_head + self.d_rope)
        attn_logits = jnp.einsum("bshd,bthd->bhst", q_final, k_final) * scale

        mask = jnp.tril(jnp.ones((seq_len, ctx_len), dtype=jnp.bool_))
        attn_logits = jnp.where(mask[None, None, :, :], attn_logits, jnp.array(-1e9, dtype=attn_logits.dtype))

        if not is_hca_layer:
            top_k = 64
            top_vals = jax.lax.top_k(attn_logits, top_k)[0]
            threshold = top_vals[..., -1:,]
            attn_logits = jnp.where(attn_logits >= threshold, attn_logits, jnp.array(-1e9, dtype=attn_logits.dtype))

        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_output = jnp.einsum("bhst,bthd->bshd", attn_weights, v_assembled)
        attn_output = attn_output.reshape(batch, seq_len, self.num_heads * self.d_head)
        return self.o_proj(attn_output)

class DualSharedSparseMoE(nn.Module):
    d_model: int
    d_ff: int
    num_routed_experts: int

    def setup(self):
        d_ff_split = self.d_ff // 2
        self.shared_1_w1 = FP8Dense(d_ff_split)
        self.shared_1_w2 = FP8Dense(self.d_model)
        self.shared_2_w1 = FP8Dense(d_ff_split)
        self.shared_2_w2 = FP8Dense(self.d_model)
        self.router = nn.Dense(self.num_routed_experts, use_bias=False, dtype=jnp.float32)
        self.routed_w1 = self.param("routed_w1", nn.initializers.lecun_normal(), (self.num_routed_experts, self.d_model, self.d_ff), jnp.bfloat16)
        self.routed_w2 = self.param("routed_w2", nn.initializers.lecun_normal(), (self.num_routed_experts, self.d_ff, self.d_model), jnp.bfloat16)

    def __call__(self, x: jnp.ndarray):
        shared_1 = self.shared_1_w2(jax.nn.silu(self.shared_1_w1(x)))
        shared_2 = self.shared_2_w2(jax.nn.silu(self.shared_2_w1(x)))
        shared_out = shared_1 + shared_2

        router_logits = self.router(x)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        expert_indices = jnp.argmax(router_probs, axis=-1)
        expert_gates = jnp.max(router_probs, axis=-1, keepdims=True)

        selected_w1 = self.routed_w1[expert_indices]
        selected_w2 = self.routed_w2[expert_indices]
        h_routed = jax.nn.silu(jnp.einsum("bsd,bsdf->bsf", x.astype(jnp.bfloat16), selected_w1))
        routed_out = jnp.einsum("bsf,bsfd->bsd", h_routed, selected_w2)
        final_out = shared_out + (routed_out * expert_gates.astype(jnp.bfloat16))

        expert_mask = jax.nn.one_hot(expert_indices, self.num_routed_experts, dtype=jnp.float32)
        f_i = jnp.mean(expert_mask, axis=(0, 1))
        P_i = jnp.mean(router_probs, axis=(0, 1))
        aux_loss = 0.01 * self.num_routed_experts * jnp.sum(f_i * P_i)
        return final_out, aux_loss.astype(jnp.bfloat16)

class ZenyxRecurrentSuperBlock(nn.Module):
    d_model: int
    num_recurrences: int
    d_latent: int
    d_rope: int
    seq_len: int
    num_heads: int
    d_ff: int
    num_routed_experts: int

    def setup(self):
        self.hybrid_attn = ZenyxHybridAttention(self.d_model, self.num_heads, self.d_latent, self.d_rope, self.seq_len)
        self.moe = DualSharedSparseMoE(self.d_model, self.d_ff, self.num_routed_experts)
        self.norm1 = nn.RMSNorm()
        self.norm2 = nn.RMSNorm()
        self.gamma_1 = self.param("gamma_1", nn.initializers.constant(1e-4), (self.d_model,))
        self.gamma_2 = self.param("gamma_2", nn.initializers.constant(1e-4), (self.d_model,))

    def __call__(self, x: jnp.ndarray):
        def recurrent_step(carry, step_idx):
            x_in = carry
            x_norm = self.norm1(x_in)
            is_hca = (step_idx % 2 == 0)
            attn_out = self.hybrid_attn(x_norm, is_hca_layer=bool(is_hca))
            x_mid = x_in + (attn_out * self.gamma_1)
            moe_out, aux_loss = self.moe(self.norm2(x_mid))
            x_out = x_mid + (moe_out * self.gamma_2)
            return x_out, aux_loss
        final_x, aux_losses = jax.lax.scan(jax.checkpoint(recurrent_step), x, jnp.arange(self.num_recurrences))
        return final_x, jnp.sum(aux_losses)

class ZenyxV3Model(nn.Module):
    vocab_size: int
    d_model: int
    d_latent: int
    d_rope: int
    seq_len: int
    num_heads: int
    d_ff: int
    num_shared_experts: int
    num_routed_experts: int
    num_recurrences: int

    def setup(self):
        self.embed = nn.Embed(self.vocab_size, self.d_model, dtype=jnp.bfloat16)
        self.superblock = ZenyxRecurrentSuperBlock(self.d_model, self.num_recurrences, self.d_latent, self.d_rope, self.seq_len, self.num_heads, self.d_ff, self.num_routed_experts)
        self.final_norm = nn.RMSNorm()
        self.lm_head = FP8Dense(self.vocab_size, use_bias=False)

    def __call__(self, input_ids: jnp.ndarray):
        x = self.embed(input_ids)
        x, aux_loss = self.superblock(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits.astype(jnp.float32), aux_loss.astype(jnp.float32)

class MuonState(struct.PyTreeNode):
    momentum: Any

def newton_schulz_iteration(G: jnp.ndarray, steps: int = 5) -> jnp.ndarray:
    if G.ndim < 2:
        return G
    X = G.astype(jnp.bfloat16)
    transpose_flag = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose_flag = True
    X = X / (jnp.linalg.norm(X.astype(jnp.float32), ord="fro") + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    def ns_step(x_curr, _):
        A = x_curr @ x_curr.T
        B = b * A + c * (A @ A)
        x_next = a * x_curr + (B @ x_curr)
        return x_next.astype(jnp.bfloat16), None
    X_final, _ = jax.lax.scan(ns_step, X, None, length=steps)
    if transpose_flag:
        X_final = X_final.T
    return X_final.astype(G.dtype)

def scale_by_muon(momentum: float = 0.95) -> optax.GradientTransformation:
    def init_fn(params):
        return MuonState(momentum=jax.tree_util.tree_map(jnp.zeros_like, params))
    def update_fn(updates, state, params=None):
        mu_next = jax.tree_util.tree_map(lambda m, g: momentum * m + g, state.momentum, updates)
        def apply_muon(m):
            return newton_schulz_iteration(m) if getattr(m, "ndim", 0) >= 2 else m
        ortho = jax.tree_util.tree_map(apply_muon, mu_next)
        return ortho, MuonState(momentum=mu_next)
    return optax.GradientTransformation(init_fn, update_fn)

def create_lr_schedule(cfg: Config):
    warmup = optax.linear_schedule(0.0, cfg.base_lr, cfg.warmup_steps)
    decay_steps = max(cfg.max_steps - cfg.warmup_steps, 1)
    cosine = optax.cosine_decay_schedule(cfg.base_lr, decay_steps, alpha=cfg.min_lr / cfg.base_lr)
    return optax.join_schedules([warmup, cosine], [cfg.warmup_steps])

def create_param_labels(params):
    return jax.tree_util.tree_map(lambda p: "muon" if getattr(p, "ndim", 0) >= 2 else "adamw", params)

def build_hybrid_optimizer(cfg: Config, params):
    schedule = create_lr_schedule(cfg)
    muon_tx = optax.chain(optax.add_decayed_weights(cfg.weight_decay), scale_by_muon(cfg.muon_momentum), optax.scale_by_schedule(schedule), optax.scale(-1.0))
    adamw_tx = optax.adamw(learning_rate=schedule, b1=cfg.adam_beta1, b2=cfg.adam_beta2, eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
    labels = create_param_labels(params)
    tx = optax.multi_transform({"muon": muon_tx, "adamw": adamw_tx}, labels)
    return tx, labels, schedule

@struct.PyTreeNode
class TrainState:
    params: Any
    opt_state: optax.OptState
    rng_key: jax.Array
    global_step: jnp.uint32
    shard_index: jnp.uint32
    offset_within_shard: jnp.uint32

def setup_mesh():
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh
    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh((len(devices),))
    return Mesh(device_mesh, axis_names=("fsdp",))

def get_partition_specs():
    from jax.sharding import PartitionSpec as P
    return {
        "Embed/embedding": P("fsdp", None),
        "q_proj/kernel": P(None, "fsdp"),
        "kv_proj/kernel": P(None, "fsdp"),
        "q_up/kernel": P(None, "fsdp"),
        "kv_up_k/kernel": P(None, "fsdp"),
        "kv_up_v/kernel": P(None, "fsdp"),
        "o_proj/kernel": P("fsdp", None),
        "q_rope_proj/kernel": P(None, "fsdp"),
        "k_rope_proj/kernel": P(None, "fsdp"),
        "shared_1_w1/kernel": P(None, "fsdp"),
        "shared_1_w2/kernel": P("fsdp", None),
        "shared_2_w1/kernel": P(None, "fsdp"),
        "shared_2_w2/kernel": P("fsdp", None),
        "router/kernel": P(None, "fsdp"),
        "routed_w1": P("fsdp", None, None),
        "routed_w2": P("fsdp", None, None),
        "gamma_1": P(None),
        "gamma_2": P(None),
    }

def calculate_masked_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray, loss_mask: jnp.ndarray) -> jnp.ndarray:
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
    loss_mask = loss_mask[:, 1:]
    xent = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    xent = xent * loss_mask
    return jnp.sum(xent) / jnp.maximum(jnp.sum(loss_mask), 1.0)

def calculate_auxiliary_loss(aux_loss: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(aux_loss, dtype=jnp.float32)

def global_norm(tree) -> jnp.ndarray:
    return optax.global_norm(tree)

def init_train_state(cfg: Config, model: ZenyxV3Model, rng: jax.Array):
    dummy_tokens = jnp.zeros((cfg.per_device_batch_size, cfg.seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_tokens)
    params = variables["params"]
    tx, labels, schedule = build_hybrid_optimizer(cfg, params)
    opt_state = tx.init(params)
    state = TrainState(params=params, opt_state=opt_state, rng_key=rng, global_step=jnp.uint32(0), shard_index=jnp.uint32(0), offset_within_shard=jnp.uint32(0))
    return state, tx, labels, schedule

def _apply_updates(params, updates):
    return optax.apply_updates(params, updates)

def train_step_fn(state: TrainState, batch: Dict[str, jnp.ndarray], model: ZenyxV3Model, tx, schedule_fn):
    rng, new_rng = jax.random.split(state.rng_key)
    def loss_fn(params):
        logits, aux_loss = model.apply({"params": params}, batch["tokens"])
        ce = calculate_masked_cross_entropy(logits, batch["tokens"], batch["loss_mask"])
        aux = calculate_auxiliary_loss(aux_loss)
        total = ce + aux
        return total, {"ce_loss": ce, "aux_loss": aux}
    (loss, metrics0), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
    new_params = _apply_updates(state.params, updates)
    grad_norm = global_norm(grads)
    lr = schedule_fn(state.global_step)
    new_state = state.replace(params=new_params, opt_state=new_opt_state, rng_key=new_rng, global_step=state.global_step + jnp.uint32(1))
    metrics = {"loss": loss, "ce_loss": metrics0["ce_loss"], "aux_loss": metrics0["aux_loss"], "grad_norm": grad_norm, "lr": lr}
    return new_state, metrics

train_step = jax.jit(train_step_fn, static_argnames=("model", "tx", "schedule_fn"))

def checkpoint_dir_for_step(cfg: Config, step: int) -> Path:
    return Path(cfg.checkpoint_root) / f"checkpoint-{step}"

def make_checkpoint_payload(state: TrainState) -> Dict[str, Any]:
    return {"params": state.params, "opt_state": state.opt_state, "global_step": state.global_step, "rng_key": state.rng_key, "shard_index": state.shard_index, "offset_within_shard": state.offset_within_shard}

def save_checkpoint(cfg: Config, state: TrainState, loss_value: float) -> None:
    step = int(state.global_step)
    ckpt_root = checkpoint_dir_for_step(cfg, step)
    params_dir = ckpt_root / "params"
    params_dir.parent.mkdir(parents=True, exist_ok=True)
    ckptr = ocp.PyTreeCheckpointer()
    payload = make_checkpoint_payload(state)
    metadata = {"global_step": step, "shard_index": int(state.shard_index), "offset_within_shard": int(state.offset_within_shard), "loss": float(loss_value), "timestamp": datetime.now(timezone.utc).isoformat(), "tokens_per_shard": cfg.tokens_per_shard}
    ckptr.save(str(params_dir), state=payload, force=True, custom_metadata=metadata)
    with open(ckpt_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    ckptr.close()
    api = HfApi(token=cfg.hf_token)
    api.upload_folder(folder_path=str(ckpt_root), repo_id=cfg.hf_repo, repo_type=cfg.checkpoint_repo_type, path_in_repo=f"checkpoint-{step}")
    print(f"Checkpoint saved to HF: step {step}")

def load_latest_checkpoint(cfg: Config):
    api = HfApi(token=cfg.hf_token)
    try:
        files = api.list_repo_files(repo_id=cfg.hf_repo, repo_type=cfg.checkpoint_repo_type)
    except Exception:
        return None
    pattern = re.compile(r"checkpoint-(\d+)/metadata\.json$")
    steps = []
    for path in files:
        m = pattern.search(path)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        return None
    step = max(steps)
    remote_dir = f"checkpoint-{step}"
    local_snapshot = snapshot_download(repo_id=cfg.hf_repo, repo_type=cfg.checkpoint_repo_type, allow_patterns=[f"{remote_dir}/params/**", f"{remote_dir}/metadata.json"], token=cfg.hf_token)
    metadata_path = Path(local_snapshot) / remote_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    params_dir = Path(local_snapshot) / remote_dir / "params"
    return params_dir, metadata

def restore_train_state_from_checkpoint(cfg: Config, model: ZenyxV3Model, tx, params_dir: Path, metadata: Dict[str, Any], rng: jax.Array):
    dummy_tokens = jnp.zeros((cfg.per_device_batch_size, cfg.seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_tokens)
    abstract_payload = {
        "params": jax.tree_util.tree_map(jax.ShapeDtypeStruct, variables["params"]),
        "opt_state": jax.tree_util.tree_map(jax.ShapeDtypeStruct, tx.init(variables["params"])),
        "global_step": jax.ShapeDtypeStruct((), jnp.uint32),
        "rng_key": jax.ShapeDtypeStruct((2,), jnp.uint32),
        "shard_index": jax.ShapeDtypeStruct((), jnp.uint32),
        "offset_within_shard": jax.ShapeDtypeStruct((), jnp.uint32),
    }
    ckptr = ocp.PyTreeCheckpointer()
    restored = ckptr.restore(str(params_dir), state=abstract_payload)
    ckptr.close()
    return TrainState(params=restored["params"], opt_state=restored["opt_state"], rng_key=restored["rng_key"], global_step=restored["global_step"], shard_index=jnp.uint32(metadata.get("shard_index", 0)), offset_within_shard=jnp.uint32(metadata.get("offset_within_shard", 0)))

def main():
    cfg = CFG
    Path(cfg.checkpoint_root).mkdir(parents=True, exist_ok=True)
    tokenizer = load_sentencepiece_tokenizer(cfg)
    model = ZenyxV3Model(cfg.vocab_size, cfg.d_model, cfg.d_latent, cfg.d_rope, cfg.seq_len, cfg.num_heads, cfg.d_ff, cfg.num_shared_experts, cfg.num_routed_experts, cfg.num_recurrences)
    rng = jax.random.PRNGKey(cfg.seed)
    mesh = setup_mesh()
    _ = get_partition_specs()

    latest = load_latest_checkpoint(cfg)

    with mesh:
        if latest is not None:
            params_dir, metadata = latest
            temp_vars = model.init(rng, jnp.zeros((cfg.per_device_batch_size, cfg.seq_len), dtype=jnp.int32))
            tx, labels, schedule_fn = build_hybrid_optimizer(cfg, temp_vars["params"])
            state = restore_train_state_from_checkpoint(cfg, model, tx, params_dir, metadata, rng)
            tokens_consumed = int(state.global_step) * cfg.global_batch_size * cfg.seq_len
            shard_index = tokens_consumed // cfg.tokens_per_shard
            offset_within_shard = tokens_consumed % cfg.tokens_per_shard
            print(f"Resuming from step {int(state.global_step)}, shard {shard_index}, offset {offset_within_shard}")
            state = state.replace(shard_index=jnp.uint32(shard_index), offset_within_shard=jnp.uint32(offset_within_shard))
        else:
            state, tx, labels, schedule_fn = init_train_state(cfg, model, rng)
            print("Starting from scratch")

        stream = build_training_stream(cfg, tokenizer, shard_index=int(state.shard_index), offset_within_shard=int(state.offset_within_shard))
        step_timer = time.perf_counter()
        tokens_per_step = cfg.global_batch_size * cfg.seq_len
        last_loss = 0.0

        for _, batch_np in enumerate(stream, start=int(state.global_step) + 1):
            if int(state.global_step) >= cfg.max_steps:
                break
            batch = {"tokens": jnp.asarray(batch_np["tokens"], dtype=jnp.int32), "loss_mask": jnp.asarray(batch_np["loss_mask"], dtype=jnp.float32)}
            state, metrics = train_step(state, batch, model=model, tx=tx, schedule_fn=schedule_fn)
            last_loss = float(metrics["loss"])
            now = time.perf_counter()
            step_time = now - step_timer
            step_timer = now
            tokens_sec = float(tokens_per_step) / max(step_time, 1e-6)
            step = int(state.global_step)
            if step % cfg.log_every == 0:
                print(f"step {step} | loss {float(metrics['loss']):.4f} | aux_loss {float(metrics['aux_loss']):.4f} | tokens/sec {tokens_sec:.2f} | step_time {step_time:.2f}s")
            if step % cfg.log_lr_every == 0:
                print(f"lr {float(metrics['lr']):.8f} | grad_norm {float(metrics['grad_norm']):.4f}")
            if step % cfg.checkpoint_every == 0:
                save_checkpoint(cfg, state, last_loss)

        if int(state.global_step) % cfg.checkpoint_every != 0:
            save_checkpoint(cfg, state, last_loss)

if __name__ == "__main__":
    main()
