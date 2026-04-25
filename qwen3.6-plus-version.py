# zenyx_v3_train.py
# === IMPORTS ===
import os
import json
import time
import math
import numpy as np
from typing import Any, Dict, Tuple, Optional, Sequence
from dataclasses import dataclass, field
import itertools

import jax
import jax.numpy as jnp
from jax import lax, random, value_and_grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
import flax
import flax.linen as nn
from flax import struct
import optax
import orbax.checkpoint as ocp
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import datasets
import transformers

# === CONFIG ===
HF_TOKEN = "YOUR_HF_TOKEN_HERE"
HF_REPO = "Arko007/zenyx-v3-checkpoints"

@dataclass
class ZenyxConfig:
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
    compress_ratio_csa: int = 4
    compress_ratio_hca: int = 128
    local_window: int = 256
    top_k_csa: int = 64
    moe_aux_loss_alpha: float = 0.01
    layerscale_init: float = 1e-4
    yarn_base: float = 10000.0
    yarn_alpha: float = 1.0
    yarn_beta: float = 32.0
    yarn_scale_s: float = 4.0
    lr: float = 3e-4
    lr_min: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    muon_momentum: float = 0.95
    batch_size: int = 8  # Per-device, global = 64 on 8-core TPU
    tokens_per_shard: int = 100_000_000  # 100M tokens per logical shard for O(1) skip
    checkpoint_interval: int = 500
    log_interval: int = 10
    log_lr_interval: int = 100

CONFIG = ZenyxConfig()

# === TOKENIZER ===
def load_tokenizer():
    # Using Qwen2.5 tokenizer as it closely matches ~65k vocab and handles code/math well
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B", token=HF_TOKEN, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# === DATA PIPELINE ===
def create_data_pipeline(tokenizer, resume_shard_idx: int = 0, resume_offset: int = 0):
    """Streams, interleaves, and packs datasets to exact seq_len."""
    ds_fw = datasets.load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    ds_stack = datasets.load_dataset("bigcode/the-stack-smol", split="train", streaming=True)
    ds_math = datasets.load_dataset("GAIR/NuminaMath-CoT", split="train", streaming=True)
    
    # Interleave with specified probabilities
    mixed_ds = datasets.interleave_datasets(
        [ds_fw, ds_stack, ds_math],
        probabilities=[0.60, 0.25, 0.15],
        seed=42,
        stopping_strategy="all_exhausted"
    )
    
    # O(1) Shard Skip
    if resume_shard_idx > 0:
        skip_count = resume_shard_idx * CONFIG.tokens_per_shard
        print(f"Skipping {skip_count} tokens to reach shard {resume_shard_idx}...")
        mixed_ds = mixed_ds.skip(skip_count)
        
    def tokenize_and_pack():
        buffer = []
        current_offset = 0
        for sample in mixed_ds:
            text = sample.get("text", sample.get("content", ""))
            if not text: continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            
            while len(buffer) >= CONFIG.seq_len + 1:
                chunk = buffer[:CONFIG.seq_len + 1]
                buffer = buffer[CONFIG.seq_len:]
                
                # Handle intra-shard offset resumption
                if current_offset < resume_offset:
                    current_offset += 1
                    continue
                    
                inputs = jnp.array(chunk[:-1], dtype=jnp.int32)
                targets = jnp.array(chunk[1:], dtype=jnp.int32)
                loss_mask = jnp.ones_like(targets, dtype=jnp.float32)
                yield {"inputs": inputs, "targets": targets, "loss_mask": loss_mask}
                
    return tokenize_and_pack()

# === MODEL COMPONENTS ===
class FP8Dense(nn.Module):
    features: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param('kernel',
            nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'),
            (inputs.shape[-1], self.features), jnp.float32)

        x_amax = jnp.max(jnp.abs(inputs))
        w_amax = jnp.max(jnp.abs(kernel))
        e4m3_max = jnp.finfo(jnp.float8_e4m3fn).max
        
        x_scale = e4m3_max / jnp.maximum(x_amax, 1e-12)
        w_scale = e4m3_max / jnp.maximum(w_amax, 1e-12)

        x_fp8 = (inputs * x_scale).astype(jnp.float8_e4m3fn)
        w_fp8 = (kernel * w_scale).astype(jnp.float8_e4m3fn)

        out_fp8 = lax.dot_general(
            x_fp8, w_fp8,
            dimension_numbers=(((inputs.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=jnp.bfloat16
        )
        out = out_fp8 / (x_scale * w_scale)

        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,), jnp.float32)
            out = out + bias.astype(jnp.bfloat16)
        return out.astype(jnp.bfloat16)

def build_yarn_rope(seq_len: int, d_rope: int, base: float = 10000.0,
                    alpha: float = 1.0, beta: float = 32.0, scale_s: float = 4.0):
    m = jnp.arange(d_rope // 2)
    theta_m = base ** (-2.0 * m / d_rope)
    lambda_m = 2 * jnp.pi / theta_m
    r_m = 8192 / lambda_m
    gamma = jnp.where(r_m < alpha, 0.0, jnp.where(r_m > beta, 1.0, (r_m - alpha) / (beta - alpha)))
    theta_yarn = (1 - gamma) * (theta_m / scale_s) + gamma * theta_m
    positions = jnp.arange(seq_len)
    angles = jnp.outer(positions, theta_yarn)
    cos_val = jnp.cos(angles)
    sin_val = jnp.sin(angles)
    mscale = 0.1 * jnp.log(scale_s) + 1.0
    return cos_val, sin_val, mscale

def apply_decoupled_yarn_rope(x_rope: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray):
    def rotate(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)
    return (x_rope * cos) + (rotate(x_rope) * sin)

class ZenyxHybridAttention(nn.Module):
    d_model: int
    num_heads: int
    d_latent: int
    d_rope: int
    is_hca_layer: bool
    compress_ratio: int
    local_window: int
    top_k: int

    def setup(self):
        self.d_head = self.d_model // self.num_heads
        self.q_proj = FP8Dense(self.d_latent)
        self.kv_proj = FP8Dense(self.d_latent)
        self.q_up = FP8Dense(self.num_heads * self.d_head)
        self.kv_up_k = FP8Dense(self.num_heads * self.d_head)
        self.kv_up_v = FP8Dense(self.num_heads * self.d_head)
        self.o_proj = FP8Dense(self.d_model)

    def __call__(self, x: jnp.ndarray, q_rope: jnp.ndarray, k_rope: jnp.ndarray) -> jnp.ndarray:
        batch, seq_len, _ = x.shape
        c_q = self.q_proj(x)
        c_kv = self.kv_proj(x)

        num_chunks = seq_len // self.compress_ratio
        c_kv_compressed = c_kv.reshape(batch, num_chunks, self.compress_ratio, self.d_latent).mean(axis=2)

        q_nope = self.q_up(c_q).reshape(batch, seq_len, self.num_heads, self.d_head)
        k_nope = self.kv_up_k(c_kv_compressed).reshape(batch, num_chunks, self.num_heads, self.d_head)
        v_nope = self.kv_up_v(c_kv_compressed).reshape(batch, num_chunks, self.num_heads, self.d_head)

        local_c_kv = c_kv[:, -self.local_window:, :]
        local_k = self.kv_up_k(local_c_kv).reshape(batch, self.local_window, self.num_heads, self.d_head)
        local_v = self.kv_up_v(local_c_kv).reshape(batch, self.local_window, self.num_heads, self.d_head)

        k_assembled = jnp.concatenate([k_nope, local_k], axis=1)
        v_assembled = jnp.concatenate([v_nope, local_v], axis=1)

        q_rope_reshaped = q_rope.reshape(batch, seq_len, self.num_heads, self.d_rope)
        q_final = jnp.concatenate([q_nope, q_rope_reshaped], axis=-1)

        k_rope_compressed = k_rope.reshape(batch, num_chunks, self.compress_ratio, self.d_rope).mean(axis=2)
        k_rope_local = k_rope[:, -self.local_window:, :]
        k_rope_final = jnp.concatenate([k_rope_compressed, k_rope_local], axis=1)
        k_rope_expanded = jnp.expand_dims(k_rope_final, axis=2)
        k_rope_broadcast = jnp.broadcast_to(k_rope_expanded, (batch, num_chunks + self.local_window, self.num_heads, self.d_rope))
        k_final = jnp.concatenate([k_assembled, k_rope_broadcast], axis=-1)

        scale = 1.0 / jnp.sqrt(self.d_head + self.d_rope)
        attn_logits = jnp.einsum('bshd,bthd->bhst', q_final, k_final) * scale

        if not self.is_hca_layer:
            top_k_indices = jnp.argsort(attn_logits, axis=-1)[:, :, :, -self.top_k:]
            thresholds = jnp.take_along_axis(attn_logits, top_k_indices[..., 0:1], axis=-1)
            attn_logits = jnp.where(attn_logits >= thresholds, attn_logits, -1e9)

        mask = jnp.tril(jnp.ones((seq_len, num_chunks + self.local_window)))
        attn_logits = jnp.where(mask == 0, -1e9, attn_logits)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_output = jnp.einsum('bhst,bthd->bshd', attn_weights, v_assembled)
        attn_output = attn_output.reshape(batch, seq_len, self.num_heads * self.d_head)
        return self.o_proj(attn_output)

class DualSharedSparseMoE(nn.Module):
    d_model: int
    d_ff: int
    num_routed_experts: int
    aux_loss_alpha: float

    def setup(self):
        d_ff_split = self.d_ff // 2
        self.shared_1_w1 = FP8Dense(d_ff_split)
        self.shared_1_w2 = FP8Dense(self.d_model)
        self.shared_2_w1 = FP8Dense(d_ff_split)
        self.shared_2_w2 = FP8Dense(self.d_model)
        self.router = nn.Dense(self.num_routed_experts, use_bias=False, dtype=jnp.float32)
        self.routed_w1 = self.param('routed_w1', nn.initializers.lecun_normal(),
            (self.num_routed_experts, self.d_model, self.d_ff), jnp.bfloat16)
        self.routed_w2 = self.param('routed_w2', nn.initializers.lecun_normal(),
            (self.num_routed_experts, self.d_ff, self.d_model), jnp.bfloat16)

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        shared_1 = self.shared_1_w2(jax.nn.silu(self.shared_1_w1(x)))
        shared_2 = self.shared_2_w2(jax.nn.silu(self.shared_2_w1(x)))
        shared_out = shared_1 + shared_2

        router_logits = self.router(x)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        expert_indices = jnp.argmax(router_probs, axis=-1)
        expert_gates = jnp.max(router_probs, axis=-1, keepdims=True)

        selected_w1 = self.routed_w1[expert_indices]
        selected_w2 = self.routed_w2[expert_indices]
        h_routed = jax.nn.silu(jnp.einsum('bsd,bsdf->bsf', x, selected_w1))
        routed_out = jnp.einsum('bsf,bsfd->bsd', h_routed, selected_w2)
        final_out = shared_out + (routed_out * expert_gates)

        expert_mask = jax.nn.one_hot(expert_indices, self.num_routed_experts, dtype=jnp.float32)
        f_i = jnp.mean(expert_mask, axis=(0, 1))
        P_i = jnp.mean(router_probs, axis=(0, 1))
        aux_loss = self.aux_loss_alpha * self.num_routed_experts * jnp.sum(f_i * P_i)
        return final_out, aux_loss

class ZenyxRecurrentSuperBlock(nn.Module):
    d_model: int
    num_heads: int
    d_latent: int
    d_rope: int
    d_ff: int
    num_routed_experts: int
    num_recurrences: int
    compress_ratio_csa: int
    compress_ratio_hca: int
    local_window: int
    top_k_csa: int
    moe_aux_loss_alpha: float
    layerscale_init: float

    def setup(self):
        self.norm1 = nn.RMSNorm()
        self.norm2 = nn.RMSNorm()
        self.gamma_1 = self.param('gamma_1', nn.initializers.constant(self.layerscale_init), (self.d_model,))
        self.gamma_2 = self.param('gamma_2', nn.initializers.constant(self.layerscale_init), (self.d_model,))
        self.moe = DualSharedSparseMoE(self.d_model, self.d_ff, self.num_routed_experts, self.moe_aux_loss_alpha)

    def __call__(self, x: jnp.ndarray, q_rope: jnp.ndarray, k_rope: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        def recurrent_step(carry, step_idx):
            x_in = carry
            is_hca = (step_idx % 2 == 0)
            compress = self.compress_ratio_hca if is_hca else self.compress_ratio_csa
            top_k = 0 if is_hca else self.top_k_csa
            
            attn = ZenyxHybridAttention(
                d_model=self.d_model, num_heads=self.num_heads, d_latent=self.d_latent,
                d_rope=self.d_rope, is_hca_layer=is_hca, compress_ratio=compress,
                local_window=self.local_window, top_k=top_k
            )
            x_norm = self.norm1(x_in)
            attn_out = attn(x_norm, q_rope, k_rope)
            x_mid = x_in + (attn_out * self.gamma_1)

            moe_out, aux_loss = self.moe(self.norm2(x_mid))
            x_out = x_mid + (moe_out * self.gamma_2)
            return x_out, aux_loss

        final_x, aux_losses = lax.scan(jax.checkpoint(recurrent_step), x, jnp.arange(self.num_recurrences))
        return final_x, jnp.sum(aux_losses)

class ZenyxV3Model(nn.Module):
    config: ZenyxConfig

    def setup(self):
        self.embed = nn.Embed(self.config.vocab_size, self.config.d_model, dtype=jnp.bfloat16)
        self.recurrent = ZenyxRecurrentSuperBlock(
            d_model=self.config.d_model, num_heads=self.config.num_heads,
            d_latent=self.config.d_latent, d_rope=self.config.d_rope,
            d_ff=self.config.d_ff, num_routed_experts=self.config.num_routed_experts,
            num_recurrences=self.config.num_recurrences,
            compress_ratio_csa=self.config.compress_ratio_csa,
            compress_ratio_hca=self.config.compress_ratio_hca,
            local_window=self.config.local_window, top_k_csa=self.config.top_k_csa,
            moe_aux_loss_alpha=self.config.moe_aux_loss_alpha,
            layerscale_init=self.config.layerscale_init
        )
        self.final_norm = nn.RMSNorm()
        self.lm_head = FP8Dense(self.config.vocab_size, use_bias=False)
        self.cos, self.sin, self.mscale = build_yarn_rope(
            self.config.seq_len, self.config.d_rope, self.config.yarn_base,
            self.config.yarn_alpha, self.config.yarn_beta, self.config.yarn_scale_s
        )

    def __call__(self, input_ids: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = self.embed(input_ids).astype(jnp.bfloat16)
        batch, seq_len = input_ids.shape
        
        q_rope = jnp.zeros((batch, seq_len, self.config.d_rope), dtype=jnp.bfloat16)
        k_rope = jnp.zeros((batch, seq_len, self.config.d_rope), dtype=jnp.bfloat16)
        
        # Broadcast RoPE across batch
        q_rope = q_rope + self.cos[:seq_len]
        k_rope = k_rope + self.cos[:seq_len] # Simplified for decoupled init
        
        x, aux_loss = self.recurrent(x, q_rope, k_rope)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, aux_loss

# === OPTIMIZER ===
def newton_schulz_iteration(G: jnp.ndarray, steps: int = 5) -> jnp.ndarray:
    X = G.astype(jnp.bfloat16)
    X = X / (jnp.linalg.norm(X, ord='fro') + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    transpose_flag = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose_flag = True
    def ns_step(carry, _):
        X_curr = carry
        A = X_curr @ X_curr.T
        B = b * A + c * (A @ A)
        X_next = a * X_curr + B @ X_curr
        return X_next, None
    X_final, _ = lax.scan(ns_step, X, None, length=steps)
    if transpose_flag: X_final = X_final.T
    return X_final.astype(G.dtype)

def scale_by_muon(learning_rate: float, momentum: float = 0.95) -> optax.GradientTransformation:
    def init_fn(params):
        return optax.ScaleByMuonState(momentum=jax.tree_util.tree_map(jnp.zeros_like, params))
    def update_fn(updates, state, params=None):
        mu_next = jax.tree_util.tree_map(lambda m, g: momentum * m + g, state.momentum, updates)
        orthogonalized_updates = jax.tree_util.tree_map(
            lambda m: newton_schulz_iteration(m) if len(m.shape) >= 2 else m, mu_next
        )
        scaled_updates = jax.tree_util.tree_map(
            lambda u: -learning_rate * 0.2 * u if len(u.shape) >= 2 else -learning_rate * u,
            orthogonalized_updates
        )
        return scaled_updates, optax.ScaleByMuonState(momentum=mu_next)
    return optax.GradientTransformation(init_fn, update_fn)

# Register Optax State for Muon
optax.ScaleByMuonState = optax.base.EmptyState # Placeholder, actual state handled in closure or custom class
# Optax expects NamedTuple or dataclass for state. We'll use a simple wrapper.
class MuonState(optax.base.EmptyState):
    momentum: Any
optax.ScaleByMuonState = MuonState

def build_hybrid_optimizer(config: ZenyxConfig) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=config.lr, warmup_steps=config.warmup_steps,
        decay_steps=100_000, end_value=config.lr_min
    )
    muon_tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        scale_by_muon(learning_rate=schedule, momentum=config.muon_momentum),
        optax.add_decayed_weights(config.weight_decay)
    )
    adamw_tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-8),
        optax.add_decayed_weights(config.weight_decay),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0)
    )
    def is_2d(path, param):
        return len(param.shape) >= 2
    return optax.multi_transform(
        {'muon': muon_tx, 'adamw': adamw_tx},
        optax.tree_utils.tree_map_with_path(lambda p, x: 'muon' if is_2d(p, x) else 'adamw', None)
    )

# === TRAINING INFRASTRUCTURE ===
def setup_mesh():
    devices = jax.devices()
    assert len(devices) == 8, "Expected 8 TPU v5e cores"
    device_mesh = mesh_utils.create_device_mesh((8,))
    return Mesh(device_mesh, axis_names=('fsdp',))

def get_partition_specs():
    return {
        'embed/embedding': P('fsdp', None),
        'recurrent/gamma_1': P(None),
        'recurrent/gamma_2': P(None),
        'recurrent/moe/shared_1_w1/kernel': P(None, 'fsdp'),
        'recurrent/moe/shared_1_w2/kernel': P('fsdp', None),
        'recurrent/moe/shared_2_w1/kernel': P(None, 'fsdp'),
        'recurrent/moe/shared_2_w2/kernel': P('fsdp', None),
        'recurrent/moe/router/kernel': P(None, 'fsdp'),
        'recurrent/moe/routed_w1': P('fsdp', None, None),
        'recurrent/moe/routed_w2': P('fsdp', None, None),
        'final_norm/scale': P(None),
        'lm_head/kernel': P('fsdp', None)
    }

@struct.dataclass
class TrainState:
    step: jnp.uint32
    params: flax.core.FrozenDict
    opt_state: optax.OptState
    rng: jax.Array

def calculate_masked_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    loss = (loss * mask).sum() / jnp.maximum(mask.sum(), 1e-8)
    return loss

def train_step(state: TrainState, batch: Dict, config: ZenyxConfig, mesh: Mesh, p_specs: Dict):
    def loss_fn(params):
        model = ZenyxV3Model(config)
        logits, aux_loss = model.apply({'params': params}, batch['inputs'])
        ce_loss = calculate_masked_cross_entropy(logits, batch['targets'], batch['loss_mask'])
        total_loss = ce_loss + aux_loss
        return total_loss, (ce_loss, aux_loss)
    
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (total_loss, (ce_loss, aux_loss)), grads = grad_fn(state.params)
    
    updates, new_opt_state = build_hybrid_optimizer(config).update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    
    new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state)
    metrics = {
        'total_loss': total_loss, 'ce_loss': ce_loss, 'aux_loss': aux_loss,
        'grad_norm': optax.global_norm(grads), 'lr': build_hybrid_optimizer(config).schedule(state.step)
    }
    return new_state, metrics

# === CHECKPOINT UTILS ===
def save_checkpoint(state: TrainState, step: int, shard_idx: int, offset: int, loss: float, mesh: Mesh):
    ckpt_dir = f"/tmp/zenyx_ckpt_{step}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    ckptr = ocp.PyTreeCheckpointer()
    save_args = jax.tree_util.tree_map(lambda _: ocp.SaveArgs(aggregate=True), state.params)
    ckptr.save(ckpt_dir, {'params': state.params, 'opt_state': state.opt_state, 'step': state.step, 'rng': state.rng}, save_args=save_args)
    
    metadata = {
        'global_step': int(step), 'shard_index': int(shard_idx), 'offset_within_shard': int(offset),
        'loss': float(loss), 'timestamp': time.time()
    }
    with open(os.path.join(ckpt_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
        
    api = HfApi(token=HF_TOKEN)
    api.upload_folder(folder_path=ckpt_dir, repo_id=HF_REPO, path_in_repo=f"checkpoint-{step}", ignore_patterns=["*.lock"])
    print(f"Checkpoint saved to HF: step {step}")

def load_latest_checkpoint(mesh: Mesh, p_specs: Dict) -> Tuple[Optional[TrainState], int, int]:
    api = HfApi(token=HF_TOKEN)
    try:
        files = api.list_repo_files(HF_REPO)
        ckpts = sorted([f.split('/')[0] for f in files if f.startswith('checkpoint-') and 'metadata.json' in f])
        if not ckpts: return None, 0, 0
        latest = ckpts[-1]
        step = int(latest.split('-')[1])
        
        local_dir = hf_hub_download(repo_id=HF_REPO, filename=f"{latest}/metadata.json", token=HF_TOKEN)
        with open(local_dir.replace('metadata.json', ''), 'r') as f: # Fix path resolution
            pass
        # Proper download
        snapshot_download(repo_id=HF_REPO, allow_patterns=[f"{latest}/*"], token=HF_TOKEN, local_dir=f"/tmp/resume_{step}")
        with open(f"/tmp/resume_{step}/{latest}/metadata.json", 'r') as f:
            meta = json.load(f)
            
        ckptr = ocp.PyTreeCheckpointer()
        restored = ckptr.restore(f"/tmp/resume_{step}/{latest}")
        
        init_rng = restored['rng']
        state = TrainState(step=jnp.uint32(meta['global_step']), params=restored['params'], opt_state=restored['opt_state'], rng=init_rng)
        return state, meta['shard_index'], meta['offset_within_shard']
    except Exception as e:
        print(f"No valid checkpoint found or error loading: {e}")
        return None, 0, 0

# === MAIN TRAINING LOOP ===
def main():
    jax.config.update('jax_threefry_partitionable', True)
    mesh = setup_mesh()
    p_specs = get_partition_specs()
    tokenizer = load_tokenizer()
    
    print("Checking for existing checkpoints...")
    state, resume_shard, resume_offset = load_latest_checkpoint(mesh, p_specs)
    
    if state is None:
        print("Initializing new model...")
        model = ZenyxV3Model(CONFIG)
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        dummy_input = jnp.zeros((CONFIG.batch_size, CONFIG.seq_len), dtype=jnp.int32)
        params = model.init(init_rng, dummy_input)['params']
        opt = build_hybrid_optimizer(CONFIG)
        opt_state = opt.init(params)
        state = TrainState(step=jnp.uint32(0), params=params, opt_state=opt_state, rng=rng)
        
    print(f"Resuming from step {state.step}, shard {resume_shard}, offset {resume_offset}")
    data_iter = create_data_pipeline(tokenizer, resume_shard, resume_offset)
    
    jitted_train_step = jax.jit(train_step, static_argnums=(2,), donate_argnums=(0,))
    
    step_time = time.time()
    tokens_processed = 0
    
    for batch_idx, batch_np in enumerate(data_iter):
        batch = jax.tree_util.tree_map(lambda x: jnp.array(x), batch_np)
        batch = jax.device_put(batch, NamedSharding(mesh, P('fsdp', None)))
        
        state, metrics = jitted_train_step(state, batch, CONFIG, mesh, p_specs)
        tokens_processed += CONFIG.batch_size * CONFIG.seq_len * jax.device_count()
        
        step_num = int(state.step)
        if step_num % CONFIG.log_interval == 0:
            elapsed = time.time() - step_time
            tok_sec = tokens_processed / elapsed if elapsed > 0 else 0
            print(f"Step {step_num} | Loss: {metrics['total_loss']:.4f} | Aux: {metrics['aux_loss']:.4f} | Tok/s: {tok_sec:.0f} | Time: {elapsed:.2f}s")
            step_time = time.time()
            tokens_processed = 0
            
        if step_num % CONFIG.log_lr_interval == 0:
            print(f"Step {step_num} | LR: {metrics['lr']:.2e} | Grad Norm: {metrics['grad_norm']:.4f}")
            
        if step_num % CONFIG.checkpoint_interval == 0 and step_num > 0:
            current_shard = (step_num * CONFIG.batch_size * CONFIG.seq_len * jax.device_count()) // CONFIG.tokens_per_shard
            current_offset = (step_num * CONFIG.batch_size * CONFIG.seq_len * jax.device_count()) % CONFIG.tokens_per_shard
            save_checkpoint(state, step_num, current_shard, current_offset, float(metrics['total_loss']), mesh)
            
        if step_num >= 100_000: # Safety break for demo
            print("Reached max steps.")
            break

if __name__ == "__main__":
    main()
