# zenyx_v3_train.py
# === IMPORTS ===
import os
import time
import json
import numpy as np
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

import flax.linen as nn
from flax.training import train_state
import optax

import orbax.checkpoint as ocp
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, interleave_datasets
import tiktoken

# === CONFIG ===
class Config:
    hf_token = "YOUR_HF_TOKEN_HERE"
    hf_repo = "Arko007/zenyx-v3-checkpoints"
    d_model = 1536
    num_heads = 12
    d_latent = 256
    d_rope = 64
    d_ff = 4096
    num_shared_experts = 2
    num_routed_experts = 64
    num_recurrences = 12
    seq_len = 32768
    vocab_size = 65536
    batch_size = 2
    lr = 3e-4
    min_lr = 3e-5
    warmup_steps = 2000
    weight_decay = 0.05
    grad_clip = 1.0
    tokens_per_shard = 10_000_000_000

cfg = Config()

# === TOKENIZER ===
enc = tiktoken.get_encoding("cl100k_base")
def tokenize(text: str):
    return [t % cfg.vocab_size for t in enc.encode(text)]

# === DATA PIPELINE ===
def get_dataset(shard_index=0, offset=0):
    fineweb = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    stack = load_dataset("bigcode/the-stack-dedup", split="train", streaming=True)
    numina = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    ds = interleave_datasets([fineweb, stack, numina], probabilities=[0.6, 0.25, 0.15], seed=42)
    ds = ds.skip(shard_index)
    return ds

def data_generator(ds):
    buffer = []
    for ex in ds:
        text = ex.get("text") or ex.get("content") or ""
        buffer.extend(tokenize(text))
        while len(buffer) >= cfg.seq_len + 1:
            chunk = buffer[:cfg.seq_len + 1]
            buffer = buffer[cfg.seq_len + 1:]
            yield {
                "input_ids": np.array(chunk[:-1], dtype=np.int32),
                "targets": np.array(chunk[1:], dtype=np.int32)
            }

# === MODEL COMPONENTS ===
class FP8Dense(nn.Module):
    features: int
    use_bias: bool = False
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param('kernel', nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'),
                            (inputs.shape[-1], self.features), jnp.float32)
        x_amax = jnp.max(jnp.abs(inputs))
        w_amax = jnp.max(jnp.abs(kernel))
        e4m3_max = jnp.finfo(jnp.float8_e4m3fn).max
        x_scale = e4m3_max / jnp.maximum(x_amax, 1e-12)
        w_scale = e4m3_max / jnp.maximum(w_amax, 1e-12)
        x_fp8 = (inputs * x_scale).astype(jnp.float8_e4m3fn)
        w_fp8 = (kernel * w_scale).astype(jnp.float8_e4m3fn)
        out_fp8 = jax.lax.dot_general(x_fp8, w_fp8,
            dimension_numbers=(((inputs.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=jnp.bfloat16)
        out = out_fp8 / (x_scale * w_scale)
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,), jnp.float32)
            out = out + bias.astype(jnp.bfloat16)
        return out.astype(jnp.bfloat16)

def build_yarn_rope(seq_len, d_rope, base=10000.0, alpha=1.0, beta=32.0, scale_s=4.0):
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

def apply_decoupled_yarn_rope(x_rope, cos, sin):
    x1, x2 = jnp.split(x_rope, 2, axis=-1)
    x_rot = jnp.concatenate([-x2, x1], axis=-1)
    return (x_rope * cos) + (x_rot * sin)

class ZenyxHybridAttention(nn.Module):
    d_model: int
    num_heads: int
    d_latent: int
    d_rope: int
    is_hca_layer: bool
    def setup(self):
        self.d_head = self.d_model // self.num_heads
        self.q_proj = FP8Dense(self.d_latent)
        self.kv_proj = FP8Dense(self.d_latent)
        self.q_up = FP8Dense(self.num_heads * self.d_head)
        self.kv_up_k = FP8Dense(self.num_heads * self.d_head)
        self.kv_up_v = FP8Dense(self.num_heads * self.d_head)
        self.o_proj = FP8Dense(self.d_model)
        self.compress_ratio = 128 if self.is_hca_layer else 4
        self.local_window = 256
    def __call__(self, x, q_rope, k_rope):
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
        q_rope_r = q_rope.reshape(batch, seq_len, self.num_heads, self.d_rope)
        q_final = jnp.concatenate([q_nope, q_rope_r], axis=-1)
        k_rope_c = k_rope.reshape(batch, num_chunks, self.compress_ratio, self.d_rope).mean(axis=2)
        k_rope_l = k_rope[:, -self.local_window:, :]
        k_rope_f = jnp.concatenate([k_rope_c, k_rope_l], axis=1)
        k_rope_b = jnp.broadcast_to(k_rope_f[:, :, None, :], (batch, num_chunks + self.local_window, self.num_heads, self.d_rope))
        k_final = jnp.concatenate([k_assembled, k_rope_b], axis=-1)
        scale = 1.0 / jnp.sqrt(self.d_head + self.d_rope)
        attn_logits = jnp.einsum('bshd,bthd->bhst', q_final, k_final) * scale
        if not self.is_hca_layer:
            top_k = 64
            top_k_idx = jnp.argsort(attn_logits, axis=-1)[..., -top_k:]
            thresh = jnp.take_along_axis(attn_logits, top_k_idx[..., 0:1], axis=-1)
            attn_logits = jnp.where(attn_logits >= thresh, attn_logits, -1e9)
        mask = jnp.tril(jnp.ones((seq_len, num_chunks + self.local_window)))
        attn_logits = jnp.where(mask == 0, -1e9, attn_logits)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_out = jnp.einsum('bhst,bthd->bshd', attn_weights, v_assembled)
        attn_out = attn_out.reshape(batch, seq_len, self.num_heads * self.d_head)
        return self.o_proj(attn_out)

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
        self.routed_w1 = self.param('routed_w1', nn.initializers.lecun_normal(),
            (self.num_routed_experts, self.d_model, self.d_ff), jnp.bfloat16)
        self.routed_w2 = self.param('routed_w2', nn.initializers.lecun_normal(),
            (self.num_routed_experts, self.d_ff, self.d_model), jnp.bfloat16)
    def __call__(self, x):
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
        f_i = jnp.mean(expert_mask, axis=(0,1))
        P_i = jnp.mean(router_probs, axis=(0,1))
        aux_loss = 0.01 * self.num_routed_experts * jnp.sum(f_i * P_i)
        return final_out, aux_loss

class ZenyxRecurrentSuperBlock(nn.Module):
    d_model: int
    num_recurrences: int
    def setup(self):
        self.norm1 = nn.RMSNorm()
        self.norm2 = nn.RMSNorm()
        self.gamma_1 = self.param('gamma_1', nn.initializers.constant(1e-4), (self.d_model,))
        self.gamma_2 = self.param('gamma_2', nn.initializers.constant(1e-4), (self.d_model,))
    def __call__(self, x, q_rope, k_rope):
        def step(carry, idx):
            x_in = carry
            is_hca = (idx % 2 == 0)
            attn = ZenyxHybridAttention(self.d_model, cfg.num_heads, cfg.d_latent, cfg.d_rope, is_hca)
            moe = DualSharedSparseMoE(self.d_model, cfg.d_ff, cfg.num_routed_experts)
            x_norm = self.norm1(x_in)
            attn_out = attn(x_norm, q_rope, k_rope)
            x_mid = x_in + attn_out * self.gamma_1
            moe_out, aux = moe(self.norm2(x_mid))
            x_out = x_mid + moe_out * self.gamma_2
            return x_out, aux
        step_remat = nn.remat(step)
        final_x, aux_losses = jax.lax.scan(step_remat, x, jnp.arange(self.num_recurrences))
        return final_x, jnp.sum(aux_losses)

class ZenyxV3Model(nn.Module):
    def setup(self):
        self.embed = nn.Embed(cfg.vocab_size, cfg.d_model)
        self.block = ZenyxRecurrentSuperBlock(cfg.d_model, cfg.num_recurrences)
        self.norm_f = nn.RMSNorm()
        self.lm_head = FP8Dense(cfg.vocab_size)
    def __call__(self, input_ids):
        x = self.embed(input_ids)
        cos, sin, mscale = build_yarn_rope(x.shape[1], cfg.d_rope)
        q_rope = apply_decoupled_yarn_rope(jnp.zeros((x.shape[0], x.shape[1], cfg.d_rope)), cos, sin)
        k_rope = q_rope
        x, aux_loss = self.block(x, q_rope, k_rope)
        x = self.norm_f(x)
        logits = self.lm_head(x) * mscale
        return logits, aux_loss

# === OPTIMIZER ===
def newton_schulz_iteration(G: jnp.ndarray, steps: int = 5) -> jnp.ndarray:
    X = G.astype(jnp.bfloat16)
    X = X / (jnp.linalg.norm(X, ord='fro') + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True
    def ns_step(carry, _):
        A = carry @ carry.T
        B = b * A + c * (A @ A)
        return a * carry + B @ carry, None
    X_final, _ = jax.lax.scan(ns_step, X, None, length=steps)
    if transpose:
        X_final = X_final.T
    return X_final.astype(G.dtype)

def scale_by_muon(lr: float, momentum: float = 0.95):
    def init_fn(params):
        return optax.TraceState(trace=jax.tree_util.tree_map(jnp.zeros_like, params))
    def update_fn(updates, state, params=None):
        mu = jax.tree_util.tree_map(lambda m, g: momentum * m + g, state.trace, updates)
        ortho = jax.tree_util.tree_map(lambda m: newton_schulz_iteration(m) if m.ndim >= 2 else m, mu)
        scaled = jax.tree_util.tree_map(lambda u: -lr * 0.2 * u if u.ndim >= 2 else -lr * u, ortho)
        return scaled, optax.TraceState(trace=mu)
    return optax.GradientTransformation(init_fn, update_fn)

def build_hybrid_optimizer():
    schedule = optax.warmup_cosine_decay_schedule(cfg.lr, cfg.min_lr, cfg.warmup_steps, 1_000_000)
    muon_tx = optax.chain(scale_by_muon(1.0), optax.add_decayed_weights(cfg.weight_decay), optax.scale_by_schedule(lambda step: schedule(step)))
    adamw_tx = optax.chain(optax.scale_by_adam(0.9, 0.95), optax.add_decayed_weights(cfg.weight_decay), optax.scale_by_schedule(lambda step: -schedule(step)))
    def label_fn(params):
        return jax.tree_util.tree_map_with_path(lambda p, _: 'muon' if _.ndim >= 2 else 'adamw', params)
    return optax.multi_transform({'muon': muon_tx, 'adamw': adamw_tx}, label_fn)

# === TRAINING INFRASTRUCTURE ===
def setup_mesh():
    devices = jax.devices()
    mesh = Mesh(mesh_utils.create_device_mesh((len(devices),)), ('fsdp',))
    return mesh

def calculate_loss(logits, targets):
    logits = logits.astype(jnp.float32)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.mean(loss)

@partial(jax.jit, static_argnums=(3,))
def train_step(state, batch, rng, model):
    def loss_fn(params):
        logits, aux = model.apply(params, batch['input_ids'])
        loss = calculate_loss(logits, batch['targets'])
        return loss + aux, (loss, aux)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (loss, aux)), grads = grad_fn(state.params)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -cfg.grad_clip, cfg.grad_clip), grads)
    state = state.apply_gradients(grads=grads)
    return state, loss, aux

# === CHECKPOINT UTILS ===
def save_checkpoint(state, step, shard_index, offset):
    ckptr = ocp.PyTreeCheckpointer()
    path = f"/tmp/checkpoint-{step}"
    ckptr.save(path, {'params': state.params, 'opt': state.opt_state, 'step': step})
    metadata = {"global_step": int(step), "shard_index": int(shard_index), "offset_within_shard": int(offset), "timestamp": time.time()}
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    api = HfApi(token=cfg.hf_token)
    api.upload_folder(repo_id=cfg.hf_repo, folder_path=path, path_in_repo=f"checkpoint-{step}", repo_type="model")
    print(f"Checkpoint saved to HF: step {step}")

def load_latest_checkpoint():
    api = HfApi(token=cfg.hf_token)
    try:
        files = api.list_repo_files(cfg.hf_repo, repo_type="model")
        steps = [int(f.split('-')[1].split('/')[0]) for f in files if f.startswith("checkpoint-")]
        if not steps:
            return 0, 0, 0
        latest = max(steps)
        meta_path = hf_hub_download(cfg.hf_repo, f"checkpoint-{latest}/metadata.json", token=cfg.hf_token)
        meta = json.load(open(meta_path))
        print(f"Resuming from step {meta['global_step']}, shard {meta['shard_index']}, offset {meta['offset_within_shard']}")
        return meta['global_step'], meta['shard_index'], meta['offset_within_shard']
    except:
        return 0, 0, 0

# === MAIN TRAINING LOOP ===
def main():
    mesh = setup_mesh()
    model = ZenyxV3Model()
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, cfg.seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy)
    tx = build_hybrid_optimizer()
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    global_step, shard_idx, offset = load_latest_checkpoint()
    tokens_consumed = global_step * cfg.batch_size * cfg.seq_len
    shard_index = tokens_consumed // cfg.tokens_per_shard
    offset_within_shard = tokens_consumed % cfg.tokens_per_shard
    ds = get_dataset(shard_index, offset_within_shard)
    gen = data_generator(ds)
    step = global_step
    start_time = time.time()
    for batch in gen:
        step += 1
        batch_jax = {k: jnp.array(v)[None, :] for k, v in batch.items()}
        state, loss, aux = train_step(state, batch_jax, rng, model)
        if step % 10 == 0:
            tokens_sec = cfg.batch_size * cfg.seq_len * 10 / (time.time() - start_time)
            print(f"step {step}, loss {float(loss):.4f}, aux_loss {float(aux):.4f}, tokens/sec {tokens_sec:.0f}")
            start_time = time.time()
        if step % 100 == 0:
            print(f"lr {float(cfg.lr)}, grad norm clipped at {cfg.grad_clip}")
        if step % 500 == 0:
            save_checkpoint(state, step, shard_index, offset_within_shard)
        if step > 1_000_000:
            break

if __name__ == "__main__":
    main()
