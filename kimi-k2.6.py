# zenyx_v3_train.py
# === IMPORTS ===
import os
import shutil
import json
import time
import datetime
from functools import partial
from typing import Any, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
import flax.linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import optax
import orbax.checkpoint
from datasets import load_dataset, interleave_datasets
from huggingface_hub import HfApi, hf_hub_download, create_repo
import sentencepiece as spm  # type: ignore

# === CONFIG ===
HF_TOKEN = "YOUR_HF_TOKEN_HERE"
HF_REPO = "Arko007/zenyx-v3-checkpoints"
TOKENIZER_REPO = "Arko007/zenyx-v3-checkpoints"
TOKENIZER_FILE = "tokenizer.model"

# Architecture
D_MODEL = 1536
NUM_HEADS = 12
D_LATENT = 256
D_ROPE = 64
D_FF = 4096
NUM_SHARED_EXPERTS = 2
NUM_ROUTED_EXPERTS = 64
NUM_RECURRENCES = 12
SEQ_LEN = 32768
VOCAB_SIZE = 65536

# Training
GLOBAL_BATCH_SIZE = 64       # ~2.1M tokens per global step
PER_DEVICE_BATCH_SIZE = GLOBAL_BATCH_SIZE // 8
TOTAL_STEPS = 400_000
WARMUP_STEPS = 2000
LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
MUON_MOMENTUM = 0.95

# Data / Checkpointing
TOKENS_PER_SHARD = 50_000_000
CHECKPOINT_EVERY = 500
LOG_EVERY = 10
LOG_LR_EVERY = 100


# === TOKENIZER ===
def load_tokenizer():
    tokenizer_path = hf_hub_download(
        repo_id=TOKENIZER_REPO,
        filename=TOKENIZER_FILE,
        token=HF_TOKEN
    )
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)  # type: ignore
    return sp


# === DATA PIPELINE ===
def create_data_pipeline(tokenizer, start_shard: int = 0, start_offset: int = 0):
    fineweb = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True, trust_remote_code=True)
    stack = load_dataset("bigcode/the-stack", split="train", streaming=True, trust_remote_code=True)
    numina = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True, trust_remote_code=True)

    def skip_to_shard(dataset, shard_idx: int):
        if hasattr(dataset, "n_shards") and dataset.n_shards > 0:
            n = dataset.n_shards
            dataset = dataset.shard(num_shards=n, index=shard_idx % n, contiguous=True)
        return dataset

    fineweb = skip_to_shard(fineweb, start_shard)
    stack = skip_to_shard(stack, start_shard)
    numina = skip_to_shard(numina, start_shard)

    dataset = interleave_datasets(
        [fineweb, stack, numina],
        probabilities=[0.6, 0.25, 0.15],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    def extract_text(example: dict) -> str:
        if "text" in example:
            return str(example["text"])
        if "content" in example:
            return str(example["content"])
        if "code" in example:
            return str(example["code"])
        if "messages" in example:
            return "\n".join([str(m.get("content", "")) for m in example["messages"]])
        return "\n".join([str(v) for v in example.values() if isinstance(v, str)])

    def token_generator():
        buffer = []
        tokens_consumed = 0
        for ex in dataset:
            text = extract_text(ex)
            if not text:
                continue
            tokens = tokenizer.encode(text, out_type=int)
            if tokens_consumed + len(tokens) <= start_offset:
                tokens_consumed += len(tokens)
                continue
            elif tokens_consumed < start_offset:
                tokens = tokens[start_offset - tokens_consumed :]
                tokens_consumed = start_offset

            buffer.extend(tokens)
            while len(buffer) >= SEQ_LEN + 1:
                yield {
                    "input_ids": np.array(buffer[: SEQ_LEN + 1], dtype=np.int32),
                    "loss_mask": np.ones(SEQ_LEN + 1, dtype=np.float32),
                }
                buffer = buffer[SEQ_LEN:]

    return token_generator()


def batch_generator(data_gen, batch_size: int):
    batch = []
    for example in data_gen:
        batch.append(example)
        if len(batch) == batch_size:
            input_ids = np.stack([b["input_ids"] for b in batch])
            loss_mask = np.stack([b["loss_mask"] for b in batch])
            yield {"input_ids": input_ids, "loss_mask": loss_mask}
            batch = []


# === MODEL COMPONENTS ===
class FP8Dense(nn.Module):
    features: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param(
            "kernel",
            nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
            (inputs.shape[-1], self.features),
            jnp.float32,
        )

        x_amax = jnp.max(jnp.abs(inputs))
        w_amax = jnp.max(jnp.abs(kernel))
        e4m3_max = jnp.finfo(jnp.float8_e4m3fn).max
        x_scale = e4m3_max / jnp.maximum(x_amax, 1e-12)
        w_scale = e4m3_max / jnp.maximum(w_amax, 1e-12)

        x_fp8 = (inputs * x_scale).astype(jnp.float8_e4m3fn)
        w_fp8 = (kernel * w_scale).astype(jnp.float8_e4m3fn)

        out_fp8 = jax.lax.dot_general(
            x_fp8,
            w_fp8,
            dimension_numbers=(((inputs.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=jnp.bfloat16,
        )
        out = out_fp8 / (x_scale * w_scale)

        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,), jnp.float32)
            out = out + bias.astype(jnp.bfloat16)

        return out.astype(jnp.bfloat16)


class RMSNorm(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(var + self.epsilon)
        return normed * scale


def build_yarn_rope(
    seq_len: int,
    d_rope: int,
    base: float = 10000.0,
    alpha: float = 1.0,
    beta: float = 32.0,
    scale_s: float = 4.0,
):
    m = jnp.arange(d_rope // 2)
    theta_m = base ** (-2.0 * m / d_rope)
    lambda_m = 2 * jnp.pi / theta_m
    r_m = 8192 / lambda_m
    gamma = jnp.where(
        r_m < alpha,
        0.0,
        jnp.where(r_m > beta, 1.0, (r_m - alpha) / (beta - alpha)),
    )
    theta_yarn = (1 - gamma) * (theta_m / scale_s) + gamma * theta_m
    positions = jnp.arange(seq_len)
    angles = jnp.outer(positions, theta_yarn)
    cos_val = jnp.cos(angles)
    sin_val = jnp.sin(angles)
    mscale = 0.1 * jnp.log(scale_s) + 1.0
    return cos_val, sin_val, mscale


def apply_decoupled_yarn_rope(x_rope: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray):
    cos_full = jnp.repeat(cos, 2, axis=-1)[None, :, None, :]
    sin_full = jnp.repeat(sin, 2, axis=-1)[None, :, None, :]

    def rotate(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    return (x_rope * cos_full) + (rotate(x_rope) * sin_full)


class ZenyxHybridAttention(nn.Module):
    d_model: int
    num_heads: int
    d_latent: int
    d_rope: int

    def setup(self):
        self.d_head = self.d_model // self.num_heads
        self.q_proj = FP8Dense(self.d_latent)
        self.kv_proj = FP8Dense(self.d_latent)
        self.q_up = FP8Dense(self.num_heads * self.d_head)
        self.kv_up_k = FP8Dense(self.num_heads * self.d_head)
        self.kv_up_v = FP8Dense(self.num_heads * self.d_head)
        self.o_proj = FP8Dense(self.d_model)
        self.local_window = 256
        self.top_k = 64

    def __call__(
        self,
        x: jnp.ndarray,
        q_rope: jnp.ndarray,
        k_rope: jnp.ndarray,
        is_hca_layer: bool,
    ) -> jnp.ndarray:
        batch, seq_len, _ = x.shape
        compress_ratio = 128 if is_hca_layer else 4

        c_q = self.q_proj(x)
        c_kv = self.kv_proj(x)

        num_chunks = seq_len // compress_ratio
        c_kv_compressed = c_kv.reshape(batch, num_chunks, compress_ratio, self.d_latent).mean(axis=2)

        q_nope = self.q_up(c_q).reshape(batch, seq_len, self.num_heads, self.d_head)
        k_nope = self.kv_up_k(c_kv_compressed).reshape(batch, num_chunks, self.num_heads, self.d_head)
        v_nope = self.kv_up_v(c_kv_compressed).reshape(batch, num_chunks, self.num_heads, self.d_head)

        local_len = min(self.local_window, seq_len)
        local_c_kv = c_kv[:, -local_len:, :]
        local_k = self.kv_up_k(local_c_kv).reshape(batch, local_len, self.num_heads, self.d_head)
        local_v = self.kv_up_v(local_c_kv).reshape(batch, local_len, self.num_heads, self.d_head)

        q_rope_reshaped = q_rope.reshape(batch, seq_len, self.num_heads, self.d_rope)
        q_final = jnp.concatenate([q_nope, q_rope_reshaped], axis=-1)

        k_rope_compressed = k_rope.reshape(batch, num_chunks, compress_ratio, self.d_rope).mean(axis=2)
        k_rope_local = k_rope[:, -local_len:, :]

        scale = 1.0 / jnp.sqrt(self.d_head + self.d_rope)

        if not is_hca_layer:
            q_pooled = q_nope.mean(axis=1)
            scores = jnp.einsum("bhd,bchd->bhc", q_pooled, k_nope) / jnp.sqrt(self.d_head)
            _, top_k_indices = jax.lax.top_k(scores, k=self.top_k)

            batch_idx = jnp.arange(batch)[:, None, None, None]
            head_idx = jnp.arange(self.num_heads)[None, :, None, None]
            top_k_idx = top_k_indices[..., None]

            k_nope_selected = k_nope[batch_idx, top_k_idx, head_idx, :]
            v_nope_selected = v_nope[batch_idx, top_k_idx, head_idx, :]
            k_rope_selected = k_rope_compressed[batch_idx, top_k_idx, head_idx, :]

            local_k_t = jnp.moveaxis(local_k, 2, 1)
            local_v_t = jnp.moveaxis(local_v, 2, 1)
            local_k_rope_t = jnp.moveaxis(k_rope_local, 2, 1)

            k_nope_final = jnp.concatenate([k_nope_selected, local_k_t], axis=2)
            v_final = jnp.concatenate([v_nope_selected, local_v_t], axis=2)
            k_rope_final = jnp.concatenate([k_rope_selected, local_k_rope_t], axis=2)

            k_nope_bc = jnp.broadcast_to(
                jnp.expand_dims(k_nope_final, 1),
                (batch, seq_len, self.num_heads, self.top_k + local_len, self.d_head),
            )
            v_bc = jnp.broadcast_to(
                jnp.expand_dims(v_final, 1),
                (batch, seq_len, self.num_heads, self.top_k + local_len, self.d_head),
            )
            k_rope_bc = jnp.broadcast_to(
                jnp.expand_dims(k_rope_final, 1),
                (batch, seq_len, self.num_heads, self.top_k + local_len, self.d_rope),
            )

            k_final = jnp.concatenate([k_nope_bc, k_rope_bc], axis=-1)
            attn_logits = jnp.einsum("bshd,bhtd->bhst", q_final, k_final) * scale

            q_chunk_idx = jnp.arange(seq_len) // compress_ratio
            chunk_mask = q_chunk_idx[None, None, :, None] >= top_k_indices[:, :, None, :]
            local_mask = jnp.ones((batch, self.num_heads, seq_len, local_len))
            mask = jnp.concatenate([chunk_mask, local_mask], axis=-1)
            attn_logits = jnp.where(mask, attn_logits, -1e9)

            attn_weights = jax.nn.softmax(attn_logits, axis=-1)
            attn_output = jnp.einsum("bhst,bshtd->bshd", attn_weights, v_bc)
        else:
            k_nope_assembled = jnp.concatenate([k_nope, local_k], axis=1)
            v_assembled = jnp.concatenate([v_nope, local_v], axis=1)
            k_rope_assembled = jnp.concatenate([k_rope_compressed, k_rope_local], axis=1)
            k_rope_expanded = jnp.expand_dims(k_rope_assembled, axis=2)
            k_rope_broadcast = jnp.broadcast_to(
                k_rope_expanded, (batch, num_chunks + local_len, self.num_heads, self.d_rope)
            )
            k_final = jnp.concatenate([k_nope_assembled, k_rope_broadcast], axis=-1)

            attn_logits = jnp.einsum("bshd,bhtd->bhst", q_final, k_final) * scale
            mask = jnp.tril(jnp.ones((seq_len, num_chunks + local_len)))
            attn_logits = jnp.where(mask[None, None, :, :] == 0, -1e9, attn_logits)

            attn_weights = jax.nn.softmax(attn_logits, axis=-1)
            attn_output = jnp.einsum("bhst,bhtd->bshd", attn_weights, v_assembled)

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
        self.router = nn.Dense(
            self.num_routed_experts, use_bias=False, dtype=jnp.float32
        )
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

    def __call__(self, x: jnp.ndarray):
        shared_1 = self.shared_1_w2(jax.nn.silu(self.shared_1_w1(x)))
        shared_2 = self.shared_2_w2(jax.nn.silu(self.shared_2_w1(x)))
        shared_out = shared_1 + shared_2

        router_logits = self.router(x)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        expert_indices = jnp.argmax(router_probs, axis=-1)
        expert_gates = jnp.max(router_probs, axis=-1)

        x_flat = x.reshape(-1, self.d_model)
        idx_flat = expert_indices.reshape(-1)
        gate_flat = expert_gates.reshape(-1)

        def apply_one(x_t, idx, gate):
            h = jax.nn.silu(jnp.dot(x_t, self.routed_w1[idx]))
            o = jnp.dot(h, self.routed_w2[idx])
            return o * gate

        routed_flat = jax.vmap(apply_one)(x_flat, idx_flat, gate_flat)
        routed_out = routed_flat.reshape(x.shape)

        final_out = shared_out + routed_out

        expert_mask = jax.nn.one_hot(idx_flat, self.num_routed_experts, dtype=jnp.float32)
        f_i = jnp.mean(expert_mask, axis=0)
        P_i = jnp.mean(router_probs.reshape(-1, self.num_routed_experts), axis=0)
        aux_loss = 0.01 * self.num_routed_experts * jnp.sum(f_i * P_i)

        return final_out, aux_loss


class ZenyxRecurrentSuperBlock(nn.Module):
    d_model: int
    num_heads: int
    d_latent: int
    d_rope: int
    d_ff: int
    num_shared_experts: int
    num_routed_experts: int
    num_recurrences: int

    def setup(self):
        self.hybrid_attn = ZenyxHybridAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_latent=self.d_latent,
            d_rope=self.d_rope,
        )
        self.moe = DualSharedSparseMoE(
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_routed_experts=self.num_routed_experts,
        )
        self.norm1 = RMSNorm()
        self.norm2 = RMSNorm()
        self.gamma_1 = self.param("gamma_1", nn.initializers.constant(1e-4), (self.d_model,))
        self.gamma_2 = self.param("gamma_2", nn.initializers.constant(1e-4), (self.d_model,))

    def __call__(self, x: jnp.ndarray, q_rope: jnp.ndarray, k_rope: jnp.ndarray):
        @partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_saveable)
        def forward(x_in):
            total_aux = 0.0
            for i in range(self.num_recurrences):
                x_norm = self.norm1(x_in)
                is_hca = (i % 2 == 0)
                attn_out = self.hybrid_attn(x_norm, q_rope, k_rope, is_hca_layer=is_hca)
                x_mid = x_in + attn_out * self.gamma_1
                moe_out, aux = self.moe(self.norm2(x_mid))
                x_in = x_mid + moe_out * self.gamma_2
                total_aux += aux
            return x_in, total_aux

        return forward(x)


class ZenyxV3Model(nn.Module):
    vocab_size: int
    d_model: int
    num_heads: int
    d_latent: int
    d_rope: int
    d_ff: int
    num_shared_experts: int
    num_routed_experts: int
    num_recurrences: int

    def setup(self):
        self.embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
        )
        self.recurrent_block = ZenyxRecurrentSuperBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_latent=self.d_latent,
            d_rope=self.d_rope,
            d_ff=self.d_ff,
            num_shared_experts=self.num_shared_experts,
            num_routed_experts=self.num_routed_experts,
            num_recurrences=self.num_recurrences,
        )
        self.norm_final = RMSNorm()
        self.lm_head = nn.Dense(
            self.vocab_size,
            use_bias=False,
            kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
        )
        self.q_rope_proj = FP8Dense(self.num_heads * self.d_rope)
        self.k_rope_proj = FP8Dense(self.num_heads * self.d_rope)

    def __call__(self, input_ids: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray):
        x = self.embed(input_ids)
        batch, seq_len = input_ids.shape

        q_rope_raw = self.q_rope_proj(x)
        k_rope_raw = self.k_rope_proj(x)

        q_rope = apply_decoupled_yarn_rope(
            q_rope_raw.reshape(batch, seq_len, self.num_heads, self.d_rope), cos, sin
        )
        k_rope = apply_decoupled_yarn_rope(
            k_rope_raw.reshape(batch, seq_len, self.num_heads, self.d_rope), cos, sin
        )

        x, aux_loss = self.recurrent_block(x, q_rope, k_rope)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits, aux_loss


# === OPTIMIZER ===
class MuonState(NamedTuple):
    momentum: Any


def newton_schulz_iteration(G: jnp.ndarray, steps: int = 5) -> jnp.ndarray:
    X = G.astype(jnp.bfloat16)
    X = X / (jnp.linalg.norm(X, ord="fro") + 1e-7)

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

    X_final, _ = jax.lax.scan(ns_step, X, None, length=steps)

    if transpose_flag:
        X_final = X_final.T

    return X_final.astype(G.dtype)


def scale_by_muon(momentum: float = 0.95):
    def init_fn(params):
        return MuonState(momentum=jax.tree_util.tree_map(jnp.zeros_like, params))

    def update_fn(updates, state, params=None):
        mu_next = jax.tree_util.tree_map(
            lambda m, g: momentum * m + g, state.momentum, updates
        )
        orthogonalized = jax.tree_util.tree_map(
            lambda m: newton_schulz_iteration(m) if m.ndim >= 2 else m, mu_next
        )
        return orthogonalized, MuonState(momentum=mu_next)

    return optax.GradientTransformation(init_fn, update_fn)


def build_hybrid_optimizer(lr_schedule, weight_decay: float, params):
    muon_tx = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        scale_by_muon(momentum=MUON_MOMENTUM),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-0.2),
        optax.add_decayed_weights(weight_decay),
    )
    adamw_tx = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-8),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
        optax.add_decayed_weights(weight_decay),
    )

    def label_fn(path, x):
        return "muon" if x.ndim >= 2 else "adamw"

    return optax.multi_transform(
        {"muon": muon_tx, "adamw": adamw_tx},
        jax.tree_util.tree_map_with_path(label_fn, params),
    )


# === TRAINING INFRASTRUCTURE ===
def setup_mesh(devices):
    device_mesh = mesh_utils.create_device_mesh((len(devices),))
    mesh = Mesh(device_mesh, axis_names=("fsdp",))
    return mesh


def get_partition_specs():
    rules = {
        ("embed", "embedding"): P("fsdp", None),
        ("q_proj", "kernel"): P(None, "fsdp"),
        ("kv_proj", "kernel"): P(None, "fsdp"),
        ("q_up", "kernel"): P(None, "fsdp"),
        ("kv_up_k", "kernel"): P(None, "fsdp"),
        ("kv_up_v", "kernel"): P(None, "fsdp"),
        ("o_proj", "kernel"): P("fsdp", None),
        ("shared_1_w1", "kernel"): P(None, "fsdp"),
        ("shared_1_w2", "kernel"): P("fsdp", None),
        ("shared_2_w1", "kernel"): P(None, "fsdp"),
        ("shared_2_w2", "kernel"): P("fsdp", None),
        ("router", "kernel"): P(None, "fsdp"),
        ("routed_w1",): P("fsdp", None, None),
        ("routed_w2",): P("fsdp", None, None),
        ("lm_head", "kernel"): P("fsdp", None),
        ("q_rope_proj", "kernel"): P(None, "fsdp"),
        ("k_rope_proj", "kernel"): P(None, "fsdp"),
    }
    return rules


def match_partition_spec(path_tuple, rules):
    for key_pattern, spec in rules.items():
        if all(k in path_tuple for k in key_pattern):
            return spec
    return P(None)


def create_train_state(rng, mesh, cos, sin):
    model = ZenyxV3Model(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_latent=D_LATENT,
        d_rope=D_ROPE,
        d_ff=D_FF,
        num_shared_experts=NUM_SHARED_EXPERTS,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        num_recurrences=NUM_RECURRENCES,
    )

    dummy_input = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
    variables = model.init(rng, dummy_input, cos, sin)
    params = variables["params"]

    def lr_schedule(step):
        step_f = step.astype(jnp.float32)
        warmup = jnp.minimum(step_f / WARMUP_STEPS, 1.0)
        decay = 0.5 * (1 + jnp.cos(jnp.pi * step_f / TOTAL_STEPS))
        decay = jnp.maximum(decay, MIN_LR / LR)
        return LR * warmup * decay

    tx = build_hybrid_optimizer(lr_schedule, WEIGHT_DECAY, params)
    opt_state = tx.init(params)

    rules = get_partition_specs()
    flat_params = flatten_dict(params, sep="/")
    flat_specs = {k: match_partition_spec(tuple(k.split("/")), rules) for k in flat_params.keys()}
    param_specs = unflatten_dict(flat_specs, sep="/")

    flat_opt = flatten_dict(opt_state, sep="/")
    flat_opt_specs = {}
    for k in flat_opt.keys():
        matched = False
        for pk, spec in flat_specs.items():
            if pk in k:
                flat_opt_specs[k] = spec
                matched = True
                break
        if not matched:
            flat_opt_specs[k] = P(None)
    opt_state_specs = unflatten_dict(flat_opt_specs, sep="/")

    param_sharding = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), param_specs)
    opt_state_sharding = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), opt_state_specs)

    params = jax.tree_util.tree_map(lambda p, s: jax.device_put(p, s), params, param_sharding)
    opt_state = jax.tree_util.tree_map(lambda o, s: jax.device_put(o, s), opt_state, opt_state_sharding)

    return params, opt_state, tx, param_sharding, opt_state_sharding


def calculate_masked_cross_entropy(logits, targets, loss_mask):
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    loss = -target_log_probs * loss_mask
    return jnp.sum(loss) / jnp.maximum(jnp.sum(loss_mask), 1.0)


def train_step(params, opt_state, tx, batch, rng, cos, sin):
    input_ids = batch["input_ids"]
    loss_mask = batch["loss_mask"]

    targets = input_ids[:, 1:]
    inputs = input_ids[:, :-1]
    loss_mask = loss_mask[:, 1:]

    def loss_fn(p):
        logits, aux_loss = ZenyxV3Model(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_latent=D_LATENT,
            d_rope=D_ROPE,
            d_ff=D_FF,
            num_shared_experts=NUM_SHARED_EXPERTS,
            num_routed_experts=NUM_ROUTED_EXPERTS,
            num_recurrences=NUM_RECURRENCES,
        ).apply({"params": p}, inputs, cos, sin)

        ce_loss = calculate_masked_cross_entropy(logits, targets, loss_mask)
        total_loss = ce_loss + aux_loss
        return total_loss, (ce_loss, aux_loss)

    (loss, (ce_loss, aux_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
        lambda acc, g: acc + jnp.sum(jnp.square(g)), grads, initializer=0.0
    ))

    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    lr = LR * jnp.minimum(jnp.float32(opt_state[0].count) / WARMUP_STEPS, 1.0) * \
         jnp.maximum(0.5 * (1 + jnp.cos(jnp.pi * jnp.float32(opt_state[0].count) / TOTAL_STEPS)), MIN_LR / LR)

    metrics = {
        "loss": loss,
        "ce_loss": ce_loss,
        "aux_loss": aux_loss,
        "grad_norm": grad_norm,
        "lr": lr,
    }
    return new_params, new_opt_state, metrics


train_step_jit = jax.jit(
    train_step,
    static_argnums=(2,),
    donate_argnums=(0, 1),
)


# === CHECKPOINT UTILS ===
def save_checkpoint(
    params,
    opt_state,
    global_step: int,
    rng_key,
    shard_index: int,
    offset_within_shard: int,
    loss: float,
    checkpoint_dir: str = "/tmp/zenyx_checkpoints",
):
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_path, exist_ok=True)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax.checkpoint.SaveArgs(aggregate=True)
    ckpt = {
        "params": params,
        "opt_state": opt_state,
        "global_step": global_step,
        "rng_key": rng_key,
        "shard_index": shard_index,
        "offset_within_shard": offset_within_shard,
    }
    checkpointer.save(ckpt_path, ckpt, save_args=jax.tree_util.tree_map(lambda _: save_args, ckpt))

    metadata = {
        "global_step": global_step,
        "shard_index": shard_index,
        "offset_within_shard": offset_within_shard,
        "loss": float(loss),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(ckpt_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    api = HfApi(token=HF_TOKEN)
    try:
        create_repo(HF_REPO, repo_type="model", private=False, exist_ok=True)
    except Exception:
        pass

    api.upload_folder(
        repo_id=HF_REPO,
        folder_path=ckpt_path,
        path_in_repo=f"checkpoint-{global_step}",
        commit_message=f"Zenyx v3 checkpoint step {global_step}",
    )

    print(f"Checkpoint saved to HF: step {global_step}")
    shutil.rmtree(ckpt_path, ignore_errors=True)
    return ckpt_path


def load_latest_checkpoint(checkpoint_dir: str = "/tmp/zenyx_checkpoints"):
    api = HfApi(token=HF_TOKEN)
    try:
        repo_files = api.list_repo_files(HF_REPO, repo_type="model")
    except Exception:
        return None, 0, 0, 0

    ckpt_dirs = [f for f in repo_files if f.startswith("checkpoint-")]
    if not ckpt_dirs:
        return None, 0, 0, 0

    steps = [int(d.split("-")[1]) for d in ckpt_dirs if d.split("-")[1].isdigit()]
    if not steps:
        return None, 0, 0, 0

    latest_step = max(steps)
    ckpt_name = f"checkpoint-{latest_step}"
    local_ckpt = os.path.join(checkpoint_dir, ckpt_name)
    os.makedirs(local_ckpt, exist_ok=True)

    for f in repo_files:
        if f.startswith(ckpt_name + "/"):
            hf_hub_download(
                repo_id=HF_REPO,
                filename=f,
                local_dir=checkpoint_dir,
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
            )

    metadata_path = os.path.join(local_ckpt, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        global_step = metadata.get("global_step", latest_step)
        shard_index = metadata.get("shard_index", 0)
        offset_within_shard = metadata.get("offset_within_shard", 0)
    else:
        global_step = latest_step
        tokens_consumed = global_step * GLOBAL_BATCH_SIZE * SEQ_LEN
        shard_index = int(tokens_consumed // TOKENS_PER_SHARD)
        offset_within_shard = int(tokens_consumed % TOKENS_PER_SHARD)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = checkpointer.restore(local_ckpt)

    return restored, global_step, shard_index, offset_within_shard


# === MAIN TRAINING LOOP ===
def main():
    print("Initializing Zenyx v3 training...")
    devices = jax.devices()
    print(f"Devices: {devices}")

    mesh = setup_mesh(devices)
    tokenizer = load_tokenizer()

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    cos, sin, _ = build_yarn_rope(SEQ_LEN, D_ROPE)
    cos = jax.device_put(cos, NamedSharding(mesh, P(None, None)))
    sin = jax.device_put(sin, NamedSharding(mesh, P(None, None)))

    restored, resume_step, resume_shard, resume_offset = load_latest_checkpoint()

    if restored is not None:
        params = restored["params"]
        opt_state = restored["opt_state"]
        rng = restored["rng_key"]
        global_step = int(resume_step)
        shard_index = int(resume_shard)
        offset_within_shard = int(resume_offset)
        print(f"Resuming from step {global_step}, shard {shard_index}, offset {offset_within_shard}")

        def lr_schedule(step):
            step_f = step.astype(jnp.float32)
            warmup = jnp.minimum(step_f / WARMUP_STEPS, 1.0)
            decay = 0.5 * (1 + jnp.cos(jnp.pi * step_f / TOTAL_STEPS))
            decay = jnp.maximum(decay, MIN_LR / LR)
            return LR * warmup * decay

        tx = build_hybrid_optimizer(lr_schedule, WEIGHT_DECAY, params)
    else:
        with mesh:
            params, opt_state, tx, param_sharding, opt_state_sharding = create_train_state(init_rng, mesh, cos, sin)
        global_step = 0
        shard_index = 0
        offset_within_shard = 0
        print("Starting fresh training from step 0")

    data_gen = create_data_pipeline(tokenizer, start_shard=shard_index, start_offset=offset_within_shard)
    batched_data = batch_generator(data_gen, PER_DEVICE_BATCH_SIZE)

    step_times = []
    tokens_processed = 0

    for batch in batched_data:
        if global_step >= TOTAL_STEPS:
            break

        step_start = time.time()

        batch_jax = {
            "input_ids": jnp.array(batch["input_ids"]),
            "loss_mask": jnp.array(batch["loss_mask"]),
        }

        rng, step_rng = jax.random.split(rng)
        params, opt_state, metrics = train_step_jit(params, opt_state, tx, batch_jax, step_rng, cos, sin)

        step_time = time.time() - step_start
        step_times.append(step_time)
        tokens_processed += GLOBAL_BATCH_SIZE * SEQ_LEN

        if global_step % LOG_EVERY == 0:
            tokens_per_sec = (GLOBAL_BATCH_SIZE * SEQ_LEN) / step_time
            print(
                f"Step {global_step} | Loss: {float(metrics['loss']):.4f} | "
                f"Aux: {float(metrics['aux_loss']):.4f} | "
                f"Tokens/sec: {tokens_per_sec:.1f} | Step time: {step_time:.3f}s"
            )

        if global_step % LOG_LR_EVERY == 0:
            print(
                f"Step {global_step} | LR: {float(metrics['lr']):.2e} | "
                f"Grad Norm: {float(metrics['grad_norm']):.4f}"
            )

        if global_step > 0 and global_step % CHECKPOINT_EVERY == 0:
            tokens_consumed = global_step * GLOBAL_BATCH_SIZE * SEQ_LEN
            current_shard = int(tokens_consumed // TOKENS_PER_SHARD)
            current_offset = int(tokens_consumed % TOKENS_PER_SHARD)

            save_checkpoint(
                params,
                opt_state,
                global_step,
                rng,
                current_shard,
                current_offset,
                float(metrics["loss"]),
            )

        global_step += 1

    print("Training complete.")


if __name__ == "__main__":
    main()
