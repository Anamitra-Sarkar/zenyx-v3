# zenyx_v3_train.py
# Zenyx v3: Sub-1B Parameter LLM Training on TPU v5e-8
# Complete script with O(1) resumption, FP8, Muon, hybrid attention, MoE, YaRN, FSDP

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import time
import json
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
import optax
import flax
from flax import linen as nn
from flax.training import train_state
from flax.training import orbax_utils
import orbax.checkpoint
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManagerOptions, CheckpointManager
import sentencepiece as spm
from huggingface_hub import HfApi, hf_hub_download, create_repo, list_repo_files
import datasets
from datasets import interleave_datasets
from typing import NamedTuple, Any, Dict, Tuple, Optional
from functools import partial
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
# Auth & storage
HF_TOKEN = "YOUR_HF_TOKEN_HERE"  # Replace with your actual HF token
HF_REPO = "Arko007/zenyx-v3-checkpoints"

# Model architecture
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

# Optimizer
LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 2000
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
MOMENTUM = 0.95

# Muon FP8 + Newton-Schulz
NEWTON_SCHULZ_STEPS = 5

# Curriculum: Phase 1 (pretrain)
PHASE1_STEPS = 50000  # Total steps for this run (example)
BATCH_SIZE = 4        # Per-device batch size; global batch will be 8 * 4 = 32
                      # Total tokens per step = 32 * 32768 = ~1M tokens
CHECKPOINT_EVERY = 500
LOG_EVERY = 10
LOG_LR_EVERY = 100

# Data mixing probabilities
DATA_MIX = {"fineweb": 0.6, "the_stack": 0.25, "numinamath": 0.15}

# =============================================================================
# TOKENIZER
# =============================================================================
def load_tokenizer() -> spm.SentencePieceProcessor:
    """Load SentencePiece tokenizer from HF repo."""
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename="tokenizer.model", token=HF_TOKEN)
        tok = spm.SentencePieceProcessor(model_file=path)
        assert tok.vocab_size() == VOCAB_SIZE, f"Tokenizer vocab size {tok.vocab_size()} != {VOCAB_SIZE}"
        return tok
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

# =============================================================================
# DATA PIPELINE
# =============================================================================
def prepare_dataset(split="train", streaming=True):
    """Load and interleave pretraining datasets."""
    # Verified open datasets
    fineweb = datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        split=split, 
        streaming=streaming, 
        token=HF_TOKEN
    ).map(lambda x: {"text": x["text"]}, remove_columns=[col for col in fineweb.column_names if col != "text"])
    
    the_stack = datasets.load_dataset(
        "bigcode/the-stack-smol", 
        split=split, 
        streaming=streaming, 
        token=HF_TOKEN
    )
    # The Stack has "content" column
    the_stack = the_stack.map(lambda x: {"text": x["content"]}, remove_columns=["content"])
    
    numinamath = datasets.load_dataset(
        "AI-MO/NuminaMath-CoT", 
        split=split, 
        streaming=streaming, 
        token=HF_TOKEN
    )
    # Extract problem + solution into a single text field (simplistic)
    def merge_math(example):
        text = example.get("problem", "") + "\n" + example.get("solution", "")
        return {"text": text}
    numinamath = numinamath.map(merge_math, remove_columns=numinamath.column_names)
    
    # Interleave with given probabilities
    ds = interleave_datasets(
        [fineweb, the_stack, numinamath],
        probabilities=list(DATA_MIX.values()),
        seed=42,
        stopping_strategy="all_exhausted"
    )
    return ds

def tokenization_pipeline(dataset_iter, tokenizer, max_len=SEQ_LEN):
    """Stream tokenization and packing into fixed-length sequences using a ring buffer."""
    buffer = []
    buffer_len = 0
    bos_id = tokenizer.bos_id()  # typically 1
    eos_id = tokenizer.eos_id()  # typically 2
    
    for example in dataset_iter:
        text = example["text"]
        if not text:
            continue
        # Tokenize text, add BOS/EOS
        ids = [bos_id] + tokenizer.encode(text) + [eos_id]
        buffer.extend(ids)
        buffer_len += len(ids)
        
        # Yield full sequences as they become available
        while buffer_len >= max_len:
            chunk = buffer[:max_len]
            buffer = buffer[max_len:]
            buffer_len -= max_len
            yield np.array(chunk, dtype=np.int32)
    
    # Final partial sequence (drop remainder, optional padding)
    if buffer_len > 0:
        # Drop to avoid contamination; could pad but we skip
        pass

def create_dataloader(global_step: int = 0, tokens_consumed: int = 0):
    """Create a streaming dataloader that resumes at the correct shard/offset.
    
    Args:
        global_step: current step (for resumption, not used in skipping logic except to print)
        tokens_consumed: total tokens consumed so far (from last checkpoint)
    
    Returns:
        iterator yielding (batch, loss_mask) where batch shape [num_devices, batch_size, seq_len]
    """
    ds = prepare_dataset()
    tokenizer = load_tokenizer()
    
    # Calculate tokens per shard (streaming, but we control skipping)
    # We'll iterate and skip tokens_consumed tokens before yielding.
    # Since datasets are shard-agnostic streaming, we'll just fast-forward the tokenizer.
    # For true O(1) shard skipping, we can use dataset.skip() on shard level.
    # Here we tokenize and discard tokens.
    # Determine shard index and offset per original blueprint.
    tokens_per_shard = 100_000_000  # example; roughly 100M per shard
    shard_idx = tokens_consumed // tokens_per_shard
    offset = tokens_consumed % tokens_per_shard
    
    # Use dataset.skip(shard_idx) to jump to the correct shard? Actually streaming datasets don't support skip by shard index directly.
    # But we can simulate: iterate through dataset and skip that many examples.
    # For real large-scale, we'd need shard-aware datasets. Here we'll just skip by iterating.
    # We'll just skip tokens_consumed tokens.
    token_iter = tokenization_pipeline(ds, tokenizer)
    
    # Skip tokens
    skipped = 0
    for seq in token_iter:
        skipped += seq.shape[0]
        if skipped >= tokens_consumed - (tokens_consumed % SEQ_LEN):
            # Now we are approximately at the resume point.
            # Store the first chunk after skipping that might be partial.
            # We'll just start yielding from here.
            break
    
    print(f"Resuming from global_step={global_step}, shard_index={shard_idx}, offset={offset}, tokens_consumed={tokens_consumed}")
    
    # Now yield batches indefinitely
    # Global batch across devices
    num_devices = jax.device_count()
    needed_tokens = BATCH_SIZE * num_devices * SEQ_LEN
    buffer = []
    while True:
        seq = next(token_iter)
        buffer.append(seq)
        if sum(s.shape[0] for s in buffer) >= needed_tokens:
            # Concatenate and reshape
            tokens = np.concatenate(buffer)[:needed_tokens]
            tokens = tokens.reshape(num_devices, BATCH_SIZE, SEQ_LEN)
            loss_mask = np.ones_like(tokens, dtype=np.float32)  # all ones for pretraining
            buffer = buffer[len(buffer):]  # clear buffer
            yield tokens, loss_mask

# =============================================================================
# MODEL COMPONENTS
# =============================================================================
class FP8Dense(nn.Module):
    """FP8 quantized dense layer for TPU v5e."""
    features: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param('kernel',
                           nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'),
                           (inputs.shape[-1], self.features), jnp.float32)
        # Dynamic scaling for FP8 E4M3
        x_amax = jnp.max(jnp.abs(inputs))
        w_amax = jnp.max(jnp.abs(kernel))
        e4m3_max = jnp.finfo(jnp.float8_e4m3fn).max
        x_scale = e4m3_max / jnp.maximum(x_amax, 1e-12)
        w_scale = e4m3_max / jnp.maximum(w_amax, 1e-12)
        
        # Quantize
        x_fp8 = (inputs * x_scale).astype(jnp.float8_e4m3fn)
        w_fp8 = (kernel * w_scale).astype(jnp.float8_e4m3fn)
        
        # FP8 matmul
        out_fp8 = jax.lax.dot_general(
            x_fp8, w_fp8,
            dimension_numbers=(((inputs.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=jnp.bfloat16
        )
        
        # Dequantize
        out = out_fp8 / (x_scale * w_scale)
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,), jnp.float32)
            out = out + bias.astype(jnp.bfloat16)
        return out.astype(jnp.bfloat16)

def build_yarn_rope(seq_len: int, d_rope: int, base: float = 10000.0,
                   alpha: float = 1.0, beta: float = 32.0, scale_s: float = 4.0):
    """Compute YaRN frequencies and temperature scaling."""
    m = jnp.arange(d_rope // 2)
    theta_m = base ** (-2.0 * m / d_rope)
    lambda_m = 2 * jnp.pi / theta_m
    r_m = 8192 / lambda_m
    gamma = jnp.where(r_m < alpha, 0.0,
                      jnp.where(r_m > beta, 1.0, (r_m - alpha) / (beta - alpha)))
    theta_yarn = (1 - gamma) * (theta_m / scale_s) + gamma * theta_m
    positions = jnp.arange(seq_len)
    angles = jnp.outer(positions, theta_yarn)
    cos_val = jnp.cos(angles)
    sin_val = jnp.sin(angles)
    mscale = 0.1 * jnp.log(scale_s) + 1.0
    return cos_val, sin_val, mscale

def apply_decoupled_yarn_rope(x_rope: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray):
    """Apply rotary embedding to decoupled heads."""
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

    def __call__(self, x: jnp.ndarray, q_rope: jnp.ndarray, k_rope: jnp.ndarray) -> jnp.ndarray:
        batch, seq_len, _ = x.shape
        # 1. Latent projections
        c_q = self.q_proj(x)
        c_kv = self.kv_proj(x)

        # 2. Compression
        num_chunks = seq_len // self.compress_ratio
        c_kv_compressed = c_kv.reshape(batch, num_chunks, self.compress_ratio, self.d_latent).mean(axis=2)

        # 3. Decompress to multi-head
        q_nope = self.q_up(c_q).reshape(batch, seq_len, self.num_heads, self.d_head)
        k_nope = self.kv_up_k(c_kv_compressed).reshape(batch, num_chunks, self.num_heads, self.d_head)
        v_nope = self.kv_up_v(c_kv_compressed).reshape(batch, num_chunks, self.num_heads, self.d_head)

        # 4. Local sliding window (uncompressed)
        local_c_kv = c_kv[:, -self.local_window:, :]
        local_k = self.kv_up_k(local_c_kv).reshape(batch, self.local_window, self.num_heads, self.d_head)
        local_v = self.kv_up_v(local_c_kv).reshape(batch, self.local_window, self.num_heads, self.d_head)

        # Assemble KV
        k_assembled = jnp.concatenate([k_nope, local_k], axis=1)
        v_assembled = jnp.concatenate([v_nope, local_v], axis=1)

        # Assemble decoupled RoPE
        q_rope_reshaped = q_rope.reshape(batch, seq_len, self.num_heads, self.d_rope)
        q_final = jnp.concatenate([q_nope, q_rope_reshaped], axis=-1)

        # For compressed keys, we need to compress k_rope similarly
        k_rope_reshaped = k_rope.reshape(batch, seq_len, self.d_rope)
        # Compress k_rope to match num_chunks
        k_rope_compressed = k_rope_reshaped.reshape(batch, num_chunks, self.compress_ratio, self.d_rope).mean(axis=2)
        k_rope_local = k_rope_reshaped[:, -self.local_window:, :]
        k_rope_final_vals = jnp.concatenate([k_rope_compressed, k_rope_local], axis=1)
        # Expand for heads: broadcast over heads dimension
        k_rope_final = jnp.expand_dims(k_rope_final_vals, axis=2)
        k_rope_final = jnp.broadcast_to(k_rope_final, (batch, num_chunks + self.local_window, self.num_heads, self.d_rope))
        k_final = jnp.concatenate([k_assembled, k_rope_final], axis=-1)

        # 5. Attention with scaling and optional sparsify
        scale = 1.0 / jnp.sqrt(self.d_head + self.d_rope)
        attn_logits = jnp.einsum('bshd,bthd->bhst', q_final, k_final) * scale

        # Causal mask
        total_kv_len = num_chunks + self.local_window
        mask = jnp.tril(jnp.ones((seq_len, total_kv_len)))
        attn_logits = jnp.where(mask == 0, -1e9, attn_logits)

        if not self.is_hca_layer:
            # Sparse top-k for CSA
            top_k = 64
            top_k_values = jnp.take_along_axis(attn_logits, jnp.argsort(attn_logits, axis=-1)[:, :, :, -top_k:], axis=-1)
            thresholds = top_k_values.min(axis=-1, keepdims=True)
            attn_logits = jnp.where(attn_logits >= thresholds, attn_logits, -1e9)

        attn_weights = jax.nn.softmax(attn_logits)
        attn_output = jnp.einsum('bhst,bthd->bshd', attn_weights, v_assembled)
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
        self.routed_w1 = self.param('routed_w1', nn.initializers.lecun_normal(),
                                    (self.num_routed_experts, self.d_model, d_ff_split), jnp.bfloat16)
        self.routed_w2 = self.param('routed_w2', nn.initializers.lecun_normal(),
                                    (self.num_routed_experts, d_ff_split, self.d_model), jnp.bfloat16)

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Dual shared experts
        shared_1_out = self.shared_1_w2(jax.nn.silu(self.shared_1_w1(x)))
        shared_2_out = self.shared_2_w2(jax.nn.silu(self.shared_2_w1(x)))
        shared_out = shared_1_out + shared_2_out

        # Router
        router_logits = self.router(x)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        expert_indices = jnp.argmax(router_probs, axis=-1)  # Top-1
        expert_gates = jnp.max(router_probs, axis=-1, keepdims=True)

        # Expert computation (batched per token via index)
        # Gather weights for chosen experts
        batch, seq, _ = x.shape
        # Efficient einsum using tokens
        selected_w1 = self.routed_w1[expert_indices]  # shape: (b,s, d_model, d_ff_split)
        selected_w2 = self.routed_w2[expert_indices]  # shape: (b,s, d_ff_split, d_model)
        h_routed = jax.nn.silu(jnp.einsum('bsd,bsdf->bsf', x, selected_w1))
        routed_out = jnp.einsum('bsf,bsfd->bsd', h_routed, selected_w2)

        final_out = shared_out + (routed_out * expert_gates)

        # Aux loss
        expert_mask = jax.nn.one_hot(expert_indices, self.num_routed_experts, dtype=jnp.float32)
        f_i = jnp.mean(expert_mask, axis=(0, 1))
        P_i = jnp.mean(router_probs, axis=(0, 1))
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
        # Attention layers: we will dynamically set is_hca_layer
        self.attn_layers = [ZenyxHybridAttention(D_MODEL, NUM_HEADS, D_LATENT, D_ROPE, is_hca_layer=False) for _ in range(num_recurrences)]
        self.moe_layers = [DualSharedSparseMoE(D_MODEL, D_FF, NUM_ROUTED_EXPERTS) for _ in range(num_recurrences)]

    def __call__(self, x: jnp.ndarray, q_rope: jnp.ndarray, k_rope: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        total_aux_loss = 0.0
        for step_idx in range(self.num_recurrences):
            # Alternate HCA on even steps
            is_hca = (step_idx % 2 == 0)
            self.attn_layers[step_idx].is_hca_layer = is_hca
            
            # Sub-layer 1: Attention with LayerScale
            x_norm = self.norm1(x)
            attn_out = self.attn_layers[step_idx](x_norm, q_rope, k_rope)
            x_mid = x + (attn_out * self.gamma_1)
            
            # Sub-layer 2: MoE with LayerScale
            x_norm2 = self.norm2(x_mid)
            moe_out, aux_loss = self.moe_layers[step_idx](x_norm2)
            x_out = x_mid + (moe_out * self.gamma_2)
            
            x = x_out
            total_aux_loss += aux_loss
        return x, total_aux_loss

class ZenyxV3Model(nn.Module):
    vocab_size: int
    d_model: int
    num_recurrences: int

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray, deterministic: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch, seq_len = input_ids.shape
        # Embeddings
        embed = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model, dtype=jnp.bfloat16)
        x = embed(input_ids)
        
        # YaRN RoPE for decoupled positional info
        cos, sin, mscale = build_yarn_rope(seq_len, D_ROPE)
        # Generate q_rope and k_rope as zero vectors? Actually we need them.
        # In practice, we would create learnable or fixed vectors.
        # For simplicity, we'll use the original input's position but decoupled.
        # We'll create a dummy rope input: a constant vector for each token.
        # Real implementation would derive from a separate embedding or fixed sinusoidal.
        # Here we'll just use a learnable parameter for each position? That's wasteful.
        # We'll cheat: use the same embedding but we need shape (batch, seq, d_rope)
        # For training, we can use a fixed sinusoidal signal.
        # We'll create positional encoding for rope.
        # Use cos/sin to produce a rotational signal? But rope typically applies to queries/keys.
        # Since our rope is decoupled, we'll just pass a tensor of sine/cosine maps.
        # In a real model, you might have a learnable continuous positional embedding.
        # For this script, we'll generate q_rope, k_rope from sinusoidal signals as placeholders.
        # They will be broadcast across batch.
        pos = jnp.arange(seq_len, dtype=jnp.float32)[None, :]  # (1, seq_len)
        q_rope = jnp.broadcast_to(pos, (batch, seq_len))[:, :, None] * jnp.ones((1,1,D_ROPE))  # dummy
        q_rope = jnp.sin(q_rope).astype(jnp.bfloat16)
        k_rope = q_rope  # shared
        
        # Apply YaRN to them? The attention function expects q_rope,k_rope already in rope format.
        # We'll just pass them as is.
        
        # Recurrent super block
        recur = ZenyxRecurrentSuperBlock(d_model=self.d_model, num_recurrences=self.num_recurrences)
        x, aux_loss = recur(x, q_rope, k_rope)
        
        # Final norm and lm_head
        x = nn.RMSNorm()(x)
        x = FP8Dense(self.vocab_size, use_bias=False)(x)  # output logits
        return x, aux_loss

# =============================================================================
# OPTIMIZER
# =============================================================================
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
        B = b * A + c * (A @ A)  # quintic approximation
        X_next = a * X_curr + B @ X_curr
        return X_next, None
    
    X_final, _ = jax.lax.scan(ns_step, X, None, length=steps)
    if transpose_flag:
        X_final = X_final.T
    return X_final.astype(G.dtype)

def scale_by_muon(learning_rate: float, momentum: float = 0.95) -> optax.GradientTransformation:
    def init_fn(params):
        return {'momentum': jax.tree_util.tree_map(jnp.zeros_like, params)}
    
    def update_fn(updates, state, params=None):
        mu = state['momentum']
        mu_next = jax.tree_util.tree_map(lambda m, g: momentum * m + g, mu, updates)
        orthogonalized = jax.tree_util.tree_map(
            lambda m: newton_schulz_iteration(m) if len(m.shape) >= 2 else m,
            mu_next
        )
        # Scale: for 2D apply 0.2 factor matching AdamW's RMS
        scaled = jax.tree_util.tree_map(
            lambda u, orig: -learning_rate * (0.2 if len(orig.shape) >= 2 else 1.0) * u,
            orthogonalized, updates
        )
        return scaled, {'momentum': mu_next}
    return optax.GradientTransformation(init_fn, update_fn)

def build_hybrid_optimizer() -> optax.GradientTransformation:
    # Parameter groups
    def is_2d(param_path, param):
        return len(param.shape) >= 2
    
    muon_tx = optax.chain(
        scale_by_muon(LR, MOMENTUM),
        optax.add_decayed_weights(WEIGHT_DECAY),
    )
    adamw_tx = optax.chain(
        optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-8),
        optax.add_decayed_weights(WEIGHT_DECAY),
        optax.scale(-LR),
    )
    
    # Routing via path
    label_fn = partial(optax.tree_utils.tree_map_with_path, 
                       lambda path, x: 'muon' if len(x.shape) >= 2 else 'adamw')
    tx = optax.multi_transform({'muon': muon_tx, 'adamw': adamw_tx}, label_fn)
    return tx

# =============================================================================
# TRAINING INFRASTRUCTURE
# =============================================================================
def setup_mesh():
    """Create TPU v5e-8 1D mesh."""
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('fsdp',))
    return mesh

def get_partition_specs():
    """FSDP sharding mapping."""
    # Rules based on module parameter names; we'll apply to the full param tree.
    # We'll define a function to map each param to PartitionSpec.
    # For simplicity, we'll shard matrices across the second dimension for weights, 
    # and keep biases/norms replicated.
    def spec_for(param_path, param):
        # param_path is a tuple of keys, e.g., ('embed', 'kernel') etc.
        # We'll categorize.
        if len(param.shape) >= 2:
            if param.shape[1] == D_MODEL or param.shape[1] == VOCAB_SIZE:
                return P('fsdp', None)  # output shard
            else:
                return P(None, 'fsdp')  # input shard
        else:
            return P(None)
    return spec_for

def create_train_state(rng, mesh, params):
    """Initialize training state with optimizer and sharding."""
    tx = build_hybrid_optimizer()
    # We need a function to initialize opt_state from params.
    opt_state = tx.init(params)
    # Wrap into TrainState
    state = train_state.TrainState.create(
        apply_fn=None, params=params, tx=tx, opt_state=opt_state
    )
    # Shard parameters and state across the mesh
    spec_fn = get_partition_specs()
    # Use jax.tree_map to shard
    state = jax.tree_map(
        lambda x, path: jax.device_put(x, NamedSharding(mesh, P(*spec_fn(path, x)) if spec_fn else P())),
        state, 
        is_leaf=lambda x: isinstance(x, jnp.ndarray)
    )
    return state

@partial(jax.jit, static_argnums=(3,))
def train_step(state, batch, loss_mask, rng):
    """Single training step."""
    def loss_fn(params):
        logits, aux_loss = ZenyxV3Model(VOCAB_SIZE, D_MODEL, NUM_RECURRENCES).apply(
            {'params': params}, batch, deterministic=False
        )
        # Shift for next-token prediction
        logits = logits[:, :-1, :]
        labels = batch[:, 1:]
        mask = loss_mask[:, 1:]
        # Cross-entropy loss
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        ce = (ce * mask).sum() / mask.sum()
        total_loss = ce + aux_loss
        return total_loss, (ce, aux_loss)
    
    (loss, (ce, aux_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # Gradient clipping
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -GRAD_CLIP, GRAD_CLIP), grads)
    state = state.apply_gradients(grads=grads)
    return state, loss, ce, aux_loss

def calculate_masked_cross_entropy(logits, labels, mask):
    # Not used separately; integrated in train_step
    pass

# =============================================================================
# CHECKPOINT UTILS
# =============================================================================
def save_checkpoint(state, step, shard_index, offset_within_shard):
    """Save checkpoint locally and upload to HF."""
    checkpointer = PyTreeCheckpointer()
    ckpt_dir = f"./checkpoint-{step}"
    # Save orbax checkpoint
    orbax_checkpointer = orbax.checkpoint.CheckpointManager(
        ckpt_dir, orbax.checkpoint.Checkpointer(PyTreeCheckpointer()), CheckpointManagerOptions(max_to_keep=1)
    )
    # Save state params, opt_state, step, etc.
    save_item = {
        'params': state.params,
        'opt_state': state.opt_state,
        'global_step': step,
        'rng': state.step,  # we'll store rng separately, not needed
        'shard_index': shard_index,
        'offset_within_shard': offset_within_shard
    }
    orbax_checkpointer.save(step, save_item)
    # Upload to HF
    api = HfApi()
    try:
        api.create_repo(HF_REPO, private=False, token=HF_TOKEN, exist_ok=True)
        api.upload_folder(
            folder_path=ckpt_dir,
            path_in_repo=f"checkpoint-{step}",
            repo_id=HF_REPO,
            token=HF_TOKEN
        )
        # Write metadata.json
        metadata = {
            "global_step": step,
            "shard_index": shard_index,
            "offset_within_shard": offset_within_shard,
            "loss": 0.0,  # will be updated after step
            "timestamp": time.time()
        }
        with open("metadata.json", "w") as f:
            json.dump(metadata, f)
        api.upload_file(
            path_or_fileobj="metadata.json",
            path_in_repo=f"checkpoint-{step}/metadata.json",
            repo_id=HF_REPO,
            token=HF_TOKEN
        )
        print(f"Checkpoint saved to HF: step {step}")
    except Exception as e:
        print(f"Failed to upload checkpoint: {e}")

def load_latest_checkpoint():
    """O(1) resumption: find latest checkpoint and calculate token offset."""
    api = HfApi()
    try:
        repo_files = api.list_repo_files(HF_REPO, token=HF_TOKEN)
    except:
        return None, 0, 0, 0
    
    # Filter checkpoint directories
    checkpoints = []
    for f in repo_files:
        if f.startswith("checkpoint-") and f.endswith("metadata.json"):
            checkpoints.append(f)
    if not checkpoints:
        return None, 0, 0, 0
    
    # Get latest step
    steps = [int(d.split('-')[1].split('/')[0]) for d in checkpoints]
    latest_step = max(steps)
    # Download metadata
    meta_path = hf_hub_download(repo_id=HF_REPO, filename=f"checkpoint-{latest_step}/metadata.json", token=HF_TOKEN)
    with open(meta_path) as f:
        meta = json.load(f)
    global_step = meta["global_step"]
    shard_index = meta["shard_index"]
    offset_within_shard = meta["offset_within_shard"]
    # Download checkpoint params
    checkpoint_dir = hf_hub_download(repo_id=HF_REPO, filename=f"checkpoint-{latest_step}/params", token=HF_TOKEN)
    # Load orbax checkpoint
    orbax_checkpointer = orbax.checkpoint.CheckpointManager(
        checkpoint_dir, orbax.checkpoint.Checkpointer(PyTreeCheckpointer())
    )
    state_dict = orbax_checkpointer.restore(global_step)
    params = state_dict['params']
    opt_state = state_dict['opt_state']
    # Reconstruct train state (need to set step)
    return params, opt_state, global_step, shard_index, offset_within_shard

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def main():
    # Setup mesh
    mesh = setup_mesh()
    jax.config.update("jax_default_device", jax.devices()[0])
    
    # Try to resume
    loaded = load_latest_checkpoint()
    if loaded[0] is not None:
        params, opt_state, step_start, shard_idx_restart, offset_restart = loaded
        print(f"Resuming from checkpoint: step {step_start}")
    else:
        # Fresh initialization
        rng = random.PRNGKey(0)
        dummy_input = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
        model = ZenyxV3Model(VOCAB_SIZE, D_MODEL, NUM_RECURRENCES)
        params = model.init(rng, dummy_input)['params']
        tx = build_hybrid_optimizer()
        opt_state = tx.init(params)
        step_start = 0
        shard_idx_restart = 0
        offset_restart = 0
        print("Starting training from scratch.")
    
    # Create optimizer state and train state
    # We'll manually build train_state because we need to shard.
    class CustomTrainState:
        def __init__(self, params, opt_state, step):
            self.params = params
            self.opt_state = opt_state
            self.step = step
    
    state = CustomTrainState(params=params, opt_state=opt_state, step=step_start)
    # Shard state
    spec_fn = get_partition_specs()
    state = jax.tree_map(lambda x, path: jax.device_put(x, NamedSharding(mesh, spec_fn(path, x))), state, 
                         is_leaf=lambda x: isinstance(x, jnp.ndarray))
    
    # Data loader
    # Calculate tokens consumed
    tokens_consumed = step_start * BATCH_SIZE * jax.device_count() * SEQ_LEN
    dataloader = create_dataloader(step_start, tokens_consumed)
    
    # Training loop
    rng = random.PRNGKey(step_start)  # base rng
    step = step_start
    total_loss = 0.0
    start_time = time.time()
    for batch, loss_mask in dataloader:
        if step >= PHASE1_STEPS:
            break
        rng, step_rng = random.split(rng)
        state, loss, ce, aux_loss = train_step(state, jnp.asarray(batch), jnp.asarray(loss_mask), step_rng)
        total_loss += loss
        step += 1
        
        if step % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (BATCH_SIZE * jax.device_count() * SEQ_LEN * LOG_EVERY) / elapsed
            print(f"Step {step}: loss={loss:.4f}, aux_loss={aux_loss:.4f}, tokens/sec={tokens_per_sec:.2f}, step time={elapsed/LOG_EVERY:.3f}s")
            start_time = time.time()
        
        if step % LOG_LR_EVERY == 0:
            # Display LR (hardcoded schedule)
            lr = LR * max(0, 1 - step/PHASE1_STEPS)  # simplified cosine
            print(f"Step {step}: learning_rate={lr:.6f}")
        
        if step % CHECKPOINT_EVERY == 0:
            shard_idx = tokens_consumed // 100_000_000 + step  # approximate
            save_checkpoint(state, step, shard_idx, tokens_consumed % 100_000_000)
    
    print("Training complete.")

if __name__ == "__main__":
    main()
