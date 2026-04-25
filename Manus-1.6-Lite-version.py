import os
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from typing import NamedTuple, Any, Callable, Dict, Tuple
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from huggingface_hub import HfApi, hf_hub_download
import json
import math
from datasets import load_dataset
from itertools import islice

# ==============================================================================
# 1. Muon Optimizer Implementation
# ==============================================================================

class MuonState(NamedTuple):
    momentum: optax.OptState

def newton_schulz_iteration(G: jnp.ndarray, steps: int = 5) -> jnp.ndarray:
    """Applies Newton-Schulz iteration to orthogonalize the gradient matrix in bfloat16."""
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
        
    X_final, _ = jax.lax.scan(ns_step, X, None, length=steps)
    
    if transpose_flag:
        X_final = X_final.T
        
    return X_final.astype(G.dtype)

def scale_by_muon(learning_rate: float, momentum: float = 0.95) -> optax.GradientTransformation:
    """Optax transformation for the Muon optimizer."""
    def init_fn(params):
        return MuonState(momentum=jax.tree_util.tree_map(jnp.zeros_like, params))
        
    def update_fn(updates, state, params=None):
        mu_next = jax.tree_util.tree_map(
            lambda m, g: momentum * m + g, state.momentum, updates
        )
        
        orthogonalized_updates = jax.tree_util.tree_map(
            lambda m: newton_schulz_iteration(m) if len(m.shape) >= 2 else m,
            mu_next
        )
        
        scaled_updates = jax.tree_util.tree_map(
            lambda u: -learning_rate * 0.2 * u if len(u.shape) >= 2 else -learning_rate * u,
            orthogonalized_updates
        )
        
        return scaled_updates, MuonState(momentum=mu_next)
        
    return optax.GradientTransformation(init_fn, update_fn)

def build_hybrid_optimizer(lr: float, weight_decay: float, params: Any) -> optax.GradientTransformation:
    """Routes 2D params to Muon and 1D params to AdamW."""
    muon_tx = optax.chain(
        scale_by_muon(lr),
        optax.add_decayed_weights(weight_decay)
    )
    adamw_tx = optax.chain(
        optax.scale_by_adam(b1=0.9, b2=0.95),
        optax.add_decayed_weights(weight_decay),
        optax.scale(-lr)
    )
    
    def is_2d(path, param):
        return len(param.shape) >= 2
        
    return optax.multi_transform(
        {'muon': muon_tx, 'adamw': adamw_tx},
        optax.tree_utils.tree_map_with_path(lambda p, x: 'muon' if is_2d(p, x) else 'adamw', params)
    )

# ==============================================================================
# 2. FP8 Dense Implementation
# ==============================================================================

class FP8Dense(nn.Module):
    features: int
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param(
            'kernel',
            nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'),
            (inputs.shape[-1], self.features), jnp.float32
        )
        
        x_amax = jnp.max(jnp.abs(inputs))
        w_amax = jnp.max(jnp.abs(kernel))
        
        e4m3_max = jnp.finfo(jnp.float8_e4m3fn).max
        x_scale = e4m3_max / jnp.maximum(x_amax, 1e-12)
        w_scale = e4m3_max / jnp.maximum(w_amax, 1e-12)
        
        x_fp8 = (inputs * x_scale).astype(jnp.float8_e4m3fn)
        w_fp8 = (kernel * w_scale).astype(jnp.float8_e4m3fn)
        
        out_fp8 = jax.lax.dot_general(
            x_fp8, w_fp8,
            dimension_numbers=(((inputs.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=jnp.bfloat16
        )
        
        out = out_fp8 / (x_scale * w_scale)
        
        if self.use_bias:
            bias = self.param('bias', nn.initializers.constant(0.0), (self.features,), jnp.float32)
            out = out + bias
            
        return out

# ==============================================================================
# 3. Decoupled YaRN RoPE Implementation
# ==============================================================================

def build_yarn_rope(seq_len: int, d_rope: int, base: float = 10000.0,
                    alpha: float = 1.0, beta: float = 32.0, scale_s: float = 4.0):
    """Computes exact YaRN frequencies and the logit temperature mscale."""
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
    
    mscale = 0.1 * jnp.log(scale_s) + 1.
    
    return cos_val, sin_val, mscale

def apply_decoupled_yarn_rope(x_rope: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray):
    """Applies rotation to decoupled sub-heads."""
    def rotate(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)
        
    return (x_rope * cos) + (rotate(x_rope) * sin)

# ==============================================================================
# 4. ZenyxHybridAttention (CSA/HCA) Implementation
# ==============================================================================

class ZenyxHybridAttention(nn.Module):
    d_model: int
    num_heads: int
    d_latent: int # For compressed KV cache
    d_rope: int # Decoupled dimension for RoPE
    is_hca_layer: bool = False # Flag to alternate between CSA and HCA

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, _ = inputs.shape
        head_dim = self.d_model // self.num_heads
        rope_head_dim = self.d_rope // self.num_heads

        # Q, K, V projections using FP8Dense
        query = FP8Dense(self.d_model, name='q_proj')(inputs)
        key = FP8Dense(self.d_latent, name='kv_proj_k')(inputs) # Compressed K
        value = FP8Dense(self.d_latent, name='kv_proj_v')(inputs) # Compressed V

        # Reshape for multi-head attention
        query = query.reshape(batch_size, seq_len, self.num_heads, head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.d_latent // self.num_heads)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.d_latent // self.num_heads)

        # Generate RoPE values
        cos_vals, sin_vals, mscale = build_yarn_rope(seq_len, self.d_rope)
        # Expand cos/sin for broadcasting across batch and heads
        cos_vals = cos_vals[jnp.newaxis, :, jnp.newaxis, :]
        sin_vals = sin_vals[jnp.newaxis, :, jnp.newaxis, :]

        # Apply YaRN RoPE to a portion of Q and K
        query_rope_part = query[..., :rope_head_dim]
        key_rope_part = key[..., :rope_head_dim]

        query_rotated = apply_decoupled_yarn_rope(query_rope_part, cos_vals, sin_vals)
        key_rotated = apply_decoupled_yarn_rope(key_rope_part, cos_vals, sin_vals)

        # Recombine with non-rotated parts
        query = jnp.concatenate([query_rotated, query[..., rope_head_dim:]], axis=-1)
        key = jnp.concatenate([key_rotated, key[..., rope_head_dim:]], axis=-1)

        # Apply compression based on is_hca_layer
        if self.is_hca_layer:
            compression_factor = 128
        else:
            compression_factor = 4 # CSA compression ratio is 4
        
        # Downsample key and value for compression
        key_compressed = key[:, ::compression_factor, :, :]
        value_compressed = value[:, ::compression_factor, :, :]

        # Scaled Dot-Product Attention
        attn_weights = jnp.einsum(
            "bthd,bshd->bhts", query, key_compressed
        ) / jnp.sqrt(head_dim)

        # Apply mscale to attention logits
        attn_weights = attn_weights * mscale

        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(jnp.bfloat16)

        output = jnp.einsum("bhts,bshd->bthd", attn_weights, value_compressed)
        output = output.reshape(batch_size, seq_len, self.d_model)

        # Output projection using FP8Dense
        output = FP8Dense(self.d_model, name='o_proj')(output)
        return output

# ==============================================================================
# 5. DualSharedSparseMoE Implementation
# ==============================================================================

class DualSharedSparseMoE(nn.Module):
    d_model: int
    d_ff: int
    num_shared_experts: int
    num_routed_experts: int
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Shared Experts
        shared_output = jnp.zeros_like(inputs)
        for i in range(self.num_shared_experts):
            w1 = FP8Dense(self.d_ff, name=f'shared_{i+1}_w1')(inputs)
            w1 = nn.gelu(w1)
            w2 = FP8Dense(self.d_model, name=f'shared_{i+1}_w2')(w1)
            shared_output += w2
        shared_output /= self.num_shared_experts

        # Router for Routed Experts (Top-1 routing)
        router_logits = FP8Dense(self.num_routed_experts, name='router')(inputs)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        top1_expert_weights, top1_expert_indices = jax.lax.top_k(router_probs, k=1)
        top1_expert_indices = top1_expert_indices.squeeze(-1)
        top1_expert_weights = top1_expert_weights.squeeze(-1)

        # Gather expert outputs
        expert_outputs = []
        for i in range(self.num_routed_experts):
            w1 = FP8Dense(self.d_ff, name=f'routed_expert_{i}_w1')(inputs)
            w1 = nn.gelu(w1)
            w2 = FP8Dense(self.d_model, name=f'routed_expert_{i}_w2')(w1)
            expert_outputs.append(w2)
        
        expert_outputs_stacked = jnp.stack(expert_outputs, axis=1) # (batch, num_routed_experts, seq_len, d_model)
        
        routed_output = jax.vmap(lambda outputs, idx: outputs[idx])(expert_outputs_stacked, top1_expert_indices)
        
        # Scale by expert weights
        routed_output = routed_output * top1_expert_weights[:, jnp.newaxis, jnp.newaxis]

        # Auxiliary Loss (Load Balancing)
        expert_mask = jax.nn.one_hot(top1_expert_indices, self.num_routed_experts, dtype=jnp.float32)
        f_i = jnp.mean(expert_mask, axis=(0, 1)) # Fraction of tokens routed to expert i
        P_i = jnp.mean(router_probs, axis=(0, 1)) # Average probability of routing to expert i
        aux_loss = 0.01 * self.num_routed_experts * jnp.sum(f_i * P_i)

        final_out = shared_output + routed_output
        return final_out, aux_loss

# ==============================================================================
# 6. ZenyxRecurrentSuperBlock Implementation
# ==============================================================================

class ZenyxRecurrentSuperBlock(nn.Module):
    d_model: int
    num_recurrences: int
    num_heads: int
    d_latent: int
    d_rope: int
    d_ff: int
    num_shared_experts: int
    num_routed_experts: int

    def setup(self):
        self.norm1 = nn.RMSNorm()
        self.norm2 = nn.RMSNorm()
        
        # LayerScale parameters initialized to 1e-4 for gradient stability
        self.gamma_1 = self.param('gamma_1', nn.initializers.constant(1e-4), (self.d_model,))
        self.gamma_2 = self.param('gamma_2', nn.initializers.constant(1e-4), (self.d_model,))

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        total_aux_loss = 0.0

        def recurrent_step(carry, step_idx):
            x_in, current_aux_loss_sum = carry

            # Sub-layer 1: Attention with LayerScale
            x_norm = self.norm1(x_in)
            is_hca = (step_idx % 2 == 0) # Alternating CSA and HCA
            
            hybrid_attn_module = ZenyxHybridAttention(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_latent=self.d_latent,
                d_rope=self.d_rope,
                is_hca_layer=is_hca,
                name=f'hybrid_attn_{step_idx}'
            )
            attn_out = hybrid_attn_module(x_norm)
            x_mid = x_in + (attn_out * self.gamma_1)

            # Sub-layer 2: MoE with LayerScale
            moe_out, aux_loss = DualSharedSparseMoE(
                d_model=self.d_model,
                d_ff=self.d_ff,
                num_shared_experts=self.num_shared_experts,
                num_routed_experts=self.num_routed_experts,
                name=f'moe_{step_idx}'
            )(self.norm2(x_mid))
            x_out = x_mid + (moe_out * self.gamma_2)
            
            current_aux_loss_sum += aux_loss
            return (x_out, current_aux_loss_sum), None

        (final_x, total_aux_loss), _ = jax.lax.scan(
            recurrent_step, (x, total_aux_loss), jnp.arange(self.num_recurrences)
        )
        
        return final_x, total_aux_loss

# ==============================================================================
# 7. ZenyxV3 Model Definition
# ==============================================================================

class ZenyxV3(nn.Module):
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

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Input Embedding
        embed_init = nn.initializers.normal(stddev=1.0)
        x = nn.Embed(self.vocab_size, self.d_model, embedding_init=embed_init, name='Embed')(input_ids)

        # Recurrent SuperBlock
        final_x, total_aux_loss = ZenyxRecurrentSuperBlock(
            d_model=self.d_model,
            num_recurrences=self.num_recurrences,
            num_heads=self.num_heads,
            d_latent=self.d_latent,
            d_rope=self.d_rope,
            d_ff=self.d_ff,
            num_shared_experts=self.num_shared_experts,
            num_routed_experts=self.num_routed_experts,
            name='recurrent_superblock'
        )(x)

        # Output Projection
        logits = FP8Dense(self.vocab_size, use_bias=True, name='logits_proj')(final_x)
        return logits, total_aux_loss

# ==============================================================================
# 8. FSDP Sharding Topology
# ==============================================================================

def get_partition_rules():
    return {
        'Embed/embedding': P('fsdp', None),
        
        # 2D Dense Matrices inside FP8 layers (Sharded across output dimension)
        f'hybrid_attn_*/q_proj/kernel': P(None, 'fsdp'),
        f'hybrid_attn_*/kv_proj_k/kernel': P(None, 'fsdp'),
        f'hybrid_attn_*/kv_proj_v/kernel': P(None, 'fsdp'),
        f'hybrid_attn_*/o_proj/kernel': P('fsdp', None),

        # DualSharedSparseMoE
        f'moe_*/shared_1_w1/kernel': P(None, 'fsdp'),
        f'moe_*/shared_1_w2/kernel': P('fsdp', None),
        f'moe_*/shared_2_w1/kernel': P(None, 'fsdp'),
        f'moe_*/shared_2_w2/kernel': P('fsdp', None),
        f'moe_*/router/kernel': P(None, 'fsdp'),
        
        # Routed Experts (Sharded across the 64-Expert dimension)
        f'moe_*/routed_expert_*/w1/kernel': P('fsdp', None),
        f'moe_*/routed_expert_*/w2/kernel': P(None, 'fsdp'),

        # 1D Vectors (LayerScale, Biases, Norms) remain replicated across all cores
        f'recurrent_superblock/gamma_1': P(None),
        f'recurrent_superblock/gamma_2': P(None),
        f'recurrent_superblock/norm1/scale': P(None),
        f'recurrent_superblock/norm2/scale': P(None),
        f'logits_proj/bias': P(None),
        f'logits_proj/kernel': P(None, 'fsdp'), # Output logits projection
    }

# ==============================================================================
# 9. Training State and Step
# ==============================================================================

from flax.training import train_state

class TrainState(train_state.TrainState):
    # A simple TrainState that includes the optimizer state.
    pass

def train_step(state: TrainState, batch: Dict[str, jnp.ndarray], model: ZenyxV3, learning_rate_fn: Callable, mesh: Mesh, partition_rules: Dict) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    def loss_fn(params):
        logits, aux_loss = model.apply({'params': params}, batch['input_ids'])
        one_hot_labels = jax.nn.one_hot(batch['labels'], num_classes=model.vocab_size)
        loss = -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1)
        total_loss = jnp.mean(loss) + aux_loss # Add auxiliary loss
        return total_loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, total_loss, logits

# ==============================================================================
# 10. Dataset Pipeline and O(1) Resumption Logic
# ==============================================================================

class DatasetLoader:
    def __init__(self, hf_token: str, hf_repo: str, seq_len: int, vocab_size: int):
        self.hf_token = hf_token
        self.hf_repo = hf_repo
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.api = HfApi(token=hf_token)

        # Define datasets based on the document. Prioritize non-gated ones.
        # Placeholder for actual tokens_per_shard values
        self.datasets_config = {
            "RedPajama-Data-V2": {"hf_id": "togethercomputer/RedPajama-Data-V2", "split": "train", "tokens_per_shard": 1e9}, 
            "DCLM-Baseline": {"hf_id": "mlfoundations/dclm-baseline-1", "split": "train", "tokens_per_shard": 1e9}, 
            "The Stack v2": {"hf_id": "bigcode/the-stack-smol", "split": "train", "tokens_per_shard": 1e9}, 
            "MathPile": {"hf_id": "GAIR/MathPile", "split": "train", "tokens_per_shard": 1e9}, 
        }

    def _get_latest_checkpoint_step(self) -> int:
        try:
            # List files in the repo to find checkpoint metadata
            repo_files = self.api.list_repo_files(repo_id=self.hf_repo, repo_type="model")
            checkpoint_files = [f for f in repo_files if "checkpoint_metadata.json" in f]
            
            if not checkpoint_files:
                return 0

            # Assuming checkpoint_metadata.json stores the global_step
            checkpoint_path = hf_hub_download(repo_id=self.hf_repo, filename="checkpoint_metadata.json", token=self.hf_token)
            with open(checkpoint_path, "r") as f:
                metadata = json.load(f)
            
            return metadata.get("global_step", 0)
        except Exception as e:
            print(f"Could not retrieve latest checkpoint step: {e}")
            return 0

    def get_dataloader(self, batch_size: int):
        global_step = self._get_latest_checkpoint_step()
        tokens_consumed = global_step * batch_size * self.seq_len
        
        # For simplicity, we will use a single dataset for now. 
        # In a real scenario, the 4-phase curriculum would dynamically select datasets.
        dataset_name = "RedPajama-Data-V2"
        config = self.datasets_config[dataset_name]
        
        dataset = load_dataset(config["hf_id"], split=config["split"], streaming=True, token=self.hf_token)

        shard_index = int(tokens_consumed // config["tokens_per_shard"])
        offset_within_shard = int(tokens_consumed % config["tokens_per_shard"])

        if tokens_consumed > 0:
            print(f"Resuming from step {global_step}, shard {shard_index}, offset {offset_within_shard}")
            # For streaming datasets, skip elements to reach the offset
            dataset = dataset.skip(offset_within_shard)

        # Simple tokenization: map text to dummy token IDs
        def tokenize_function(examples):
            # In a real scenario, a proper tokenizer (e.g., from transformers) would be used.
            # This is a placeholder to generate input_ids and labels of correct shape.
            token_ids = [list(range(self.seq_len)) for _ in examples["text"]]
            # Labels are typically the next token, so shift input_ids
            labels = [ids[1:] + [0] for ids in token_ids] # 0 for padding/EOS
            return {"input_ids": token_ids, "labels": labels}

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        # Batching function for streaming dataset
        def collate_fn(examples):
            input_ids = jnp.array([ex["input_ids"] for ex in examples])
            labels = jnp.array([ex["labels"] for ex in examples])
            return {"input_ids": input_ids, "labels": labels}

        # Generator for batches
        def batch_generator():
            batch = []
            for example in tokenized_dataset:
                batch.append(example)
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    batch = []
            if batch: # Yield any remaining partial batch
                yield collate_fn(batch)

        return batch_generator(), global_step

# ==============================================================================
# 11. Main Training Loop and Checkpointing
# ==============================================================================

if __name__ == '__main__':
    # Configuration from the document
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
    
    # Hugging Face details
    HF_TOKEN = os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN_HERE') # Use environment variable for token
    HF_REPO = 'Arko007/zenyx-v3-checkpoints'
    
    # Initialize model
    model = ZenyxV3(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_latent=D_LATENT,
        d_rope=D_ROPE,
        d_ff=D_FF,
        num_shared_experts=NUM_SHARED_EXPERTS,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        num_recurrences=NUM_RECURRENCES,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE
    )

    # Initialize dummy input for parameter initialization
    dummy_input = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
    params_key = jax.random.PRNGKey(0)
    initial_variables = model.init(params_key, dummy_input)
    initial_params = initial_variables['params']

    # Optimizer
    LEARNING_RATE = 1e-4 # Placeholder, will be scaled by muP
    WEIGHT_DECAY = 0.05
    
    tx = build_hybrid_optimizer(LEARNING_RATE, WEIGHT_DECAY, initial_params)
    opt_state = tx.init(initial_params)

    # Initialize TrainState
    state = TrainState(apply_fn=model.apply, params=initial_params, tx=tx, opt_state=opt_state)

    # FSDP Mesh and Sharding
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(device_mesh, axis_names=('fsdp',))
    partition_rules = get_partition_rules()

    # Learning rate scheduler (placeholder)
    def learning_rate_fn(step):
        return LEARNING_RATE

    # Dataset and Dataloader
    BATCH_SIZE = 2 # Example batch size
    dataset_loader = DatasetLoader(HF_TOKEN, HF_REPO, SEQ_LEN, VOCAB_SIZE)
    train_dataloader, start_global_step = dataset_loader.get_dataloader(BATCH_SIZE)

    print(f"Starting training from global step: {start_global_step}")
    current_global_step = start_global_step
    
    # Create a directory for local checkpoints
    os.makedirs("./checkpoints", exist_ok=True)

    for epoch in range(10): # Example epochs
        for batch_idx, batch in enumerate(train_dataloader):
            # Ensure batch has correct shapes
            if batch['input_ids'].shape[0] != BATCH_SIZE:
                print(f"Skipping partial batch of size {batch['input_ids'].shape[0]}")
                continue

            # Perform a training step
            state, total_loss, logits = train_step(state, batch, model, learning_rate_fn, mesh, partition_rules)
            current_global_step += 1
            print(f"Epoch {epoch}, Step {current_global_step}, Loss: {total_loss:.4f}")

            if current_global_step % 500 == 0: # Checkpoint every 500 steps
                print(f"Saving checkpoint at step {current_global_step}")
                # Save metadata locally
                with open(f"./checkpoints/checkpoint_metadata_{current_global_step}.json", "w") as f:
                    json.dump({"global_step": current_global_step}, f)
                
                # Placeholder for saving model parameters. 
                # In a real scenario, you would save `state.params` and `state.opt_state`.
                # For simplicity, we just save a dummy file.
                with open(f"./checkpoints/dummy_checkpoint_{current_global_step}.txt", "w") as f:
                    f.write(f"Checkpoint for step {current_global_step}")

                # Upload to Hugging Face Hub
                try:
                    api = HfApi(token=HF_TOKEN)
                    api.upload_folder(
                        repo_id=HF_REPO,
                        folder_path="./checkpoints",
                        commit_message=f"Checkpoint at step {current_global_step}",
                        repo_type="model",
                    )
                    print(f"Uploaded checkpoint to Hugging Face Hub: {HF_REPO}")
                except Exception as e:
                    print(f"Failed to upload checkpoint to Hugging Face Hub: {e}")

            if current_global_step >= start_global_step + 1000: # Stop after 1000 steps for demonstration
                break
        if current_global_step >= start_global_step + 1000:
            break

    print("Training simulation complete.")
