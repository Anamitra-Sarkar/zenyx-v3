#!/usr/bin/env python3
"""CPU inference demo for Zenyx-V3 with a T4 GPU script appendix."""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from zenyx_v3_final import ZenyxConfig, ZenyxV3Model, precompute_freqs_cis


HF_REPO_ID = "Arko007/Zenyx-V3-Base"
TOKENIZER_REPO_ID = "deepseek-ai/DeepSeek-V4-Pro"
PROMPT = "The future of artificial intelligence is"


def list_latest_step(repo_id: str) -> int:
    api = HfApi()
    files = api.list_repo_files(repo_id)
    steps = []
    for filename in files:
        match = re.fullmatch(r"checkpoints/step_(\d+)/params\.pkl", filename)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise RuntimeError(f"No checkpoints found in {repo_id}")
    return max(steps)


def download_latest_params(repo_id: str = HF_REPO_ID) -> tuple[int, str]:
    step = list_latest_step(repo_id)
    remote_path = f"checkpoints/step_{step}/params.pkl"
    local_path = hf_hub_download(repo_id=repo_id, filename=remote_path)
    return step, local_path


def load_tokenizer() -> AutoTokenizer:
    for repo_id, kwargs in (
        (TOKENIZER_REPO_ID, {}),
        (HF_REPO_ID, {"subfolder": "tokenizer"}),
    ):
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=False, **kwargs)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception:
            continue
    raise RuntimeError("Failed to load tokenizer from both tokenizer sources")


def load_params(params_path: str):
    with open(params_path, "rb") as f:
        params_np = pickle.load(f)

    if isinstance(params_np, dict) and set(params_np.keys()) == {"params"}:
        params_np = params_np["params"]

    def convert(x):
        if isinstance(x, np.ndarray):
            arr = jnp.asarray(x)
            if np.issubdtype(x.dtype, np.floating):
                return arr.astype(jnp.bfloat16)
            return arr
        return x

    params = jax.tree_util.tree_map(convert, params_np)
    return params


def generate_greedy(model, params, tokenizer, prompt: str, num_new_tokens: int) -> tuple[list[int], str]:
    cfg = ZenyxConfig()
    input_ids = tokenizer(prompt, return_tensors="np", add_special_tokens=False).input_ids.astype(np.int32)
    token_ids = jnp.asarray(input_ids)
    generated_ids: list[int] = []

    for _ in range(num_new_tokens):
        seq_len = int(token_ids.shape[1])
        cos, sin = precompute_freqs_cis(cfg.rope_head_dim, seq_len, cfg.rope_theta)
        logits, _, _ = model.apply({"params": params}, token_ids, cos, sin, deterministic=True)
        next_token = int(jnp.argmax(logits[:, -1, :], axis=-1)[0])
        generated_ids.append(next_token)
        token_ids = jnp.concatenate([token_ids, jnp.array([[next_token]], dtype=jnp.int32)], axis=1)

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return generated_ids, decoded


def main() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    cfg = ZenyxConfig()
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())

    tokenizer = load_tokenizer()
    step, params_path = download_latest_params()
    print(f"latest checkpoint step: {step}")
    print(f"params path: {params_path}")

    params = load_params(params_path)
    model = ZenyxV3Model(cfg)

    generated_ids, decoded = generate_greedy(model, params, tokenizer, PROMPT, 2)
    token_pieces = [
        tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for token_id in generated_ids
    ]

    print(f"prompt: {PROMPT}")
    print(f"generated token ids: {generated_ids}")
    print(f"generated token pieces: {token_pieces}")
    print(f"generated text: {decoded!r}")


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# T4 GPU / Kaggle / Colab standalone script
# -----------------------------------------------------------------------------
T4_GPU_SCRIPT = r'''
#!/usr/bin/env python3
# pip install -U "jax[cuda12]" flax optax transformers datasets huggingface_hub sentencepiece

import os
import pickle
import re

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from zenyx_v3_final import ZenyxConfig, ZenyxV3Model, precompute_freqs_cis

HF_REPO_ID = "Arko007/Zenyx-V3-Base"
TOKENIZER_REPO_ID = "deepseek-ai/DeepSeek-V4-Pro"
PROMPT = "The future of artificial intelligence is"


def latest_step(repo_id: str) -> int:
    files = HfApi().list_repo_files(repo_id)
    steps = [
        int(m.group(1))
        for path in files
        if (m := re.fullmatch(r"checkpoints/step_(\d+)/params\.pkl", path))
    ]
    return max(steps)


def load_tokenizer():
    for repo_id, kwargs in (
        (TOKENIZER_REPO_ID, {}),
        (HF_REPO_ID, {"subfolder": "tokenizer"}),
    ):
        try:
            tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=False, **kwargs)
            if tok.pad_token_id is None and tok.eos_token_id is not None:
                tok.pad_token = tok.eos_token
            return tok
        except Exception:
            pass
    raise RuntimeError("Tokenizer load failed")


def load_params(repo_id: str, step: int):
    path = hf_hub_download(repo_id=repo_id, filename=f"checkpoints/step_{step}/params.pkl")
    with open(path, "rb") as f:
        params_np = pickle.load(f)

    if isinstance(params_np, dict) and set(params_np.keys()) == {"params"}:
        params_np = params_np["params"]

    def convert(x):
        if isinstance(x, np.ndarray):
            arr = jnp.asarray(x)
            if np.issubdtype(x.dtype, np.floating):
                return arr.astype(jnp.bfloat16)
            return arr
        return x

    params = jax.tree_util.tree_map(convert, params_np)
    return jax.device_put(params, jax.devices()[0])


def greedy_generate(model, params, tokenizer, prompt: str, steps: int):
    cfg = ZenyxConfig()
    ids = tokenizer(prompt, return_tensors="np", add_special_tokens=False).input_ids.astype(np.int32)
    tokens = jnp.asarray(ids)
    out_ids = []
    for _ in range(steps):
        seq_len = int(tokens.shape[1])
        cos, sin = precompute_freqs_cis(cfg.rope_head_dim, seq_len, cfg.rope_theta)
        logits, _, _ = model.apply({"params": params}, tokens, cos, sin, deterministic=True)
        next_id = int(jnp.argmax(logits[:, -1, :], axis=-1)[0])
        out_ids.append(next_id)
        tokens = jnp.concatenate([tokens, jnp.array([[next_id]], dtype=jnp.int32)], axis=1)
    return out_ids, tokenizer.decode(out_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def main():
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    tokenizer = load_tokenizer()
    step = latest_step(HF_REPO_ID)
    params = load_params(HF_REPO_ID, step)
    model = ZenyxV3Model(ZenyxConfig())
    two_ids, two_text = greedy_generate(model, params, tokenizer, PROMPT, 2)
    ten_ids, ten_text = greedy_generate(model, params, tokenizer, PROMPT, 10)
    print("latest step:", step)
    print("2-token ids:", two_ids)
    print("2-token text:", repr(two_text))
    print("10-token ids:", ten_ids)
    print("10-token text:", repr(ten_text))


if __name__ == "__main__":
    main()
'''

# Verified CPU run:
# backend: cpu
# devices: [CpuDevice(id=0)]
# latest checkpoint step: 6900
# generated token ids: [260, 1017]
# generated token pieces: [' a', ' new']
# generated text: ' a new'
