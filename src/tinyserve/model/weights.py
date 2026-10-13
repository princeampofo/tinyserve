"""Load HuggingFace Llama checkpoints into our own weight dataclasses.

A HF checkpoint is a dict of named tensors (`model.layers.3.self_attn.q_proj.weight`
...). All this file does is pick out each tensor by name, move it to the device
in the requested dtype, and put it in the matching field.
"""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from tinyserve.model.llama import LayerWeights, LlamaConfig, LlamaWeights

# A gated repo: request access on its HuggingFace page, then `hf auth login`.
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def weights_from_state_dict(
    config: LlamaConfig,
    state: dict[str, torch.Tensor],
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LlamaWeights:
    def get(name: str) -> torch.Tensor:
        return state[name].to(device=device, dtype=dtype)

    layers = []
    for i in range(config.num_layers):
        p = f"model.layers.{i}."
        layers.append(
            LayerWeights(
                attn_norm=get(p + "input_layernorm.weight"),
                wq=get(p + "self_attn.q_proj.weight"),
                wk=get(p + "self_attn.k_proj.weight"),
                wv=get(p + "self_attn.v_proj.weight"),
                wo=get(p + "self_attn.o_proj.weight"),
                mlp_norm=get(p + "post_attention_layernorm.weight"),
                w_gate=get(p + "mlp.gate_proj.weight"),
                w_up=get(p + "mlp.up_proj.weight"),
                w_down=get(p + "mlp.down_proj.weight"),
            )
        )

    embed = get("model.embed_tokens.weight")
    # Llama 3.2 1B/3B tie the output projection to the input embedding (saves
    # 128256 x 2048 = 262M parameters on 1B), so checkpoints omit lm_head.
    lm_head = embed if config.tie_word_embeddings else get("lm_head.weight")
    return LlamaWeights(
        embed=embed, layers=layers, final_norm=get("model.norm.weight"), lm_head=lm_head
    )


def load_model(
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[LlamaConfig, LlamaWeights]:
    """Load from a local directory or a HuggingFace repo id (downloaded and cached)."""
    path = Path(model)
    if not path.is_dir():
        path = Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors"]))

    generation_path = path / "generation_config.json"
    config = LlamaConfig.from_hf_dict(
        json.loads((path / "config.json").read_text()),
        json.loads(generation_path.read_text()) if generation_path.exists() else None,
    )
    state: dict[str, torch.Tensor] = {}
    for f in sorted(path.glob("*.safetensors")):
        state.update(load_file(f))
    return config, weights_from_state_dict(config, state, device, dtype)
