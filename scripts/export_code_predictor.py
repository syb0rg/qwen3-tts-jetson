#!/usr/bin/env python3
"""Export Qwen3-TTS code predictor transformer layers to ONNX for TRT conversion.

Exports the 5 shared transformer layers with explicit KV cache I/O.
The per-codebook embedding lookups and lm_heads stay in C++.

Usage:
    python3 export_code_predictor.py -i models/Qwen3-TTS-12Hz-0.6B-Base -o models/code_pred.onnx
"""

import argparse
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from pathlib import Path


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CodePredLayersExport(nn.Module):
    """Exportable 5-layer transformer decoder with explicit KV cache."""

    def __init__(self, config, state_dict_prefix, state_dict):
        super().__init__()
        self.n_layers = config["num_hidden_layers"]
        self.hidden_size = config["hidden_size"]
        self.n_heads = config["num_attention_heads"]
        self.n_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.intermediate_size = config["intermediate_size"]
        self.rms_norm_eps = config["rms_norm_eps"]
        self.rope_theta = config.get("rope_theta", 1000000.0)

        # Pre-compute RoPE frequencies for max_seq=16
        self.max_seq = 16
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        ))
        t = torch.arange(self.max_seq, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs).unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", torch.sin(freqs).unsqueeze(0).unsqueeze(0))

        p = state_dict_prefix
        self.input_layernorms = nn.ModuleList()
        self.post_attn_layernorms = nn.ModuleList()
        self.q_projs = nn.ModuleList()
        self.k_projs = nn.ModuleList()
        self.v_projs = nn.ModuleList()
        self.o_projs = nn.ModuleList()
        self.q_norms = nn.ModuleList()
        self.k_norms = nn.ModuleList()
        self.gate_projs = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        self.down_projs = nn.ModuleList()

        for il in range(self.n_layers):
            lp = f"{p}layers.{il}."

            iln = RMSNorm(self.hidden_size, self.rms_norm_eps)
            iln.weight.data = state_dict[lp + "input_layernorm.weight"].float()
            self.input_layernorms.append(iln)

            paln = RMSNorm(self.hidden_size, self.rms_norm_eps)
            paln.weight.data = state_dict[lp + "post_attention_layernorm.weight"].float()
            self.post_attn_layernorms.append(paln)

            q = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
            q.weight.data = state_dict[lp + "self_attn.q_proj.weight"].float()
            self.q_projs.append(q)

            k = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
            k.weight.data = state_dict[lp + "self_attn.k_proj.weight"].float()
            self.k_projs.append(k)

            v = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
            v.weight.data = state_dict[lp + "self_attn.v_proj.weight"].float()
            self.v_projs.append(v)

            o = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)
            o.weight.data = state_dict[lp + "self_attn.o_proj.weight"].float()
            self.o_projs.append(o)

            qn = RMSNorm(self.head_dim, self.rms_norm_eps)
            qn.weight.data = state_dict[lp + "self_attn.q_norm.weight"].float()
            self.q_norms.append(qn)

            kn = RMSNorm(self.head_dim, self.rms_norm_eps)
            kn.weight.data = state_dict[lp + "self_attn.k_norm.weight"].float()
            self.k_norms.append(kn)

            gate = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            gate.weight.data = state_dict[lp + "mlp.gate_proj.weight"].float()
            self.gate_projs.append(gate)

            up = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            up.weight.data = state_dict[lp + "mlp.up_proj.weight"].float()
            self.up_projs.append(up)

            down = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            down.weight.data = state_dict[lp + "mlp.down_proj.weight"].float()
            self.down_projs.append(down)

        self.output_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.output_norm.weight.data = state_dict[f"{p}norm.weight"].float()

    def forward(self, hidden_states, position_id,
                past_key_0, past_value_0,
                past_key_1, past_value_1,
                past_key_2, past_value_2,
                past_key_3, past_value_3,
                past_key_4, past_value_4):
        """Fixed-size KV cache: write new K/V at position_id, attend to 0..position_id.

        Args:
            hidden_states: [1, 1, hidden_size]
            position_id: [1] int64 - write position in KV cache
            past_key_i: [1, n_kv_heads, MAX_KV, head_dim] - fixed-size KV cache
            past_value_i: [1, n_kv_heads, MAX_KV, head_dim]

        Returns:
            output: [1, 1, hidden_size]
            present_key_i: [1, n_kv_heads, MAX_KV, head_dim] - updated KV cache (same shape)
            present_value_i: [1, n_kv_heads, MAX_KV, head_dim]
        """
        past_keys = [past_key_0, past_key_1, past_key_2, past_key_3, past_key_4]
        past_values = [past_value_0, past_value_1, past_value_2, past_value_3, past_value_4]
        present_keys = []
        present_values = []

        pos_idx = position_id.long()
        cos = self.cos_cached[:, :, pos_idx, :]
        sin = self.sin_cached[:, :, pos_idx, :]
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

        cur = hidden_states

        for il in range(self.n_layers):
            residual = cur
            cur = self.input_layernorms[il](cur)

            q = self.q_projs[il](cur)
            k_new = self.k_projs[il](cur)
            v_new = self.v_projs[il](cur)

            q = q.view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
            k_new = k_new.view(1, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v_new = v_new.view(1, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)

            q = self.q_norms[il](q)
            k_new = self.k_norms[il](k_new)

            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

            # Write new K/V into fixed-size cache at position_id using scatter
            k_cache = past_keys[il].clone()
            v_cache = past_values[il].clone()
            idx = pos_idx.view(1, 1, 1, 1).expand(1, self.n_kv_heads, 1, self.head_dim)
            k_cache.scatter_(2, idx, k_new)
            v_cache.scatter_(2, idx, v_new)

            present_keys.append(k_cache)
            present_values.append(v_cache)

            # GQA expansion
            n_rep = self.n_heads // self.n_kv_heads
            if n_rep > 1:
                k_expanded = k_cache.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
                k_expanded = k_expanded.reshape(1, self.n_heads, self.max_seq, self.head_dim)
                v_expanded = v_cache.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
                v_expanded = v_expanded.reshape(1, self.n_heads, self.max_seq, self.head_dim)
            else:
                k_expanded = k_cache
                v_expanded = v_cache

            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale

            # Causal mask: attend only to positions 0..position_id (inclusive)
            kv_positions = torch.arange(self.max_seq, device=hidden_states.device)
            mask = kv_positions.unsqueeze(0) > pos_idx.unsqueeze(-1)
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v_expanded)

            attn_output = attn_output.transpose(1, 2).contiguous().view(1, 1, -1)
            cur = self.o_projs[il](attn_output)
            cur = cur + residual

            residual = cur
            cur = self.post_attn_layernorms[il](cur)
            gate = F.silu(self.gate_projs[il](cur))
            up = self.up_projs[il](cur)
            cur = self.down_projs[il](gate * up)
            cur = cur + residual

        cur = self.output_norm(cur)

        return (cur,
                present_keys[0], present_values[0],
                present_keys[1], present_values[1],
                present_keys[2], present_values[2],
                present_keys[3], present_values[3],
                present_keys[4], present_values[4])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Model directory")
    parser.add_argument("-o", "--output", required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    model_dir = Path(args.input)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    cp_config = config["talker_config"]["code_predictor_config"]
    print(f"Code predictor: {cp_config['num_hidden_layers']} layers, "
          f"hidden={cp_config['hidden_size']}, heads={cp_config['num_attention_heads']}/"
          f"{cp_config['num_key_value_heads']}, head_dim={cp_config['head_dim']}, "
          f"ffn={cp_config['intermediate_size']}")

    import glob
    st_files = sorted(glob.glob(str(model_dir / "model*.safetensors")))
    state_dict = {}
    for sf in st_files:
        sd = load_file(sf)
        for k, v in sd.items():
            if "code_predictor.model." in k:
                short_key = k.replace("talker.code_predictor.model.", "")
                state_dict[short_key] = v.float()

    print(f"Loaded {len(state_dict)} code predictor model tensors")

    model = CodePredLayersExport(cp_config, "", state_dict)
    model.float()
    model.eval()

    hidden_size = cp_config["hidden_size"]
    n_kv_heads = cp_config["num_key_value_heads"]
    head_dim = cp_config["head_dim"]

    # Use fixed KV cache size of 16 (max possible: 15 codebook steps + 1 prefill)
    # We'll pad shorter sequences with zeros; causal mask handles correctness
    MAX_KV = 16
    n_past = MAX_KV

    dummy_hidden = torch.randn(1, 1, hidden_size)
    dummy_pos = torch.tensor([2], dtype=torch.int64)  # position index
    dummy_past_k = [torch.randn(1, n_kv_heads, n_past, head_dim) for _ in range(5)]
    dummy_past_v = [torch.randn(1, n_kv_heads, n_past, head_dim) for _ in range(5)]

    dummy_inputs = (dummy_hidden, dummy_pos,
                    dummy_past_k[0], dummy_past_v[0],
                    dummy_past_k[1], dummy_past_v[1],
                    dummy_past_k[2], dummy_past_v[2],
                    dummy_past_k[3], dummy_past_v[3],
                    dummy_past_k[4], dummy_past_v[4])

    with torch.no_grad():
        outputs = model(*dummy_inputs)
    print(f"Forward pass OK: output shape = {outputs[0].shape}")
    print(f"Present KV shape: {outputs[1].shape}")

    input_names = ["hidden_states", "position_id"]
    output_names = ["output"]

    for il in range(5):
        input_names.extend([f"past_key_{il}", f"past_value_{il}"])
        output_names.extend([f"present_key_{il}", f"present_value_{il}"])

    # Use legacy torch.onnx exporter (PyTorch 2.10 dynamo exporter has issues with fixed shapes)
    print(f"Exporting to ONNX (fixed KV={MAX_KV}): {args.output}")
    torch.onnx.export(
        model,
        dummy_inputs,
        args.output,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,  # use legacy exporter
    )
    print(f"ONNX export complete: {args.output}")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.output)
        feed = {
            "hidden_states": dummy_hidden.numpy(),
            "position_id": dummy_pos.numpy(),
        }
        for il in range(5):
            feed[f"past_key_{il}"] = dummy_past_k[il].numpy()
            feed[f"past_value_{il}"] = dummy_past_v[il].numpy()

        ort_out = sess.run(None, feed)
        pt_out = outputs[0].numpy()
        diff = abs(ort_out[0] - pt_out).max()
        print(f"ONNX verification: max diff = {diff:.6f}")
    except Exception as e:
        print(f"ONNX verification skipped: {e}")


if __name__ == "__main__":
    main()
