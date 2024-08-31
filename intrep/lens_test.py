import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import numpy as np

import transformer_lens
import einops

from searchformer.transformer import JustDecoder, HyperParams

def just_decoder_to_lens_format(model, cfg: transformer_lens.HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = model.decoder.embedding.weight
    # state_dict["pos_embed.w_pos"] = model.decoder.pos_emb.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = model.decoder.layers[l].attention_norm.weight

        W_Q = model.decoder.layers[l].attention.wq.weight.T
        W_K = model.decoder.layers[l].attention.wk.weight.T
        W_V = model.decoder.layers[l].attention.wv.weight.T

        #rearrange
        W_Q = einops.rearrange(W_Q, "m (i h) -> i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m (i h) -> i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m (i h) -> i m h", i=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        W_O = model.decoder.layers[l].attention.wo.weight.T
        W_O = einops.rearrange(W_O,"(i h) m -> i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        #zeros
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros((cfg.n_heads, cfg.d_head))
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros((cfg.n_heads, cfg.d_head))
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros((cfg.n_heads, cfg.d_head))
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros((cfg.d_model))

        if cfg.attn_only:
            continue
        state_dict[f"blocks.{l}.ln2.w"] = model.decoder.layers[l].ffn_norm.weight

        W_gate = model.decoder.layers[l].feed_forward.w1.weight
        W_out = model.decoder.layers[l].feed_forward.w2.weight
        W_in = model.decoder.layers[l].feed_forward.w3.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = W_in.T
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = W_gate.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(W_in.shape[0])
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(W_out.shape[0])
        state_dict[f"blocks.{l}.mlp.b_gate"] = torch.zeros(W_gate.shape[0])

    state_dict["unembed.W_U"] = model.decoder.output[1].weight.T
    state_dict["unembed.b_U"] = torch.zeros(model.decoder.output[1].weight.shape[0])
    state_dict["ln_final.w"] = model.decoder.output[0].weight

    #save it to a checkpoint
    torch.save(state_dict, "checkpoints/lens_decoder_searchformer.ckpt")
    return state_dict

@dataclass
class SearchFormerDecoderConfig(transformer_lens.HookedTransformerConfig):
    n_layers: int = 6 # 4
    d_model: int = 192 # 124
    n_ctx: int = 2**16 # 10000
    d_head: int = 64
    n_heads: int = 3 # 2
    d_mlp: int = int(256 * ((8 * 192/3 + 256 - 1)//256)) #make 192 124
    act_fn: str = "silu"
    d_vocab: int = 116+2
    # weight_init_mode:str = "kaiming_uniform"
    normalization_type="RMS"
    positional_embedding_type: str = "rotary"
    attn_only: bool = False
    # rotary_dim: int = 192//3 
    # rotary_base: int = 10000

def load_pretrained_decoder_searchformer(checkpoint_dir="checkpoints/onlydecoderrun1.ckpt"):
    lens_cfg = SearchFormerDecoderConfig()
    hp_cfg = HyperParams.from_name("dec-s", 116+2)

    #setup some stuff here
    model = JustDecoder(hp_cfg)
    state_dict = torch.load(checkpoint_dir)["model"]
    #rename the keys by removing the module.model prefix
    state_dict = {k.replace("module.model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    lens_state_dict = just_decoder_to_lens_format(model, lens_cfg)

def maze_move(plan, maze, start, goal, idx=0):
    curr_color = np.array([255, 255, 0])

    cur_pos = np.where((maze[0] == 255) & (maze[1] == 255))
    maze[cur_pos[0], cur_pos[1]] = (255, 255, 255)

    maze[plan[idx][1], plan[idx][2]] = curr_color
    return maze


if __name__=="__main__":
    load_pretrained_decoder_searchformer()
    