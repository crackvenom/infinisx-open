# InfinisX — Flux ControlNet for the Forge Neo engine (AGPL-3.0, like the engine).
#
# Forge Neo's ControlNet loader only builds SD 1.5 / SDXL ControlNets, so a
# Guide on a Flux model did nothing (founder, 2026-10-22). This is our own
# implementation of the Flux ControlNet architecture (the one InstantX and
# Shakker-Labs Union Pro 2.0 are trained in, diffusers' FluxControlNetModel):
#
#   a short Flux: N double-stream blocks, no single-stream blocks, whose
#   image input is   img_in(noisy latent) + controlnet_x_embedder(drawing latent)
#   and whose N outputs, each through a zero-initialised linear
#   (controlnet_blocks), are added to the main Flux's double blocks — block j
#   gets output j // ceil(19 / N) (diffusers' "interval" rule).
#
# The weights are the user's own download; nothing here is copied from them.
# The blocks are the engine's own Flux blocks (backend/nn/flux.py), so attention,
# RoPE, modulation and low-VRAM casting behave exactly as in the main model.

import math

import torch
from einops import rearrange, repeat
from torch import nn

from backend.nn.flux import DoubleStreamBlock, EmbedND, MLPEmbedder, timestep_embedding

# Key prefixes that only a Flux ControlNet (diffusers layout) has.
SIGNATURE_KEYS = ("controlnet_x_embedder.weight", "transformer_blocks.0.attn.to_q.weight")


def is_flux_controlnet(sd: dict) -> bool:
    return all(k in sd for k in SIGNATURE_KEYS) and "controlnet_blocks.0.weight" in sd


def config_from_state_dict(sd: dict) -> dict:
    """Architecture from the weights alone (no config.json needed)."""
    hidden = sd["x_embedder.weight"].shape[0]
    head_dim = sd["transformer_blocks.0.attn.norm_q.weight"].shape[0]
    depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("transformer_blocks."))
    single = sum(1 for k in sd if k.startswith("single_transformer_blocks.") and k.endswith(".proj_out.weight"))
    mode = sd.get("controlnet_mode_embedder.weight")
    return dict(
        in_channels=sd["x_embedder.weight"].shape[1],
        hidden_size=hidden,
        num_heads=hidden // head_dim,
        depth=depth,
        depth_single=single,
        context_in_dim=sd["context_embedder.weight"].shape[1],
        vec_in_dim=sd["time_text_embed.text_embedder.linear_1.weight"].shape[1],
        mlp_ratio=sd["transformer_blocks.0.ff.net.0.proj.weight"].shape[0] / hidden,
        guidance_embed="time_text_embed.guidance_embedder.linear_1.weight" in sd,
        num_modes=None if mode is None else mode.shape[0],
        # Flux's RoPE split of the 128-dim head (time / height / width).
        axes_dim=[16, 56, 56] if head_dim == 128 else _axes_for(head_dim),
    )


# Flux.1's own shape: a ControlNet for anything else can't add into its blocks.
FLUX1 = dict(in_channels=64, hidden_size=3072, num_heads=24, context_in_dim=4096, vec_in_dim=768)


def validate_flux1(cfg: dict) -> None:
    """Refuse at load time what would only fail mid-render (a Flux 2 / Chroma ControlNet, a broken file)."""
    wrong = [f"{k} {cfg.get(k)} (Flux.1: {v})" for k, v in FLUX1.items() if cfg.get(k) != v]
    if wrong:
        raise ValueError("not a Flux.1 ControlNet — " + ", ".join(wrong))
    if not 1 <= cfg["depth"] <= 19:
        raise ValueError(f"{cfg['depth']} ControlNet blocks (Flux.1 has 19)")
    if cfg["depth_single"]:
        raise ValueError("ControlNets with single-stream blocks are not supported yet")


# Union Pro 1.0's control modes (InstantX / Shakker-Labs model card); Pro 2.0 has none.
UNION_MODES = {"canny": 0, "tile": 1, "depth": 2, "blur": 3, "pose": 4, "gray": 5, "lq": 6}


def union_mode_for(module: str | None) -> int:
    """The Union Pro 1.0 mode for a Forge preprocessor: edges and line art read as canny."""
    m = (module or "").lower()
    if "depth" in m:
        return UNION_MODES["depth"]
    if "pose" in m:
        return UNION_MODES["pose"]
    if "tile" in m:
        return UNION_MODES["tile"]
    if "blur" in m:
        return UNION_MODES["blur"]
    if "gray" in m or "recolor" in m:
        return UNION_MODES["gray"]
    return UNION_MODES["canny"]


def _axes_for(head_dim: int) -> list[int]:
    # Flux's 1 : 3.5 : 3.5 split (16, 56, 56 for 128), every part even for RoPE.
    a = 2 * round(head_dim / 16)
    b = 2 * ((head_dim - a) // 4)
    return [a, b, head_dim - a - b]


def convert_diffusers(sd: dict, depth: int) -> dict:
    """diffusers FluxControlNetModel keys → this module's (the engine's Flux names)."""
    out = {}

    def take(src, dst):
        out[dst] = sd[src]

    def lin(src, dst):
        take(f"{src}.weight", f"{dst}.weight")
        if f"{src}.bias" in sd:
            take(f"{src}.bias", f"{dst}.bias")

    lin("x_embedder", "img_in")
    lin("controlnet_x_embedder", "controlnet_x_embedder")
    lin("context_embedder", "txt_in")
    lin("time_text_embed.timestep_embedder.linear_1", "time_in.in_layer")
    lin("time_text_embed.timestep_embedder.linear_2", "time_in.out_layer")
    lin("time_text_embed.text_embedder.linear_1", "vector_in.in_layer")
    lin("time_text_embed.text_embedder.linear_2", "vector_in.out_layer")
    if "time_text_embed.guidance_embedder.linear_1.weight" in sd:
        lin("time_text_embed.guidance_embedder.linear_1", "guidance_in.in_layer")
        lin("time_text_embed.guidance_embedder.linear_2", "guidance_in.out_layer")
    if "controlnet_mode_embedder.weight" in sd:
        take("controlnet_mode_embedder.weight", "controlnet_mode_embedder.weight")

    for i in range(depth):
        s = f"transformer_blocks.{i}"
        d = f"double_blocks.{i}"
        # q, k, v are one fused linear in the engine's blocks.
        for side, (q, k, v) in (("img_attn", ("to_q", "to_k", "to_v")), ("txt_attn", ("add_q_proj", "add_k_proj", "add_v_proj"))):
            out[f"{d}.{side}.qkv.weight"] = torch.cat([sd[f"{s}.attn.{n}.weight"] for n in (q, k, v)], dim=0)
            out[f"{d}.{side}.qkv.bias"] = torch.cat([sd[f"{s}.attn.{n}.bias"] for n in (q, k, v)], dim=0)
        take(f"{s}.attn.norm_q.weight", f"{d}.img_attn.norm.query_norm.weight")
        take(f"{s}.attn.norm_k.weight", f"{d}.img_attn.norm.key_norm.weight")
        take(f"{s}.attn.norm_added_q.weight", f"{d}.txt_attn.norm.query_norm.weight")
        take(f"{s}.attn.norm_added_k.weight", f"{d}.txt_attn.norm.key_norm.weight")
        lin(f"{s}.attn.to_out.0", f"{d}.img_attn.proj")
        lin(f"{s}.attn.to_add_out", f"{d}.txt_attn.proj")
        lin(f"{s}.ff.net.0.proj", f"{d}.img_mlp.0")
        lin(f"{s}.ff.net.2", f"{d}.img_mlp.2")
        lin(f"{s}.ff_context.net.0.proj", f"{d}.txt_mlp.0")
        lin(f"{s}.ff_context.net.2", f"{d}.txt_mlp.2")
        # AdaLayerNormZero and Flux's Modulation emit shift, scale, gate ×2 in the same order.
        lin(f"{s}.norm1.linear", f"{d}.img_mod.lin")
        lin(f"{s}.norm1_context.linear", f"{d}.txt_mod.lin")
        lin(f"controlnet_blocks.{i}", f"controlnet_blocks.{i}")
    return out


def expand_to_blocks(samples: list, main_depth: int) -> list:
    """N ControlNet outputs over the main model's double blocks (diffusers' interval rule)."""
    if not samples:
        return []
    interval = int(math.ceil(main_depth / len(samples)))
    return [samples[j // interval] for j in range(main_depth)]


class FluxControlNet(nn.Module):
    def __init__(self, in_channels, hidden_size, num_heads, depth, depth_single, context_in_dim, vec_in_dim, mlp_ratio, guidance_embed, num_modes, axes_dim, theta=10000):
        super().__init__()
        if depth_single:
            # No released Flux ControlNet we target has them (Union Pro 2.0, InstantX canny/depth: 0).
            raise ValueError("Flux ControlNets with single-stream blocks are not supported yet")
        self.depth = depth
        self.guidance_embed = guidance_embed
        self.hidden_size = hidden_size
        self.pe_embedder = EmbedND(dim=hidden_size // num_heads, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(in_channels, hidden_size)
        self.controlnet_x_embedder = nn.Linear(in_channels, hidden_size)
        self.time_in = MLPEmbedder(256, hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, hidden_size)
        self.guidance_in = MLPEmbedder(256, hidden_size) if guidance_embed else None
        self.txt_in = nn.Linear(context_in_dim, hidden_size)
        self.controlnet_mode_embedder = nn.Embedding(num_modes, hidden_size) if num_modes else None
        self.double_blocks = nn.ModuleList([DoubleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)])
        self.controlnet_blocks = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])

    @staticmethod
    def pack(latent: torch.Tensor) -> torch.Tensor:
        """B,16,H,W latent → B,(H/2·W/2),64 tokens — the main Flux's own patchify."""
        return rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    @staticmethod
    def image_ids(b: int, h: int, w: int, device) -> torch.Tensor:
        ids = torch.zeros((h // 2, w // 2, 3), device=device, dtype=torch.float32)
        ids[..., 1] = torch.arange(h // 2, device=device, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(w // 2, device=device, dtype=torch.float32)[None, :]
        return repeat(ids, "h w c -> b (h w) c", b=b)

    def embed_hint(self, hint):
        """The drawing's embedding: the same every step of a render, so callers may compute it once."""
        return self.controlnet_x_embedder(hint)

    def embed_text(self, txt):
        """The prompt's projection: the same every step of a render too."""
        return self.txt_in(txt)

    def forward(self, img, hint, timesteps, txt, y, guidance=None, img_ids=None, txt_ids=None, mode=None, transformer_options={}, hint_emb=None, txt_emb=None):
        """img, hint: packed tokens (B, L, 64). Returns the N per-block corrections (B, L, hidden).
        hint_emb / txt_emb: embed_hint / embed_text results reused across steps."""
        x = self.img_in(img) + (hint_emb if hint_emb is not None else self.embed_hint(hint))
        vec = self.time_in(timestep_embedding(timesteps, 256).to(x.dtype))
        if self.guidance_in is not None and guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(x.dtype))
        vec = vec + self.vector_in(y.to(x.dtype))
        txt = txt_emb if txt_emb is not None else self.embed_text(txt)
        if self.controlnet_mode_embedder is not None and mode is not None:
            # Union (Pro 1) models: the control type as one extra text token.
            m = self.controlnet_mode_embedder(mode.to(x.device)).unsqueeze(1).to(txt.dtype)
            txt = torch.cat([m, txt], dim=1)
            txt_ids = torch.cat([txt_ids[:, :1], txt_ids], dim=1)
        pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))
        out = []
        for block, proj in zip(self.double_blocks, self.controlnet_blocks):
            x, txt = block(img=x, txt=txt, vec=vec, pe=pe, transformer_options=transformer_options)
            out.append(proj(x))
        return out


def keep_small_params_precise(model, dtype):
    """Biases and norm scales back to the compute dtype: fp8 storage is for the big matrices only."""
    for m in model.modules():
        for name, p in list(m.named_parameters(recurse=False)):
            if p is not None and p.ndim < 2 and p.dtype != dtype:
                setattr(m, name, torch.nn.Parameter(p.detach().to(dtype), requires_grad=False))


def build(state_dict: dict, cfg: dict, storage: torch.dtype, compute_dtype: torch.dtype):
    """The model on the engine's own layers (low-VRAM offload, weights cast to the compute dtype as used)."""
    from backend.operations import using_forge_operations

    with using_forge_operations(dtype=storage, manual_cast_enabled=True):
        model = FluxControlNet(**cfg)
    model.to(dtype=storage)
    missing, unexpected = model.load_state_dict(convert_diffusers(state_dict, cfg["depth"]), strict=False)
    keep_small_params_precise(model, compute_dtype)
    return model, missing, unexpected
