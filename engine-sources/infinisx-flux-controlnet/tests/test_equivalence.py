"""
Our Flux ControlNet against diffusers' reference FluxControlNetModel, on the CPU
with small random weights: same inputs → same per-block outputs. Then the whole
path — diffusers' Flux transformer fed the ControlNet's samples vs the engine's
Flux fed our expanded list — gives the same prediction.

Run with the engine's Python, from the engine folder:
  python.exe <this file> <engine>/renderer
"""

import sys

RENDERER = sys.argv[1]
sys.argv = sys.argv[:1]  # the engine's argument parser reads sys.argv
sys.path[:0] = [RENDERER, RENDERER + "/modules_forge/packages", __file__.rsplit("tests", 1)[0]]

import os

os.chdir(RENDERER)

import torch
from diffusers import FluxControlNetModel, FluxTransformer2DModel

from lib_flux_cn.model import FLUX1, FluxControlNet, build, config_from_state_dict, convert_diffusers, expand_to_blocks, is_flux_controlnet, union_mode_for, validate_flux1

torch.manual_seed(0)
torch.set_grad_enabled(False)

HEADS, HEAD_DIM, AXES = 2, 16, (4, 6, 6)
TINY = dict(patch_size=1, in_channels=16, attention_head_dim=HEAD_DIM, num_attention_heads=HEADS, joint_attention_dim=24, pooled_projection_dim=12, guidance_embeds=True, axes_dims_rope=AXES)


def randomize(m):
    # The released ControlNet's output linears start at zero; random weights test every path.
    for p in m.parameters():
        p.copy_(torch.randn_like(p) * 0.2)
    return m.float().eval()


def max_err(a, b):
    return (a - b).abs().max().item() / max(1e-6, b.abs().max().item())


ref = randomize(FluxControlNetModel(num_layers=3, num_single_layers=0, **TINY))
sd = ref.state_dict()
assert is_flux_controlnet(sd), "signature keys"
cfg = config_from_state_dict(sd)
cfg["axes_dim"] = list(AXES)
ours = FluxControlNet(**cfg).float().eval()
missing, unexpected = ours.load_state_dict(convert_diffusers(sd, cfg["depth"]), strict=False)
assert not missing and not unexpected, (missing, unexpected)

B, H, W, T = 2, 8, 6, 5  # latent 16×H×W → (H/2·W/2) tokens of 64
latent = torch.randn(B, 4, H, W)  # in_channels 16 = 4 channels × 2×2 patch
hint = torch.randn(B, 4, H, W)
img, cond = FluxControlNet.pack(latent), FluxControlNet.pack(hint)
txt = torch.randn(B, T, 24)
y = torch.randn(B, 12)
t = torch.tensor([0.7, 0.3])  # sigma: Flux's timestep in 0..1
g = torch.tensor([3.5, 3.5])
img_ids = FluxControlNet.image_ids(B, H, W, "cpu")
txt_ids = torch.zeros(B, T, 3)

# diffusers multiplies timestep and guidance by 1000 inside; it takes ids without the batch axis.
want = ref(hidden_states=img, controlnet_cond=cond, timestep=t, guidance=g, encoder_hidden_states=txt, pooled_projections=y, img_ids=img_ids[0], txt_ids=txt_ids[0], return_dict=False)[0]
got = ours(img, cond, t, txt, y, guidance=g, img_ids=img_ids, txt_ids=txt_ids)
assert len(got) == len(want) == 3
errs = [max_err(a, b) for a, b in zip(got, want)]
print("controlnet blocks, relative error:", ["%.2e" % e for e in errs])
assert max(errs) < 1e-4, errs

# Whole path: a 7-block Flux, as the 19-block one, takes 3 samples at interval ceil(7/3) = 3.
from backend.nn.flux import IntegratedFluxTransformer2DModel

main_ref = randomize(FluxTransformer2DModel(num_layers=7, num_single_layers=2, out_channels=None, **TINY))
engine_main = IntegratedFluxTransformer2DModel(
    in_channels=4, out_channels=4, vec_in_dim=12, context_in_dim=24, hidden_size=HEADS * HEAD_DIM, mlp_ratio=4.0, num_heads=HEADS,
    depth=7, depth_single_blocks=2, axes_dim=list(AXES), theta=10000, patch_size=2, qkv_bias=True, guidance_embed=True,
).float().eval()
dsd = main_ref.state_dict()
# Double blocks and embedders share the ControlNet's naming: reuse its conversion.
dsd_cn = dict(dsd)
for i in range(7):  # convert_diffusers also maps controlnet_blocks.i: give it dummies, dropped below
    dsd_cn[f"controlnet_blocks.{i}.weight"] = torch.zeros(1)
    dsd_cn[f"controlnet_blocks.{i}.bias"] = torch.zeros(1)
dsd_cn["controlnet_x_embedder.weight"] = dsd_cn["controlnet_x_embedder.bias"] = torch.zeros(1)
conv = {k: v for k, v in convert_diffusers(dsd_cn, 7).items() if not k.startswith("controlnet")}
for i in range(2):  # single blocks: q, k, v and the MLP input are one fused linear
    s_, d_ = f"single_transformer_blocks.{i}", f"single_blocks.{i}"
    conv[f"{d_}.linear1.weight"] = torch.cat([dsd[f"{s_}.attn.to_q.weight"], dsd[f"{s_}.attn.to_k.weight"], dsd[f"{s_}.attn.to_v.weight"], dsd[f"{s_}.proj_mlp.weight"]])
    conv[f"{d_}.linear1.bias"] = torch.cat([dsd[f"{s_}.attn.to_q.bias"], dsd[f"{s_}.attn.to_k.bias"], dsd[f"{s_}.attn.to_v.bias"], dsd[f"{s_}.proj_mlp.bias"]])
    conv[f"{d_}.linear2.weight"] = dsd[f"{s_}.proj_out.weight"]
    conv[f"{d_}.linear2.bias"] = dsd[f"{s_}.proj_out.bias"]
    conv[f"{d_}.norm.query_norm.weight"] = dsd[f"{s_}.attn.norm_q.weight"]
    conv[f"{d_}.norm.key_norm.weight"] = dsd[f"{s_}.attn.norm_k.weight"]
    conv[f"{d_}.modulation.lin.weight"] = dsd[f"{s_}.norm.linear.weight"]
    conv[f"{d_}.modulation.lin.bias"] = dsd[f"{s_}.norm.linear.bias"]
# diffusers' final norm emits (scale, shift); the engine's (shift, scale).
swap = lambda t: torch.cat(t.chunk(2, dim=0)[::-1])
conv["final_layer.adaLN_modulation.1.weight"] = swap(dsd["norm_out.linear.weight"])
conv["final_layer.adaLN_modulation.1.bias"] = swap(dsd["norm_out.linear.bias"])
conv["final_layer.linear.weight"] = dsd["proj_out.weight"]
conv["final_layer.linear.bias"] = dsd["proj_out.bias"]
missing, unexpected = engine_main.load_state_dict(conv, strict=False)
assert not missing and not unexpected, (missing[:6], unexpected[:6])

samples = [s * 0.8 for s in want]  # conditioning scale = Guide strength
want_pred = main_ref(hidden_states=img, timestep=t, guidance=g, encoder_hidden_states=txt, pooled_projections=y, img_ids=img_ids[0], txt_ids=txt_ids[0], controlnet_block_samples=samples, return_dict=False)[0]
control = {"input": expand_to_blocks([s * 0.8 for s in got], 7), "output": []}
got_pred = engine_main.forward_orig(img, img_ids, txt, txt_ids, t, y, g, control)
no_cn = engine_main.forward_orig(img, img_ids, txt, txt_ids, t, y, g, None)
e = max_err(got_pred, want_pred)
moved = max_err(no_cn, want_pred)
print("whole prediction, relative error: %.2e (without the ControlNet it would be off by %.2e)" % (e, moved))
assert e < 1e-4 and moved > 1e-2

# As the engine builds it: its own layers, weights stored fp8 / bf16, computed in float32 here.
for storage, tol in ((torch.bfloat16, 3e-2), (torch.float8_e4m3fn, 1.5e-1)):
    m, missing, unexpected = build(sd, cfg, storage, torch.float32)
    assert not missing and not unexpected, (missing, unexpected)
    got_s = m(img, cond, t, txt, y, guidance=g, img_ids=img_ids, txt_ids=txt_ids)
    errs = [max_err(a.float(), b) for a, b in zip(got_s, want)]
    print("engine layers, stored %s: relative error %s" % (str(storage).split(".")[-1], ["%.1e" % e for e in errs]))
    assert max(errs) < tol, errs

# Cached drawing / prompt embeddings (computed once per render) change nothing.
cached = ours(img, cond, t, txt, y, guidance=g, img_ids=img_ids, txt_ids=txt_ids, hint_emb=ours.embed_hint(cond), txt_emb=ours.embed_text(txt))
assert all(torch.equal(a, b) for a, b in zip(cached, got)), "cached embeddings"
print("cached embeddings: identical")

# Union Pro 1.0: the control mode as an extra text token, as diffusers does it.
ref_u = randomize(FluxControlNetModel(num_layers=2, num_single_layers=0, num_mode=10, **TINY))
sd_u = ref_u.state_dict()
cfg_u = config_from_state_dict(sd_u)
cfg_u["axes_dim"] = list(AXES)
assert cfg_u["num_modes"] == 10
ours_u = FluxControlNet(**cfg_u).float().eval()
missing, unexpected = ours_u.load_state_dict(convert_diffusers(sd_u, cfg_u["depth"]), strict=False)
assert not missing and not unexpected, (missing, unexpected)
mode = torch.tensor([union_mode_for("depth_anything_v2")] * B)
want_u = ref_u(hidden_states=img, controlnet_cond=cond, controlnet_mode=mode.view(-1, 1), timestep=t, guidance=g, encoder_hidden_states=txt, pooled_projections=y, img_ids=img_ids[0], txt_ids=txt_ids[0], return_dict=False)[0]
got_u = ours_u(img, cond, t, txt, y, guidance=g, img_ids=img_ids, txt_ids=txt_ids, mode=mode)
errs = [max_err(a, b) for a, b in zip(got_u, want_u)]
print("union pro 1 (mode %d), relative error: %s" % (mode[0], ["%.2e" % e for e in errs]))
assert max(errs) < 1e-4, errs
assert union_mode_for("lineart_realistic") == 0 and union_mode_for("dw_openpose_full") == 4 and union_mode_for(None) == 0

# A ControlNet that isn't Flux.1's shape is refused at load time, not mid-render.
try:
    validate_flux1(cfg)
    raise AssertionError("tiny config should be refused")
except ValueError as e:
    print("refused:", str(e)[:70], "...")
validate_flux1(dict(FLUX1, depth=6, depth_single=0))
print("OK")
