# InfinisX — registers the Flux ControlNet (lib_flux_cn/model.py) with Forge Neo's
# ControlNet extension. A Flux ControlNet file (InstantX / Shakker-Labs Union
# Pro 2.0 layout) picked as a unit's model now loads and steers a Flux render;
# every other file falls through to Forge's own SD 1.5 / SDXL loader.

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import memory_management  # noqa: E402
from backend.nn.flux import IntegratedFluxTransformer2DModel  # noqa: E402
from backend.patcher.base import ModelPatcher  # noqa: E402
from backend.patcher.controlnet import ControlBase, apply_controlnet_advanced, logger  # noqa: E402
from lib_flux_cn.model import FluxControlNet, build, config_from_state_dict, expand_to_blocks, is_flux_controlnet, union_mode_for, validate_flux1  # noqa: E402
from modules import shared  # noqa: E402
from modules_forge.shared import add_supported_control_model  # noqa: E402
from modules_forge.supported_controlnet import ControlModelPatcher, ModelCache  # noqa: E402

_CACHE = ModelCache()


def flux1_transformer(sd_model):
    """The loaded model's Flux.1 transformer, or None (SD, SDXL, Flux 2, Chroma… can't take this ControlNet)."""
    try:
        dm = sd_model.forge_objects.unet.model.diffusion_model
    except AttributeError:
        return None
    if not isinstance(dm, IntegratedFluxTransformer2DModel) or getattr(dm, "global_modulation", False):
        return None
    return dm if dm.hidden_size == 3072 and len(dm.double_blocks) == 19 else None


class FluxControl(ControlBase):
    """One Flux ControlNet unit during sampling: the drawing's latent in, per-block corrections out."""

    def __init__(self, control_model, load_device, compute_dtype):
        super().__init__()
        self.control_model = control_model
        self.load_device = load_device
        self.compute_dtype = compute_dtype
        self.control_model_wrapped = ModelPatcher(control_model, load_device=load_device, offload_device=memory_management.unet_offload_device())
        self.hint_latent = None  # B,16,h,w — set before sampling (FluxControlNetPatcher)
        self.mode = None  # Union Pro 1.0 control mode (Pro 2.0: None)
        self.main_depth = 19
        self.model_sampling_current = None
        # Per render: what stays the same every step (the drawing, the prompt) computed once.
        self._hint_tokens = None
        self._hint_emb = None
        self._txt = None

    def copy(self):
        c = FluxControl(self.control_model, self.load_device, self.compute_dtype)
        c.control_model_wrapped = self.control_model_wrapped
        c.hint_latent = self.hint_latent
        c.mode = self.mode
        self.copy_to(c)
        return c

    def get_models(self):
        return super().get_models() + [self.control_model_wrapped]

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        self.model_sampling_current = model.predictor
        self.main_depth = len(model.diffusion_model.double_blocks)

    def cleanup(self):
        self.model_sampling_current = None
        self._hint_tokens = self._hint_emb = self._txt = None
        super().cleanup()

    def _hint_for(self, x_noisy, batched_number):
        b, _, h, w = x_noisy.shape
        key = (b, h, w)
        if self._hint_tokens is None or self._hint_tokens[0] != key:
            lat = self.hint_latent
            if lat.shape[-2:] != (h, w):  # e.g. a Hi-Res pass: the same drawing at the new size
                lat = torch.nn.functional.interpolate(lat.float(), size=(h, w), mode="bilinear", align_corners=False)
            if lat.shape[0] != b:
                lat = lat[:1].repeat(b, 1, 1, 1)
            self._hint_tokens = (key, FluxControlNet.pack(lat.to(self.load_device, self.compute_dtype)))
            self._hint_emb = None
        return self._hint_tokens[1]

    def _text_for(self, context):
        # Held by reference, so an equal data pointer means the same tensor, never a reused one.
        if self._txt is None or self._txt[0].data_ptr() != context.data_ptr() or self._txt[0].shape != context.shape:
            self._txt = (context, self.control_model.embed_text(context))
        return self._txt[1]

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number) if self.previous_controlnet is not None else None
        if self.timestep_range is not None and (t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]):
            return control_prev
        if self.hint_latent is None:
            return control_prev

        dtype = self.compute_dtype
        b, _, h, w = x_noisy.shape
        # Exactly what the main Flux receives this step (k_model.apply_model).
        x = self.model_sampling_current.calculate_input(t, x_noisy).to(dtype)
        timestep = self.model_sampling_current.timestep(t).float()
        context = cond["c_crossattn"].to(dtype)
        y = cond.get("y")
        guidance = cond.get("guidance")
        txt_ids = torch.zeros((b, context.shape[1], 3), device=x.device, dtype=torch.float32)
        img_ids = FluxControlNet.image_ids(b, h, w, x.device)

        hint = self._hint_for(x_noisy, batched_number)
        if self._hint_emb is None:
            self._hint_emb = self.control_model.embed_hint(hint)
        mode = None if self.mode is None else torch.full((b,), self.mode, dtype=torch.long)
        samples = self.control_model(
            FluxControlNet.pack(x), hint, timestep, context,
            y.to(dtype) if y is not None else torch.zeros((b, 768), device=x.device, dtype=dtype),
            guidance=guidance.to(x.device) if guidance is not None else None,
            img_ids=img_ids, txt_ids=txt_ids, mode=mode, transformer_options=self.transformer_options,
            hint_emb=self._hint_emb, txt_emb=self._text_for(context),
        )
        out_dtype = x_noisy.dtype
        blocks = expand_to_blocks([s.to(out_dtype) * self.strength for s in samples], self.main_depth)
        out = {"input": blocks, "middle": [], "output": []}
        if control_prev is not None:  # another Flux unit: corrections add up
            for i, prev in enumerate(control_prev.get("input", [])):
                if prev is None:
                    continue
                if i < len(out["input"]):
                    out["input"][i] = out["input"][i] + prev
                else:
                    out["input"].append(prev)
        return out


class FluxControlNetPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if not is_flux_controlnet(state_dict):
            return None
        name = os.path.basename(ckpt_path)
        try:
            cfg = config_from_state_dict(state_dict)
            validate_flux1(cfg)
        except (KeyError, IndexError, ValueError) as e:
            # Refused here, plainly, rather than failing halfway through a render.
            logger.error(f"Flux ControlNet {name} can't be used: {e}")
            return None
        compute_dtype = shared.sd_model.forge_objects.unet.model.computation_dtype
        load_device = memory_management.get_torch_device()
        # The file's size and date are in the key: a file replaced under the same name loads afresh.
        st = os.stat(ckpt_path)
        cache_key = (ckpt_path, st.st_size, st.st_mtime_ns)
        model = _CACHE.get(cache_key)
        if model is not None:
            logger.info("Reusing Flux ControlNet Model...")
        else:
            # A 16 GB card holds Flux and a 4 GB bf16 ControlNet only just: store it in fp8
            # (as most shared Flux ControlNets already are) and compute in the model's dtype.
            already_fp8 = any(v.dtype == torch.float8_e4m3fn for v in state_dict.values())
            total_vram = memory_management.total_vram  # MB
            storage = torch.float8_e4m3fn if already_fp8 or total_vram < 20 * 1024 else torch.bfloat16
            logger.info(f"Creating Flux ControlNet Model ({cfg['depth']} blocks, stored {str(storage).split('.')[-1]})...")
            model, missing, unexpected = build(state_dict, cfg, storage, compute_dtype)
            if missing:
                logger.warning(f"Missing Flux ControlNet keys: {missing}")
            if unexpected:
                logger.warning(f"Unexpected Flux ControlNet keys: {unexpected}")
            _CACHE.put(cache_key, model)
        return FluxControlNetPatcher(FluxControl(model, load_device, compute_dtype))

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        if flux1_transformer(process.sd_model) is None:
            logger.warning("Flux ControlNet skipped: the loaded model is not Flux.1")
            return
        # The drawing through the Flux VAE once, before sampling, so no model swap happens mid-render.
        vae = process.sd_model.forge_objects.vae
        pixels = cond.movedim(1, -1).float().clamp(0, 1)
        latent = vae.first_stage_model.process_in(vae.encode(pixels))
        self.model_patcher.hint_latent = latent.cpu()
        # Union Pro 1.0 wants the control kind as a mode; read from the unit's preprocessor.
        unit = kwargs.get("unit")
        if self.model_patcher.control_model.controlnet_mode_embedder is not None:
            self.model_patcher.mode = union_mode_for(getattr(unit, "module", None))
        unet = apply_controlnet_advanced(
            unet=process.sd_model.forge_objects.unet,
            controlnet=self.model_patcher,
            image_bchw=cond,
            strength=self.strength,
            start_percent=self.start_percent,
            end_percent=self.end_percent,
        )
        process.sd_model.forge_objects.unet = unet
        logger.info("Flux ControlNet patched.")


add_supported_control_model(FluxControlNetPatcher)
