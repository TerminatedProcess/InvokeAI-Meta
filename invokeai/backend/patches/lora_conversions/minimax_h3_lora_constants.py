# MiniMax H3 LoRA prefix constants and key-shape detection helpers.
#
# H3 LoRAs in the wild (e.g. larryvrh/MiniMax-H3-Turbo-Lora) are published against
# the original remote-code / Comfy single-file transformer layout with plain PEFT
# ``lora_A`` / ``lora_B`` suffixes and no ``.alpha`` tensors (alpha == rank by
# convention, i.e. ``W_eff = W + lora_B @ lora_A`` with no extra scaling):
#
#     blocks.0.attn.qkv_proj.lora_A.weight            (fused QKV)
#     blocks.0.attn.out_proj.lora_A.weight
#     blocks.0.mlp.fc1.lora_A.weight                  (fused SwiGLU input projection)
#     blocks.0.mlp.fc2.lora_A.weight
#     blocks.0.adaln_proj.linear.lora_A.weight        (AdaLN time-conditioning)
#     token_refiner.blocks.0.attn.qkv_proj.lora_A.weight
#     final_layer.adaln_proj.linear.lora_A.weight
#
# ComfyUI users sometimes re-key these with a ``diffusion_model.`` prefix; a bare
# ``transformer.`` / ``base_model.model.transformer.`` PEFT prefix is also accepted.
# Kohya/sd-scripts trainers flatten the same paths with underscores and use
# ``lora_down`` / ``lora_up`` / ``alpha`` (``lora_unet_blocks_0_attn_qkv_proj.lora_down.weight``);
# ``minimax_h3_native_key`` rewrites those to the native layout before detection and conversion.
#
# The detection helpers below are shared with ``configs/lora.py`` so the probe and
# the conversion code agree on what counts as an H3 LoRA. They keep this file
# circular-import-free.

import re

# Prefix for H3 transformer LoRA layers in the ModelPatchRaw layer dict. Same
# convention as Wan / Anima / QwenImage — the LayerPatcher uses this prefix to
# resolve patches against the loaded transformer's parameter paths.
MINIMAX_H3_LORA_TRANSFORMER_PREFIX = "lora_transformer-"

# Optional prefixes seen on PEFT-style keys. Anchored to the key start so detection admits
# exactly what the converter's single prefix-strip can handle — an unanchored alternative
# would also match nested prefixes like ``diffusion_model.transformer.blocks...``, which the
# probe would then admit but the converter would map onto nonexistent module paths (the LoRA
# would silently apply zero layers).
_PEFT_PREFIX_RE = r"^(?:(?:diffusion_model|transformer|base_model\.model\.transformer)\.)?"

# Submodules the H3 transformer exposes for LoRA, in the checkpoint's native naming.
# ``attn.qkv_proj`` (a fused projection under a bare ``attn.``) and ``adaln_proj.linear``
# are unique to MiniMax H3 among the supported architectures:
# - Wan native uses ``self_attn.q`` / ``cross_attn.k`` (unfused, no ``_proj`` tail);
# - Anima/Cosmos uses ``self_attn.q_proj`` / ``mlp.layer_0`` / ``adaln_modulation``;
# - timm-style DiTs use ``attn.qkv`` / ``attn.proj`` (no ``_proj`` on the fused QKV);
# - Krea-2 native uses ``mlp.{down,gate,up}``.
# ``mlp.fc1`` alone would NOT be safe (timm-style MLPs use it too), so detection
# requires at least one *exclusive* H3 marker.
_H3_EXCLUSIVE_RE = re.compile(
    _PEFT_PREFIX_RE + r"(?:token_refiner\.)?blocks\.\d+\.(?:attn\.qkv_proj|adaln_proj\.linear)\."
)
_H3_FINAL_LAYER_RE = re.compile(_PEFT_PREFIX_RE + r"final_layer\.adaln_proj\.linear\.")

# Kohya layer names for every H3 LoRA target. Matched against the whole layer name (not by
# splitting on ``_``) because the module names themselves contain underscores.
_KOHYA_BLOCK_RE = re.compile(
    r"^lora_unet_(token_refiner_)?blocks_(\d+)_(attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2|adaln_proj_linear)\."
)
_KOHYA_FINAL_LAYER = "lora_unet_final_layer_adaln_proj_linear."
_KOHYA_MODULES = {
    "attn_qkv_proj": "attn.qkv_proj",
    "attn_out_proj": "attn.out_proj",
    "mlp_fc1": "mlp.fc1",
    "mlp_fc2": "mlp.fc2",
    "adaln_proj_linear": "adaln_proj.linear",
}

# Any of these indicates a different architecture; an H3 LoRA never carries them.
_NON_H3_ANTI_RES = (
    re.compile(r"(self_attn|cross_attn)[\._]"),  # Wan / Anima attention naming
    re.compile(r"attn[12]\."),  # Wan diffusers naming
    re.compile(r"(^|\.|_)(double_blocks|single_blocks|single_transformer_blocks)[\._]\d+"),  # FLUX
    re.compile(r"(^|\.)transformer_blocks\.\d+\."),  # QwenImage / already-diffusers keys
    re.compile(r"diffusion_model\.layers\.\d+\."),  # Z-Image
    re.compile(r"mlp[\._]layer|adaln_modulation"),  # Anima/Cosmos
)

# LyCORIS variants the H3 pipeline does NOT support: LoKR/LoHA cannot be row-split across the
# fused qkv/SwiGLU tensors (a Kronecker/Hadamard factorization does not distribute over row
# slices), and DoRA's per-output-row magnitudes would need the same split/swap treatment plus
# original-weight access that the int8 sidecar path cannot provide. Rejecting these at probe
# time turns a would-be generation-time crash (or silent half-swap corruption) into a clear
# install-time "not supported".
_UNSUPPORTED_H3_VARIANT_SUFFIXES = (
    ".lokr_w1",
    ".lokr_w2",
    ".hada_w1_a",
    ".hada_w2_a",
    ".dora_scale",
)


def minimax_h3_native_key(key: str) -> str:
    """Rewrite a Kohya-format H3 LoRA key to the native dotted layout; other keys pass through."""
    m = _KOHYA_BLOCK_RE.match(key)
    if m:
        refiner = "token_refiner." if m.group(1) else ""
        return f"{refiner}blocks.{m.group(2)}.{_KOHYA_MODULES[m.group(3)]}.{key[m.end() :]}"
    if key.startswith(_KOHYA_FINAL_LAYER):
        return "final_layer.adaln_proj.linear." + key[len(_KOHYA_FINAL_LAYER) :]
    return key


def has_unsupported_minimax_h3_lora_variant_keys(str_keys: list[str]) -> bool:
    """True if the state dict carries LyCORIS-variant tensors (LoKR/LoHA/DoRA) that the H3
    conversion pipeline cannot apply to the fused transformer layers."""
    return any(k.endswith(suffix) for k in str_keys for suffix in _UNSUPPORTED_H3_VARIANT_SUFFIXES)


def has_minimax_h3_lora_keys(str_keys: list[str]) -> bool:
    """PEFT-style keys naming H3-exclusive submodules (fused ``attn.qkv_proj`` or
    ``adaln_proj.linear``) in the checkpoint's native or Kohya layout."""
    native_keys = map(minimax_h3_native_key, str_keys)
    return any(_H3_EXCLUSIVE_RE.search(k) or _H3_FINAL_LAYER_RE.search(k) for k in native_keys)


def has_non_minimax_h3_architecture_keys(str_keys: list[str]) -> bool:
    """True if any key indicates a non-H3 architecture (Wan, Anima, FLUX, QwenImage, Z-Image).

    Used as an exclusion guard — an H3 LoRA never carries these patterns, so finding
    them is grounds to reject the H3 probe.
    """
    return any(anti.search(k) for k in str_keys for anti in _NON_H3_ANTI_RES)
