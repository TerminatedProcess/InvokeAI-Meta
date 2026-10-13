# Qwen Image Edit LoRA prefix constants
# These prefixes are used for key mapping when applying LoRA patches to Qwen Image Edit models

import re

# Prefix for Qwen Image Edit transformer LoRA layers
QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX = "lora_transformer-"

# ComfyUI saves Qwen Image LoRAs against the single-file checkpoint, prefixing every key with
# ``diffusion_model.``. LTX-2 and Wan's diffusers layout share that prefix and the
# ``transformer_blocks.N`` path, so the prefix only counts as Qwen Image alongside a dual-stream
# block module no other supported architecture names this way.
QWEN_IMAGE_COMFY_PREFIX = "diffusion_model."
_QWEN_IMAGE_COMFY_RE = re.compile(r"^diffusion_model\.transformer_blocks\.\d+\.(?:img_mod|txt_mod|img_mlp|txt_mlp)\.")


def has_comfy_qwen_image_lora_keys(str_keys: list[str]) -> bool:
    """True if the keys are a ComfyUI-format (``diffusion_model.``-prefixed) Qwen Image LoRA."""
    return any(_QWEN_IMAGE_COMFY_RE.match(k) for k in str_keys)
