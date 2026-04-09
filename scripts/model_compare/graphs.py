"""Execution graph builders for InvokeAI model types."""

import uuid


def build_sdxl_graph(
    model: dict,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    cfg_rescale: float,
    scheduler: str,
    loras: list[dict],
) -> dict:
    """Build an SDXL text-to-image execution graph."""
    ml = f"sdxl_model_loader:{uuid.uuid4().hex[:10]}"
    pp = f"positive_prompt:{uuid.uuid4().hex[:10]}"
    np_ = f"negative_prompt:{uuid.uuid4().hex[:10]}"
    pc = f"pos_cond:{uuid.uuid4().hex[:10]}"
    pcc = f"pos_cond_collect:{uuid.uuid4().hex[:10]}"
    nc = f"neg_cond:{uuid.uuid4().hex[:10]}"
    ncc = f"neg_cond_collect:{uuid.uuid4().hex[:10]}"
    sd = f"seed:{uuid.uuid4().hex[:10]}"
    ns = f"noise:{uuid.uuid4().hex[:10]}"
    dn = f"denoise_latents:{uuid.uuid4().hex[:10]}"
    cm = f"core_metadata:{uuid.uuid4().hex[:10]}"
    out = f"canvas_output:{uuid.uuid4().hex[:10]}"

    nodes = {
        ml: {
            "type": "sdxl_model_loader", "id": ml, "is_intermediate": True, "use_cache": True,
            "model": model,
        },
        pp: {"type": "string", "id": pp, "is_intermediate": True, "use_cache": True, "value": positive_prompt},
        np_: {"type": "string", "id": np_, "is_intermediate": True, "use_cache": True, "value": negative_prompt},
        pc: {
            "type": "sdxl_compel_prompt", "id": pc, "is_intermediate": True, "use_cache": True,
            "prompt": "", "style": "",
            "original_width": width, "original_height": height,
            "crop_top": 0, "crop_left": 0, "target_width": width, "target_height": height,
        },
        pcc: {"type": "collect", "id": pcc, "is_intermediate": True, "use_cache": True, "collection": []},
        nc: {
            "type": "sdxl_compel_prompt", "id": nc, "is_intermediate": True, "use_cache": True,
            "prompt": negative_prompt, "style": negative_prompt,
            "original_width": width, "original_height": height,
            "crop_top": 0, "crop_left": 0, "target_width": width, "target_height": height,
        },
        ncc: {"type": "collect", "id": ncc, "is_intermediate": True, "use_cache": True, "collection": []},
        sd: {"type": "integer", "id": sd, "is_intermediate": True, "use_cache": True, "value": seed},
        ns: {
            "type": "noise", "id": ns, "is_intermediate": True, "use_cache": True,
            "seed": 0, "width": width, "height": height, "use_cpu": True,
        },
        dn: {
            "type": "denoise_latents", "id": dn, "is_intermediate": True, "use_cache": True,
            "steps": steps, "cfg_scale": cfg_scale, "denoising_start": 0.0, "denoising_end": 1.0,
            "scheduler": scheduler, "cfg_rescale_multiplier": cfg_rescale,
        },
        cm: {
            "type": "core_metadata", "id": cm, "is_intermediate": True, "use_cache": True,
            "generation_mode": "sdxl_txt2img", "negative_prompt": negative_prompt,
            "width": width, "height": height, "rand_device": "cpu",
            "cfg_scale": cfg_scale, "cfg_rescale_multiplier": cfg_rescale,
            "steps": steps, "scheduler": scheduler,
            "seamless_x": False, "seamless_y": False,
            "model": model,
            "loras": [{"model": l["model"], "weight": l["weight"]} for l in loras],
            "ref_images": [],
        },
        out: {
            "type": "l2i", "id": out, "is_intermediate": False, "use_cache": False,
            "tiled": False, "tile_size": 0, "fp32": True,
        },
    }

    edges = [
        {"source": {"node_id": pp, "field": "value"}, "destination": {"node_id": pc, "field": "prompt"}},
        {"source": {"node_id": pp, "field": "value"}, "destination": {"node_id": pc, "field": "style"}},
        {"source": {"node_id": pc, "field": "conditioning"}, "destination": {"node_id": pcc, "field": "item"}},
        {"source": {"node_id": pcc, "field": "collection"}, "destination": {"node_id": dn, "field": "positive_conditioning"}},
        {"source": {"node_id": nc, "field": "conditioning"}, "destination": {"node_id": ncc, "field": "item"}},
        {"source": {"node_id": ncc, "field": "collection"}, "destination": {"node_id": dn, "field": "negative_conditioning"}},
        {"source": {"node_id": sd, "field": "value"}, "destination": {"node_id": ns, "field": "seed"}},
        {"source": {"node_id": ns, "field": "noise"}, "destination": {"node_id": dn, "field": "noise"}},
        {"source": {"node_id": dn, "field": "latents"}, "destination": {"node_id": out, "field": "latents"}},
        {"source": {"node_id": sd, "field": "value"}, "destination": {"node_id": cm, "field": "seed"}},
        {"source": {"node_id": pp, "field": "value"}, "destination": {"node_id": cm, "field": "positive_prompt"}},
        {"source": {"node_id": cm, "field": "metadata"}, "destination": {"node_id": out, "field": "metadata"}},
        {"source": {"node_id": ml, "field": "vae"}, "destination": {"node_id": out, "field": "vae"}},
    ]

    if loras:
        lc = f"lora_collector:{uuid.uuid4().hex[:10]}"
        ll = f"sdxl_lora_collection_loader:{uuid.uuid4().hex[:10]}"
        nodes[lc] = {"type": "collect", "id": lc, "is_intermediate": True, "use_cache": True, "collection": []}
        nodes[ll] = {"type": "sdxl_lora_collection_loader", "id": ll, "is_intermediate": True, "use_cache": True}

        for i, lora in enumerate(loras):
            ls = f"lora_selector_{i}:{uuid.uuid4().hex[:10]}"
            nodes[ls] = {
                "type": "lora_selector", "id": ls, "is_intermediate": True, "use_cache": True,
                "lora": lora["model"], "weight": lora["weight"],
            }
            edges.append({"source": {"node_id": ls, "field": "lora"}, "destination": {"node_id": lc, "field": "item"}})

        edges.extend([
            {"source": {"node_id": lc, "field": "collection"}, "destination": {"node_id": ll, "field": "loras"}},
            {"source": {"node_id": ml, "field": "unet"}, "destination": {"node_id": ll, "field": "unet"}},
            {"source": {"node_id": ml, "field": "clip"}, "destination": {"node_id": ll, "field": "clip"}},
            {"source": {"node_id": ml, "field": "clip2"}, "destination": {"node_id": ll, "field": "clip2"}},
            {"source": {"node_id": ll, "field": "unet"}, "destination": {"node_id": dn, "field": "unet"}},
            {"source": {"node_id": ll, "field": "clip"}, "destination": {"node_id": pc, "field": "clip"}},
            {"source": {"node_id": ll, "field": "clip"}, "destination": {"node_id": nc, "field": "clip"}},
            {"source": {"node_id": ll, "field": "clip2"}, "destination": {"node_id": pc, "field": "clip2"}},
            {"source": {"node_id": ll, "field": "clip2"}, "destination": {"node_id": nc, "field": "clip2"}},
        ])
    else:
        edges.extend([
            {"source": {"node_id": ml, "field": "unet"}, "destination": {"node_id": dn, "field": "unet"}},
            {"source": {"node_id": ml, "field": "clip"}, "destination": {"node_id": pc, "field": "clip"}},
            {"source": {"node_id": ml, "field": "clip"}, "destination": {"node_id": nc, "field": "clip"}},
            {"source": {"node_id": ml, "field": "clip2"}, "destination": {"node_id": pc, "field": "clip2"}},
            {"source": {"node_id": ml, "field": "clip2"}, "destination": {"node_id": nc, "field": "clip2"}},
        ])

    return {"id": uuid.uuid4().hex, "nodes": nodes, "edges": edges}


def build_sd1_graph(
    model: dict,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    cfg_rescale: float,
    scheduler: str,
    loras: list[dict],
) -> dict:
    """Build an SD1.5 text-to-image execution graph."""
    ml = f"main_model_loader:{uuid.uuid4().hex[:10]}"
    pp = f"positive_prompt:{uuid.uuid4().hex[:10]}"
    np_ = f"negative_prompt:{uuid.uuid4().hex[:10]}"
    pc = f"pos_cond:{uuid.uuid4().hex[:10]}"
    pcc = f"pos_cond_collect:{uuid.uuid4().hex[:10]}"
    nc = f"neg_cond:{uuid.uuid4().hex[:10]}"
    ncc = f"neg_cond_collect:{uuid.uuid4().hex[:10]}"
    sd = f"seed:{uuid.uuid4().hex[:10]}"
    ns = f"noise:{uuid.uuid4().hex[:10]}"
    dn = f"denoise_latents:{uuid.uuid4().hex[:10]}"
    cm = f"core_metadata:{uuid.uuid4().hex[:10]}"
    out = f"canvas_output:{uuid.uuid4().hex[:10]}"

    nodes = {
        ml: {
            "type": "main_model_loader", "id": ml, "is_intermediate": True, "use_cache": True,
            "model": model,
        },
        pp: {"type": "string", "id": pp, "is_intermediate": True, "use_cache": True, "value": positive_prompt},
        np_: {"type": "string", "id": np_, "is_intermediate": True, "use_cache": True, "value": negative_prompt},
        pc: {"type": "compel", "id": pc, "is_intermediate": True, "use_cache": True, "prompt": ""},
        pcc: {"type": "collect", "id": pcc, "is_intermediate": True, "use_cache": True, "collection": []},
        nc: {"type": "compel", "id": nc, "is_intermediate": True, "use_cache": True, "prompt": negative_prompt},
        ncc: {"type": "collect", "id": ncc, "is_intermediate": True, "use_cache": True, "collection": []},
        sd: {"type": "integer", "id": sd, "is_intermediate": True, "use_cache": True, "value": seed},
        ns: {
            "type": "noise", "id": ns, "is_intermediate": True, "use_cache": True,
            "seed": 0, "width": width, "height": height, "use_cpu": True,
        },
        dn: {
            "type": "denoise_latents", "id": dn, "is_intermediate": True, "use_cache": True,
            "steps": steps, "cfg_scale": cfg_scale, "denoising_start": 0.0, "denoising_end": 1.0,
            "scheduler": scheduler, "cfg_rescale_multiplier": cfg_rescale,
        },
        cm: {
            "type": "core_metadata", "id": cm, "is_intermediate": True, "use_cache": True,
            "generation_mode": "txt2img", "negative_prompt": negative_prompt,
            "width": width, "height": height, "rand_device": "cpu",
            "cfg_scale": cfg_scale, "cfg_rescale_multiplier": cfg_rescale,
            "steps": steps, "scheduler": scheduler,
            "seamless_x": False, "seamless_y": False,
            "model": model,
            "loras": [{"model": l["model"], "weight": l["weight"]} for l in loras],
            "ref_images": [],
        },
        out: {
            "type": "l2i", "id": out, "is_intermediate": False, "use_cache": False,
            "tiled": False, "tile_size": 0, "fp32": True,
        },
    }

    edges = [
        {"source": {"node_id": pp, "field": "value"}, "destination": {"node_id": pc, "field": "prompt"}},
        {"source": {"node_id": pc, "field": "conditioning"}, "destination": {"node_id": pcc, "field": "item"}},
        {"source": {"node_id": pcc, "field": "collection"}, "destination": {"node_id": dn, "field": "positive_conditioning"}},
        {"source": {"node_id": nc, "field": "conditioning"}, "destination": {"node_id": ncc, "field": "item"}},
        {"source": {"node_id": ncc, "field": "collection"}, "destination": {"node_id": dn, "field": "negative_conditioning"}},
        {"source": {"node_id": sd, "field": "value"}, "destination": {"node_id": ns, "field": "seed"}},
        {"source": {"node_id": ns, "field": "noise"}, "destination": {"node_id": dn, "field": "noise"}},
        {"source": {"node_id": dn, "field": "latents"}, "destination": {"node_id": out, "field": "latents"}},
        {"source": {"node_id": sd, "field": "value"}, "destination": {"node_id": cm, "field": "seed"}},
        {"source": {"node_id": pp, "field": "value"}, "destination": {"node_id": cm, "field": "positive_prompt"}},
        {"source": {"node_id": cm, "field": "metadata"}, "destination": {"node_id": out, "field": "metadata"}},
        {"source": {"node_id": ml, "field": "vae"}, "destination": {"node_id": out, "field": "vae"}},
    ]

    if loras:
        lc = f"lora_collector:{uuid.uuid4().hex[:10]}"
        ll = f"lora_collection_loader:{uuid.uuid4().hex[:10]}"
        nodes[lc] = {"type": "collect", "id": lc, "is_intermediate": True, "use_cache": True, "collection": []}
        nodes[ll] = {"type": "lora_collection_loader", "id": ll, "is_intermediate": True, "use_cache": True}

        for i, lora in enumerate(loras):
            ls = f"lora_selector_{i}:{uuid.uuid4().hex[:10]}"
            nodes[ls] = {
                "type": "lora_selector", "id": ls, "is_intermediate": True, "use_cache": True,
                "lora": lora["model"], "weight": lora["weight"],
            }
            edges.append({"source": {"node_id": ls, "field": "lora"}, "destination": {"node_id": lc, "field": "item"}})

        edges.extend([
            {"source": {"node_id": lc, "field": "collection"}, "destination": {"node_id": ll, "field": "loras"}},
            {"source": {"node_id": ml, "field": "unet"}, "destination": {"node_id": ll, "field": "unet"}},
            {"source": {"node_id": ml, "field": "clip"}, "destination": {"node_id": ll, "field": "clip"}},
            {"source": {"node_id": ll, "field": "unet"}, "destination": {"node_id": dn, "field": "unet"}},
            {"source": {"node_id": ll, "field": "clip"}, "destination": {"node_id": pc, "field": "clip"}},
            {"source": {"node_id": ll, "field": "clip"}, "destination": {"node_id": nc, "field": "clip"}},
        ])
    else:
        edges.extend([
            {"source": {"node_id": ml, "field": "unet"}, "destination": {"node_id": dn, "field": "unet"}},
            {"source": {"node_id": ml, "field": "clip"}, "destination": {"node_id": pc, "field": "clip"}},
            {"source": {"node_id": ml, "field": "clip"}, "destination": {"node_id": nc, "field": "clip"}},
        ])

    return {"id": uuid.uuid4().hex, "nodes": nodes, "edges": edges}


def build_flux_graph(
    model: dict,
    positive_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    scheduler: str,
    t5_encoder_model: dict,
    clip_embed_model: dict,
    vae_model: dict,
) -> dict:
    """Build a standard FLUX text-to-image execution graph."""
    ml = f"flux_model_loader:{uuid.uuid4().hex[:10]}"
    pp = f"positive_prompt:{uuid.uuid4().hex[:10]}"
    te = f"flux_text_encoder:{uuid.uuid4().hex[:10]}"
    sd = f"seed:{uuid.uuid4().hex[:10]}"
    dn = f"flux_denoise:{uuid.uuid4().hex[:10]}"
    out = f"flux_vae_decode:{uuid.uuid4().hex[:10]}"

    nodes = {
        ml: {
            "type": "flux_model_loader", "id": ml, "is_intermediate": True, "use_cache": True,
            "model": model, "t5_encoder_model": t5_encoder_model,
            "clip_embed_model": clip_embed_model, "vae_model": vae_model,
        },
        pp: {"type": "string", "id": pp, "is_intermediate": True, "use_cache": True, "value": positive_prompt},
        te: {"type": "flux_text_encoder", "id": te, "is_intermediate": True, "use_cache": True},
        sd: {"type": "rand_int", "id": sd, "is_intermediate": True, "use_cache": False, "low": seed, "high": seed + 1},
        dn: {
            "type": "flux_denoise", "id": dn, "is_intermediate": True, "use_cache": True,
            "width": width, "height": height, "num_steps": steps, "guidance": guidance,
            "scheduler": scheduler, "denoising_start": 0.0, "denoising_end": 1.0,
        },
        out: {"type": "flux_vae_decode", "id": out, "is_intermediate": False, "use_cache": False},
    }

    edges = [
        {"source": {"node_id": ml, "field": "clip"}, "destination": {"node_id": te, "field": "clip"}},
        {"source": {"node_id": ml, "field": "t5_encoder"}, "destination": {"node_id": te, "field": "t5_encoder"}},
        {"source": {"node_id": ml, "field": "max_seq_len"}, "destination": {"node_id": te, "field": "t5_max_seq_len"}},
        {"source": {"node_id": pp, "field": "value"}, "destination": {"node_id": te, "field": "prompt"}},
        {"source": {"node_id": te, "field": "conditioning"}, "destination": {"node_id": dn, "field": "positive_text_conditioning"}},
        {"source": {"node_id": ml, "field": "transformer"}, "destination": {"node_id": dn, "field": "transformer"}},
        {"source": {"node_id": sd, "field": "value"}, "destination": {"node_id": dn, "field": "seed"}},
        {"source": {"node_id": dn, "field": "latents"}, "destination": {"node_id": out, "field": "latents"}},
        {"source": {"node_id": ml, "field": "vae"}, "destination": {"node_id": out, "field": "vae"}},
    ]

    return {"id": uuid.uuid4().hex, "nodes": nodes, "edges": edges}


def build_flux2_graph(
    model: dict,
    positive_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
) -> dict:
    """Build a FLUX.2 Klein text-to-image execution graph."""
    ml = f"flux2_klein_model_loader:{uuid.uuid4().hex[:10]}"
    pp = f"positive_prompt:{uuid.uuid4().hex[:10]}"
    te = f"flux2_klein_text_encoder:{uuid.uuid4().hex[:10]}"
    sd = f"seed:{uuid.uuid4().hex[:10]}"
    dn = f"flux2_denoise:{uuid.uuid4().hex[:10]}"
    out = f"flux2_vae_decode:{uuid.uuid4().hex[:10]}"

    nodes = {
        ml: {
            "type": "flux2_klein_model_loader", "id": ml, "is_intermediate": True, "use_cache": True,
            "model": model,
        },
        pp: {"type": "string", "id": pp, "is_intermediate": True, "use_cache": True, "value": positive_prompt},
        te: {"type": "flux2_klein_text_encoder", "id": te, "is_intermediate": True, "use_cache": True},
        sd: {"type": "rand_int", "id": sd, "is_intermediate": True, "use_cache": False, "low": seed, "high": seed + 1},
        dn: {
            "type": "flux2_denoise", "id": dn, "is_intermediate": True, "use_cache": True,
            "width": width, "height": height, "num_steps": steps,
        },
        out: {"type": "flux2_vae_decode", "id": out, "is_intermediate": False, "use_cache": False},
    }

    edges = [
        {"source": {"node_id": ml, "field": "qwen3_encoder"}, "destination": {"node_id": te, "field": "qwen3_encoder"}},
        {"source": {"node_id": ml, "field": "max_seq_len"}, "destination": {"node_id": te, "field": "max_seq_len"}},
        {"source": {"node_id": pp, "field": "value"}, "destination": {"node_id": te, "field": "prompt"}},
        {"source": {"node_id": te, "field": "conditioning"}, "destination": {"node_id": dn, "field": "positive_text_conditioning"}},
        {"source": {"node_id": ml, "field": "transformer"}, "destination": {"node_id": dn, "field": "transformer"}},
        {"source": {"node_id": ml, "field": "vae"}, "destination": {"node_id": dn, "field": "vae"}},
        {"source": {"node_id": sd, "field": "value"}, "destination": {"node_id": dn, "field": "seed"}},
        {"source": {"node_id": dn, "field": "latents"}, "destination": {"node_id": out, "field": "latents"}},
        {"source": {"node_id": ml, "field": "vae"}, "destination": {"node_id": out, "field": "vae"}},
    ]

    return {"id": uuid.uuid4().hex, "nodes": nodes, "edges": edges}


def build_zimage_graph(
    model: dict,
    positive_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float = 1.0,
    scheduler: str = "euler",
) -> dict:
    """Build a Z-Image text-to-image execution graph."""
    ml = f"z_image_model_loader:{uuid.uuid4().hex[:10]}"
    pp = f"positive_prompt:{uuid.uuid4().hex[:10]}"
    te = f"z_image_text_encoder:{uuid.uuid4().hex[:10]}"
    sd = f"seed:{uuid.uuid4().hex[:10]}"
    dn = f"z_image_denoise:{uuid.uuid4().hex[:10]}"
    out = f"z_image_l2i:{uuid.uuid4().hex[:10]}"

    nodes = {
        ml: {
            "type": "z_image_model_loader", "id": ml, "is_intermediate": True, "use_cache": True,
            "model": model,
        },
        pp: {"type": "string", "id": pp, "is_intermediate": True, "use_cache": True, "value": positive_prompt},
        te: {"type": "z_image_text_encoder", "id": te, "is_intermediate": True, "use_cache": True},
        sd: {"type": "rand_int", "id": sd, "is_intermediate": True, "use_cache": False, "low": seed, "high": seed + 1},
        dn: {
            "type": "z_image_denoise", "id": dn, "is_intermediate": True, "use_cache": True,
            "width": width, "height": height, "steps": steps,
            "guidance_scale": cfg_scale, "scheduler": scheduler,
        },
        out: {"type": "z_image_l2i", "id": out, "is_intermediate": False, "use_cache": False},
    }

    edges = [
        {"source": {"node_id": ml, "field": "qwen3_encoder"}, "destination": {"node_id": te, "field": "qwen3_encoder"}},
        {"source": {"node_id": pp, "field": "value"}, "destination": {"node_id": te, "field": "prompt"}},
        {"source": {"node_id": te, "field": "conditioning"}, "destination": {"node_id": dn, "field": "positive_conditioning"}},
        {"source": {"node_id": ml, "field": "transformer"}, "destination": {"node_id": dn, "field": "transformer"}},
        {"source": {"node_id": ml, "field": "vae"}, "destination": {"node_id": dn, "field": "vae"}},
        {"source": {"node_id": sd, "field": "value"}, "destination": {"node_id": dn, "field": "seed"}},
        {"source": {"node_id": dn, "field": "latents"}, "destination": {"node_id": out, "field": "latents"}},
        {"source": {"node_id": ml, "field": "vae"}, "destination": {"node_id": out, "field": "vae"}},
    ]

    return {"id": uuid.uuid4().hex, "nodes": nodes, "edges": edges}
