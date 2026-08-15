#!/usr/bin/env python3
"""Model Compare — companion web service for InvokeAI.

Reads generation settings from InvokeAI's saved state and lets you
generate one image per model with the same prompt/settings, all in one click.
Results appear in InvokeAI's gallery automatically.

Usage (from InvokeAI project root, with venv active):
    python scripts/model_compare/server.py
    python scripts/model_compare/server.py --port 9091
"""

import argparse
import json
import random
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).parent))
from graphs import (
    build_anima_graph,
    build_ernie_graph,
    build_flux2_graph,
    build_flux_graph,
    build_krea2_graph,
    build_qwen_image_graph,
    build_sd1_graph,
    build_sdxl_graph,
    build_wan_graph,
    build_zimage_graph,
)

# ── Defaults ─────────────────────────────────────────────────────────────────

INVOKEAI_API_DEFAULT = "http://127.0.0.1:9090"
DEFAULT_PORT = 9091

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Model Compare")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Set via CLI args at startup
invokeai_api_url: str = INVOKEAI_API_DEFAULT


# ── Models ───────────────────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    model_keys: list[str]
    width: int | None = None
    height: int | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────


# client_state slices needed to reconstruct generation settings
CLIENT_STATE_KEYS = ("params", "canvas", "loras")


async def read_client_state() -> dict:
    """Read client_state from the live InvokeAI server via its API.

    Reading the running instance (rather than a guessed SQLite path) guarantees
    Model Compare reuses whatever settings are currently active in InvokeAI — the
    same instance it enqueues jobs to.
    """
    state: dict = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for key in CLIENT_STATE_KEYS:
            resp = await client.get(
                f"{invokeai_api_url}/api/v1/client_state/default/get_by_key",
                params={"key": key},
            )
            resp.raise_for_status()
            raw = resp.json()  # endpoint returns the stringified slice, or null
            state[key] = json.loads(raw) if raw else {}
    return state


def remap_loras(loras: list[dict], installed: list[dict], base: str) -> tuple[list[dict], list[str]]:
    """Remap saved LoRAs to currently-installed models for a given base.

    Saved client_state holds stale keys when LoRAs are re-installed (same hash/name,
    new UUID). Match installed records by hash first, then (name, base), and rewrite
    the model dict so InvokeAI can resolve it. Unresolvable LoRAs are skipped.
    """
    by_hash = {m["hash"]: m for m in installed if m.get("hash")}
    by_name = {(m["name"], m["base"]): m for m in installed}

    remapped: list[dict] = []
    warnings: list[str] = []
    for entry in loras:
        model = entry["model"]
        if model.get("base") != base:
            continue
        match = by_hash.get(model.get("hash")) or by_name.get((model.get("name"), model.get("base")))
        if not match:
            warnings.append(f"LoRA '{model.get('name')}' not installed — skipped")
            continue
        remapped.append({
            "model": {
                "key": match["key"], "hash": match.get("hash", ""),
                "name": match["name"], "base": match["base"], "type": "lora",
            },
            "weight": entry["weight"],
        })
    return remapped, warnings


def _submodel_field(m: dict) -> dict:
    """Build a ModelIdentifierField dict from an InvokeAI /api/v2/models/ record."""
    return {"key": m["key"], "hash": m.get("hash", ""), "name": m["name"], "base": m["base"], "type": m["type"]}


def pick_submodels(
    vaes: list[dict], encoders: list[dict], vae_bases: tuple[str, ...], vae_hint_key: str | None = None
) -> tuple[dict | None, dict | None]:
    """Pick a compatible VAE + text encoder from installed models (mirrors invokepush._pick_submodels).

    `vae_bases` is ordered by preference. A VAE whose key matches `vae_hint_key` (the VAE currently set
    in InvokeAI) wins if it's compatible; otherwise prefer a Qwen-Image-named VAE in the most-preferred
    base, then the first of that base. The encoder is the first installed model of the requested type.
    """
    compatible = [v for v in vaes if v.get("base") in vae_bases]
    vae = None
    if compatible:
        if vae_hint_key:
            vae = next((v for v in compatible if v["key"] == vae_hint_key), None)
        if vae is None:
            compatible.sort(key=lambda v: vae_bases.index(v["base"]))
            top_base = [v for v in compatible if v["base"] == compatible[0]["base"]]
            named = [v for v in top_base if "qwen" in v["name"].lower() and "vae" in v["name"].lower()]
            vae = (named or top_base)[0]
    return (_submodel_field(vae) if vae else None), (_submodel_field(encoders[0]) if encoders else None)


def parse_settings(state: dict) -> dict:
    """Extract generation settings from client_state."""
    params = state.get("params", {})
    canvas = state.get("canvas", {})
    loras_state = state.get("loras", {})

    bbox = canvas.get("bbox", {})
    rect = bbox.get("rect", {})
    scaled = bbox.get("scaledSize", {})

    # Use scaled size if available (SDXL optimal), fall back to rect
    width = scaled.get("width") or rect.get("width", 1024)
    height = scaled.get("height") or rect.get("height", 1024)

    # Parse LoRAs
    loras = []
    for lora_entry in loras_state.get("loras", []):
        if lora_entry.get("isEnabled", True):
            loras.append({
                "model": lora_entry["model"],
                "weight": lora_entry.get("weight", 0.75),
            })

    aspect_ratio = bbox.get("aspectRatio", {})

    return {
        "positivePrompt": params.get("positivePrompt") or "",
        "negativePrompt": params.get("negativePrompt") or "",
        "steps": params.get("steps", 30),
        "cfgScale": params.get("cfgScale", 7.0),
        "cfgRescaleMultiplier": params.get("cfgRescaleMultiplier", 0.0),
        "scheduler": params.get("scheduler", "euler"),
        "seed": params.get("seed", 0),
        "shouldRandomizeSeed": params.get("shouldRandomizeSeed", True),
        "model": params.get("model"),
        "vae": params.get("vae"),
        "width": width,
        "height": height,
        "aspectRatioId": aspect_ratio.get("id", "Free"),
        "loras": loras,
        # Flux-specific
        "guidance": params.get("guidance", 4),
        "fluxScheduler": params.get("fluxScheduler", "euler"),
        "fluxVAE": params.get("fluxVAE"),
        "t5EncoderModel": params.get("t5EncoderModel"),
        "clipEmbedModel": params.get("clipEmbedModel"),
        # Base-specific submodel overrides. InvokeAI stores the Krea-2 / Anima / Qwen-Image / Wan
        # VAE + text-encoder picks under per-base keys (NOT the generic `vae`), so we forward them
        # verbatim and let the generation branches prefer them over auto-picked submodels.
        "krea2VaeModel": params.get("krea2VaeModel"),
        "krea2Qwen3VlEncoderModel": params.get("krea2Qwen3VlEncoderModel"),
        "animaVaeModel": params.get("animaVaeModel"),
        "animaQwen3EncoderModel": params.get("animaQwen3EncoderModel"),
        "qwenImageVaeModel": params.get("qwenImageVaeModel"),
        "qwenImageQwenVLEncoderModel": params.get("qwenImageQwenVLEncoderModel"),
        "wanVaeModel": params.get("wanVaeModel"),
        "wanT5EncoderModel": params.get("wanT5EncoderModel"),
    }




# ── API Routes ───────────────────────────────────────────────────────────────


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/settings")
async def get_settings():
    """Read current generation settings from InvokeAI's saved state."""
    try:
        state = await read_client_state()
        return parse_settings(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_models():
    """Get available main models from InvokeAI."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{invokeai_api_url}/api/v2/models/", params={"model_type": "main"})
        resp.raise_for_status()
        data = resp.json()

    models = []
    for m in data.get("models", []):
        models.append({
            "key": m["key"],
            "name": m["name"],
            "base": m["base"],
            "format": m.get("format", ""),
            "hash": m.get("hash", ""),
            "type": m.get("type", "main"),
        })

    models.sort(key=lambda x: (x["base"], x["name"].lower()))
    return {"models": models}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """Build generation graphs and enqueue them in InvokeAI's queue."""
    # Read fresh settings from InvokeAI's saved state
    state = await read_client_state()
    settings = parse_settings(state)

    positive_prompt = settings["positivePrompt"]
    negative_prompt = settings["negativePrompt"]
    steps = settings["steps"]
    cfg_scale = settings["cfgScale"]
    cfg_rescale = settings["cfgRescaleMultiplier"]
    scheduler = settings["scheduler"]
    width = req.width if req.width else settings["width"]
    height = req.height if req.height else settings["height"]
    loras = settings["loras"]

    # Use saved seed, or generate random if InvokeAI is set to randomize
    base_seed = random.randint(0, 2**32 - 1) if settings["shouldRandomizeSeed"] else settings["seed"]

    # Fetch model details for each key, plus installed LoRAs for key remapping
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{invokeai_api_url}/api/v2/models/", params={"model_type": "main"})
        resp.raise_for_status()
        all_models = {m["key"]: m for m in resp.json().get("models", [])}

        lora_resp = await client.get(f"{invokeai_api_url}/api/v2/models/", params={"model_type": "lora"})
        lora_resp.raise_for_status()
        installed_loras = lora_resp.json().get("models", [])

        # Standalone submodels for Krea-2 / Anima (single-file & GGUF mains carry no bundled VAE/encoder).
        vae_resp = await client.get(f"{invokeai_api_url}/api/v2/models/", params={"model_type": "vae"})
        vae_resp.raise_for_status()
        installed_vaes = vae_resp.json().get("models", [])
        qwen3vl_resp = await client.get(f"{invokeai_api_url}/api/v2/models/", params={"model_type": "qwen3_vl_encoder"})
        qwen3vl_resp.raise_for_status()
        qwen3_vl_encoders = qwen3vl_resp.json().get("models", [])
        qwen3_resp = await client.get(f"{invokeai_api_url}/api/v2/models/", params={"model_type": "qwen3_encoder"})
        qwen3_resp.raise_for_status()
        qwen3_encoders = qwen3_resp.json().get("models", [])
        # Qwen-Image uses a Qwen2.5-VL encoder (distinct from Krea-2's qwen3_vl_encoder).
        qwenvl_resp = await client.get(f"{invokeai_api_url}/api/v2/models/", params={"model_type": "qwen_vl_encoder"})
        qwenvl_resp.raise_for_status()
        qwen_vl_encoders = qwenvl_resp.json().get("models", [])
        # Wan encodes with a UMT5 (wan_t5_encoder) and decodes with a Wan VAE.
        want5_resp = await client.get(f"{invokeai_api_url}/api/v2/models/", params={"model_type": "wan_t5_encoder"})
        want5_resp.raise_for_status()
        wan_t5_encoders = want5_resp.json().get("models", [])

    vae_hint_key = (settings.get("vae") or {}).get("key")

    enqueued = 0
    errors = []

    async with httpx.AsyncClient(timeout=60) as client:
        for i, key in enumerate(req.model_keys):
            model_info = all_models.get(key)
            if not model_info:
                errors.append(f"Model {key} not found")
                continue

            model_ref = {
                "key": model_info["key"],
                "hash": model_info.get("hash", ""),
                "name": model_info["name"],
                "base": model_info["base"],
                "type": "main",
            }

            seed = base_seed

            # Filter LoRAs to same base as model, remapping stale keys to installed models
            compatible_loras, lora_warnings = remap_loras(loras, installed_loras, model_info["base"])
            errors.extend(f"{model_info['name']}: {w}" for w in lora_warnings)

            base = model_info["base"]
            if base in ("sdxl", "sdxl-refiner"):
                graph = build_sdxl_graph(
                    model=model_ref, positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt, seed=seed,
                    width=width, height=height, steps=steps,
                    cfg_scale=cfg_scale, cfg_rescale=cfg_rescale,
                    scheduler=scheduler, loras=compatible_loras,
                )
            elif base in ("sd-1", "sd-2"):
                graph = build_sd1_graph(
                    model=model_ref, positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt, seed=seed,
                    width=width, height=height, steps=steps,
                    cfg_scale=cfg_scale, cfg_rescale=cfg_rescale,
                    scheduler=scheduler, loras=compatible_loras,
                )
            elif base == "flux":
                if not settings["t5EncoderModel"] or not settings["clipEmbedModel"] or not settings["fluxVAE"]:
                    errors.append(f"Skipped {model_info['name']} (needs T5 encoder, CLIP embed, and VAE configured in InvokeAI)")
                    continue
                graph = build_flux_graph(
                    model=model_ref, positive_prompt=positive_prompt,
                    seed=seed, width=width, height=height,
                    steps=steps, guidance=settings["guidance"],
                    scheduler=settings["fluxScheduler"],
                    t5_encoder_model=settings["t5EncoderModel"],
                    clip_embed_model=settings["clipEmbedModel"],
                    vae_model=settings["fluxVAE"],
                )
            elif base == "flux2":
                graph = build_flux2_graph(
                    model=model_ref, positive_prompt=positive_prompt,
                    seed=seed, width=width, height=height,
                    steps=steps,
                )
            elif base == "z-image":
                graph = build_zimage_graph(
                    model=model_ref, positive_prompt=positive_prompt,
                    seed=seed, width=width, height=height,
                    steps=steps, cfg_scale=cfg_scale,
                    scheduler=settings.get("zImageScheduler", "euler"),
                )
            elif base == "krea-2":
                # Prefer the VAE + Qwen3-VL encoder explicitly selected in InvokeAI; auto-pick only
                # fills gaps. Diffusers mains bundle both, so single-file/GGUF need them supplied.
                vae_model = settings.get("krea2VaeModel")
                qwen3_vl_encoder_model = settings.get("krea2Qwen3VlEncoderModel")
                if model_info.get("format") != "diffusers" and not (vae_model and qwen3_vl_encoder_model):
                    picked_vae, picked_enc = pick_submodels(
                        installed_vaes, qwen3_vl_encoders, ("qwen-image", "anima"), vae_hint_key
                    )
                    vae_model = vae_model or picked_vae
                    qwen3_vl_encoder_model = qwen3_vl_encoder_model or picked_enc
                    if not (vae_model and qwen3_vl_encoder_model):
                        errors.append(
                            f"Skipped {model_info['name']} (needs a Qwen-Image VAE and a Qwen3-VL encoder installed)"
                        )
                        continue
                graph = build_krea2_graph(
                    model=model_ref, positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt, seed=seed,
                    width=width, height=height, steps=steps, cfg_scale=cfg_scale,
                    loras=compatible_loras, vae_model=vae_model,
                    qwen3_vl_encoder_model=qwen3_vl_encoder_model,
                )
            elif base == "anima":
                # Anima always needs a 16-channel VAE + Qwen3 0.6B encoder, regardless of main format.
                # Prefer the pair selected in InvokeAI; auto-pick only fills gaps.
                vae_model = settings.get("animaVaeModel")
                qwen3_encoder_model = settings.get("animaQwen3EncoderModel")
                if not (vae_model and qwen3_encoder_model):
                    picked_vae, picked_enc = pick_submodels(
                        installed_vaes, qwen3_encoders, ("anima", "qwen-image", "flux"), vae_hint_key
                    )
                    vae_model = vae_model or picked_vae
                    qwen3_encoder_model = qwen3_encoder_model or picked_enc
                    if not (vae_model and qwen3_encoder_model):
                        errors.append(
                            f"Skipped {model_info['name']} (needs a 16-channel VAE and a Qwen3 0.6B encoder installed)"
                        )
                        continue
                graph = build_anima_graph(
                    model=model_ref, positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt, seed=seed,
                    width=width, height=height, steps=steps,
                    guidance_scale=cfg_scale, scheduler=scheduler,
                    loras=compatible_loras, vae_model=vae_model,
                    qwen3_encoder_model=qwen3_encoder_model,
                )
            elif base == "ernie-image":
                # ERNIE mains are self-contained diffusers pipelines — no standalone submodels.
                graph = build_ernie_graph(
                    model=model_ref, positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt, seed=seed,
                    width=width, height=height, steps=steps,
                    guidance_scale=cfg_scale, loras=compatible_loras,
                )
            elif base == "qwen-image":
                # Prefer the VAE + Qwen2.5-VL encoder selected in InvokeAI; auto-pick only fills gaps.
                # Diffusers mains bundle both, so single-file/GGUF need them supplied.
                vae_model = settings.get("qwenImageVaeModel")
                qwen_vl_encoder_model = settings.get("qwenImageQwenVLEncoderModel")
                if model_info.get("format") != "diffusers" and not (vae_model and qwen_vl_encoder_model):
                    picked_vae, picked_enc = pick_submodels(
                        installed_vaes, qwen_vl_encoders, ("qwen-image",), vae_hint_key
                    )
                    vae_model = vae_model or picked_vae
                    qwen_vl_encoder_model = qwen_vl_encoder_model or picked_enc
                    if not (vae_model and qwen_vl_encoder_model):
                        errors.append(
                            f"Skipped {model_info['name']} (needs a Qwen-Image VAE and a Qwen2.5-VL encoder installed)"
                        )
                        continue
                graph = build_qwen_image_graph(
                    model=model_ref, positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt, seed=seed,
                    width=width, height=height, steps=steps, cfg_scale=cfg_scale,
                    loras=compatible_loras, vae_model=vae_model,
                    qwen_vl_encoder_model=qwen_vl_encoder_model,
                )
            elif base == "wan":
                # Wan still image (num_frames=1). Prefer the VAE + UMT5 selected in InvokeAI; auto-pick
                # only fills gaps. Diffusers mains bundle both, so others need them supplied.
                vae_model = settings.get("wanVaeModel")
                wan_t5_encoder_model = settings.get("wanT5EncoderModel")
                if model_info.get("format") != "diffusers" and not (vae_model and wan_t5_encoder_model):
                    picked_vae, picked_enc = pick_submodels(
                        installed_vaes, wan_t5_encoders, ("wan",), vae_hint_key
                    )
                    vae_model = vae_model or picked_vae
                    wan_t5_encoder_model = wan_t5_encoder_model or picked_enc
                    if not (vae_model and wan_t5_encoder_model):
                        errors.append(
                            f"Skipped {model_info['name']} (needs a Wan VAE and a Wan UMT5 encoder installed)"
                        )
                        continue
                graph = build_wan_graph(
                    model=model_ref, positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt, seed=seed,
                    width=width, height=height, steps=steps, cfg_scale=cfg_scale,
                    loras=compatible_loras, vae_model=vae_model,
                    wan_t5_encoder_model=wan_t5_encoder_model,
                )
            else:
                errors.append(f"Skipped {model_info['name']} ('{base}' not supported)")
                continue

            batch = {
                "batch": {
                    "graph": graph,
                    "runs": 1,
                    "origin": "model_compare",
                },
                "prepend": False,
            }

            try:
                resp = await client.post(
                    f"{invokeai_api_url}/api/v1/queue/default/enqueue_batch",
                    json=batch,
                )
                if resp.status_code != 200:
                    errors.append(f"{model_info['name']}: {resp.status_code} {resp.text[:300]}")
                    continue
                enqueued += 1
            except Exception as e:
                errors.append(f"Failed to enqueue {model_info['name']}: {e}")

    return {
        "enqueued": enqueued,
        "total": len(req.model_keys),
        "seed": base_seed,
        "errors": errors,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    global invokeai_api_url

    parser = argparse.ArgumentParser(description="Model Compare — companion service for InvokeAI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--invokeai-api", default=INVOKEAI_API_DEFAULT)
    args = parser.parse_args()

    invokeai_api_url = args.invokeai_api

    print(f"Model Compare starting on http://127.0.0.1:{args.port}")
    print(f"InvokeAI API: {invokeai_api_url}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
