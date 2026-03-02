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
import sqlite3
import uuid
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Defaults ─────────────────────────────────────────────────────────────────

INVOKEAI_DB_DEFAULT = "/mnt/llm/hub/invokeai_data/databases/invokeai.db"
INVOKEAI_API_DEFAULT = "http://127.0.0.1:9090"
DEFAULT_PORT = 9091

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Model Compare")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Set via CLI args at startup
invokeai_db_path: str = INVOKEAI_DB_DEFAULT
invokeai_api_url: str = INVOKEAI_API_DEFAULT


# ── Models ───────────────────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    model_keys: list[str]


# ── Helpers ──────────────────────────────────────────────────────────────────


def read_client_state() -> dict:
    """Read client_state from InvokeAI's SQLite database."""
    conn = sqlite3.connect(invokeai_db_path)
    try:
        rows = conn.execute(
            "SELECT key, value FROM client_state WHERE user_id='system'"
        ).fetchall()
        return {key: json.loads(value) for key, value in rows}
    finally:
        conn.close()


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

    return {
        "positivePrompt": params.get("positivePrompt", ""),
        "negativePrompt": params.get("negativePrompt", ""),
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
        "loras": loras,
    }


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
    # Node IDs
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
            "type": "sdxl_model_loader",
            "id": ml,
            "is_intermediate": True,
            "use_cache": True,
            "model": model,
        },
        pp: {
            "type": "string",
            "id": pp,
            "is_intermediate": True,
            "use_cache": True,
            "value": positive_prompt,
        },
        np_: {
            "type": "string",
            "id": np_,
            "is_intermediate": True,
            "use_cache": True,
            "value": negative_prompt,
        },
        pc: {
            "type": "sdxl_compel_prompt",
            "id": pc,
            "is_intermediate": True,
            "use_cache": True,
            "prompt": "",
            "style": "",
            "original_width": width,
            "original_height": height,
            "crop_top": 0,
            "crop_left": 0,
            "target_width": width,
            "target_height": height,
        },
        pcc: {
            "type": "collect",
            "id": pcc,
            "is_intermediate": True,
            "use_cache": True,
            "collection": [],
        },
        nc: {
            "type": "sdxl_compel_prompt",
            "id": nc,
            "is_intermediate": True,
            "use_cache": True,
            "prompt": negative_prompt,
            "style": negative_prompt,
            "original_width": width,
            "original_height": height,
            "crop_top": 0,
            "crop_left": 0,
            "target_width": width,
            "target_height": height,
        },
        ncc: {
            "type": "collect",
            "id": ncc,
            "is_intermediate": True,
            "use_cache": True,
            "collection": [],
        },
        sd: {
            "type": "integer",
            "id": sd,
            "is_intermediate": True,
            "use_cache": True,
            "value": seed,
        },
        ns: {
            "type": "noise",
            "id": ns,
            "is_intermediate": True,
            "use_cache": True,
            "seed": 0,
            "width": width,
            "height": height,
            "use_cpu": True,
        },
        dn: {
            "type": "denoise_latents",
            "id": dn,
            "is_intermediate": True,
            "use_cache": True,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "denoising_start": 0.0,
            "denoising_end": 1.0,
            "scheduler": scheduler,
            "cfg_rescale_multiplier": cfg_rescale,
        },
        cm: {
            "type": "core_metadata",
            "id": cm,
            "is_intermediate": True,
            "use_cache": True,
            "generation_mode": "sdxl_txt2img",
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "rand_device": "cpu",
            "cfg_scale": cfg_scale,
            "cfg_rescale_multiplier": cfg_rescale,
            "steps": steps,
            "scheduler": scheduler,
            "seamless_x": False,
            "seamless_y": False,
            "model": model,
            "loras": [{"model": l["model"], "weight": l["weight"]} for l in loras],
            "ref_images": [],
        },
        out: {
            "type": "l2i",
            "id": out,
            "is_intermediate": False,
            "use_cache": False,
            "tiled": False,
            "tile_size": 0,
            "fp32": True,
        },
    }

    edges = [
        # Prompt text → compel nodes
        {"source": {"node_id": pp, "field": "value"}, "destination": {"node_id": pc, "field": "prompt"}},
        {"source": {"node_id": pp, "field": "value"}, "destination": {"node_id": pc, "field": "style"}},
        # Conditioning → collect → denoise
        {"source": {"node_id": pc, "field": "conditioning"}, "destination": {"node_id": pcc, "field": "item"}},
        {"source": {"node_id": pcc, "field": "collection"}, "destination": {"node_id": dn, "field": "positive_conditioning"}},
        {"source": {"node_id": nc, "field": "conditioning"}, "destination": {"node_id": ncc, "field": "item"}},
        {"source": {"node_id": ncc, "field": "collection"}, "destination": {"node_id": dn, "field": "negative_conditioning"}},
        # Seed → noise → denoise
        {"source": {"node_id": sd, "field": "value"}, "destination": {"node_id": ns, "field": "seed"}},
        {"source": {"node_id": ns, "field": "noise"}, "destination": {"node_id": dn, "field": "noise"}},
        # Denoise → output
        {"source": {"node_id": dn, "field": "latents"}, "destination": {"node_id": out, "field": "latents"}},
        # Metadata
        {"source": {"node_id": sd, "field": "value"}, "destination": {"node_id": cm, "field": "seed"}},
        {"source": {"node_id": pp, "field": "value"}, "destination": {"node_id": cm, "field": "positive_prompt"}},
        {"source": {"node_id": cm, "field": "metadata"}, "destination": {"node_id": out, "field": "metadata"}},
        # VAE from model loader
        {"source": {"node_id": ml, "field": "vae"}, "destination": {"node_id": out, "field": "vae"}},
    ]

    # LoRA chain or direct model connection
    if loras:
        lc = f"lora_collector:{uuid.uuid4().hex[:10]}"
        ll = f"sdxl_lora_collection_loader:{uuid.uuid4().hex[:10]}"
        nodes[lc] = {"type": "collect", "id": lc, "is_intermediate": True, "use_cache": True, "collection": []}
        nodes[ll] = {"type": "sdxl_lora_collection_loader", "id": ll, "is_intermediate": True, "use_cache": True}

        # Add lora_selector nodes
        for i, lora in enumerate(loras):
            ls = f"lora_selector_{i}:{uuid.uuid4().hex[:10]}"
            nodes[ls] = {
                "type": "lora_selector",
                "id": ls,
                "is_intermediate": True,
                "use_cache": True,
                "lora": lora["model"],
                "weight": lora["weight"],
            }
            edges.append({"source": {"node_id": ls, "field": "lora"}, "destination": {"node_id": lc, "field": "item"}})

        # Model → LoRA loader → denoise/compel
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
        # Direct model → denoise/compel (no LoRAs)
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
            "type": "main_model_loader",
            "id": ml,
            "is_intermediate": True,
            "use_cache": True,
            "model": model,
        },
        pp: {
            "type": "string",
            "id": pp,
            "is_intermediate": True,
            "use_cache": True,
            "value": positive_prompt,
        },
        np_: {
            "type": "string",
            "id": np_,
            "is_intermediate": True,
            "use_cache": True,
            "value": negative_prompt,
        },
        pc: {
            "type": "compel",
            "id": pc,
            "is_intermediate": True,
            "use_cache": True,
            "prompt": "",
        },
        pcc: {
            "type": "collect",
            "id": pcc,
            "is_intermediate": True,
            "use_cache": True,
            "collection": [],
        },
        nc: {
            "type": "compel",
            "id": nc,
            "is_intermediate": True,
            "use_cache": True,
            "prompt": negative_prompt,
        },
        ncc: {
            "type": "collect",
            "id": ncc,
            "is_intermediate": True,
            "use_cache": True,
            "collection": [],
        },
        sd: {
            "type": "integer",
            "id": sd,
            "is_intermediate": True,
            "use_cache": True,
            "value": seed,
        },
        ns: {
            "type": "noise",
            "id": ns,
            "is_intermediate": True,
            "use_cache": True,
            "seed": 0,
            "width": width,
            "height": height,
            "use_cpu": True,
        },
        dn: {
            "type": "denoise_latents",
            "id": dn,
            "is_intermediate": True,
            "use_cache": True,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "denoising_start": 0.0,
            "denoising_end": 1.0,
            "scheduler": scheduler,
            "cfg_rescale_multiplier": cfg_rescale,
        },
        cm: {
            "type": "core_metadata",
            "id": cm,
            "is_intermediate": True,
            "use_cache": True,
            "generation_mode": "txt2img",
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "rand_device": "cpu",
            "cfg_scale": cfg_scale,
            "cfg_rescale_multiplier": cfg_rescale,
            "steps": steps,
            "scheduler": scheduler,
            "seamless_x": False,
            "seamless_y": False,
            "model": model,
            "loras": [{"model": l["model"], "weight": l["weight"]} for l in loras],
            "ref_images": [],
        },
        out: {
            "type": "l2i",
            "id": out,
            "is_intermediate": False,
            "use_cache": False,
            "tiled": False,
            "tile_size": 0,
            "fp32": True,
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
                "type": "lora_selector",
                "id": ls,
                "is_intermediate": True,
                "use_cache": True,
                "lora": lora["model"],
                "weight": lora["weight"],
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


# ── API Routes ───────────────────────────────────────────────────────────────


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/settings")
async def get_settings():
    """Read current generation settings from InvokeAI's saved state."""
    try:
        state = read_client_state()
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
    state = read_client_state()
    settings = parse_settings(state)

    positive_prompt = settings["positivePrompt"]
    negative_prompt = settings["negativePrompt"]
    steps = settings["steps"]
    cfg_scale = settings["cfgScale"]
    cfg_rescale = settings["cfgRescaleMultiplier"]
    scheduler = settings["scheduler"]
    width = settings["width"]
    height = settings["height"]
    loras = settings["loras"]

    # Use saved seed, or generate random if InvokeAI is set to randomize
    base_seed = random.randint(0, 2**32 - 1) if settings["shouldRandomizeSeed"] else settings["seed"]

    # Fetch model details for each key
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{invokeai_api_url}/api/v2/models/", params={"model_type": "main"})
        resp.raise_for_status()
        all_models = {m["key"]: m for m in resp.json().get("models", [])}

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

            # Filter LoRAs to same base as model
            compatible_loras = [l for l in loras if l["model"].get("base") == model_info["base"]]

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
            else:
                errors.append(f"Unsupported base '{base}' for model {model_info['name']}")
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
                resp.raise_for_status()
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
    global invokeai_db_path, invokeai_api_url

    parser = argparse.ArgumentParser(description="Model Compare — companion service for InvokeAI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--invokeai-db", default=INVOKEAI_DB_DEFAULT)
    parser.add_argument("--invokeai-api", default=INVOKEAI_API_DEFAULT)
    args = parser.parse_args()

    invokeai_db_path = args.invokeai_db
    invokeai_api_url = args.invokeai_api

    print(f"Model Compare starting on http://127.0.0.1:{args.port}")
    print(f"InvokeAI API: {invokeai_api_url}")
    print(f"InvokeAI DB:  {invokeai_db_path}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
