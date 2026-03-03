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
from graphs import build_flux2_graph, build_flux_graph, build_sd1_graph, build_sdxl_graph

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
        "loras": loras,
        # Flux-specific
        "guidance": params.get("guidance", 4),
        "fluxScheduler": params.get("fluxScheduler", "euler"),
        "fluxVAE": params.get("fluxVAE"),
        "t5EncoderModel": params.get("t5EncoderModel"),
        "clipEmbedModel": params.get("clipEmbedModel"),
    }




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
