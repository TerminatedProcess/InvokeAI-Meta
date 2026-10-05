# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project policy, review tiers, commands, and hygiene rules live in `AGENTS.md` files — the root one plus scoped ones under `invokeai/app`, `invokeai/backend`, `invokeai/frontend/{webv2,webv1,api}`, `tests`, and `.github`. Read the scoped file for any path you edit. This file only adds orientation.

@AGENTS.md

## Running

- Backend: `uv run --no-sync invokeai-web` (entry `invokeai/app/run_app.py`, FastAPI app in `invokeai/app/api_app.py`, default port 9090). Serves the built webv2 bundle by default; `--web-legacy` serves webv1.
- Frontend dev: `pnpm -C invokeai/frontend/webv2 dev` — Vite proxies to `INVOKEAI_DEV_BACKEND` (default `http://127.0.0.1:9090`).
- Single Python test: `uv run --no-sync pytest tests/path/test_file.py::test_name`
- Single webv2 test: `pnpm -C invokeai/frontend/webv2 test <path>` (unit) or `test:browser <path>` (Chromium).
- API contract regeneration after backend model/route changes: `make frontend-openapi && make frontend-typegen` (never hand-edit `invokeai/frontend/api/schema.ts`).

## Architecture

**Request → graph → invocation.** The frontend builds a node graph and enqueues it (`services/session_queue`). `services/session_processor` pulls queue items and runs them through `GraphExecutionState` (`services/shared/graph.py` + `execution_engine/`), which schedules ready nodes; control-flow (`If`, `Iterate`, `Collect`, saved-workflow calls) has dedicated planners/runtimes in `services/shared/`. Read `services/shared/README.md` before touching scheduling — many graph shapes intentionally fall back to a compatibility scheduler.

**Invocations** (`invokeai/app/invocations/`) are Pydantic node classes (`baseinvocation.py`) whose fields/versions form the saved-workflow and OpenAPI contract. They are thin: they call into `invokeai/backend/` for inference and use `InvocationContext` (`services/shared/invocation_context.py`) for services (images, models, boards, config). One invocation set per model family (`flux_*`, `flux2_*`, `sd3_*`, `cogview4_*`, `anima_*`, `z_image_*`, …) mirrors a package under `invokeai/backend/`.

**Services** (`invokeai/app/services/`) follow a base/default/sqlite triad per domain and are wired once in `api/dependencies.py` into `InvocationServices`. SQLite schema changes go through `services/shared/sqlite_migrator/`. Multi-user: auth/users services plus per-account isolation on every account-owned record.

**Model management** (`invokeai/backend/model_manager/`): `configs/` identifies model files into typed configs (base + type + format); `load/` loaders produce models held by the model cache, governed by `max_cache_ram_gb` / `max_cache_vram_gb` in `invokeai.yaml`. Device/dtype handling goes through `backend/util/devices.py`.

**Frontend.** `webv2` (React 19, Chakra 3, TanStack Query/Router) is active; layering `app → workbench/features → platform` is enforced by `pnpm architecture:check`. See `invokeai/frontend/webv2/ARCHITECTURE.md`. `webv1` is legacy. `frontend/api` owns generated `openapi.json`/`schema.ts` shared by both.

## Fork notes (branch `mryan`)

This checkout is a personal fork tracking upstream `main`. Keep fork-specific work additive — avoid patching upstream files where an addition will do — so upstream merges stay clean.
