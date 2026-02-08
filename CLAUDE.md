# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InvokeAI is an AI-powered image generation application with a Python/FastAPI backend and React/TypeScript frontend. It supports Stable Diffusion (1.5, 2.x, XL), Flux, SD3, CogView4, and other diffusion models through a node-based graph execution system.

## Common Commands

### Backend (run from repo root)

```bash
# Install dev + test dependencies
pip install ".[dev,test]"

# Run backend server (with hot reload for development)
invokeai-web --dev_reload

# Lint & format Python code
make ruff                    # or: ruff check . --fix && ruff format .

# Type checking
make mypy                    # mypy scripts/invokeai-web.py

# Run all fast tests
pytest tests/

# Run a single test file
pytest tests/app/services/model_install/test_something.py

# Run slow tests (model-dependent)
pytest tests/ -m "slow"

# Run all tests (fast + slow)
pytest tests/ -m ""

# Test coverage
pytest tests/ --cov

# Generate OpenAPI schema
python scripts/generate_openapi_schema.py
```

### Frontend (run from `invokeai/frontend/web/`)

```bash
pnpm install                 # Install dependencies
pnpm dev                     # Dev server on localhost:5173 (proxies to backend :9090)
pnpm build                   # Production build (includes lint)
pnpm lint                    # Run all lint checks (eslint, prettier, tsc, knip, dpdm)
pnpm fix                     # Auto-fix eslint & prettier issues
pnpm test:no-watch           # Run vitest once
pnpm typegen                 # Generate TS types from OpenAPI schema
```

Full type generation from backend schema:

```bash
# From repo root
cd invokeai/frontend/web && python ../../../scripts/generate_openapi_schema.py | pnpm typegen
```

## Architecture

### Backend

**Entry point**: `invokeai.app.run_app:run_app` (registered as `invokeai-web` script)

**API layer** (`invokeai/app/api/`): FastAPI app with Socket.IO for real-time events. `api_app.py` sets up the lifespan, middleware, and routers. `dependencies.py` contains `ApiDependencies` which initializes all services as a singleton during startup and provides the `Invoker` instance to routers.

**Invocations system** (`invokeai/app/invocations/`): The core extensibility mechanism. Each node is a class decorated with `@invocation(type, title, tags, category, version, classification)` that extends `BaseInvocation` and implements `invoke(context) -> SomeOutput`. Fields use `InputField()` and `OutputField()` wrappers with UI metadata. Outputs extend `BaseInvocationOutput` with `@invocation_output` decorator. All invocations auto-register via `InvocationRegistry` at import time.

**Services layer** (`invokeai/app/services/`): Abstract base class + default implementation pattern throughout. Key services:
- `SessionProcessor` / `SessionRunner` - executes node graphs from the queue
- `SessionQueue` (SQLite-backed) - manages batch/single generation jobs
- `ModelManagerService` - coordinates model record storage, installation, loading, and download
- `ImageService` - image CRUD with SQLite metadata + disk file storage
- `InvocationContext` (`shared/invocation_context.py`) - safe wrapper providing boards, images, models, tensors, conditioning, and logging interfaces to invocations

**Model Manager** (`invokeai/backend/model_manager/`): Three-tier system:
1. `ModelRecordService` - CRUD for model metadata (key, name, type, format, base, path, hash)
2. `ModelInstallService` - downloads, probes, and registers models
3. `ModelLoadService` - lazy loads into VRAM with caching, context-manager pattern for device management

**Stable Diffusion core** (`invokeai/backend/stable_diffusion/`): `StableDiffusionGeneratorPipeline` coordinates UNet/VAE/CLIP/scheduler. Extensions system handles LoRA, ControlNet, IP-Adapter, T2I-Adapter, InPaint, FreeU, Seamless, RescaleCFG.

### Frontend

**Package**: `@invoke-ai/invoke-ai-ui`, built with Vite + React 18 + TypeScript 5.8 (strict mode)

**State management**: Redux Toolkit with 21 slices. Persistence via `redux-remember` to IndexedDB. Undo/redo via `redux-undo` on `canvas` and `nodes` slices. Side effects handled by listener middleware (not sagas). Nanostores used for non-Redux reactive state.

**API client**: RTK Query generated from backend OpenAPI schema. 60+ cache invalidation tags. Socket.IO for real-time progress/queue/model events.

**Key feature modules** (`src/features/`):
- `controlLayers/` - Konva-based canvas editor with layer management
- `nodes/` - XYFlow-based workflow/node editor
- `gallery/` - Virtual-scrolling image browser with boards
- `queue/` - Job queue management with real-time status
- `modelManagerV2/` - Model installation and management UI

**UI framework**: `@invoke-ai/ui-library` (custom Chakra-based), `dockview` for panel layout, `@atlaskit/pragmatic-drag-and-drop` for DnD.

## Code Conventions

### Python
- **Line length**: 120 chars
- **Linter**: ruff (rules: B, C, E, F, W, I, TID)
- **No relative imports** (enforced by ruff `ban-relative-imports = "all"`)
- **Type checking**: mypy (strict) + pyright (strict mode, warnings)
- **Python version**: 3.11-3.12 only

### TypeScript/React
- **No relative imports** - use path aliases (e.g., `import { x } from 'features/nodes/...'`)
- **Use `import type`** for type-only imports
- **Use `nanoid()`** not `crypto.randomUUID()`
- **Use `es-toolkit`** not `lodash-es`
- **Use `useClipboard` hook** not `navigator.clipboard`
- **Use `zod`** not `zod/v3`
- **Print width**: 120 chars, single quotes, trailing commas
- **Tests**: colocated `.test.ts` files with vitest

### Invocation Development
- Class name should end in `Invocation`
- Use `@invocation` decorator with unique type string and semver version
- Use `InputField()` / `OutputField()` with appropriate `Input` enum and `UIType`
- Access services only through `InvocationContext`, never directly
- Custom nodes go in a `nodes/` subfolder at the InvokeAI root, each with `__init__.py` importing the node classes
- Classification levels: `Stable`, `Beta`, `Prototype`, `Deprecated`, `Internal`

### Test Organization
- Backend tests mirror `invokeai/` structure under `tests/`
- Tests needing models must be marked `@pytest.mark.slow`
- Coverage threshold: 85%
- Frontend tests are unit-only (no UI/integration tests), colocated with source
