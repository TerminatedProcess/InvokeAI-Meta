#!/usr/bin/env bash
# InvokeAI-Meta Development Setup
# Gets InvokeAI running in development mode
clear
set -e

# INVOKE_DIR is exported by .salias, which is only sourced once the project has an
# `sw` tag — not the case on a fresh clone, which is precisely when this script runs.
# Fall back to the script's own location so it works standalone.
PROJECT_DIR="${INVOKE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_DIR="$PROJECT_DIR/invokeai_data"

echo "🚀 Setting up InvokeAI development environment..."

echo "$INVOKE_DIR"
echo "$PROJECT_DIR"

cd "$PROJECT_DIR"

# 1. Setup git-lfs and pull large files
echo "Setting up git-lfs..."
git lfs install
echo "Downloading large files with git-lfs..."
git lfs pull

# 2. Ensure the companion tooling repo is checked out beside this one.
# scripts/hub_import.py, scripts/hub_update.py and scripts/model_compare are
# relative symlinks into a sibling clone of invokeai-claude-fixes. Without it they
# dangle, and `hubup` / model-compare.service fail with a confusing ENOENT.
TOOLS_DIR="$(dirname "$PROJECT_DIR")/invokeai-claude-fixes"
TOOLS_REPO="https://github.com/TerminatedProcess/invokeai-claude-fixes.git"
if [ ! -d "$TOOLS_DIR/.git" ]; then
    echo "Cloning companion tooling repo to $TOOLS_DIR..."
    git clone "$TOOLS_REPO" "$TOOLS_DIR"
else
    echo "Companion tooling repo present, updating..."
    git -C "$TOOLS_DIR" pull --ff-only
fi

# Resolve the symlinks now, so a missing sibling fails here with a clear message
# rather than hours later inside a service.
for link in scripts/hub_import.py scripts/hub_update.py scripts/model_compare; do
    if [ ! -e "$link" ]; then
        echo "Error: $link does not resolve. Expected its target under $TOOLS_DIR" >&2
        exit 1
    fi
done

# 3. Create venv if it doesn't exist
#if [ ! -d ".venv" ]; then
#    echo "Creating virtual environment..."
#    uv venv --relocatable --prompt invoke-meta --python 3.12 --python-preference only-managed .venv
#fi

# 4. Activate venv
#echo "Activating virtual environment..."
#source .venv/bin/activate

# 5. Install InvokeAI in editable mode with dev dependencies
echo "Installing InvokeAI with dev dependencies (this may take a while)..."
uv pip install -e ".[dev,test,docs]" --python 3.12 --python-preference only-managed --torch-backend=cu128 --reinstall

# 6. Create data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory at $DATA_DIR..."
    mkdir -p "$DATA_DIR"
fi

# 7. Create initial config file
echo "Creating invokeai.yaml config..."
cat > "$DATA_DIR/invokeai.yaml" << EOF
# InvokeAI-Meta Configuration
# Data directory: $DATA_DIR

schema_version: 4.0.2

# Use persistent database (set to true for in-memory/ephemeral database)
use_memory_db: false

# Scan models on startup when using memory database
scan_models_on_startup: true

# Models directory - we'll symlink this to ComfyUI later
models_dir: $DATA_DIR/models

# Host and port
host: 0.0.0.0
port: 9090

# Log level
log_level: info
EOF

# 8. Install Node.js dependencies for frontend
echo "Installing frontend dependencies..."
cd invokeai/frontend/web
pnpm i

# 9. Build frontend (--mode test skips vite-plugin-eslint, avoids Node 25 V8 crash)
echo "Building frontend (this may take a few minutes)..."
pnpm exec vite build --mode test

# 10. Install pypatchmatch
uv pip install pypatchmatch

cd "$PROJECT_DIR"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start InvokeAI:"
echo "  1. Activate venv: source .venv/bin/activate"
echo "  2. Run server: invokeai-web --root $DATA_DIR"
echo ""
echo "Server will be available at: http://127.0.0.1:9090"
echo ""
echo "For frontend development mode (with hot reload):"
echo "  cd invokeai/frontend/web && pnpm dev"
echo "  (Server will be at http://127.0.0.1:5173)"
