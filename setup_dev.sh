#!/usr/bin/env bash
# InvokeAI-Meta (v7) development setup. Idempotent: safe to re-run after a pull.
#   ./setup_dev.sh            backend + webv2 (the default UI)
#   ./setup_dev.sh --legacy   also build webv1, needed only for `invokeai-web --web-legacy`
set -euo pipefail

BUILD_LEGACY=false
[ "${1:-}" = "--legacy" ] && BUILD_LEGACY=true

# INVOKE_DIR comes from .salias, which isn't sourced on a fresh clone; fall back to
# the script's own location.
PROJECT_DIR="${INVOKE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_DIR="$PROJECT_DIR/invokeai_data"
cd "$PROJECT_DIR"

echo "==> git-lfs (test fixtures under tests/*/stripped_models)"
git lfs install --local
git lfs pull

# Fork tooling lives in a sibling clone of invokeai-claude-fixes and is symlinked into
# scripts/ (upstream owns that dir; we only add to it).
echo "==> Companion tooling repo"
TOOLS_DIR="$(dirname "$PROJECT_DIR")/invokeai-claude-fixes"
if [ ! -d "$TOOLS_DIR/.git" ]; then
    git clone https://github.com/TerminatedProcess/invokeai-claude-fixes.git "$TOOLS_DIR"
else
    git -C "$TOOLS_DIR" pull --ff-only
fi
TOOL_LINKS=(hub_import.py hub_update.py)
for name in "${TOOL_LINKS[@]}"; do
    ln -sfn "../../invokeai-claude-fixes/scripts/$name" "scripts/$name"
    [ -e "scripts/$name" ] || { echo "Error: scripts/$name does not resolve into $TOOLS_DIR/scripts" >&2; exit 1; }
done

# Keep local-only paths out of `git status` without touching upstream's .gitignore.
# .git/info/exclude is per-clone and untracked, so mryan stays additions-only.
echo "==> Excluding local-only paths"
EXCLUDE_FILE="$(git rev-parse --git-path info/exclude)"
mkdir -p "$(dirname "$EXCLUDE_FILE")"
for path in invokeai_data/ .env .env.keys .envrc .claude/ "${TOOL_LINKS[@]/#/scripts/}"; do
    grep -qxF "$path" "$EXCLUDE_FILE" 2>/dev/null || echo "$path" >> "$EXCLUDE_FILE"
done

# Upstream's dev install: `uv sync` against the committed uv.lock. --frozen installs the
# lock as-is; --locked would re-resolve, which the global `exclude-newer` in
# ~/.config/uv/uv.toml turns into a hard failure. The cuda extra pins torch to the cu130
# index; the project is installed editable, so Python edits are live.
echo "==> Python environment (uv sync; first run downloads torch)"
uv sync --frozen --python 3.12 --managed-python --extra cuda --extra dev --extra test

# Only write a config if absent — the two machines need different VRAM cache settings,
# and overwriting would silently reset them.
mkdir -p "$DATA_DIR"
if [ -f "$DATA_DIR/invokeai.yaml" ]; then
    echo "==> Keeping existing invokeai.yaml"
else
    echo "==> Writing invokeai.yaml"
    cat > "$DATA_DIR/invokeai.yaml" << EOF
# Internal metadata - do not edit:
schema_version: 4.0.3

# Put user settings here - see https://invoke-ai.github.io/InvokeAI/configuration/:
host: 0.0.0.0
port: 9090
log_level: info
EOF
fi

# Node: .nvmrc pins v22; the system node here is newer, so build under fnm's v22 if present.
pnpm_node22() {
    if command -v fnm >/dev/null 2>&1; then
        fnm exec --using=22 -- pnpm "$@"
    else
        pnpm "$@"
    fi
}
command -v fnm >/dev/null 2>&1 \
    || echo "Warning: fnm not found; using system node $(node -v) (.nvmrc wants $(cat .nvmrc))" >&2

# webv2 is mounted at /. `vite build` directly, skipping the lint gate in `pnpm build`.
echo "==> Building webv2"
pnpm_node22 -C invokeai/frontend/webv2 install --frozen-lockfile
pnpm_node22 -C invokeai/frontend/webv2 exec vite build

# webv1's vite config runs vite-plugin-eslint outside test mode, which crashes on newer
# Node; --mode test skips it.
if $BUILD_LEGACY; then
    echo "==> Building webv1 (legacy)"
    pnpm_node22 -C invokeai/frontend/webv1 install --frozen-lockfile
    pnpm_node22 -C invokeai/frontend/webv1 exec vite build --mode test
fi

# pypatchmatch (a core dependency) compiles libpatchmatch.so on first import if it's
# missing, but its Makefile only asks pkg-config for `opencv`/`opencv4`. Arch ships
# OpenCV 5 as `opencv5`, so build it here with a shim mapping opencv4 -> opencv5.
# uv sync reinstalling the package removes the .so; re-running this script rebuilds it.
echo "==> Building patchmatch"
PM_DIR="$(.venv/bin/python -c 'import os, importlib.util; print(os.path.dirname(importlib.util.find_spec("patchmatch").origin))')"
if [ -f "$PM_DIR/libpatchmatch.so" ]; then
    echo "libpatchmatch.so present"
elif pkg-config --exists opencv4; then
    make -C "$PM_DIR" >/dev/null
elif pkg-config --exists opencv5; then
    PC_SHIM="$(mktemp -d)"
    ln -s "$(pkg-config --variable=pcfiledir opencv5)/opencv5.pc" "$PC_SHIM/opencv4.pc"
    PKG_CONFIG_PATH="$PC_SHIM" make -C "$PM_DIR" >/dev/null
    rm -rf "$PC_SHIM"
else
    echo "Warning: system OpenCV not found; patchmatch infill disabled (Arch: pacman -S opencv)" >&2
fi

cat << EOF

Setup complete.
  Run:      uv run --no-sync invokeai-web --root $DATA_DIR
  UI:       http://127.0.0.1:9090
  Dev UI:   pnpm -C invokeai/frontend/webv2 dev   (http://127.0.0.1:5173, proxies to :9090)
EOF
