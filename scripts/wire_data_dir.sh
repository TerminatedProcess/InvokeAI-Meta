#!/usr/bin/env bash
# Link the tracked parts of the data directory into invokeai_data.
#
# invokeai_data/ holds models, databases and outputs, so it is gitignored wholesale.
# But some of what lives under it is source we want versioned — custom nodes, dynamic
# prompt wildcards. Those are kept in ./invokeai_data_git (tracked) and symlinked into
# invokeai_data, which is the only path InvokeAI itself looks at.
#
# Without this, a fresh clone silently loads no custom nodes and no wildcards, with
# nothing to explain why. Called by setup_dev.sh and by the `wiredata` shell function,
# so there is one implementation rather than a bash copy and a fish copy to keep in sync.
#
# Idempotent: safe to run on every environment rebuild.
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_DIR/invokeai_data"
GIT_DATA_DIR="$PROJECT_DIR/invokeai_data_git"

if [ ! -d "$GIT_DATA_DIR" ]; then
    echo "Error: $GIT_DATA_DIR does not exist — nothing to wire." >&2
    exit 1
fi

mkdir -p "$DATA_DIR"

for target in "$GIT_DATA_DIR"/*/; do
    [ -d "$target" ] || continue            # no-op if invokeai_data_git is empty
    name="$(basename "$target")"
    link="$DATA_DIR/$name"
    # Relative, so the pair keeps working if the repo is cloned to a different path.
    rel="../invokeai_data_git/$name"

    if [ -L "$link" ]; then
        # Re-point a stale link. Removing a symlink discards nothing.
        if [ "$(readlink "$link")" != "$rel" ]; then
            echo "Re-pointing $link -> $rel"
            rm "$link" && ln -s "$rel" "$link"
        fi
    elif [ -d "$link" ]; then
        # A real directory here predates the split and may hold files that exist
        # nowhere else. Never clobber it — tell the user how to migrate by hand.
        echo "Warning: $link is a real directory, not a link to $rel." >&2
        echo "         Its contents are untracked. To migrate:" >&2
        echo "           mv $link/* $target && rmdir $link && ln -s $rel $link" >&2
    else
        echo "Linking $link -> $rel"
        ln -s "$rel" "$link"
    fi
done
