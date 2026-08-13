#!/usr/bin/env python3
"""Reconcile InvokeAI's model database with the hub.

`hubupdate` runs four maintenance steps:

  1. Remove models whose hub source is gone (broken symlinks / missing files).
  2. Delete duplicate DB entries (more than one row for the same file hash —
     an InvokeAI bug that sometimes triggers when the same model is re-imported).
  3. Remove models InvokeAI classified as Unknown (unrecognized files).
  4. Import models that are new in the hub (delegates to hub_import.import_models).

Steps 1-3 are pure SQLite + filesystem and need no torch/GPU. They run first so
step 4 sees a clean database and reports accurate "new" counts.

Only model directories under the InvokeAI models dir are ever deleted from disk;
entries that point directly at a hub source file (absolute path outside the
models dir) are unregistered from the DB but their files are left untouched.

Usage (from the InvokeAI project root, with venv active):
    python scripts/hub_update.py --dry-run
    python scripts/hub_update.py
    python scripts/hub_update.py --yes            # skip the confirmation prompt
    python scripts/hub_update.py --no-add         # cleanup only, don't import new
"""

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

# Disable CUDA — the cleanup phases never touch a GPU, and the import phase only
# reads safetensors headers. Prevents SIGSEGV from torch/CUDA init on bulk runs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Reuse hub_import's defaults and its import routine (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hub_import  # noqa: E402


def _full_path(models_dir: Path, path_str: str) -> Path:
    """Resolve a model's DB `path` to an on-disk location.

    Relative paths are resolved against models_dir; absolute paths are used as-is.
    """
    p = Path(path_str)
    return p if p.is_absolute() else models_dir / p


def _ondisk_dir_to_remove(models_dir: Path, path_str: str) -> Path | None:
    """Return the top-level model directory under models_dir to delete.

    Returns None when the model lives outside models_dir — those are hub sources
    registered by absolute path and must never be deleted from disk.
    """
    p = Path(path_str)
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(models_dir.resolve())
        except ValueError:
            return None  # outside models_dir — leave the file alone
        parts = rel.parts
    else:
        parts = p.parts
    if not parts:
        return None
    return models_dir / parts[0]


def _remove_model(conn: sqlite3.Connection, models_dir: Path, key: str, path_str: str) -> None:
    """Delete a model's DB row and, if it lives under models_dir, its symlink dir."""
    conn.execute("DELETE FROM models WHERE id = ?", (key,))
    conn.commit()

    target = _ondisk_dir_to_remove(models_dir, path_str)
    if target is not None and target.exists():
        shutil.rmtree(target, ignore_errors=True)


def cleanup(args: argparse.Namespace) -> None:
    invokeai_db = Path(args.invokeai_db)
    models_dir = Path(args.invokeai_models)

    hub_import.validate_invokeai_db(invokeai_db)

    conn = sqlite3.connect(str(invokeai_db))
    conn.execute("PRAGMA foreign_keys = ON")  # cascade-delete model_relationships rows

    rows = conn.execute("SELECT id, hash, type, path, name, created_at FROM models ORDER BY created_at").fetchall()
    print(f"InvokeAI DB: {len(rows)} models registered\n")

    broken: list[tuple] = []  # (id, path, name) — hub source gone
    duplicates: list[tuple] = []  # (id, path, name) — extra rows sharing a hash
    unknown: list[tuple] = []  # (id, path, name) — classified as Unknown

    removed_ids: set[str] = set()

    # ── Phase 1: broken symlinks / missing sources ──
    for key, _hash, _type, path_str, name, _created in rows:
        if not _full_path(models_dir, path_str).exists():
            broken.append((key, path_str, name))
            removed_ids.add(key)

    # ── Phase 2: duplicate entries (same hash), keep the oldest surviving row ──
    seen_hash: dict[str, str] = {}
    for key, hash_, _type, path_str, name, _created in rows:  # rows are ordered by created_at
        if key in removed_ids:
            continue
        if hash_ in seen_hash:
            duplicates.append((key, path_str, name))
            removed_ids.add(key)
        else:
            seen_hash[hash_] = key

    # ── Phase 3: unknown-type models ──
    for key, _hash, type_, path_str, name, _created in rows:
        if key in removed_ids:
            continue
        if type_ == "unknown":
            unknown.append((key, path_str, name))
            removed_ids.add(key)

    def _report(title: str, items: list[tuple]) -> None:
        print(f"{title}: {len(items)}")
        for _key, _path, name in items:
            print(f"    - {name}")
        if items:
            print()

    _report("Broken (hub source gone)", broken)
    _report("Duplicates (same hash)", duplicates)
    _report("Unknown (unrecognized)", unknown)

    total = len(broken) + len(duplicates) + len(unknown)
    if total == 0:
        print("Nothing to clean up.\n")
        conn.close()
        return

    if args.dry_run:
        print(f"Dry run — would remove {total} models.\n")
        conn.close()
        return

    if not args.yes:
        try:
            reply = input(f"Remove {total} models? (y/N) ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            reply = ""
        if reply != "y":
            print("Aborted — no changes made.")
            conn.close()
            sys.exit(1)

    for key, path_str, _name in broken + duplicates + unknown:
        _remove_model(conn, models_dir, key, path_str)

    conn.close()
    print(f"\nRemoved {total} models ({len(broken)} broken, {len(duplicates)} duplicate, {len(unknown)} unknown).\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile InvokeAI's model DB with the hub: prune broken/duplicate/"
        "unknown entries, then import new hub models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hub-db", default=hub_import.HUB_DB_DEFAULT)
    parser.add_argument("--hub-models", default=hub_import.HUB_MODELS_DEFAULT)
    parser.add_argument("--invokeai-db", default=hub_import.INVOKEAI_DB_DEFAULT)
    parser.add_argument("--invokeai-models", default=hub_import.INVOKEAI_MODELS_DEFAULT)
    parser.add_argument("--dry-run", action="store_true", help="Show what would change; make no changes")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip the cleanup confirmation prompt")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup; only import new models")
    parser.add_argument("--no-add", action="store_true", help="Skip importing new models; only clean up")
    parser.add_argument("--limit", type=int, help="Import only the first N eligible new models")
    parser.add_argument("--verbose", action="store_true", help="Show full tracebacks on probe failures")

    args = parser.parse_args()

    if not args.no_cleanup:
        cleanup(args)

    if not args.no_add:
        print("=" * 60)
        print("Importing new hub models")
        print("=" * 60)
        hub_import.import_models(args)


if __name__ == "__main__":
    main()
    # Flush then force-exit to avoid SIGSEGV during torch/CUDA cleanup (import phase).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
