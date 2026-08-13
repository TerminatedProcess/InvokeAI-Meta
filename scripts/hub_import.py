#!/usr/bin/env python3
"""Import models from hubrootv3 hub into InvokeAI as symlinks.

Reads the hubrootv3 database, filters to InvokeAI-compatible models, creates
symlinks in InvokeAI's models directory, probes each model using InvokeAI's
own ModelConfigFactory, and registers them in InvokeAI's SQLite database.

Usage (from the InvokeAI project root, with venv active):
    python scripts/hub_import.py
    python scripts/hub_import.py --dry-run
    python scripts/hub_import.py --limit 10 --verbose
"""

import argparse
import json
import os
import sqlite3
import sys
import time
import traceback
import uuid
from pathlib import Path

# Disable CUDA — probing only reads safetensors headers, no GPU needed.
# Prevents SIGSEGV from torch/CUDA initialization during bulk imports.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ─── Default paths ───────────────────────────────────────────────────────────

HUB_DB_DEFAULT = "/mnt/llm/hub/hubmodels/hubrootv3.db"
HUB_MODELS_DEFAULT = "/mnt/llm/hub/hubmodels/models"
INVOKEAI_DB_DEFAULT = "/mnt/llm/hub/invokeai_data/databases/invokeai.db"
INVOKEAI_MODELS_DEFAULT = "/mnt/llm/hub/invokeai_data/models"

# ─── Filtering tables ────────────────────────────────────────────────────────

# Hub model_type values that InvokeAI cannot use at all
SKIP_MODEL_TYPES = frozenset({
    "audio",
    "audio_separation",
    "depth_estimation",
    "face_detection",
    "image_translation",
    "llm",
    "optical_flow",
    "projection",
    "segmentation",
    "unknown",
})

# Skip base_model if it contains any of these substrings (video/unsupported arch InvokeAI can't run).
# Wan is NOT here — InvokeAI supports Wan 2.2 (T2V/I2V) natively; it's handled via WAN_BASE_PREFIX below.
SKIP_BASE_KEYWORDS = frozenset({"ltxv", "hunyuan", "cogvideo", "chroma"})

# Any hub base starting with this maps to InvokeAI's single "wan" base. The hub uses many wan labels
# (wan, wan 2.1, wan 2.2, wan video, wan video 2.2 t2v-a14b, ...); the probe classifies the exact variant.
WAN_BASE_PREFIX = "wan"

# Skip base_model if it exactly matches one of these
SKIP_BASE_EXACT = frozenset({
    "bs-roformer",
    "cyclegan",
    "depth anything",
    "gemma",
    "other",
    "raft",
    "sam",
    "unknown",
    "yolo",
})

# Text-encoder bases InvokeAI uses as Krea-2 / Anima submodels. The hub labels the Qwen3-VL encoder with
# type 'llm' (which we'd otherwise skip), so allow these bases through regardless of type — a --reset then
# re-imports the encoders those bases need. The probe classifies the exact encoder type.
ENCODER_BASES = frozenset({"qwen3", "qwen3-vl"})

# Hub base_model → InvokeAI BaseModelType string value
# Models whose base_model is NOT in this map (and not skipped above) are skipped
BASE_MODEL_MAP = {
    "sd 1.5": "sd-1",
    "sd 1.5 lcm": "sd-1",
    "sd 1.5 hyper": "sd-1",
    "sdxl 1.0": "sdxl",
    "sdxl hyper": "sdxl",
    "sdxl lightning": "sdxl",
    "sdxl turbo": "sdxl",
    "sdxl 1.0 lcm": "sdxl",
    "illustrious": "sdxl",
    "noobai": "sdxl",
    "pony": "sdxl",
    "flux.1 d": "flux",
    "flux.1 s": "flux",
    "flux.2": "flux2",
    "flux.2 d": "flux2",
    "flux.2 klein 4b": "flux2",
    "flux.2 klein 4b-base": "flux2",
    "flux.2 klein 9b": "flux2",
    "flux.2 klein 9b-base": "flux2",
    "z-image": "z-image",
    "zimageturbo": "z-image",
    "zimagebase": "z-image",
    # Qwen-Image family (main + LoRA + VAE). The hub labels some as bare "qwen".
    "qwen-image": "qwen-image",
    "qwen": "qwen-image",
    # Krea-2 (12B MMDiT) — checkpoints, GGUF, and LoKr/standard LoRAs.
    "krea 2": "krea-2",
    "krea-2": "krea-2",
    # Anima (Cosmos Predict2 DiT) — checkpoints, LoRAs, VAEs.
    "anima": "anima",
    # ERNIE-Image.
    "ernie": "ernie-image",
    "ernie-image": "ernie-image",
    "any": "any",
}


# ─── Filtering logic ─────────────────────────────────────────────────────────


def should_skip(model_type: str, base_model: str) -> tuple[bool, str]:
    """Check if a hub model should be skipped. Returns (skip, reason)."""
    mt = model_type.lower().strip()
    bm = base_model.lower().strip()

    # Allow Krea-2 / Anima text encoders through even when the hub mislabels their type (e.g. 'llm').
    if bm in ENCODER_BASES:
        return False, ""

    if mt in SKIP_MODEL_TYPES:
        return True, f"unsupported type '{model_type}'"

    if bm in SKIP_BASE_EXACT:
        return True, f"unsupported base '{base_model}'"

    for kw in SKIP_BASE_KEYWORDS:
        if kw in bm:
            return True, f"unsupported base '{base_model}' (matches '{kw}')"

    # Wan (2.1/2.2 + video variants) — eligible under the single InvokeAI "wan" base.
    if bm.startswith(WAN_BASE_PREFIX):
        return False, ""

    if bm not in BASE_MODEL_MAP:
        return True, f"unmapped base '{base_model}'"

    return False, ""


# ─── Database helpers ─────────────────────────────────────────────────────────


def validate_invokeai_db(db_path: Path) -> None:
    """Exit with error if the InvokeAI DB is missing or not initialized."""
    if not db_path.exists():
        print(f"Error: InvokeAI database not found at {db_path}", file=sys.stderr)
        print("Start InvokeAI at least once to initialize the database.", file=sys.stderr)
        sys.exit(1)

    if db_path.stat().st_size == 0:
        print(f"Error: InvokeAI database is empty (0 bytes): {db_path}", file=sys.stderr)
        print("Start InvokeAI at least once to initialize the database.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='models'"
        ).fetchone()
        if not row:
            print("Error: 'models' table not found in InvokeAI database.", file=sys.stderr)
            print("Start InvokeAI at least once to initialize the database.", file=sys.stderr)
            sys.exit(1)
    finally:
        conn.close()


def load_existing_models(db_path: Path) -> tuple[set[str], set[str]]:
    """Load existing model hashes and paths from InvokeAI DB.

    Returns (hashes, paths) sets for duplicate detection.
    """
    hashes: set[str] = set()
    paths: set[str] = set()
    conn = sqlite3.connect(str(db_path))
    try:
        for (config_json,) in conn.execute("SELECT config FROM models").fetchall():
            try:
                cfg = json.loads(config_json)
                if h := cfg.get("hash"):
                    hashes.add(h)
                if p := cfg.get("path"):
                    paths.add(p)
            except (json.JSONDecodeError, TypeError):
                pass
    finally:
        conn.close()
    return hashes, paths


# ─── Main ─────────────────────────────────────────────────────────────────────


def import_models(args: argparse.Namespace) -> None:
    hub_db = Path(args.hub_db)
    hub_models_dir = Path(args.hub_models)
    invokeai_db = Path(args.invokeai_db)
    invokeai_models_dir = Path(args.invokeai_models)

    # Validate source paths
    if not hub_db.exists():
        print(f"Error: Hub database not found: {hub_db}", file=sys.stderr)
        sys.exit(1)
    if not hub_models_dir.exists():
        print(f"Error: Hub models directory not found: {hub_models_dir}", file=sys.stderr)
        sys.exit(1)

    # Validate InvokeAI DB
    validate_invokeai_db(invokeai_db)
    invokeai_models_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Query hub and filter (no InvokeAI imports needed) ──

    existing_hashes, existing_paths = load_existing_models(invokeai_db)
    print(f"InvokeAI DB: {len(existing_hashes)} existing models")

    hub_conn = sqlite3.connect(str(hub_db))
    hub_conn.row_factory = sqlite3.Row
    rows = hub_conn.execute("""
        SELECT id, hash_blake3, filename, file_size, model_type, base_model, name
        FROM models
        WHERE deleted = 0
        ORDER BY base_model, model_type, filename
    """).fetchall()
    hub_conn.close()

    print(f"Hub DB: {len(rows)} non-deleted models")

    eligible = []
    skip_reasons: dict[str, int] = {}

    for row in rows:
        skip, reason = should_skip(row["model_type"], row["base_model"])
        if skip:
            bucket = reason.split("'")[0].strip()  # group by reason prefix
            skip_reasons[bucket] = skip_reasons.get(bucket, 0) + 1
            continue

        invoke_hash = f"blake3:{row['hash_blake3']}"
        if invoke_hash in existing_hashes:
            skip_reasons["already in InvokeAI"] = skip_reasons.get("already in InvokeAI", 0) + 1
            continue

        eligible.append(dict(row))

    print(f"\nFilter results:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f"  {count:>5}  {reason}")
    print(f"  {len(eligible):>5}  eligible for import")

    if args.limit and args.limit < len(eligible):
        eligible = eligible[: args.limit]
        print(f"  (limited to first {args.limit})")

    if not eligible:
        print("\nNothing to import.")
        return

    # ── Dry run stops here ──

    if args.dry_run:
        print(f"\nDry run — would import {len(eligible)} models:")
        for row in eligible:
            print(f"  {row['filename']:<50} {row['model_type']:<15} {row['base_model']}")
        return

    # ── Phase 2: Import (needs InvokeAI imports for probing) ──

    # Set INVOKEAI_ROOT so InvokeAI's get_config() works during import
    project_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("INVOKEAI_ROOT", str(project_root / "invokeai_data"))
    sys.path.insert(0, str(project_root))

    print("\nLoading InvokeAI model classification system (importing torch, ~1-2 min)...", flush=True)
    try:
        from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
        from invokeai.backend.model_manager.configs.unknown import Unknown_Config
    except ImportError as e:
        print(f"Error: Cannot import InvokeAI modules: {e}", file=sys.stderr)
        print("Activate InvokeAI's virtual environment first.", file=sys.stderr)
        sys.exit(1)

    invokeai_conn = sqlite3.connect(str(invokeai_db))
    imported = 0
    failed = 0
    skipped = 0
    t0 = time.time()

    print(f"\nImporting {len(eligible)} models...\n", flush=True)

    try:
        for i, row in enumerate(eligible, 1):
            filename = row["filename"]
            hub_hash = row["hash_blake3"]
            invoke_hash = f"blake3:{hub_hash}"
            hub_name = Path(filename).stem
            file_size = row["file_size"]

            source_path = hub_models_dir / hub_hash / filename

            # Verify source exists
            if not source_path.exists():
                print(f"  [{i}/{len(eligible)}] MISS  {filename} (source file not found)")
                failed += 1
                continue

            # Generate UUID key and build paths
            key = str(uuid.uuid4())
            link_dir = invokeai_models_dir / key
            link_path = link_dir / filename
            rel_path = f"{key}/{filename}"

            # Create symlink
            try:
                link_dir.mkdir(parents=True, exist_ok=True)
                link_path.symlink_to(source_path)
            except OSError as e:
                print(f"  [{i}/{len(eligible)}] FAIL  {filename} (symlink error: {e})")
                failed += 1
                continue

            # Probe using InvokeAI's classification system
            try:
                result = ModelConfigFactory.from_model_on_disk(
                    link_path,
                    override_fields={
                        "key": key,
                        "hash": invoke_hash,
                        "name": hub_name,
                        "file_size": file_size,
                        "source": str(source_path),
                        "source_type": "path",
                    },
                    allow_unknown=False,
                )

                if result.config is None:
                    link_path.unlink(missing_ok=True)
                    link_dir.rmdir()
                    if args.verbose:
                        print(f"  [{i}/{len(eligible)}] SKIP  {filename} (no matching config class)")
                    skipped += 1
                    continue

                config = result.config

                # Check if InvokeAI classified this as Unknown
                if isinstance(config, Unknown_Config):
                    link_path.unlink(missing_ok=True)
                    link_dir.rmdir()
                    if args.verbose:
                        print(f"  [{i}/{len(eligible)}] SKIP  {filename} (classified as Unknown)")
                    skipped += 1
                    continue

            except Exception as e:
                link_path.unlink(missing_ok=True)
                link_dir.rmdir()
                err = f"{type(e).__name__}: {e}"
                print(f"  [{i}/{len(eligible)}] FAIL  {filename} (probe: {err})")
                if args.verbose:
                    traceback.print_exc()
                failed += 1
                continue

            # Set path relative to models_dir (InvokeAI resolves from models_path)
            config.path = rel_path

            # Register in InvokeAI DB
            try:
                invokeai_conn.execute(
                    "INSERT INTO models (id, config) VALUES (?, ?)",
                    (config.key, config.model_dump_json()),
                )
                invokeai_conn.commit()
                existing_hashes.add(invoke_hash)
                imported += 1

                info = f"{config.type.value}/{config.base.value}/{config.format.value}"
                print(f"  [{i}/{len(eligible)}] OK    {hub_name:<40} {info}")

            except sqlite3.IntegrityError as e:
                link_path.unlink(missing_ok=True)
                link_dir.rmdir()
                print(f"  [{i}/{len(eligible)}] DUP   {filename} ({e})")
                skipped += 1

            except Exception as e:
                link_path.unlink(missing_ok=True)
                link_dir.rmdir()
                print(f"  [{i}/{len(eligible)}] FAIL  {filename} (db: {e})")
                failed += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial import preserved.")

    finally:
        invokeai_conn.close()

    elapsed = time.time() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'=' * 60}")
    print(f"Import complete ({minutes}m {seconds}s)")
    print(f"  Imported:  {imported}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {len(eligible)}")


def reset_invokeai_models(args: argparse.Namespace) -> None:
    """Reset InvokeAI models: clear DB models table and wipe models directory."""
    invokeai_db = Path(args.invokeai_db)
    invokeai_models_dir = Path(args.invokeai_models)

    if not invokeai_db.exists():
        print(f"Error: InvokeAI database not found at {invokeai_db}", file=sys.stderr)
        sys.exit(1)

    # Count existing models
    conn = sqlite3.connect(str(invokeai_db))
    try:
        count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    finally:
        conn.close()

    print(f"InvokeAI DB: {count} models registered")
    print(f"Models dir:  {invokeai_models_dir}")

    if args.dry_run:
        print(f"\nDry run — would delete {count} model records and wipe {invokeai_models_dir}")
        return

    # Clear models table
    conn = sqlite3.connect(str(invokeai_db))
    try:
        conn.execute("DELETE FROM models")
        conn.execute("DELETE FROM model_relationships")
        conn.commit()
        print(f"Cleared {count} models from database")
    finally:
        conn.close()

    # Wipe models directory (all symlinks and UUID dirs)
    if invokeai_models_dir.exists():
        import shutil
        removed = 0
        for child in invokeai_models_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
                removed += 1
            elif child.is_symlink() or child.is_file():
                child.unlink()
                removed += 1
        print(f"Removed {removed} entries from {invokeai_models_dir}")
    else:
        print(f"Models directory doesn't exist, nothing to wipe")

    print("\nReset complete. Run hubimport to re-import models.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import models from hubrootv3 hub into InvokeAI as symlinks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/hub_import.py --dry-run        Show what would be imported
  python scripts/hub_import.py --limit 5         Import first 5 eligible models
  python scripts/hub_import.py --verbose          Show detailed error info
  python scripts/hub_import.py --reset            Wipe models and DB, then re-import
  python scripts/hub_import.py --reset --dry-run  Show what reset would do
""",
    )
    parser.add_argument(
        "--hub-db",
        default=HUB_DB_DEFAULT,
        help=f"Hub database path (default: {HUB_DB_DEFAULT})",
    )
    parser.add_argument(
        "--hub-models",
        default=HUB_MODELS_DEFAULT,
        help=f"Hub models directory (default: {HUB_MODELS_DEFAULT})",
    )
    parser.add_argument(
        "--invokeai-db",
        default=INVOKEAI_DB_DEFAULT,
        help=f"InvokeAI database path (default: {INVOKEAI_DB_DEFAULT})",
    )
    parser.add_argument(
        "--invokeai-models",
        default=INVOKEAI_MODELS_DEFAULT,
        help=f"InvokeAI models directory (default: {INVOKEAI_MODELS_DEFAULT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without making changes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Import only the first N eligible models",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full tracebacks on probe failures",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe InvokeAI models DB and symlinks before importing",
    )
    parser.add_argument(
        "--no-import",
        action="store_true",
        help="Skip the import step (useful with --reset to wipe only)",
    )

    args = parser.parse_args()

    if args.reset:
        reset_invokeai_models(args)
        if args.dry_run:
            return
        print()

    if not args.no_import:
        import_models(args)


if __name__ == "__main__":
    main()
    # Flush output then force-exit to avoid SIGSEGV during torch/CUDA cleanup
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
