"""
Utility: generate_code_changes.py
Author: Adaptive Framework Generator

Description:
    This script updates the repository with new/modified files for the
    self-learning adaptive system, while preserving full safety and auditability.

Features:
    ✔ Writes all generated files in correct paths
    ✔ Creates .bak backups of old versions before overwriting
    ✔ Creates git-compatible .patch files
    ✔ Supports dry-run mode for previewing changes
    ✔ Logs output to data/logs/code_changes.log
    ✔ Auto-creates missing folders
    ✔ Safe for production repositories

Usage:
    python generate_code_changes.py --apply
    python generate_code_changes.py --dry-run
"""

import os
import difflib
import argparse
from datetime import datetime

# Root of repo
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))

# Paths and file contents will be filled dynamically
# You will insert content placeholders for each updated/new file.
FILES_TO_WRITE = {
    "app/adaptive/model_hub.py": "...",
    "app/adaptive/reward.py": "...",
    "app/adaptive/learner.py": "...",
    "app/adaptive/optimizer.py": "...",
    "app/adaptive/run_offline_rl.py": "...",
    "app/ml/trainer.py": "...",
    "app/strategy/engine.py": "...",
    # Add more files here if needed
}

LOG_PATH = os.path.join("data", "logs", "code_changes.log")
PATCH_DIR = os.path.join("data", "patches")


def ensure_dirs():
    """Ensure output folders exist."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(PATCH_DIR, exist_ok=True)


def backup_file(path):
    """Create a .bak backup file if original exists."""
    if os.path.exists(path):
        bak_path = path + ".bak"
        with open(path, "r", encoding="utf-8") as src:
            with open(bak_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
        return bak_path
    return None


def write_log(message):
    """Append message to log file."""
    ts = datetime.utcnow().strftime("[%Y-%m-%d %H:%M:%S] ")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(ts + message + "\n")


def create_patch(old_text, new_text, patch_path, file_path):
    """Create a git-style patch file."""
    diff = difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        fromfile=file_path + ".old",
        tofile=file_path + ".new",
        lineterm=""
    )
    patch = "\n".join(diff)

    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(patch)


def apply_file(path, new_content, dry_run=False):
    """
    Apply update to a file:
      - Create folder if missing
      - Backup existing file
      - Create patch
      - Write new content
    """
    full_path = os.path.join(REPO_ROOT, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    old_content = ""
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            old_content = f.read()

    # Always create patch file
    patch_name = os.path.basename(path) + ".patch"
    patch_path = os.path.join(PATCH_DIR, patch_name)
    create_patch(old_content, new_content, patch_path, path)

    if dry_run:
        write_log(f"[DRY RUN] Would update: {path}")
        return

    # Backup original
    backup_file(full_path)

    # Write new content
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    write_log(f"Updated: {path}")
    write_log(f"Patch saved: {patch_path}")


def main():
    parser = argparse.ArgumentParser(description="Apply adaptive system code updates.")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    args = parser.parse_args()

    ensure_dirs()

    if not (args.dry_run or args.apply):
        print("ERROR: Use --dry-run or --apply")
        return

    for file_path, new_content in FILES_TO_WRITE.items():
        apply_file(
            path=file_path,
            new_content=new_content,
            dry_run=args.dry_run
        )

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] No files were changed.\n")
    else:
        print("\n[UPDATE COMPLETE] All files updated successfully.\n")


if __name__ == "__main__":
    main()
