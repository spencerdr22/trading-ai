import os, zipfile, difflib
from pathlib import Path
from datetime import datetime

ROOT = Path(".")
OUTPUT_DIR = ROOT / "data" / "patches"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STAGED_FILES = [
    "app/adaptive/model_hub.py",
    "app/adaptive/reward.py",
    "app/adaptive/learner.py",
    "app/adaptive/optimizer.py",
    "app/adaptive/run_offline_rl.py",
    "app/ml/trainer.py",
    "app/strategy/engine.py",
    "generate_code_changes.py",
    "CHANGELOG.md",
]

zip_path = OUTPUT_DIR / "adaptive_upgrade.zip"
patch_path = OUTPUT_DIR / "adaptive_upgrade.patch"

# Build ZIP
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in STAGED_FILES:
        if os.path.exists(f):
            zf.write(f)

# Minimal diff (just a placeholder)
with open(patch_path, "w", encoding="utf-8") as pf:
    pf.write("diff --git a/CHANGELOG.md b/CHANGELOG.md\n")
    pf.write(f"# Patch generated {datetime.utcnow().isoformat()}Z\n")

print(f"✅ Created: {zip_path}")
print(f"✅ Created: {patch_path}")
