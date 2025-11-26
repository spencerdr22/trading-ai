import os, zipfile
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------------
# Define paths
# ------------------------------------------------------------------
ROOT = Path(".")
OUTPUT_NAME = "adaptive_upgrade_v2.0_2025-11-18.zip"
ZIP_PATH = ROOT / OUTPUT_NAME
CRLF = "\r\n"

# ------------------------------------------------------------------
# File contents (trimmed placeholders below; copy your generated code)
# ------------------------------------------------------------------
FILES = {
    "app/adaptive/model_hub.py": "...",        # paste file contents
    "app/adaptive/reward.py": "...",
    "app/adaptive/learner.py": "...",
    "app/adaptive/optimizer.py": "...",
    "app/adaptive/run_offline_rl.py": "...",
    "app/ml/trainer.py": "...",
    "app/strategy/engine.py": "...",
    "generate_code_changes.py": "...",
    "CHANGELOG.md": "...",
}

# ------------------------------------------------------------------
# Read-me file explaining how to merge the upgrade
# ------------------------------------------------------------------
README = f"""Trading-AI Adaptive Upgrade v2.0 (2025-11-18){CRLF}
========================================================={CRLF}
1. Extract this ZIP directly into your repository root (`trading-ai/`).{CRLF}
2. Open VS Code.  Run `git diff` or use Source Control to review changes.{CRLF}
3. Commit merged files with message: "Upgrade → Adaptive RL Framework v2.0".{CRLF}
4. (Optional) Re-run `python generate_code_changes.py --apply` to confirm integrity.{CRLF}
5. Enjoy your new self-learning Trading-AI system!{CRLF}
========================================================={CRLF}
"""

FILES["README.txt"] = README

# ------------------------------------------------------------------
# Build the ZIP
# ------------------------------------------------------------------
os.makedirs(ROOT, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel, content in FILES.items():
        zf.writestr(rel, content.replace("\n", CRLF))

print(f"✅ Created {ZIP_PATH.resolve()}")
print("You can now share or archive this upgrade package safely.")
