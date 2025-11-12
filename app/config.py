import os
import yaml
from typing import Any, Dict
from dotenv import load_dotenv

# ✅ Load environment variables from .env at startup
load_dotenv()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yml")

def load_config() -> Dict[str, Any]:
    path = os.path.abspath(CONFIG_PATH)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # override with env vars if present
    default = cfg.get("default", {})
    default["db"] = {
        "user": os.getenv("DB_USER", "spencerdruckenbroad"),
        "password": os.getenv("DB_PASSWORD", "rV@,^e,#!d9VWNj"),  # ✅ fixed var name
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "db": os.getenv("DB_NAME", "trading_ai"),
    }
    default["tradovate"] = {
        "client_id": os.getenv("TRADOVATE_CLIENT_ID"),
        "client_secret": os.getenv("TRADOVATE_CLIENT_SECRET"),
        "access_token": os.getenv("TRADOVATE_ACCESS_TOKEN"),
    }

    return default
