"""
Module: model_hub.py
Centralized registry and persistence layer for all models.
"""

import os
import joblib
import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, Session

from app.db.init import get_engine
from app.monitor.logger import get_logger

logger = get_logger(__name__)
Base = declarative_base()


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    model_name   = Column(String, nullable=False)
    model_type   = Column(String, nullable=False)
    version      = Column(String, nullable=False)
    accuracy     = Column(Float,  default=None)
    reward_score = Column(Float,  default=None)
    file_path    = Column(String, nullable=False)
    timestamp    = Column(DateTime, default=datetime.datetime.utcnow)
    meta         = Column(JSON, default={})


class ModelHub:
    """
    Manages saving, loading, and versioning of ML and RL models.
    Supports sklearn (joblib) and PyTorch (state_dict) models.
    """

    def __init__(self, base_dir: str = "data/models/"):
        self.base_dir = base_dir
        self.engine   = get_engine()
        Base.metadata.create_all(self.engine)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_model(
        self,
        model,
        model_name: str,
        model_type:  str  = "RandomForest",
        metrics:     dict = None,
        version:     str  = None,
    ) -> Optional[str]:
        """
        Persist model to disk and log metadata to the registry table.

        For PyTorch models (model_type='RLPolicy') the model must have
        a .state_dict() method.  All other models are saved with joblib.
        Passing a plain dict (e.g. hyperparameter records) is also fine
        and will be stored with joblib.
        """
        os.makedirs(self.base_dir, exist_ok=True)
        session = Session(self.engine)
        try:
            version   = version or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            is_torch  = model_type.lower() == "rlpolicy"
            file_ext  = "pt" if is_torch else "pkl"
            file_name = f"{model_name}_{version}.{file_ext}"
            file_path = os.path.join(self.base_dir, file_name)

            if is_torch:
                # Lazy import so torch is only required when actually used
                import torch
                if hasattr(model, "state_dict"):
                    torch.save(model.state_dict(), file_path)
                else:
                    # Fallback: save whatever was passed
                    torch.save(model, file_path)
            else:
                joblib.dump(model, file_path)

            entry = ModelRegistry(
                model_name   = model_name,
                model_type   = model_type,
                version      = version,
                accuracy     = (metrics or {}).get("accuracy"),
                reward_score = (metrics or {}).get("reward"),
                file_path    = file_path,
                meta         = metrics or {},
            )
            session.add(entry)
            session.commit()
            logger.info("ModelHub: saved %s (%s) -> %s", model_name, model_type, file_path)
            return file_path

        except Exception as e:
            session.rollback()
            logger.error("ModelHub: failed to save %s: %s", model_name, e)
            return None
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load_model(
        self,
        model_name: str,
        model_type: str  = None,
        version:    str  = None,
    ):
        """
        Load the latest (or a specific) version of a model from disk.
        Returns the deserialized object, or None if not found.
        """
        session = Session(self.engine)
        try:
            q = session.query(ModelRegistry).filter(
                ModelRegistry.model_name == model_name
            )
            if model_type:
                q = q.filter(ModelRegistry.model_type == model_type)
            if version:
                q = q.filter(ModelRegistry.version == version)

            entry = q.order_by(ModelRegistry.timestamp.desc()).first()
            if not entry:
                logger.warning("ModelHub: no record found for '%s'.", model_name)
                return None

            if not os.path.exists(entry.file_path):
                logger.error("ModelHub: file missing at %s", entry.file_path)
                return None

            if entry.model_type.lower() == "rlpolicy":
                import torch
                logger.info("ModelHub: loading RL model %s", entry.file_path)
                return torch.load(entry.file_path, map_location="cpu")
            else:
                logger.info("ModelHub: loading ML model %s", entry.file_path)
                return joblib.load(entry.file_path)

        except Exception as e:
            logger.error("ModelHub: error loading '%s': %s", model_name, e)
            return None
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def get_model_metadata(self, model_name: str, latest: bool = True):
        session = Session(self.engine)
        try:
            q = session.query(ModelRegistry).filter(
                ModelRegistry.model_name == model_name
            )
            if latest:
                q = q.order_by(ModelRegistry.timestamp.desc())
            result = q.first()
            return result.meta if result else None
        finally:
            session.close()

    def list_versions(self, model_name: str):
        session = Session(self.engine)
        try:
            rows = (
                session.query(ModelRegistry.version)
                .filter(ModelRegistry.model_name == model_name)
                .order_by(ModelRegistry.timestamp.desc())
                .all()
            )
            return [r[0] for r in rows]
        finally:
            session.close()
