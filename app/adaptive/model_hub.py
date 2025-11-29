"""
Module: model_hub.py
Author: Adaptive Framework Generator (Corrected)
Description:
    Centralized registry and persistence layer for managing all models
    (baseline, adaptive, and experimental) within the Trading-AI System.
    Handles saving, versioning, and loading models, along with reward
    metrics and metadata synchronization in the database.
"""

import os
import json
import joblib
import torch
import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, Session
from app.db.init import get_engine
from app.monitor.logger import get_logger

logger = get_logger(__name__)
Base = declarative_base()
 
# ============================================================
# DATABASE TABLE DEFINITIONS
# ============================================================

class ModelRegistry(Base):
    """
    ORM model representing metadata for stored ML or RL models.
    """
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # e.g., RandomForest, RLPolicy
    version = Column(String, nullable=False)
    accuracy = Column(Float, default=None)
    reward_score = Column(Float, default=None)
    file_path = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    meta = Column(JSON, default={})  # ✅ FIX: renamed from 'metadata'

# ============================================================
# MODEL HUB CLASS
# ============================================================

class ModelHub:
    """
    Manages the saving, loading, and tracking of model versions and metadata.
    Supports both sklearn and torch models.
    """

    def __init__(self, base_dir: str = "data/models/"):
        self.base_dir = base_dir
        self.engine = get_engine()
        Base.metadata.create_all(self.engine)

    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------
    def save_model(self, model, model_name: str, model_type: str,
                   metrics: dict = None, version: str = None):
        """
        Persist model to disk and log metadata to database.
        """
        os.makedirs(self.base_dir, exist_ok=True)
        session = Session(self.engine)

        try:
            version = version or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_ext = "pkl" if model_type.lower() != "rlpolicy" else "pt"
            file_name = f"{model_name}_{version}.{file_ext}"
            file_path = os.path.join(self.base_dir, file_name)

            # Save model
            if model_type.lower() == "rlpolicy":
                torch.save(model.state_dict(), file_path)
            else:
                joblib.dump(model, file_path)

            # Record metadata
            entry = ModelRegistry(
                model_name=model_name,
                model_type=model_type,
                version=version,
                accuracy=metrics.get("accuracy") if metrics else None,
                reward_score=metrics.get("reward") if metrics else None,
                file_path=file_path,
                meta=metrics or {},
            )
            session.add(entry)
            session.commit()

            logger.info(f"ModelHub: saved {model_name} ({model_type}) → {file_path}")
            return file_path

        except Exception as e:
            session.rollback()
            logger.error(f"ModelHub: failed to save model {model_name} — {e}")
            return None
        finally:
            session.close()

    # --------------------------------------------------------
    # Load Model
    # --------------------------------------------------------
    def load_model(self, model_name: str, model_type: str = None, version: str = None):
        """
        Load model by name, type, or version. Defaults to latest version.
        """
        session = Session(self.engine)
        try:
            query = session.query(ModelRegistry).filter(ModelRegistry.model_name == model_name)
            if model_type:
                query = query.filter(ModelRegistry.model_type == model_type)
            if version:
                query = query.filter(ModelRegistry.version == version)

            entry = query.order_by(ModelRegistry.timestamp.desc()).first()
            if not entry:
                logger.warning(f"ModelHub: no model found for {model_name}.")
                return None

            if not os.path.exists(entry.file_path):
                logger.error(f"ModelHub: file missing at {entry.file_path}")
                return None

            if entry.model_type.lower() == "rlpolicy":
                logger.info(f"ModelHub: found RL model {entry.model_name} @ {entry.file_path}")
                return torch.load(entry.file_path)
            else:
                logger.info(f"ModelHub: found ML model {entry.model_name} @ {entry.file_path}")
                return joblib.load(entry.file_path)

        except Exception as e:
            logger.error(f"ModelHub: error loading model {model_name} — {e}")
            return None
        finally:
            session.close()

    # --------------------------------------------------------
    # Fetch Metadata
    # --------------------------------------------------------
    def get_model_metadata(self, model_name: str, latest: bool = True):
        """
        Retrieve model metadata from the registry.
        """
        session = Session(self.engine)
        try:
            query = session.query(ModelRegistry).filter(ModelRegistry.model_name == model_name)
            if latest:
                query = query.order_by(ModelRegistry.timestamp.desc())
            result = query.first()
            return result.meta if result else None
        finally:
            session.close()

    # --------------------------------------------------------
    # Version History
    # --------------------------------------------------------
    def list_versions(self, model_name: str):
        """
        List all recorded versions for a given model.
        """
        session = Session(self.engine)
        try:
            versions = (
                session.query(ModelRegistry.version)
                .filter(ModelRegistry.model_name == model_name)
                .order_by(ModelRegistry.timestamp.desc())
                .all()
            )
            return [v[0] for v in versions]
        finally:
            session.close()
