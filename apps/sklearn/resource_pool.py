"""
Resource Pool - In-memory storage for the latest training session data.

Stores X_train, X_test, y_train, y_test from the most recent training run.
This allows YellowBrick visualizations to access the same data without
reloading from DuckDB.

Model is NOT stored here - it's loaded on-demand from MLflow via ModelCache
to ensure YellowBrick always uses the best model (same as predictions).

Thread-safe singleton pattern - only ONE session stored at a time.
New training runs replace the previous session data.
"""
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any

import pandas as pd


@dataclass
class SessionResources:
    """Data from a single training session (no model - loaded from MLflow on demand)."""

    # Training data
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    # Feature metadata
    feature_names: list[str]
    cat_feature_indices: list[int]

    # Session metadata
    project_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def train_samples(self) -> int:
        return len(self.X_train)

    @property
    def test_samples(self) -> int:
        return len(self.X_test)

    @property
    def X(self) -> pd.DataFrame:
        """Combined X (train + test) for visualizations that need full data."""
        return pd.concat([self.X_train, self.X_test], ignore_index=True)

    @property
    def y(self) -> pd.Series:
        """Combined y (train + test) for visualizations that need full data."""
        return pd.concat([self.y_train, self.y_test], ignore_index=True)


class ResourcePool:
    """
    Thread-safe singleton storing the latest training session data.

    Only ONE session is stored at a time. Each new training run
    replaces the previous session data entirely.

    Note: Model is NOT stored here. Use ModelCache to get the best
    model from MLflow for predictions and visualizations.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._session: SessionResources | None = None
                    cls._instance._session_lock = Lock()
        return cls._instance

    def store(self, resources: SessionResources) -> None:
        """
        Store a new training session, replacing any previous session.

        Args:
            resources: SessionResources from the latest training run
        """
        with self._session_lock:
            self._session = resources

    def get(self) -> SessionResources | None:
        """
        Get the current session resources.

        Returns:
            SessionResources if available, None otherwise
        """
        with self._session_lock:
            return self._session

    def clear(self) -> None:
        """Clear the current session, freeing memory."""
        with self._session_lock:
            self._session = None

    def has_session(self) -> bool:
        """Check if a session is available."""
        with self._session_lock:
            return self._session is not None

    def status(self) -> dict:
        """
        Get status summary of the resource pool.

        Returns:
            Dict with session info or empty status
        """
        with self._session_lock:
            if self._session is None:
                return {
                    "has_session": False,
                    "message": "No training session available. Run training first.",
                }
            return {
                "has_session": True,
                "project_name": self._session.project_name,
                "train_samples": self._session.train_samples,
                "test_samples": self._session.test_samples,
                "feature_count": len(self._session.feature_names),
                "feature_names": self._session.feature_names,
                "created_at": self._session.created_at.isoformat(),
            }


# Global singleton instance
resource_pool = ResourcePool()
