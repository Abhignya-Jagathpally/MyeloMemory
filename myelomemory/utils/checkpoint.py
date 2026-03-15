"""Checkpoint management: save, load, resume, and track pipeline stages.

All checkpoints follow a standard format with model state, optimizer state,
metrics, config, and git hash for full reproducibility.
"""

from __future__ import annotations

import dataclasses
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """Manages saving, loading, and tracking checkpoints across pipeline stages.

    Checkpoints are stored as .pt files in the checkpoint directory.
    Each stage has a unique name that maps to a single checkpoint file.

    Args:
        checkpoint_dir: Directory for storing checkpoint files.
        logger: Logger instance.
    """

    def __init__(self, checkpoint_dir: Path, logger: logging.Logger | None = None) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)

    def path(self, stage_name: str) -> Path:
        """Get the file path for a stage's checkpoint.

        Args:
            stage_name: Name of the pipeline stage (e.g., 'vae_pretrained').

        Returns:
            Path to the checkpoint file.
        """
        return self.checkpoint_dir / f"{stage_name}.pt"

    def exists(self, stage_name: str) -> bool:
        """Check if a checkpoint exists for a given stage.

        Args:
            stage_name: Name of the pipeline stage.

        Returns:
            True if checkpoint file exists and is non-empty.
        """
        p = self.path(stage_name)
        return p.exists() and p.stat().st_size > 0

    def save(self, stage_name: str, data: dict[str, Any]) -> Path:
        """Save a checkpoint for a pipeline stage.

        Automatically adds metadata (timestamp, git hash) to the checkpoint.

        Args:
            stage_name: Name of the pipeline stage.
            data: Dict containing model state, optimizer state, metrics, etc.

        Returns:
            Path to the saved checkpoint file.
        """
        ckpt_path = self.path(stage_name)

        # Add metadata
        data["_metadata"] = {
            "stage_name": stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_hash": self._get_git_hash(),
        }

        # Convert dataclass configs to dicts for serialization
        if "config" in data:
            config = data["config"]
            if dataclasses.is_dataclass(config):
                data["config"] = dataclasses.asdict(config)

        torch.save(data, ckpt_path)
        self.logger.info(f"Checkpoint saved: {ckpt_path}")

        return ckpt_path

    def load(self, stage_name: str, map_location: str = "cpu") -> dict[str, Any]:
        """Load a checkpoint for a pipeline stage.

        Args:
            stage_name: Name of the pipeline stage.
            map_location: Device to map tensors to (default: 'cpu').

        Returns:
            Loaded checkpoint dict.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
        """
        ckpt_path = self.path(stage_name)

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found for stage '{stage_name}' at {ckpt_path}. "
                f"Run the '{stage_name}' stage first."
            )

        data = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        self.logger.info(
            f"Checkpoint loaded: {ckpt_path} "
            f"(saved {data.get('_metadata', {}).get('timestamp', 'unknown')})"
        )

        return data

    def get_latest_completed_stage(self, stage_names: list[str]) -> str | None:
        """Find the most recently completed pipeline stage.

        Args:
            stage_names: Ordered list of stage names.

        Returns:
            Name of the latest completed stage, or None if none completed.
        """
        latest = None
        for name in stage_names:
            if self.exists(name):
                latest = name
        return latest

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all existing checkpoints with metadata.

        Returns:
            List of dicts with 'stage_name', 'path', 'size_mb', 'modified'.
        """
        checkpoints = []
        for pt_file in sorted(self.checkpoint_dir.glob("*.pt")):
            checkpoints.append({
                "stage_name": pt_file.stem,
                "path": pt_file,
                "size_mb": pt_file.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(
                    pt_file.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            })
        return checkpoints

    def delete(self, stage_name: str) -> None:
        """Delete a checkpoint file.

        Args:
            stage_name: Name of the pipeline stage.
        """
        ckpt_path = self.path(stage_name)
        if ckpt_path.exists():
            ckpt_path.unlink()
            self.logger.info(f"Checkpoint deleted: {ckpt_path}")

    @staticmethod
    def _get_git_hash() -> str:
        """Get the current git commit hash, or 'unknown' if not in a repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "unknown"
