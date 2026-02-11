"""Tests for pie_cli.model module."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from pie_cli.cli import app

runner = CliRunner()


class TestModelList:
    """Tests for model list command."""

    def test_list_empty(self, tmp_path):
        """Shows message when no models."""
        with patch("pie_backend.hf_utils.get_hf_cache_dir", return_value=tmp_path):
            result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "No models found" in result.stdout

    def test_list_models(self, tmp_path):
        """Lists downloaded models."""
        # Setup HF cache structure
        hf_cache = tmp_path

        # Model A: Compatible
        model_a_dir = hf_cache / "models--user--model-a"
        model_a_dir.mkdir()
        (model_a_dir / "snapshots").mkdir()
        snapshot_a = model_a_dir / "snapshots" / "snap1"
        snapshot_a.mkdir()
        (snapshot_a / "config.json").write_text('{"model_type": "llama"}')

        # Model B: Incompatible
        model_b_dir = hf_cache / "models--user--model-b"
        model_b_dir.mkdir()
        (model_b_dir / "snapshots").mkdir()
        snapshot_b = model_b_dir / "snapshots" / "snap1"
        snapshot_b.mkdir()
        (snapshot_b / "config.json").write_text('{"model_type": "unknown"}')

        with patch("pie_backend.hf_utils.get_hf_cache_dir", return_value=hf_cache):
            result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "user/model-a" in result.stdout
        assert "llama" in result.stdout  # displays arch
        assert "user/model-b" in result.stdout
        assert "unsupported" in result.stdout


class TestModelRemove:
    """Tests for model remove command."""

    def test_remove_model(self):
        """Removes downloaded model."""
        mock_delete_strategy = MagicMock()
        mock_repo = MagicMock()
        mock_repo.repo_id = "user/model-a"
        mock_repo.size_on_disk = 1024 * 1024 * 100  # 100 MB
        mock_repo.revisions = [MagicMock(commit_hash="hash1")]

        mock_cache_info = MagicMock()
        mock_cache_info.repos = [mock_repo]
        mock_cache_info.delete_revisions.return_value = mock_delete_strategy

        with patch("huggingface_hub.scan_cache_dir", return_value=mock_cache_info):
            # Confirm removal
            result = runner.invoke(
                app, ["model", "remove", "user/model-a"], input="y\n"
            )

        assert result.exit_code == 0
        assert "Removed user/model-a" in result.stdout
        mock_cache_info.delete_revisions.assert_called()
        mock_delete_strategy.execute.assert_called()

    def test_remove_nonexistent(self):
        """Returns error when model doesn't exist."""
        mock_cache_info = MagicMock()
        mock_cache_info.repos = []

        with patch("huggingface_hub.scan_cache_dir", return_value=mock_cache_info):
            result = runner.invoke(app, ["model", "remove", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.stdout


