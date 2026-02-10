"""Tests for pie_cli.config module."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import toml
from typer.testing import CliRunner

from pie_cli.config import defaults as config
from pie_cli.cli import app

runner = CliRunner()



class TestConfigInit:
    """Tests for config init command."""

    @patch("huggingface_hub.scan_cache_dir")
    def test_init_creates_config_file(self, mock_scan, tmp_path):
        """Creates config file at specified path."""
        # Mock cache to contain default model
        mock_repo = MagicMock()
        mock_repo.repo_id = config.DEFAULT_MODEL
        mock_scan.return_value.repos = [mock_repo]

        config_path = tmp_path / "config.toml"

        result = runner.invoke(app, ["config", "init", "--path", str(config_path)])

        assert result.exit_code == 0
        assert config_path.exists()
        assert "Configuration file created" in result.stdout
        assert "Warning" not in result.stdout

    @patch("huggingface_hub.scan_cache_dir")
    def test_init_warns_missing_model(self, mock_scan, tmp_path):
        """Warns when default model is missing."""
        mock_scan.return_value.repos = []

        config_path = tmp_path / "config.toml"

        result = runner.invoke(app, ["config", "init", "--path", str(config_path)])

        assert result.exit_code == 0
        assert config_path.exists()
        assert "Default model" in result.stdout
        assert "not found" in result.stdout


class TestConfigShow:
    """Tests for config show command."""

    def test_show_displays_config(self, tmp_path):
        """Displays config file content."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('host = "localhost"\nport = 9000\n')

        result = runner.invoke(app, ["config", "show", "--path", str(config_path)])

        assert result.exit_code == 0
        assert "localhost" in result.stdout
        assert "9000" in result.stdout

    def test_show_error_when_missing(self, tmp_path):
        """Returns error when config file doesn't exist."""
        config_path = tmp_path / "nonexistent.toml"

        result = runner.invoke(app, ["config", "show", "--path", str(config_path)])

        assert result.exit_code == 1
        # Error messages go to stderr, check combined output
        assert "not found" in result.output.lower()


class TestConfigSet:
    """Tests for config set command."""

    def test_set_host(self, tmp_path):
        """Sets host in config file."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'host = "127.0.0.1"\nport = 8080\n[[model]]\nhf_repo = "test/model"\n'
        )

        result = runner.invoke(
            app, ["config", "set", "host", "0.0.0.0", "--path", str(config_path)]
        )

        assert result.exit_code == 0
        updated = toml.loads(config_path.read_text())
        assert updated["host"] == "0.0.0.0"

    def test_set_port(self, tmp_path):
        """Sets port in config file."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'host = "127.0.0.1"\nport = 8080\n[[model]]\nhf_repo = "test/model"\n'
        )

        result = runner.invoke(
            app, ["config", "set", "port", "9090", "--path", str(config_path)]
        )

        assert result.exit_code == 0
        updated = toml.loads(config_path.read_text())
        assert updated["port"] == 9090

    def test_set_model_hf_repo(self, tmp_path):
        """Sets model hf_repo in config file."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'host = "127.0.0.1"\nport = 8080\n[[model]]\nhf_repo = "old/model"\n'
        )

        result = runner.invoke(
            app,
            [
                "config",
                "set",
                "model.0.hf_repo",
                "new/model",
                "--path",
                str(config_path),
            ],
        )

        assert result.exit_code == 0
        updated = toml.loads(config_path.read_text())
        assert updated["model"][0]["hf_repo"] == "new/model"

    def test_set_auth_enabled(self, tmp_path):
        """Sets nested auth.enabled value."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'host = "127.0.0.1"\nport = 8080\n\n[auth]\nenabled = false\n'
        )

        result = runner.invoke(
            app, ["config", "set", "auth.enabled", "true", "--path", str(config_path)]
        )

        assert result.exit_code == 0
        updated = toml.loads(config_path.read_text())
        assert updated["auth"]["enabled"] is True

    def test_set_error_when_missing(self, tmp_path):
        """Returns error when config file doesn't exist."""
        config_path = tmp_path / "nonexistent.toml"

        result = runner.invoke(
            app, ["config", "set", "host", "0.0.0.0", "--path", str(config_path)]
        )

        assert result.exit_code == 1
        # Error messages go to stderr, check combined output
        assert "not found" in result.output.lower()
