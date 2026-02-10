"""Tests for pie.config and pie_cli.config module."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import toml
from typer.testing import CliRunner

from pie_cli import defaults as config
from pie_cli.cli import app

runner = CliRunner()



class TestConfigInit:
    """Tests for config init command."""

    @patch("pie_cli.config.scan_cache_dir")
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

    @patch("pie_cli.config.scan_cache_dir")
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


class TestConfigUpdate:
    """Tests for config update command."""

    def test_update_engine_host(self, tmp_path):
        """Updates host in config file."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'host = "127.0.0.1"\nport = 8080\n[[model]]\nhf_repo = "test/model"\n'
        )

        result = runner.invoke(
            app, ["config", "update", "--host", "0.0.0.0", "--path", str(config_path)]
        )

        assert result.exit_code == 0
        updated = toml.loads(config_path.read_text())
        assert updated["host"] == "0.0.0.0"

    def test_update_engine_port(self, tmp_path):
        """Updates port in config file."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'host = "127.0.0.1"\nport = 8080\n[[model]]\nhf_repo = "test/model"\n'
        )

        result = runner.invoke(
            app, ["config", "update", "--port", "9090", "--path", str(config_path)]
        )

        assert result.exit_code == 0
        updated = toml.loads(config_path.read_text())
        assert updated["port"] == 9090

    def test_update_model_hf_repo(self, tmp_path):
        """Updates model hf_repo in config file."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'host = "127.0.0.1"\nport = 8080\n[[model]]\nhf_repo = "old/model"\n'
        )

        result = runner.invoke(
            app,
            [
                "config",
                "update",
                "--hf-repo",
                "new/model",
                "--path",
                str(config_path),
            ],
        )

        assert result.exit_code == 0
        updated = toml.loads(config_path.read_text())
        assert updated["model"][0]["hf_repo"] == "new/model"

    def test_update_no_options_warning(self, tmp_path):
        """Shows warning when no options provided."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('host = "127.0.0.1"\nport = 8080\n')

        result = runner.invoke(app, ["config", "update", "--path", str(config_path)])

        assert result.exit_code == 0
        assert "No configuration options provided" in result.stdout

    def test_update_error_when_missing(self, tmp_path):
        """Returns error when config file doesn't exist."""
        config_path = tmp_path / "nonexistent.toml"

        result = runner.invoke(
            app, ["config", "update", "--host", "0.0.0.0", "--path", str(config_path)]
        )

        assert result.exit_code == 1
        # Error messages go to stderr, check combined output
        assert "not found" in result.output.lower()
