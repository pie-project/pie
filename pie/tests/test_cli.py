"""Tests for pie_cli CLI main commands."""

import os
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from pie_cli.cli import app

runner = CliRunner()


class TestCliHelp:
    """Tests for CLI help output."""

    def test_main_help(self):
        """Shows all main commands."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "serve" in result.stdout
        assert "run" in result.stdout
        assert "config" in result.stdout
        assert "model" in result.stdout
        assert "auth" in result.stdout

    def test_serve_help(self):
        """Shows serve command options."""
        result = runner.invoke(app, ["serve", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--no-auth" in result.stdout
        assert "--verbose" in result.stdout

    def test_run_help(self):
        """Shows run command options."""
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        # Check argument name in help (typer converts underscores to uppercase usually)
        assert "INFERLET" in result.stdout or "inferlet" in result.stdout
        assert "--config" in result.stdout
        assert "--log" in result.stdout

    def test_config_help(self):
        """Shows config subcommands."""
        result = runner.invoke(app, ["config", "--help"])

        assert result.exit_code == 0
        assert "init" in result.stdout
        assert "show" in result.stdout
        assert "set" in result.stdout

    def test_config_init_help(self):
        """Shows config init options."""
        result = runner.invoke(app, ["config", "init", "--help"])

        assert result.exit_code == 0
        # --dummy was removed
        assert "--path" in result.stdout

    def test_config_set_help(self):
        """Shows config set options."""
        result = runner.invoke(app, ["config", "set", "--help"])

        assert result.exit_code == 0
        assert "KEY" in result.stdout
        assert "VALUE" in result.stdout
        assert "--path" in result.stdout

    def test_model_help(self):
        """Shows model subcommands."""
        result = runner.invoke(app, ["model", "--help"])

        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "download" in result.stdout
        assert "remove" in result.stdout

    def test_auth_help(self):
        """Shows auth subcommands."""
        result = runner.invoke(app, ["auth", "--help"])

        assert result.exit_code == 0
        assert "add" in result.stdout
        assert "remove" in result.stdout
        assert "list" in result.stdout


class TestServeCommand:
    """Tests for serve command."""

    def test_serve_missing_config(self, tmp_path):
        """Returns error when config file doesn't exist."""
        # Need to patch get_default_config_path or provide custom path
        # If we provide custom non-existent path, it should fail
        result = runner.invoke(
            app, ["serve", "--config", str(tmp_path / "missing.toml")]
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestRunCommand:
    """Tests for run command."""

    def test_run_missing_inferlet(self, tmp_path):
        """Returns error when inferlet path doesn't exist."""
        result = runner.invoke(
            app, ["run", "--path", str(tmp_path / "nonexistent.wasm")]
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()
