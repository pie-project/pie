"""Tests for pie_cli.auth module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import toml
from typer.testing import CliRunner

from pie.auth import load_authorized_users, save_authorized_users
from pie_cli.cli import app

runner = CliRunner()


class TestAuthFormat:
    """Tests for auth file format compatibility with Rust runtime."""

    def test_save_produces_rust_compatible_format(self, tmp_path):
        """Verifies saved format matches Rust AuthorizedUsers struct expectations."""
        auth_path = tmp_path / "authorized_users.toml"
        users = {"testuser": {"mykey": "ssh-ed25519 AAAAC3..."}}

        save_authorized_users(auth_path, users)

        # Check raw TOML structure matches Rust expectations
        raw = toml.loads(auth_path.read_text())
        assert "users" in raw
        assert "testuser" in raw["users"]
        assert "keys" in raw["users"]["testuser"]
        assert raw["users"]["testuser"]["keys"]["mykey"] == "ssh-ed25519 AAAAC3..."

    def test_load_handles_nested_format(self, tmp_path):
        """Loads files in the nested format expected by Rust."""
        auth_path = tmp_path / "authorized_users.toml"
        auth_path.write_text('[users.testuser.keys]\nmykey = "ssh-ed25519 AAAAC3..."')

        users = load_authorized_users(auth_path)

        assert "testuser" in users
        assert "mykey" in users["testuser"]
        assert users["testuser"]["mykey"] == "ssh-ed25519 AAAAC3..."

    def test_roundtrip_preserves_data(self, tmp_path):
        """Save then load preserves all user data."""
        auth_path = tmp_path / "authorized_users.toml"
        original = {
            "alice": {"laptop": "key1", "desktop": "key2"},
            "bob": {"server": "key3"},
        }

        save_authorized_users(auth_path, original)
        loaded = load_authorized_users(auth_path)

        assert loaded == original


class TestAuthAdd:
    """Tests for auth add command."""

    def test_add_user_with_key(self, tmp_path):
        """Adds user with public key."""
        auth_path = tmp_path / ".pie" / "authorized_users.toml"
        auth_path.parent.mkdir(parents=True, exist_ok=True)

        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path / ".pie")}):
            result = runner.invoke(
                app,
                ["auth", "add", "testuser", "mykey"],
                input="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAAB test@example.com\n",
            )

        assert result.exit_code == 0
        assert auth_path.exists()
        users = load_authorized_users(auth_path)
        assert "testuser" in users
        assert "mykey" in users["testuser"]

    def test_add_user_without_key(self, tmp_path):
        """Creates user without key when no key provided."""
        auth_path = tmp_path / ".pie" / "authorized_users.toml"
        auth_path.parent.mkdir(parents=True, exist_ok=True)

        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path / ".pie")}):
            result = runner.invoke(
                app,
                ["auth", "add", "testuser"],
                input="",  # Empty key
            )

        assert result.exit_code == 0
        assert auth_path.exists()
        users = load_authorized_users(auth_path)
        assert "testuser" in users
        assert len(users["testuser"]) == 0  # No keys

    def test_add_key_to_existing_user(self, tmp_path):
        """Adds key to existing user."""
        auth_path = tmp_path / ".pie" / "authorized_users.toml"
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        # Use nested format
        save_authorized_users(auth_path, {"testuser": {"key1": "existing-key"}})

        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path / ".pie")}):
            result = runner.invoke(
                app,
                ["auth", "add", "testuser", "key2"],
                input="new-public-key-content\n",
            )

        assert result.exit_code == 0
        users = load_authorized_users(auth_path)
        assert "key1" in users["testuser"]
        assert "key2" in users["testuser"]


class TestAuthRemove:
    """Tests for auth remove command."""

    def test_remove_user(self, tmp_path):
        """Removes entire user."""
        auth_path = tmp_path / ".pie" / "authorized_users.toml"
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        save_authorized_users(auth_path, {"testuser": {"key1": "key-content"}})

        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path / ".pie")}):
            result = runner.invoke(
                app,
                ["auth", "remove", "testuser"],
                input="y\n",  # Confirm removal
            )

        assert result.exit_code == 0
        users = load_authorized_users(auth_path)
        assert "testuser" not in users

    def test_remove_specific_key(self, tmp_path):
        """Removes specific key from user."""
        auth_path = tmp_path / ".pie" / "authorized_users.toml"
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        save_authorized_users(auth_path, {"testuser": {"key1": "content1", "key2": "content2"}})

        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path / ".pie")}):
            result = runner.invoke(app, ["auth", "remove", "testuser", "key1"])

        assert result.exit_code == 0
        users = load_authorized_users(auth_path)
        assert "testuser" in users
        assert "key1" not in users["testuser"]
        assert "key2" in users["testuser"]

    def test_remove_nonexistent_user(self, tmp_path):
        """Returns error when user doesn't exist."""
        auth_path = tmp_path / ".pie" / "authorized_users.toml"
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        auth_path.write_text("")

        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path / ".pie")}):
            result = runner.invoke(app, ["auth", "remove", "nonexistent"])

        assert result.exit_code == 1


class TestAuthList:
    """Tests for auth list command."""

    def test_list_users(self, tmp_path):
        """Lists all users and their keys."""
        auth_path = tmp_path / ".pie" / "authorized_users.toml"
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        save_authorized_users(
            auth_path,
            {"alice": {"laptop": "key1", "desktop": "key2"}, "bob": {"server": "key3"}},
        )

        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path / ".pie")}):
            result = runner.invoke(app, ["auth", "list"])

        assert result.exit_code == 0
        assert "alice" in result.stdout
        assert "bob" in result.stdout
        assert "laptop" in result.stdout
        assert "desktop" in result.stdout
        assert "server" in result.stdout

    def test_list_empty(self, tmp_path):
        """Shows message when no users."""
        auth_path = tmp_path / ".pie" / "authorized_users.toml"
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        auth_path.write_text("")

        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path / ".pie")}):
            result = runner.invoke(app, ["auth", "list"])

        assert result.exit_code == 0
        assert "No authorized users" in result.stdout

    def test_list_no_file(self, tmp_path):
        """Shows message when file doesn't exist."""
        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path / ".pie")}):
            result = runner.invoke(app, ["auth", "list"])

        assert result.exit_code == 0
        assert "No authorized users" in result.stdout
