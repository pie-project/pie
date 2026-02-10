"""Tests for pie_cli.path module."""

import os
from pathlib import Path
from unittest.mock import patch

from pie_cli import path as pie_path


class TestGetPieHome:
    """Tests for get_pie_home function."""

    def test_default_path(self):
        """Returns ~/.pie when PIE_HOME is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove PIE_HOME if it exists
            os.environ.pop("PIE_HOME", None)
            result = pie_path.get_pie_home()
            assert result == Path.home() / ".pie"

    def test_custom_path_from_env(self, tmp_path):
        """Returns PIE_HOME when set."""
        custom_path = str(tmp_path / "custom_pie")
        with patch.dict(os.environ, {"PIE_HOME": custom_path}):
            result = pie_path.get_pie_home()
            assert result == Path(custom_path)


class TestGetDefaultConfigPath:
    """Tests for get_default_config_path function."""

    def test_returns_config_toml_in_pie_home(self):
        """Returns ~/.pie/config.toml."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PIE_HOME", None)
            result = pie_path.get_default_config_path()
            assert result == Path.home() / ".pie" / "config.toml"


class TestGetAuthorizedUsersPath:
    """Tests for get_authorized_users_path function."""

    def test_returns_authorized_users_toml_in_pie_home(self):
        """Returns ~/.pie/authorized_users.toml."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PIE_HOME", None)
            result = pie_path.get_authorized_users_path()
            assert result == Path.home() / ".pie" / "authorized_users.toml"

