"""Clean interface for importing common backend components."""

import sys
from pathlib import Path

# Use repo_utils for consistent path setup
_repo_utils_path = Path(__file__).parent.parent.parent / "repo_utils.py"
sys.path.insert(0, str(_repo_utils_path.parent))

import repo_utils  # pylint: disable=wrong-import-position

repo_utils.setup_pie_imports()

# Import from common_python modules - import what's actually used by backend-python
# pylint: disable=unused-import  # These are re-exported for the module interface
# pylint: disable=wrong-import-position,wrong-import-order  # Must come after repo_utils setup
from config.common import ModelInfo
from config.l4ma import L4maArch
from config.qwen3 import Qwen3Arch
from config.gptoss import GptOssArch
from model_loader import load_model
from server_common import build_config, print_config, start_service
from handler_common import Handler
from adapter_import_utils import get_adapter_subpass, AdapterSubpass
