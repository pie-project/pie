import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from .config import START_TIMEOUT_S


def engine_version() -> str:
    pie = shutil.which("pie")
    if pie is None:
        return ""
    try:
        out = subprocess.run([pie, "--version"], capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    parts = out.split()
    return parts[-1] if parts else ""


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def start_engine(host: str, port: int) -> subprocess.Popen:
    pie = shutil.which("pie")
    if pie is None:
        raise SystemExit("the `pie` command is not on PATH; install it or use --no-engine")
    log_dir = Path(tempfile.gettempdir()) / "pie-chat"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / "serve.log", "w")  # noqa: SIM115
    print(f"starting the engine (log: {log.name}) ...", flush=True)
    engine = subprocess.Popen([pie, "serve"], stdout=log, stderr=subprocess.STDOUT)

    deadline = time.monotonic() + START_TIMEOUT_S
    while not port_open(host, port):
        if engine.poll() is not None:
            raise SystemExit(f"`pie serve` exited early; see {log.name}")
        if time.monotonic() > deadline:
            engine.terminate()
            raise SystemExit(f"the engine did not start within {START_TIMEOUT_S}s; see {log.name}")
        time.sleep(1)
    print("engine ready.", flush=True)
    return engine


def stop_engine(engine: subprocess.Popen) -> None:
    engine.terminate()
    try:
        engine.wait(timeout=15)
    except subprocess.TimeoutExpired:
        engine.kill()
