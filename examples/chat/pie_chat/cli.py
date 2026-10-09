import argparse
import os
from urllib.parse import urlsplit

from .app import build
from .backend import EngineBackend, PlaceholderBackend
from .chat import Chat
from .config import DEFAULT_URL, LOCAL_HOSTS
from .engine import port_open, start_engine, stop_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal chat on a running pie engine.")
    parser.add_argument("--url", default=DEFAULT_URL, help="address of `pie serve`")
    parser.add_argument("--no-engine", action="store_true", help="connect only; never start `pie serve`")
    parser.add_argument("--placeholder", action="store_true", help="fixed replies, no engine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = None
    if not args.placeholder:
        split = urlsplit(args.url)
        host, port = split.hostname or "127.0.0.1", split.port or 8080
        if not args.no_engine and host in LOCAL_HOSTS and not port_open(host, port):
            engine = start_engine(host, port)

    backend = PlaceholderBackend() if args.placeholder else EngineBackend(args.url)
    os.system("cls" if os.name == "nt" else "clear")
    chat = Chat(backend)
    try:
        build(chat).run()
    finally:
        if engine is not None:
            stop_engine(engine)
