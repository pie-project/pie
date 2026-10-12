"""Command line entry point: `python -m pie_openai_adapter`."""

from __future__ import annotations

import argparse
import logging

from aiohttp import web

from .backend import PieBackend
from .server import create_app
from .translate import Defaults


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pie-openai-adapter", description=__doc__)
    parser.add_argument("--pie-url", default="ws://127.0.0.1:8080", help="Pie gateway WebSocket URL")
    parser.add_argument("--identity", default="default/adapter", help="x-pie-identity value")
    parser.add_argument("--inferlet", default="chat-completion@0.1.0", help="inferlet to run per request")
    parser.add_argument("--model-name", default="pie", help="model id advertised on /v1/models")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    app = create_app(
        PieBackend(args.pie_url, inferlet=args.inferlet, identity=args.identity),
        model_name=args.model_name,
        defaults=Defaults(max_tokens=args.max_tokens),
    )
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
