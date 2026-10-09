# chat

A full-screen terminal chat that talks to a running `pie serve`.

```
pip install -r examples/chat/requirements.txt
python examples/chat/chat.py
```

If nothing is listening on a local address, the chat starts `pie serve` for you
and stops it on exit. An engine that is already running is reused and left alone.

Or, from the repo root, `./examples/chat/pie-chat` does the setup and the start in
one step (first run creates `.venv` for the chat's packages).

Replies stream from the engine's OpenAI-compatible endpoint
(`/v1/chat/completions`). The conversation is kept, so follow-up questions see
earlier turns. Flags: `--url` for another address, `--no-engine` to connect only
(never start one), `--placeholder` to run without an engine.

Enter sends a message. `/new` starts over. Ctrl-C or Ctrl-D twice quits
(the first press shows a hint; a second within two seconds exits).
