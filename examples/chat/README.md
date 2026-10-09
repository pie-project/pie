# chat

A full-screen terminal chat that talks to a running `pie serve`.

```
pie serve                                  # in one terminal, with a model configured
pip install -r examples/chat/requirements.txt
python examples/chat/chat.py               # in another terminal
```

Replies stream from the engine's OpenAI-compatible endpoint
(`/v1/chat/completions`). The conversation is kept, so follow-up questions see
earlier turns. Use `--url` to point at another address, or `--placeholder` to run
without an engine.

Enter sends a message. `/new` starts over. Ctrl-D quits.
