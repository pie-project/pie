"""Acceptance tests for the API-compatible built-in inferlets.

These run against a live `pie serve` and speak to it the way real clients
do: the official `openai`, `anthropic` and `google-genai` packages when they
are installed, and plain HTTP always. If the official client parses the
answer, the server is compatible; that is the whole definition.

    python test_compat.py --base-url http://127.0.0.1:8080

The model is whatever the server loaded. Assertions are about shape and
protocol (finish reasons, streaming framing, tool-call structure, forced
JSON), not about the words, which are model- and seed-dependent.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
import urllib.error
import urllib.request


class Skip(Exception):
    """A test that cannot run here (an official client is not installed)."""


def official(module: str):
    try:
        return importlib.import_module(module)
    except ImportError:
        raise Skip(f"the `{module}` package is not installed")


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


class Http:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def post(self, path: str, body: dict, *, stream: bool = False):
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read().decode()
        if stream:
            return resp.status, dict(resp.headers), resp
        return resp.status, dict(resp.headers), resp.read().decode()

    def get(self, path: str):
        req = urllib.request.Request(self.base_url + path, method="GET")
        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()
        return resp.status, resp.read().decode()


def sse_frames(resp) -> list[tuple[str | None, str]]:
    """(event, data) per frame, in order."""
    frames = []
    event = None
    data_lines: list[str] = []
    for raw in resp:
        line = raw.decode().rstrip("\n").rstrip("\r")
        if line == "":
            if data_lines:
                frames.append((event, "\n".join(data_lines)))
            event = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        frames.append((event, "\n".join(data_lines)))
    return frames


def chat(http: Http, *, stream: bool = False, **body):
    body.setdefault("model", "pie")
    body.setdefault("messages", [{"role": "user", "content": "Say hello in one word."}])
    body.setdefault("max_tokens", 32)
    body.setdefault("temperature", 0)
    if stream:
        body["stream"] = True
    return http.post("/v1/chat/completions", body, stream=stream)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_models(http: Http):
    status, text = http.get("/v1/models")
    assert status == 200, text
    body = json.loads(text)
    assert body["object"] == "list"
    assert body["data"] and body["data"][0]["object"] == "model", body


def test_non_streaming(http: Http):
    status, _, text = chat(http, chat_template_kwargs={"enable_thinking": False})
    assert status == 200, text
    body = json.loads(text)
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["created"] > 1_700_000_000
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str) and choice["message"]["content"]
    assert choice["finish_reason"] in ("stop", "length")
    usage = body["usage"]
    assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert "<think>" not in choice["message"]["content"]


def test_length_finish(http: Http):
    status, _, text = chat(
        http,
        messages=[{"role": "user", "content": "Count from one to one hundred, in words."}],
        max_tokens=4,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert status == 200, text
    body = json.loads(text)
    assert body["choices"][0]["finish_reason"] == "length", body
    assert body["usage"]["completion_tokens"] == 4, body["usage"]


def test_streaming(http: Http):
    status, headers, resp = chat(
        http,
        stream=True,
        stream_options={"include_usage": True},
        chat_template_kwargs={"enable_thinking": False},
    )
    assert status == 200, status
    content_type = {k.lower(): v for k, v in headers.items()}.get("content-type", "")
    assert content_type.startswith("text/event-stream"), headers
    frames = sse_frames(resp)
    assert frames[-1][1] == "[DONE]", frames[-1]
    chunks = [json.loads(data) for _, data in frames[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    ids = {c["id"] for c in chunks}
    assert len(ids) == 1, ids
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    content = "".join(
        c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"]
    )
    assert content.strip(), chunks
    finishes = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]]
    assert finishes[-1] in ("stop", "length"), finishes
    assert all(f is None for f in finishes[:-1]), finishes
    usage = chunks[-1]
    assert usage["choices"] == [] and usage["usage"]["completion_tokens"] > 0, usage


def test_reasoning_content(http: Http):
    status, _, text = chat(
        http,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        max_tokens=200,
        chat_template_kwargs={"enable_thinking": True},
    )
    assert status == 200, text
    msg = json.loads(text)["choices"][0]["message"]
    # A thinking model puts its reasoning in `reasoning_content`, never in
    # `content`; a model that does not think has neither tag anywhere.
    assert "<think>" not in (msg.get("content") or "")
    assert "</think>" not in (msg.get("content") or "")
    if "reasoning_content" in msg:
        assert msg["reasoning_content"].strip()


def test_stop_sequence(http: Http):
    status, _, text = chat(
        http,
        messages=[{"role": "user", "content": "Write the alphabet, letters separated by spaces."}],
        max_tokens=64,
        stop=["F"],
        chat_template_kwargs={"enable_thinking": False},
    )
    assert status == 200, text
    choice = json.loads(text)["choices"][0]
    assert "F" not in choice["message"]["content"], choice
    if choice["finish_reason"] == "stop":
        assert choice["message"]["content"], choice


def test_forced_tool_call(http: Http):
    status, _, text = chat(
        http,
        messages=[{"role": "user", "content": "What is the weather in Paris right now?"}],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        max_tokens=64,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert status == 200, text
    choice = json.loads(text)["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    calls = choice["message"]["tool_calls"]
    assert calls and calls[0]["type"] == "function", choice
    assert calls[0]["function"]["name"] == "get_weather", calls
    args = json.loads(calls[0]["function"]["arguments"])
    assert "city" in args, args
    assert calls[0]["id"].startswith("call_"), calls
    assert "<tool_call>" not in (choice["message"]["content"] or ""), choice


def test_named_tool_call_streaming(http: Http):
    status, _, resp = chat(
        http,
        stream=True,
        messages=[{"role": "user", "content": "Weather in Tokyo?"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_tokens=64,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert status == 200
    frames = sse_frames(resp)
    chunks = [json.loads(d) for _, d in frames if d != "[DONE]"]
    calls = [
        tc
        for c in chunks
        if c["choices"]
        for tc in (c["choices"][0]["delta"].get("tool_calls") or [])
    ]
    assert calls, chunks
    assert calls[0]["function"]["name"] == "get_weather", calls
    assert "city" in json.loads(calls[0]["function"]["arguments"]), calls
    finishes = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]]
    assert finishes[-1] == "tool_calls", finishes


def test_tool_result_round_trip(http: Http):
    status, _, text = chat(
        http,
        messages=[
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"temperature_c\": 21, \"sky\": \"sunny\"}"},
        ],
        tools=[WEATHER_TOOL],
        max_tokens=64,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert status == 200, text
    choice = json.loads(text)["choices"][0]
    assert choice["message"]["content"], choice


def test_json_schema(http: Http):
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
        "required": ["city", "population"],
        "additionalProperties": False,
    }
    status, _, text = chat(
        http,
        messages=[{"role": "user", "content": "Give the city of Paris and its population as JSON."}],
        response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": schema}},
        max_tokens=64,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert status == 200, text
    content = json.loads(text)["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert isinstance(parsed["city"], str) and isinstance(parsed["population"], int), parsed


def test_json_object(http: Http):
    status, _, text = chat(
        http,
        messages=[{"role": "user", "content": "Return a JSON object with one key, `ok`, set to true."}],
        response_format={"type": "json_object"},
        max_tokens=48,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert status == 200, text
    content = json.loads(text)["choices"][0]["message"]["content"]
    assert isinstance(json.loads(content), dict), content


def test_structural_tag(http: Http):
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    status, _, text = http.post(
        "/v1/completions",
        {
            "prompt": 'Q: Paris\nA: The city <city>{"name": "Paris"}</city>\nQ: Rome\nA:',
            "max_tokens": 40,
            "temperature": 0,
            "stop": ["\nQ:"],
            "response_format": {
                "type": "structural_tag",
                "structures": [{"begin": "<city>", "schema": schema, "end": "</city>"}],
                "triggers": ["<city>"],
            },
        },
    )
    assert status == 200, text
    out = json.loads(text)["choices"][0]["text"]
    # Free before the trigger, the schema and the end forced after it.
    assert "<city>" in out, out
    inside = out.split("<city>", 1)[1]
    assert "</city>" in inside, out
    assert isinstance(json.loads(inside.split("</city>", 1)[0])["name"], str), out


def test_bad_requests(http: Http):
    status, _, text = chat(http, messages=[])
    assert status == 400, (status, text)
    err = json.loads(text)["error"]
    assert err["type"] == "invalid_request_error" and err["param"] == "messages", err

    status, _, text = chat(http, n=3)
    assert status == 200 and len(json.loads(text)["choices"]) == 3, text
    status, _, text = chat(http, n=0)
    assert status == 400 and json.loads(text)["error"]["param"] == "n", text

    status, _, text = chat(
        http,
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
    )
    assert status == 400, (status, text)
    err = json.loads(text)["error"]
    assert err["param"] == "messages[0]" and "data:" in err["message"], err

    status, _, text = http.post("/v1/chat/completions", {"messages": "nope"})
    assert status == 400, text

    status, _, text = http.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "tool_choice": "sometimes"})
    assert status == 400 and json.loads(text)["error"]["param"] == "tool_choice", text


def test_official_openai_client(http: Http):
    client = official("openai").OpenAI(base_url=http.base_url + "/v1", api_key="unused")
    models = client.models.list()
    assert models.data, models
    completion = client.chat.completions.create(
        model=models.data[0].id,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=32,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    assert completion.choices[0].message.content
    assert completion.usage and completion.usage.total_tokens > 0

    pieces = []
    stream = client.chat.completions.create(
        model=models.data[0].id,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=32,
        temperature=0,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    finish = None
    for chunk in stream:
        if chunk.choices:
            pieces.append(chunk.choices[0].delta.content or "")
            finish = chunk.choices[0].finish_reason or finish
    assert "".join(pieces).strip(), pieces
    assert finish in ("stop", "length"), finish

    forced = client.chat.completions.create(
        model=models.data[0].id,
        messages=[{"role": "user", "content": "Weather in Rome?"}],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        max_tokens=64,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    call = forced.choices[0].message.tool_calls[0]
    assert call.function.name == "get_weather"
    assert "city" in json.loads(call.function.arguments)


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------

ANTHROPIC_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather in a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def messages(http: Http, *, stream: bool = False, **body):
    body.setdefault("model", "pie")
    body.setdefault("max_tokens", 32)
    body.setdefault("temperature", 0)
    if stream:
        body["stream"] = True
    return http.post("/v1/messages", body, stream=stream)


def test_anthropic_non_streaming(http: Http):
    status, _, text = messages(
        http,
        system="You are terse.",
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    assert status == 200, text
    body = json.loads(text)
    assert body["type"] == "message" and body["role"] == "assistant"
    assert body["id"].startswith("msg_")
    assert body["content"][0]["type"] == "text" and body["content"][0]["text"]
    assert body["stop_reason"] in ("end_turn", "max_tokens"), body
    assert body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] > 0
    assert "<think>" not in body["content"][0]["text"]


def test_anthropic_streaming(http: Http):
    status, _, resp = messages(
        http,
        stream=True,
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    assert status == 200
    frames = sse_frames(resp)
    names = [e for e, _ in frames]
    assert names[0] == "message_start" and names[-1] == "message_stop", names
    assert "content_block_start" in names and "content_block_stop" in names
    assert names.index("message_delta") == len(names) - 2, names
    for event, data in frames:
        assert json.loads(data)["type"] == event, (event, data)
    text = "".join(
        json.loads(d)["delta"]["text"]
        for e, d in frames
        if e == "content_block_delta" and json.loads(d)["delta"]["type"] == "text_delta"
    )
    assert text.strip(), frames
    delta = json.loads([d for e, d in frames if e == "message_delta"][0])
    assert delta["delta"]["stop_reason"] in ("end_turn", "max_tokens"), delta
    assert delta["usage"]["output_tokens"] > 0, delta


def test_anthropic_tool_use(http: Http):
    status, _, text = messages(
        http,
        messages=[{"role": "user", "content": "What is the weather in Paris right now?"}],
        tools=[ANTHROPIC_TOOL],
        tool_choice={"type": "any"},
        max_tokens=64,
    )
    assert status == 200, text
    body = json.loads(text)
    assert body["stop_reason"] == "tool_use", body
    uses = [b for b in body["content"] if b["type"] == "tool_use"]
    assert uses and uses[0]["name"] == "get_weather", body
    assert uses[0]["id"].startswith("toolu_") and "city" in uses[0]["input"], uses

    # And the result replayed.
    status, _, text = messages(
        http,
        messages=[
            {"role": "user", "content": "What is the weather in Paris right now?"},
            {"role": "assistant", "content": body["content"]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": uses[0]["id"], "content": "21C and sunny"}
                ],
            },
        ],
        tools=[ANTHROPIC_TOOL],
        max_tokens=64,
    )
    assert status == 200, text
    body = json.loads(text)
    assert any(b["type"] == "text" and b["text"] for b in body["content"]), body


def test_anthropic_bad_requests(http: Http):
    status, _, text = http.post("/v1/messages", {"model": "pie", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 400, text
    err = json.loads(text)
    assert err["type"] == "error" and err["error"]["type"] == "invalid_request_error", err
    assert "max_tokens" in err["error"]["message"]
    status, _, text = messages(
        http,
        messages=[{"role": "user", "content": [{"type": "image", "source": {}}]}],
    )
    assert status == 400 and "image" in json.loads(text)["error"]["message"], text


def test_official_anthropic_client(http: Http):
    client = official("anthropic").Anthropic(base_url=http.base_url, api_key="unused")
    message = client.messages.create(
        model="pie",
        max_tokens=32,
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    assert message.content[0].type == "text" and message.content[0].text
    assert message.stop_reason in ("end_turn", "max_tokens")

    pieces = []
    with client.messages.stream(
        model="pie",
        max_tokens=32,
        messages=[{"role": "user", "content": "Say hello in one word."}],
    ) as stream:
        for text in stream.text_stream:
            pieces.append(text)
        final = stream.get_final_message()
    assert "".join(pieces).strip(), pieces
    assert final.usage.output_tokens > 0

    forced = client.messages.create(
        model="pie",
        max_tokens=64,
        messages=[{"role": "user", "content": "What is the weather in Paris right now?"}],
        tools=[ANTHROPIC_TOOL],
        tool_choice={"type": "any"},
    )
    use = [b for b in forced.content if b.type == "tool_use"][0]
    assert use.name == "get_weather" and "city" in use.input


# ---------------------------------------------------------------------------
# OpenAI legacy completions
# ---------------------------------------------------------------------------


def test_completions(http: Http):
    status, _, text = http.post(
        "/v1/completions",
        {"model": "pie", "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0},
    )
    assert status == 200, text
    body = json.loads(text)
    assert body["object"] == "text_completion" and body["id"].startswith("cmpl-")
    choice = body["choices"][0]
    assert "paris" in choice["text"].lower(), choice
    assert choice["finish_reason"] == "length" and body["usage"]["completion_tokens"] == 8, body

    status, _, text = http.post(
        "/v1/completions",
        {"prompt": "The capital of France is", "max_tokens": 4, "temperature": 0, "echo": True},
    )
    assert json.loads(text)["choices"][0]["text"].startswith("The capital of France is"), text

    status, _, text = http.post("/v1/completions", {"prompt": ["a", "b"]})
    assert status == 400 and json.loads(text)["error"]["param"] == "prompt", text


SAMPLING_ARMS = [
    {"temperature": 1.0},
    {"temperature": 0.7},
    {"temperature": 1.0, "min_p": 0.05},
    {"temperature": 0.7, "top_p": 0.9},
    {"temperature": 1.0, "top_k": 40},
]


def test_sampling_arms(http: Http):
    """Every sampler arm samples: at temperature > 0 a fixed seed answers the
    same each time, different seeds answer differently, nothing degenerates to
    one repeated token (a draw over an unwritten row answers token 0 every
    step: "!!!!" on Qwen), and the drawn tokens are ones the model finds
    plausible. Without a cut, compat's draw is one fused scan of the logits;
    each cut takes another arm, so each is covered."""
    prompt = "Once upon a time"

    def draw(seed: int, **knobs) -> str:
        status, _, text = http.post(
            "/v1/completions",
            {"prompt": prompt, "max_tokens": 24, "seed": seed, "ignore_eos": True, **knobs},
        )
        assert status == 200, text
        return json.loads(text)["choices"][0]["text"]

    for knobs in SAMPLING_ARMS:
        texts = [draw(seed, **knobs) for seed in range(4)]
        assert draw(0, **knobs) == texts[0], f"{knobs}: seed 0 drew differently twice"
        assert len(set(texts)) >= 3, f"{knobs}: four seeds drew {len(set(texts))} texts: {texts}"
        for t in texts:
            stripped = t.strip()
            assert len(set(stripped)) > 3, f"{knobs}: a degenerate draw {t!r}"

        # Logprobs ride a different epilogue; the draws must still be plausible
        # under the model's own (raw) distribution.
        status, _, text = http.post(
            "/v1/completions",
            {"prompt": prompt, "max_tokens": 24, "seed": 7, "logprobs": 1, **knobs},
        )
        assert status == 200, text
        lps = json.loads(text)["choices"][0]["logprobs"]["token_logprobs"]
        mean = sum(lps) / len(lps)
        assert mean > -6.0, f"{knobs}: drawn tokens average logprob {mean:.2f}"


def test_completions_prompt_logprobs(http: Http):
    prompt = "The capital of France is"
    status, _, text = http.post(
        "/v1/completions",
        {"prompt": prompt, "max_tokens": 4, "temperature": 0, "echo": True, "logprobs": 2, "prompt_logprobs": 1},
    )
    assert status == 200, text
    choice = json.loads(text)["choices"][0]
    assert choice["text"].startswith(prompt), choice["text"]
    lp = choice["logprobs"]
    # The echoed tokens spell the prompt, and the generated ones follow it.
    assert "".join(lp["tokens"]).startswith(prompt), lp["tokens"]
    assert len(lp["tokens"]) == len(lp["token_logprobs"]) == len(lp["top_logprobs"]) == len(lp["text_offset"])
    assert lp["text_offset"][0] == 0 and lp["text_offset"] == sorted(lp["text_offset"])
    scored = [v for v in lp["token_logprobs"] if v is not None]
    assert len(scored) >= len(lp["tokens"]) - 1 and all(v <= 0.0 for v in scored), lp
    assert all(top is None or len(top) <= 2 for top in lp["top_logprobs"]), lp["top_logprobs"]
    # vLLM's shape: null first, then the prompt token's own entry with its rank.
    rows = choice["prompt_logprobs"]
    assert rows[0] is None and len(rows) >= 2, rows
    for row in rows[1:]:
        assert any(entry["rank"] >= 1 for entry in row.values()), row
    # Scoring is opt-in: without it the shape is as before.
    status, _, text = http.post("/v1/completions", {"prompt": prompt, "max_tokens": 4, "echo": True})
    assert status == 200 and json.loads(text)["choices"][0]["logprobs"] is None, text


def test_completions_streaming(http: Http):
    status, _, resp = http.post(
        "/v1/completions",
        {
            "prompt": "The capital of France is",
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
    )
    assert status == 200
    frames = sse_frames(resp)
    assert frames[-1][1] == "[DONE]"
    chunks = [json.loads(d) for _, d in frames[:-1]]
    assert all(c["object"] == "text_completion" for c in chunks)
    text = "".join(c["choices"][0]["text"] for c in chunks if c["choices"])
    assert "paris" in text.lower(), text
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "length"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 8


def tool_call_markup(city: str) -> str:
    return '<tool_call>\n{"name": "get_weather", "arguments": {"city": "%s"}}\n</tool_call>' % city


TOOL_CALL_SHOTS = f"Q: What is the weather in Rome?\nA: {tool_call_markup('Rome')}\nQ: What is the weather in Paris?\nA:"


def test_completions_verbatim(http: Http):
    for prompt in (
        TOOL_CALL_SHOTS,
        "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n",
    ):
        status, _, text = http.post(
            "/v1/completions",
            {"prompt": prompt, "max_tokens": 24, "temperature": 0, "logprobs": 0},
        )
        assert status == 200, text
        choice = json.loads(text)["choices"][0]
        tokens = choice["logprobs"]["tokens"]
        if choice["finish_reason"] == "stop":
            tokens = tokens[:-1]  # the stop token that ended the turn spells no text
        assert choice["text"] and "".join(tokens) == choice["text"], choice

    status, _, text = http.post(
        "/v1/completions", {"prompt": TOOL_CALL_SHOTS, "max_tokens": 32, "temperature": 0}
    )
    assert status == 200, text
    out = json.loads(text)["choices"][0]["text"]
    assert "<tool_call>" in out and "get_weather" in out and "Paris" in out, out


def test_completions_stop_strings(http: Http):
    base = {"prompt": TOOL_CALL_SHOTS, "max_tokens": 48, "temperature": 0, "stop": ["</tool_call>"]}
    status, _, text = http.post("/v1/completions", base)
    assert status == 200, text
    choice = json.loads(text)["choices"][0]
    assert "<tool_call>" in choice["text"] and "</tool_call>" not in choice["text"], choice
    assert choice["finish_reason"] == "stop" and choice["stop_reason"] == "</tool_call>", choice

    status, _, text = http.post("/v1/completions", {**base, "include_stop_str_in_output": True})
    assert status == 200, text
    included = json.loads(text)["choices"][0]
    assert included["text"] == choice["text"] + "</tool_call>", (included, choice)
    assert included["stop_reason"] == "</tool_call>", included

    status, _, resp = http.post(
        "/v1/completions", {**base, "include_stop_str_in_output": True, "stream": True}, stream=True
    )
    assert status == 200
    chunks = [json.loads(d) for _, d in sse_frames(resp) if d != "[DONE]"]
    assert "".join(c["choices"][0]["text"] for c in chunks) == included["text"], chunks
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop", chunks[-1]
    assert chunks[-1]["choices"][0]["stop_reason"] == "</tool_call>", chunks[-1]

    status, _, text = http.post(
        "/v1/completions",
        {"prompt": "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "max_tokens": 32, "temperature": 0},
    )
    choice = json.loads(text)["choices"][0]
    if choice["finish_reason"] == "stop":
        assert choice["stop_reason"] is None, choice


def test_chat_stop_reason(http: Http):
    body = {
        "messages": [{"role": "user", "content": "Write the alphabet, letters separated by spaces."}],
        "max_tokens": 64,
        "stop": ["F"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    status, _, text = chat(http, **body, include_stop_str_in_output=True)
    assert status == 200, text
    choice = json.loads(text)["choices"][0]
    assert choice["message"]["content"].endswith("F"), choice
    assert choice["finish_reason"] == "stop" and choice["stop_reason"] == "F", choice
    status, _, resp = chat(http, stream=True, **body)
    chunks = [json.loads(d) for _, d in sse_frames(resp) if d != "[DONE]"]
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert content == choice["message"]["content"][:-1], (content, choice)
    assert chunks[-1]["choices"][0]["stop_reason"] == "F", chunks[-1]


def stream_chunks(http: Http, path: str, body: dict) -> list[dict]:
    status, _, resp = http.post(path, {**body, "stream": True}, stream=True)
    assert status == 200, status
    return [json.loads(d) for _, d in sse_frames(resp) if d != "[DONE]"]


def test_completions_token_ids(http: Http):
    body = {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0, "return_token_ids": True}
    status, _, text = http.post(
        "/v1/completions", {**body, "logprobs": 1, "return_tokens_as_token_ids": True}
    )
    assert status == 200, text
    whole = json.loads(text)
    choice = whole["choices"][0]
    prompt_ids, ids = choice["prompt_token_ids"], choice["token_ids"]
    assert prompt_ids and all(isinstance(t, int) for t in prompt_ids + ids), choice
    assert len(ids) == whole["usage"]["completion_tokens"] == 8, whole
    assert len(prompt_ids) == whole["usage"]["prompt_tokens"], whole
    lp = choice["logprobs"]
    assert lp["tokens"] == [f"token_id:{t}" for t in ids], lp["tokens"]
    assert all(key.startswith("token_id:") for top in lp["top_logprobs"] for key in top), lp
    assert lp["text_offset"][-1] < len(choice["text"]), lp

    chunks = stream_chunks(http, "/v1/completions", body)
    streamed = [t for c in chunks if c["choices"] for t in c["choices"][0]["token_ids"]]
    assert streamed == ids, (streamed, ids)
    prompts = [c["choices"][0]["prompt_token_ids"] for c in chunks if c["choices"] and "prompt_token_ids" in c["choices"][0]]
    assert prompts == [prompt_ids], prompts

    status, _, text = http.post("/v1/completions", {k: v for k, v in body.items() if k != "return_token_ids"})
    assert "token_ids" not in json.loads(text)["choices"][0], text

    stop = ids[2]
    at = ids.index(stop)
    for include in (False, True):
        status, _, text = http.post(
            "/v1/completions",
            {**body, "stop_token_ids": [stop], "include_stop_str_in_output": include, "logprobs": 0},
        )
        assert status == 200, text
        stopped = json.loads(text)["choices"][0]
        assert stopped["token_ids"] == ids[: at + 1], (stopped, ids)
        assert stopped["finish_reason"] == "stop" and stopped["stop_reason"] == stop, stopped
        spelled = stopped["logprobs"]["tokens"][: at + 1 if include else at]
        assert stopped["text"] == "".join(spelled), stopped


def test_chat_token_ids(http: Http):
    body = {
        "model": "pie",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
        "temperature": 0,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    status, _, text = http.post("/v1/chat/completions", {**body, "logprobs": True, "return_tokens_as_token_ids": True})
    assert status == 200, text
    whole = json.loads(text)
    ids = whole["choices"][0]["token_ids"]
    assert len(ids) == whole["usage"]["completion_tokens"], whole
    assert len(whole["prompt_token_ids"]) == whole["usage"]["prompt_tokens"], whole
    entries = whole["choices"][0]["logprobs"]["content"]
    assert [e["token"] for e in entries] == [f"token_id:{t}" for t in ids], entries

    chunks = stream_chunks(http, "/v1/chat/completions", body)
    assert [t for c in chunks if c["choices"] for t in c["choices"][0]["token_ids"]] == ids, chunks
    assert [c["prompt_token_ids"] for c in chunks if "prompt_token_ids" in c] == [whole["prompt_token_ids"]], chunks


def test_logprobs_mode(http: Http):
    body = {
        "prompt": "Once upon a time",
        "max_tokens": 12,
        "seed": 3,
        "logprobs": 3,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }

    def draw(**knobs):
        status, _, text = http.post("/v1/completions", {**body, **knobs})
        assert status == 200, text
        choice = json.loads(text)["choices"][0]
        return choice["token_ids"], choice["logprobs"]["top_logprobs"]

    def gaps(tops):
        return [{(a, b): row[a] - row[b] for a in row for b in row if a < b} for row in tops]

    def close(x, y):
        return abs(x - y) <= 1e-2 + 1e-2 * abs(y)

    t = 0.5
    draw(temperature=t)
    ids, raw = draw(temperature=t)
    for mode, scale in [("processed_logprobs", 1 / t), ("raw_logits", 1.0), ("processed_logits", 1 / t)]:
        drawn, tops = draw(temperature=t, logprobs_mode=mode)
        assert drawn == ids, f"{mode} changed the draw"
        for got, want in zip(gaps(tops), gaps(raw)):
            assert got.keys() == want.keys(), (mode, got, want)
            assert all(close(got[k], want[k] * scale) for k in got), (mode, got, want)

    _, tops = draw(temperature=1.0, top_k=2, logprobs_mode="processed_logprobs")
    for row in tops:
        values = sorted(row.values(), reverse=True)
        assert abs(math.exp(values[0]) + math.exp(values[1]) - 1.0) < 1e-3, row
        assert values[2] == -9999.0, row

    _, greedy = draw(temperature=0)
    _, processed = draw(temperature=0, logprobs_mode="processed_logprobs")
    assert all(close(p[k], g[k]) for g, p in zip(greedy, processed) for k in g), (greedy, processed)

    for path, extra in [("/v1/completions", {"prompt": "Hi"}), ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}]})]:
        status, _, text = http.post(path, {**extra, "logprobs_mode": "processed"})
        assert status == 400 and json.loads(text)["error"]["param"] == "logprobs_mode", text

    chat_body = {
        "model": "pie",
        "messages": [{"role": "user", "content": "Tell me a story."}],
        "max_tokens": 8,
        "temperature": t,
        "seed": 3,
        "logprobs": True,
        "top_logprobs": 2,
        "return_tokens_as_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    rows = {}
    for mode in ("raw_logprobs", "raw_logprobs", "processed_logprobs"):
        status, _, text = http.post("/v1/chat/completions", {**chat_body, "logprobs_mode": mode})
        assert status == 200, text
        rows[mode] = [{e["token"]: e["logprob"] for e in entry["top_logprobs"]} for entry in json.loads(text)["choices"][0]["logprobs"]["content"]]
    for got, want in zip(gaps(rows["processed_logprobs"]), gaps(rows["raw_logprobs"])):
        assert got.keys() == want.keys() and all(close(got[k], want[k] / t) for k in got), (got, want)


def test_official_openai_client_vllm_fields(http: Http):
    client = official("openai").OpenAI(base_url=http.base_url + "/v1", api_key="unused")
    extra = {"include_stop_str_in_output": True, "return_token_ids": True}
    completion = client.completions.create(
        model="pie", prompt=TOOL_CALL_SHOTS, max_tokens=48, temperature=0, stop=["</tool_call>"], extra_body=extra
    )
    choice = completion.choices[0]
    assert choice.text.lstrip().startswith("<tool_call>") and choice.text.endswith("</tool_call>"), choice
    assert choice.finish_reason == "stop" and choice.model_extra["stop_reason"] == "</tool_call>", choice
    ids = choice.model_extra["token_ids"]
    assert len(ids) == completion.usage.completion_tokens, completion

    pieces, streamed = [], []
    for chunk in client.completions.create(
        model="pie", prompt=TOOL_CALL_SHOTS, max_tokens=48, temperature=0, stop=["</tool_call>"], extra_body=extra, stream=True
    ):
        pieces.append(chunk.choices[0].text)
        streamed.extend(chunk.choices[0].model_extra.get("token_ids") or [])
    assert "".join(pieces) == choice.text and streamed == ids, (pieces, streamed, ids)

    calls, streamed = [], []
    for chunk in client.chat.completions.create(
        model="pie",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[WEATHER_TOOL],
        max_tokens=128,
        temperature=0,
        stream=True,
        extra_body={"return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False}},
    ):
        if chunk.choices:
            calls.extend(chunk.choices[0].delta.tool_calls or [])
            streamed.extend(chunk.choices[0].model_extra.get("token_ids") or [])
    assert calls and calls[0].function.name == "get_weather", calls
    assert "city" in json.loads(calls[0].function.arguments), calls
    assert streamed, "the streamed tool call carried no token ids"


def test_tool_markup_is_content_without_callable_tools(http: Http):
    shots = [
        {"role": "user", "content": "What is the weather in Rome?"},
        {"role": "assistant", "content": tool_call_markup("Rome")},
        {"role": "user", "content": "What is the weather in Paris?"},
    ]
    status, _, text = chat(
        http,
        messages=shots,
        tools=[WEATHER_TOOL],
        tool_choice="none",
        max_tokens=64,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert status == 200, text
    choice = json.loads(text)["choices"][0]
    assert choice["finish_reason"] != "tool_calls" and not choice["message"].get("tool_calls"), choice
    assert choice["message"]["content"], choice


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------

RESPONSES_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def responses(http: Http, *, stream: bool = False, **body):
    body.setdefault("model", "pie")
    body.setdefault("max_output_tokens", 32)
    body.setdefault("temperature", 0)
    body.setdefault("reasoning", {"effort": "none"})
    if stream:
        body["stream"] = True
    return http.post("/v1/responses", body, stream=stream)


def test_responses_non_streaming(http: Http):
    status, _, text = responses(http, input="Say hello in one word.", instructions="You are terse.")
    assert status == 200, text
    body = json.loads(text)
    assert body["object"] == "response" and body["id"].startswith("resp_")
    assert body["status"] == "completed", body
    message = [o for o in body["output"] if o["type"] == "message"][0]
    assert message["content"][0]["type"] == "output_text" and message["content"][0]["text"]
    assert body["usage"]["total_tokens"] == body["usage"]["input_tokens"] + body["usage"]["output_tokens"]

    status, _, text = responses(http, input="Count to one hundred in words.", max_output_tokens=4)
    body = json.loads(text)
    assert body["status"] == "incomplete" and body["incomplete_details"]["reason"] == "max_output_tokens", body

    status, _, text = responses(http, input="hi", previous_response_id="resp_x")
    assert status == 400 and json.loads(text)["error"]["param"] == "previous_response_id", text


def test_responses_streaming(http: Http):
    status, _, resp = responses(http, stream=True, input="Say hello in one word.")
    assert status == 200
    frames = sse_frames(resp)
    names = [e for e, _ in frames]
    assert names[0] == "response.created" and names[-1] == "response.completed", names
    assert "response.output_item.added" in names and "response.output_text.delta" in names
    seqs = [json.loads(d)["sequence_number"] for _, d in frames]
    assert seqs == list(range(len(seqs))), seqs
    for event, data in frames:
        assert json.loads(data)["type"] == event
    text = "".join(json.loads(d)["delta"] for e, d in frames if e == "response.output_text.delta")
    assert text.strip(), frames
    final = json.loads(frames[-1][1])["response"]
    assert final["status"] == "completed" and final["output"][0]["content"][0]["text"] == text, final


def test_responses_function_call(http: Http):
    status, _, text = responses(
        http,
        input="What is the weather in Paris right now?",
        tools=[RESPONSES_TOOL],
        tool_choice="required",
        max_output_tokens=64,
    )
    assert status == 200, text
    body = json.loads(text)
    calls = [o for o in body["output"] if o["type"] == "function_call"]
    assert calls and calls[0]["name"] == "get_weather", body
    assert calls[0]["call_id"].startswith("call_") and "city" in json.loads(calls[0]["arguments"])

    status, _, text = responses(
        http,
        input=[
            {"role": "user", "content": "What is the weather in Paris right now?"},
            calls[0],
            {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": "21C and sunny"},
        ],
        tools=[RESPONSES_TOOL],
        max_output_tokens=64,
    )
    assert status == 200, text
    body = json.loads(text)
    message = [o for o in body["output"] if o["type"] == "message"]
    assert message and message[0]["content"][0]["text"], body


def test_official_openai_responses_client(http: Http):
    client = official("openai").OpenAI(base_url=http.base_url + "/v1", api_key="unused")
    response = client.responses.create(
        model="pie",
        input="Say hello in one word.",
        max_output_tokens=32,
        temperature=0,
        reasoning={"effort": "none"},
    )
    assert response.output_text.strip(), response
    pieces = []
    with client.responses.stream(
        model="pie",
        input="Say hello in one word.",
        max_output_tokens=32,
        temperature=0,
        reasoning={"effort": "none"},
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                pieces.append(event.delta)
        final = stream.get_final_response()
    assert "".join(pieces).strip() and final.status == "completed"
    completion = client.chat.completions.create(
        model="pie",
        messages=[{"role": "user", "content": "The capital of France is"}],
        max_tokens=8,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    assert completion.choices[0].message.content
    legacy = client.completions.create(model="pie", prompt="The capital of France is", max_tokens=8, temperature=0)
    assert "paris" in legacy.choices[0].text.lower()


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

GEMINI_TOOL = {
    "functionDeclarations": [
        {
            "name": "get_weather",
            "description": "Get the current weather in a city",
            "parameters": {"type": "OBJECT", "properties": {"city": {"type": "STRING"}}, "required": ["city"]},
        }
    ]
}


def gemini(http: Http, method: str = "generateContent", query: str = "", **body):
    body.setdefault("generationConfig", {})
    body["generationConfig"].setdefault("maxOutputTokens", 32)
    body["generationConfig"].setdefault("temperature", 0)
    body["generationConfig"].setdefault("thinkingConfig", {"thinkingBudget": 0})
    return http.post(f"/v1beta/models/pie:{method}{query}", body, stream=method == "streamGenerateContent")


def test_gemini_generate(http: Http):
    status, _, text = gemini(
        http,
        systemInstruction={"parts": [{"text": "You are terse."}]},
        contents=[{"role": "user", "parts": [{"text": "Say hello in one word."}]}],
    )
    assert status == 200, text
    body = json.loads(text)
    candidate = body["candidates"][0]
    assert candidate["content"]["role"] == "model" and candidate["content"]["parts"][0]["text"]
    assert candidate["finishReason"] in ("STOP", "MAX_TOKENS"), body
    assert body["usageMetadata"]["totalTokenCount"] > 0 and body["modelVersion"] == "pie"

    status, _, text = gemini(http, contents=[{"parts": [{"inlineData": {"mimeType": "image/png", "data": ""}}]}])
    assert status == 400, text
    err = json.loads(text)["error"]
    assert err["code"] == 400 and err["status"] == "INVALID_ARGUMENT", err


def test_gemini_stream(http: Http):
    status, headers, resp = gemini(
        http,
        "streamGenerateContent",
        "?alt=sse",
        contents=[{"parts": [{"text": "Say hello in one word."}]}],
    )
    assert status == 200
    frames = sse_frames(resp)
    chunks = [json.loads(d) for _, d in frames]
    text = "".join(p.get("text", "") for c in chunks for p in c["candidates"][0]["content"]["parts"])
    assert text.strip(), chunks
    assert chunks[-1]["candidates"][0]["finishReason"] in ("STOP", "MAX_TOKENS"), chunks[-1]
    assert chunks[-1]["usageMetadata"]["candidatesTokenCount"] > 0


def test_gemini_function_call(http: Http):
    status, _, text = gemini(
        http,
        contents=[{"role": "user", "parts": [{"text": "What is the weather in Paris right now?"}]}],
        tools=[GEMINI_TOOL],
        toolConfig={"functionCallingConfig": {"mode": "ANY"}},
        generationConfig={"maxOutputTokens": 64},
    )
    assert status == 200, text
    parts = json.loads(text)["candidates"][0]["content"]["parts"]
    calls = [p["functionCall"] for p in parts if "functionCall" in p]
    assert calls and calls[0]["name"] == "get_weather" and "city" in calls[0]["args"], parts

    status, _, text = gemini(
        http,
        contents=[
            {"role": "user", "parts": [{"text": "What is the weather in Paris right now?"}]},
            {"role": "model", "parts": [{"functionCall": calls[0]}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"temperature_c": 21}}}]},
        ],
        tools=[GEMINI_TOOL],
        generationConfig={"maxOutputTokens": 64},
    )
    assert status == 200, text
    parts = json.loads(text)["candidates"][0]["content"]["parts"]
    assert any(p.get("text") for p in parts), parts


def test_gemini_json_schema(http: Http):
    status, _, text = gemini(
        http,
        contents=[{"parts": [{"text": "Give the city of Paris and its population as JSON."}]}],
        generationConfig={
            "maxOutputTokens": 64,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {"city": {"type": "STRING"}, "population": {"type": "INTEGER"}},
                "required": ["city", "population"],
            },
        },
    )
    assert status == 200, text
    out = json.loads(text)["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(out)
    assert isinstance(parsed["city"], str) and isinstance(parsed["population"], int), parsed


def test_official_google_genai_client(http: Http):
    genai = official("google.genai")
    types = importlib.import_module("google.genai.types")
    client = genai.Client(api_key="unused", http_options=types.HttpOptions(base_url=http.base_url, api_version="v1beta"))
    config = types.GenerateContentConfig(
        max_output_tokens=32, temperature=0, thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
    response = client.models.generate_content(model="pie", contents="Say hello in one word.", config=config)
    assert response.text and response.text.strip(), response
    pieces = [c.text or "" for c in client.models.generate_content_stream(model="pie", contents="Say hello in one word.", config=config)]
    assert "".join(pieces).strip(), pieces


TESTS = [
    test_models,
    test_non_streaming,
    test_length_finish,
    test_streaming,
    test_reasoning_content,
    test_stop_sequence,
    test_forced_tool_call,
    test_named_tool_call_streaming,
    test_tool_result_round_trip,
    test_json_schema,
    test_json_object,
    test_structural_tag,
    test_bad_requests,
    test_official_openai_client,
    test_anthropic_non_streaming,
    test_anthropic_streaming,
    test_anthropic_tool_use,
    test_anthropic_bad_requests,
    test_official_anthropic_client,
    test_completions,
    test_sampling_arms,
    test_completions_prompt_logprobs,
    test_completions_streaming,
    test_completions_verbatim,
    test_completions_stop_strings,
    test_chat_stop_reason,
    test_completions_token_ids,
    test_chat_token_ids,
    test_logprobs_mode,
    test_official_openai_client_vllm_fields,
    test_tool_markup_is_content_without_callable_tools,
    test_responses_non_streaming,
    test_responses_streaming,
    test_responses_function_call,
    test_official_openai_responses_client,
    test_gemini_generate,
    test_gemini_stream,
    test_gemini_function_call,
    test_gemini_json_schema,
    test_official_google_genai_client,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8080")
    parser.add_argument("-k", default=None, help="only tests whose name contains this")
    args = parser.parse_args()
    http = Http(args.base_url)
    passed = total = 0
    for test in TESTS:
        name = test.__name__.removeprefix("test_").replace("_", "-")
        if args.k and args.k not in name:
            continue
        print(f"🔄 {name:32s} ", end="", flush=True)
        start = time.time()
        try:
            test(http)
            print(f"✅ ({time.time() - start:.1f}s)")
            passed += 1
            total += 1
        except Skip as e:
            print(f"⏭️  ({time.time() - start:.1f}s) SKIPPED: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"❌ ({time.time() - start:.1f}s)")
            print(f"   {str(e)[:400]}")
            total += 1
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
