"""Grammar-constrained JSON generation — Python inferlet example.

Demonstrates :meth:`Generator.collect_json` with a JSON Schema string:
the SDK compiles the schema into a stateful matcher and drives it per
generated token, so the model's output is parseable JSON conforming to
the schema by construction.

The inferlet returns the parsed dict directly — ``session.send`` and the
``main`` return value both auto-serialize structured values, so no
manual ``json.dumps`` is needed.

Note: pydantic v2 (and any package shipping a native Rust/C extension)
does not load inside the WASM runtime today — ``componentize-py`` can
bundle pure-Python packages but not native extensions. Use a JSON
Schema string + the ``parse=`` hook in :meth:`collect_json` if you need
typed output via a pure-Python validator.
"""

import json

from inferlet import Context, JsonSchema, Model, Sampler, runtime, session


PERSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "email": {"type": "string"},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
        "required": ["name", "age", "email", "skills"],
    }
)

SYSTEM_PROMPT = (
    "You are a helpful assistant that generates structured data. "
    "Output ONLY a raw JSON object — no markdown, no explanation."
)
DEFAULT_PROMPT = "Generate a profile for a fictional software engineer named Alice."


async def main(input: dict):
    prompt = input.get("prompt", DEFAULT_PROMPT)
    max_tokens = int(input.get("max_tokens", 512))

    model = Model.load(runtime.models()[0])
    ctx = Context(model)
    ctx.system(SYSTEM_PROMPT).user(prompt)

    person = await ctx.generate(
        Sampler.argmax(),
        constrain=JsonSchema(PERSON_SCHEMA),
        max_tokens=max_tokens,
    ).collect_json(schema=PERSON_SCHEMA)

    session.send(f"Hello {person['name']}, age {person['age']}.\n")
    session.send(f"Skills: {', '.join(person['skills'])}\n")
    session.send("[done]")
    return person
