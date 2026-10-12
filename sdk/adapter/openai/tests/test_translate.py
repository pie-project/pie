import pytest

from pie_openai_adapter.translate import Defaults, RequestError, to_inferlet_input


def test_single_user_message_becomes_prompt_with_default_system():
    out = to_inferlet_input({"messages": [{"role": "user", "content": "hi"}]})
    assert out == {
        "prompt": "hi",
        "system": Defaults().system,
        "max_tokens": 256,
        "temperature": 0.6,
        "top_p": 0.95,
    }


def test_system_messages_are_joined_and_history_is_folded_into_prompt():
    body = {
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "system", "content": "be kind"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
    }
    out = to_inferlet_input(body)
    assert out["system"] == "be brief\nbe kind"
    assert out["prompt"] == "user: one\nassistant: two\nuser: three"


def test_content_parts_are_flattened():
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}
    assert to_inferlet_input(body)["prompt"] == "ab"


def test_sampling_parameters_override_defaults():
    body = {
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 10,
        "temperature": 0,
        "top_p": 0.5,
    }
    out = to_inferlet_input(body)
    assert (out["max_tokens"], out["temperature"], out["top_p"]) == (10, 0.0, 0.5)


@pytest.mark.parametrize(
    "body, message",
    [
        ("not an object", "JSON object"),
        ({}, "non-empty"),
        ({"messages": []}, "non-empty"),
        ({"messages": [{"role": "tool", "content": "x"}]}, "unsupported message role"),
        ({"messages": [{"role": "assistant", "content": "x"}]}, "last message must be from the user"),
        ({"messages": [{"role": "user", "content": "x"}], "tools": [{}]}, "tool calling"),
        ({"messages": [{"role": "user", "content": "x"}], "n": 2}, "only n=1"),
        ({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, "max_tokens"),
        ({"messages": [{"role": "user", "content": "x"}], "top_p": 0}, "top_p"),
        ({"messages": [{"role": "user", "content": "x"}], "temperature": -1}, "temperature"),
        (
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]},
            "text content parts",
        ),
    ],
)
def test_invalid_requests_are_rejected(body, message):
    with pytest.raises(RequestError, match=message):
        to_inferlet_input(body)
