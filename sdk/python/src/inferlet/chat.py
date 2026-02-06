"""
Chat formatting using Jinja2 templates.
Mirrors the Rust ChatFormatter from inferlet/src/chat.rs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

try:
    from jinja2 import Environment, BaseLoader

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

if TYPE_CHECKING:
    from .model import Model


@dataclass
class ToolCall:
    """Represents a single tool call."""

    name: str
    arguments: Any


@dataclass
class Message:
    """Internal message representation."""

    role: str
    content: str
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChatFormatter:
    """
    Formats chat messages using the model's Jinja2 template.

    Provides a stateful API for building conversations message by message.
    This mirrors the Rust ChatFormatter API.
    """

    def __init__(self, model: "Model | None" = None) -> None:
        """
        Initialize the ChatFormatter.

        Args:
            model: Optional model to get the prompt template from.
                   If None, fallback formatting will be used.
        """
        self._messages: list[Message] = []
        self._template_str = model.prompt_template if model else ""
        self._template: Any = None

        if HAS_JINJA2 and self._template_str:
            env = Environment(loader=BaseLoader())
            env.globals["raise_exception"] = self._raise_exception
            try:
                self._template = env.from_string(self._template_str)
            except Exception:
                self._template = None

    @staticmethod
    def _raise_exception(msg: str) -> None:
        """Helper for templates to raise exceptions."""
        raise ValueError(msg)

    def system(self, content: str) -> None:
        """
        Add a system message to the conversation.

        Args:
            content: The system message content
        """
        self._messages.append(Message(role="system", content=content))

    def user(self, content: str) -> None:
        """
        Add a user message to the conversation.

        Args:
            content: The user message content
        """
        self._messages.append(Message(role="user", content=content))

    def assistant(self, content: str) -> None:
        """
        Add a simple assistant message with only text content.

        Args:
            content: The assistant message content
        """
        self.assistant_response(content, None, None)

    def assistant_response(
        self,
        content: str,
        reasoning: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        """
        Add an assistant message with optional reasoning and tool calls.

        Args:
            content: The main content of the assistant message
            reasoning: Optional reasoning content (for chain-of-thought)
            tool_calls: Optional list of tool calls
        """
        self._messages.append(
            Message(
                role="assistant",
                content=content,
                reasoning_content=reasoning,
                tool_calls=tool_calls,
            )
        )

    def tool(self, content: str) -> None:
        """
        Add a tool response message to the conversation.

        Args:
            content: The tool response content
        """
        self._messages.append(Message(role="tool", content=content))

    def has_messages(self) -> bool:
        """
        Check if there are any messages in the conversation.

        Returns:
            True if there are messages, False otherwise
        """
        return len(self._messages) > 0

    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self._messages.clear()

    def render(
        self,
        template: str | None = None,
        add_generation_prompt: bool = False,
        begin_of_sequence: bool = True,
    ) -> str:
        """
        Render the conversation into a single formatted string prompt.

        Args:
            template: Optional override template string. Uses the model's
                      template if None.
            add_generation_prompt: Whether to append the assistant prompt prefix
            begin_of_sequence: Whether to include the begin-of-sequence token

        Returns:
            Formatted prompt string
        """
        # Use override template or fall back to model template
        template_str = template or self._template_str
        active_template = self._template

        # If using override template, compile it
        if template is not None and HAS_JINJA2:
            try:
                env = Environment(loader=BaseLoader())
                env.globals["raise_exception"] = self._raise_exception
                active_template = env.from_string(template)
            except Exception:
                active_template = None

        # Convert messages to dict format for template
        messages_dict = self._messages_to_dict()

        if active_template is not None:
            try:
                return active_template.render(
                    messages=messages_dict,
                    add_generation_prompt=add_generation_prompt,
                    begin_of_sequence=begin_of_sequence,
                    bos_token="",
                    eos_token="",
                )
            except Exception:
                pass

        # Fallback formatting
        return self._format_fallback(messages_dict, add_generation_prompt)

    def _messages_to_dict(self) -> list[dict[str, Any]]:
        """Convert internal Message objects to dict format for templates."""
        messages_dict = []
        for msg in self._messages:
            d: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.reasoning_content is not None:
                d["reasoning_content"] = msg.reasoning_content
            if msg.tool_calls is not None:
                d["tool_calls"] = [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in msg.tool_calls
                ]
            messages_dict.append(d)
        return messages_dict

    def _format_fallback(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        """
        Fallback formatting without Jinja2.

        Detects template style and formats accordingly:
        - ChatML style: Uses <|im_start|>, <|im_end|> (Qwen, Yi, etc.)
        - Llama 3 style: Uses <|start_header_id|>, <|end_header_id|>, <|eot_id|>
        - Generic style: Uses [Role]: format
        """
        # Detect ChatML style template (Qwen, Yi, etc.)
        if self._template_str and "<|im_start|>" in self._template_str:
            return self._format_chatml(messages, add_generation_prompt)

        # Detect Llama 3 style template
        if self._template_str and "<|start_header_id|>" in self._template_str:
            return self._format_llama3(messages, add_generation_prompt)

        # Generic fallback
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"[{role.title()}]: {content}")

        result = "\n".join(parts)
        if add_generation_prompt:
            result += "\n[Assistant]:"
        return result

    def _format_chatml(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        """
        Format messages in ChatML style (Qwen, Yi, etc.).

        Format:
        <|im_start|>system
        {content}<|im_end|>
        <|im_start|>user
        {content}<|im_end|>
        <|im_start|>assistant
        """
        parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map tool role to user with wrapper (ChatML convention)
            if role == "tool":
                parts.append(
                    f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
                )
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        result = "".join(parts)

        if add_generation_prompt:
            result += "<|im_start|>assistant\n"

        return result

    def _format_llama3(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        """
        Format messages in Llama 3 chat style.

        Format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {content}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """
        parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map role names (tool -> ipython for Llama 3)
            header = "ipython" if role == "tool" else role

            parts.append(
                f"<|start_header_id|>{header}<|end_header_id|>\n\n{content}<|eot_id|>"
            )

        result = "".join(parts)

        if add_generation_prompt:
            result += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        return result

    def format(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = False,
        begin_of_sequence: bool = True,
    ) -> str:
        """
        Format a list of messages directly without modifying internal state.

        This is a stateless API for backward compatibility with code that
        passes message dicts directly.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            add_generation_prompt: Whether to append the assistant prompt prefix
            begin_of_sequence: Whether to include the begin-of-sequence token

        Returns:
            Formatted prompt string
        """
        if self._template is not None:
            try:
                return self._template.render(
                    messages=messages,
                    add_generation_prompt=add_generation_prompt,
                    begin_of_sequence=begin_of_sequence,
                    bos_token="",
                    eos_token="",
                )
            except Exception:
                pass

        # Fallback formatting
        return self._format_fallback(messages, add_generation_prompt)


def format_messages(
    messages: list[tuple[str, str]],
) -> list[dict[str, str]]:
    """
    Convert tuple-style messages to dict format.

    This is a backwards-compatible helper function.

    Args:
        messages: List of (role, content) tuples

    Returns:
        List of {'role': role, 'content': content} dicts
    """
    return [{"role": role, "content": content} for role, content in messages]
