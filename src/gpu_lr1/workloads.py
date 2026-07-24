from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NamedSchema:
    name: str
    schema: dict[str, Any]
    family: str


def benchmark_schemas(count: int = 16) -> list[NamedSchema]:
    if count <= 0:
        raise ValueError("count must be positive")
    builders = [
        _small_tool_schema,
        _nested_tool_schema,
        _enum_heavy_schema,
        _array_heavy_schema,
        _string_heavy_schema,
        _wide_object_schema,
        _deep_object_schema,
    ]
    schemas = []
    for index in range(count):
        builder = builders[index % len(builders)]
        schemas.append(builder(index))
    return schemas


def _small_tool_schema(index: int) -> NamedSchema:
    schema = {
        "type": "object",
        "properties": {
            f"enabled_{index}": {"type": "boolean"},
            f"id_{index}": {
                "type": "integer",
                "minimum": 0,
                "maximum": 31 + index,
            },
            f"mode_{index}": {
                "enum": ["fast", "safe", "balanced", f"custom-{index}"]
            },
        },
        "required": [f"enabled_{index}", f"id_{index}", f"mode_{index}"],
        "additionalProperties": False,
    }
    return NamedSchema(f"small-tool-{index}", schema, "small")


def _nested_tool_schema(index: int) -> NamedSchema:
    schema = {
        "type": "object",
        "properties": {
            f"request_{index}": {
                "type": "object",
                "properties": {
                    "action": {
                        "enum": ["create", "update", "delete", "inspect"]
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "dry_run": {"type": "boolean"},
                            "priority": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 9,
                            },
                            "region": {
                                "enum": ["us-east", "us-west", "eu", "apac"]
                            },
                        },
                        "required": ["dry_run", "priority", "region"],
                        "additionalProperties": False,
                    },
                },
                "required": ["action", "metadata"],
                "additionalProperties": False,
            },
            "version": {"enum": ["v1", "v2", "v3"]},
        },
        "required": [f"request_{index}", "version"],
        "additionalProperties": False,
    }
    return NamedSchema(f"nested-tool-{index}", schema, "nested")


def _enum_heavy_schema(index: int) -> NamedSchema:
    properties = {}
    for property_index in range(8 + index % 5):
        properties[f"choice_{index}_{property_index}"] = {
            "enum": [
                f"value-{property_index}-{choice}"
                for choice in range(4 + (property_index % 4))
            ]
        }
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return NamedSchema(f"enum-heavy-{index}", schema, "enum")


def _array_heavy_schema(index: int) -> NamedSchema:
    schema = {
        "type": "object",
        "properties": {
            f"batches_{index}": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["read", "write", "sync"]},
                        "ok": {"type": "boolean"},
                        "shard": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 15,
                        },
                    },
                    "required": ["kind", "ok", "shard"],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": 2 + index % 3,
            }
        },
        "required": [f"batches_{index}"],
        "additionalProperties": False,
    }
    return NamedSchema(f"array-heavy-{index}", schema, "array")


def _string_heavy_schema(index: int) -> NamedSchema:
    schema = {
        "type": "object",
        "properties": {
            f"description_{index}": {
                "type": "string",
                "minLength": index % 3,
            },
            "label": {"type": "string", "minLength": 1, "maxLength": 8},
            "status": {"enum": ["new", "running", "done", "failed"]},
        },
        "required": [f"description_{index}", "label", "status"],
        "additionalProperties": False,
    }
    return NamedSchema(f"string-heavy-{index}", schema, "string")


def _wide_object_schema(index: int) -> NamedSchema:
    width = 16 + (index % 4) * 4
    properties = {}
    for property_index in range(width):
        name = f"field_{index}_{property_index:02d}"
        if property_index % 3 == 0:
            properties[name] = {"type": "boolean"}
        elif property_index % 3 == 1:
            properties[name] = {
                "type": "integer",
                "minimum": 0,
                "maximum": 7,
            }
        else:
            properties[name] = {"enum": ["alpha", "beta", "gamma"]}
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return NamedSchema(f"wide-object-{index}", schema, "wide")


def _deep_object_schema(index: int) -> NamedSchema:
    node: dict[str, Any] = {"enum": ["leaf-a", "leaf-b", "leaf-c"]}
    depth = 5 + index % 4
    for level in reversed(range(depth)):
        property_name = f"level_{index}_{level}"
        node = {
            "type": "object",
            "properties": {
                property_name: node,
                f"valid_{level}": {"type": "boolean"},
            },
            "required": [property_name, f"valid_{level}"],
            "additionalProperties": False,
        }
    return NamedSchema(f"deep-object-{index}", node, "deep")

