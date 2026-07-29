from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gpu_lr1.automata import ByteDFA, Fragment, NFABuilder, determinize


class UnsupportedSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class CompilerOptions:
    unbounded_array_cap: int = 4
    max_integer_choices: int = 128
    include_optional_properties: bool = True


class CanonicalJSONSchemaCompiler:
    """Compile a useful JSON Schema subset into a canonical-output byte DFA.

    The accepted language is deliberately a subset of the schema language:
    objects use sorted property order, all declared properties are emitted by
    default, whitespace is omitted, and strings are ASCII JSON strings. Every
    accepted document still validates against the supported input schema.
    """

    def __init__(
        self,
        schema: Mapping[str, Any],
        options: CompilerOptions | None = None,
    ) -> None:
        self.schema = schema
        self.options = options or CompilerOptions()
        self.builder = NFABuilder()

    def compile(self) -> ByteDFA:
        fragment = self._compile_schema(self.schema, ())
        return determinize(self.builder, fragment)

    def _compile_schema(
        self,
        schema: Mapping[str, Any] | bool,
        ref_stack: tuple[str, ...],
    ) -> Fragment:
        if schema is False:
            return self.builder.alternate([])
        if schema is True:
            raise UnsupportedSchemaError(
                "the unconstrained true schema requires an unbounded JSON parser"
            )
        if not isinstance(schema, Mapping):
            raise UnsupportedSchemaError(f"schema must be a mapping, got {type(schema)}")

        unsupported_assertions = {
            "allOf",
            "contains",
            "dependentRequired",
            "dependentSchemas",
            "if",
            "maxContains",
            "minContains",
            "not",
            "oneOf",
            "patternProperties",
            "propertyNames",
            "then",
            "uniqueItems",
            "unevaluatedProperties",
        }
        unsupported = sorted(unsupported_assertions.intersection(schema))
        if unsupported:
            raise UnsupportedSchemaError(
                "unsupported assertion keywords: " + ", ".join(unsupported)
            )

        if "$ref" in schema:
            ref = schema["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#/"):
                raise UnsupportedSchemaError("only local JSON Pointer refs are supported")
            if ref in ref_stack:
                raise UnsupportedSchemaError("recursive $ref requires the PDA backend")
            return self._compile_schema(self._resolve_ref(ref), ref_stack + (ref,))

        if "const" in schema:
            self._reject_finite_value_siblings(schema, "const")
            value = schema["const"]
            if not self._matches_declared_type(value, schema.get("type")):
                return self.builder.alternate([])
            return self._literal_value(value)
        if "enum" in schema:
            self._reject_finite_value_siblings(schema, "enum")
            values = schema["enum"]
            if not isinstance(values, list) or not values:
                raise UnsupportedSchemaError("enum must contain at least one value")
            values = [
                value
                for value in values
                if self._matches_declared_type(value, schema.get("type"))
            ]
            return self.builder.alternate(self._literal_value(value) for value in values)

        if "anyOf" in schema:
            self._reject_any_of_siblings(schema)
            choices = schema["anyOf"]
            if not isinstance(choices, list) or not choices:
                raise UnsupportedSchemaError("anyOf must contain choices")
            return self.builder.alternate(
                self._compile_schema(choice, ref_stack) for choice in choices
            )

        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            return self.builder.alternate(
                self._compile_schema({**schema, "type": choice}, ref_stack)
                for choice in schema_type
            )
        if schema_type is None:
            if "properties" in schema:
                schema_type = "object"
            elif "items" in schema or "prefixItems" in schema:
                schema_type = "array"
            else:
                raise UnsupportedSchemaError("schema type cannot be inferred")

        if schema_type == "object":
            return self._compile_object(schema, ref_stack)
        if schema_type == "array":
            return self._compile_array(schema, ref_stack)
        if schema_type == "string":
            return self._compile_string(schema)
        if schema_type == "integer":
            return self._compile_integer(schema)
        if schema_type == "number":
            return self._compile_number(schema)
        if schema_type == "boolean":
            return self.builder.alternate(
                [self.builder.literal(b"true"), self.builder.literal(b"false")]
            )
        if schema_type == "null":
            return self.builder.literal(b"null")
        raise UnsupportedSchemaError(f"unsupported schema type: {schema_type!r}")

    def _reject_finite_value_siblings(
        self,
        schema: Mapping[str, Any],
        keyword: str,
    ) -> None:
        allowed = {
            "$anchor",
            "$comment",
            "$defs",
            "$id",
            "$schema",
            "default",
            "deprecated",
            "description",
            "examples",
            "readOnly",
            "title",
            "type",
            "writeOnly",
            keyword,
        }
        unsupported = sorted(set(schema) - allowed)
        if unsupported:
            raise UnsupportedSchemaError(
                f"{keyword} with sibling assertions requires normalization: "
                + ", ".join(unsupported)
            )

    def _reject_any_of_siblings(self, schema: Mapping[str, Any]) -> None:
        allowed = {
            "$anchor",
            "$comment",
            "$defs",
            "$id",
            "$schema",
            "anyOf",
            "default",
            "deprecated",
            "description",
            "examples",
            "readOnly",
            "title",
            "writeOnly",
        }
        unsupported = sorted(set(schema) - allowed)
        if unsupported:
            raise UnsupportedSchemaError(
                "anyOf with sibling assertions requires normalization: "
                + ", ".join(unsupported)
            )

    def _matches_declared_type(self, value: Any, declared_type: Any) -> bool:
        if declared_type is None:
            return True
        choices = declared_type if isinstance(declared_type, list) else [declared_type]
        for choice in choices:
            if choice == "null" and value is None:
                return True
            if choice == "boolean" and isinstance(value, bool):
                return True
            if (
                choice == "integer"
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                return True
            if (
                choice == "number"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                return True
            if choice == "string" and isinstance(value, str):
                return True
            if choice == "array" and isinstance(value, list):
                return True
            if choice == "object" and isinstance(value, Mapping):
                return True
        return False

    def _resolve_ref(self, ref: str) -> Mapping[str, Any] | bool:
        current: Any = self.schema
        for raw_part in ref[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if not isinstance(current, Mapping) or part not in current:
                raise UnsupportedSchemaError(f"unresolved ref: {ref}")
            current = current[part]
        if not isinstance(current, (Mapping, bool)):
            raise UnsupportedSchemaError(f"ref does not point to a schema: {ref}")
        return current

    def _literal_value(self, value: Any) -> Fragment:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return self.builder.literal(encoded)

    def _compile_object(
        self,
        schema: Mapping[str, Any],
        ref_stack: tuple[str, ...],
    ) -> Fragment:
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise UnsupportedSchemaError("object properties must be a mapping")
        min_properties = int(schema.get("minProperties", 0))
        max_properties = int(schema.get("maxProperties", len(properties)))
        if min_properties > len(properties):
            raise UnsupportedSchemaError("declared properties cannot satisfy minProperties")
        if max_properties < min_properties:
            raise UnsupportedSchemaError("object property bounds are unsatisfiable")

        required = set(schema.get("required", []))
        if not required.issubset(properties):
            raise UnsupportedSchemaError("required contains an undeclared property")
        if len(required) > max_properties:
            raise UnsupportedSchemaError("required properties exceed maxProperties")

        optional = [name for name in sorted(properties) if name not in required]
        selected = set(required)
        target_count = (
            min(len(properties), max_properties)
            if self.options.include_optional_properties
            else max(len(required), min_properties)
        )
        selected.update(optional[: max(0, target_count - len(selected))])
        names = sorted(selected)

        parts: list[Fragment] = [self.builder.literal(b"{")]
        for index, name in enumerate(names):
            if index:
                parts.append(self.builder.literal(b","))
            key = json.dumps(name, ensure_ascii=True).encode("ascii")
            parts.extend(
                [
                    self.builder.literal(key),
                    self.builder.literal(b":"),
                    self._compile_schema(properties[name], ref_stack),
                ]
            )
        parts.append(self.builder.literal(b"}"))
        return self.builder.concat(parts)

    def _compile_array(
        self,
        schema: Mapping[str, Any],
        ref_stack: tuple[str, ...],
    ) -> Fragment:
        if "prefixItems" in schema:
            prefix_items = schema["prefixItems"]
            if not isinstance(prefix_items, list):
                raise UnsupportedSchemaError("prefixItems must be a list")
            min_items = int(schema.get("minItems", 0))
            raw_max_items = schema.get("maxItems")
            additional_items_reachable = (
                raw_max_items is None
                or int(raw_max_items) > len(prefix_items)
            )
            if (
                additional_items_reachable
                and schema.get("items", True) is not False
            ):
                raise UnsupportedSchemaError(
                    "prefixItems with additional items requires normalization"
                )
            max_items = min(
                int(
                    raw_max_items
                    if raw_max_items is not None
                    else len(prefix_items)
                ),
                len(prefix_items),
            )
            if min_items > max_items:
                raise UnsupportedSchemaError(
                    "prefixItems shorter than minItems is not supported"
                )
            return self.builder.alternate(
                self._fixed_array(prefix_items[:count], ref_stack)
                for count in range(min_items, max_items + 1)
            )

        min_items = int(schema.get("minItems", 0))
        item_schema = schema.get("items", True)
        if item_schema is False:
            if min_items > 0:
                raise UnsupportedSchemaError(
                    "items: false cannot satisfy a positive minItems"
                )
            return self.builder.literal(b"[]")
        if item_schema is True:
            raise UnsupportedSchemaError("unconstrained array items require a JSON PDA")

        raw_max = schema.get("maxItems")
        max_items = (
            int(raw_max)
            if raw_max is not None
            else max(min_items, self.options.unbounded_array_cap)
        )
        if min_items < 0 or max_items < min_items:
            raise UnsupportedSchemaError("invalid array bounds")

        alternatives = [
            self._fixed_array([item_schema] * count, ref_stack)
            for count in range(min_items, max_items + 1)
        ]
        return self.builder.alternate(alternatives)

    def _fixed_array(
        self,
        item_schemas: list[Mapping[str, Any] | bool],
        ref_stack: tuple[str, ...],
    ) -> Fragment:
        parts: list[Fragment] = [self.builder.literal(b"[")]
        for index, item_schema in enumerate(item_schemas):
            if index:
                parts.append(self.builder.literal(b","))
            parts.append(self._compile_schema(item_schema, ref_stack))
        parts.append(self.builder.literal(b"]"))
        return self.builder.concat(parts)

    def _compile_string(self, schema: Mapping[str, Any]) -> Fragment:
        if "pattern" in schema:
            raise UnsupportedSchemaError("string pattern compilation is not implemented")
        min_length = int(schema.get("minLength", 0))
        raw_max = schema.get("maxLength")
        max_length = int(raw_max) if raw_max is not None else None
        if min_length < 0 or (max_length is not None and max_length < min_length):
            raise UnsupportedSchemaError("invalid string length bounds")

        content_parts = [self._string_unit() for _ in range(min_length)]
        if max_length is None:
            content_parts.append(self.builder.star(self._string_unit()))
            content = self.builder.concat(content_parts)
        else:
            variants = []
            for length in range(min_length, max_length + 1):
                variants.append(
                    self.builder.concat(self._string_unit() for _ in range(length))
                )
            content = self.builder.alternate(variants)

        return self.builder.concat(
            [
                self.builder.literal(b'"'),
                content,
                self.builder.literal(b'"'),
            ]
        )

    def _string_unit(self) -> Fragment:
        safe_ascii = [
            value
            for value in range(0x20, 0x7F)
            if value not in (ord('"'), ord("\\"))
        ]
        simple_escape = self.builder.alternate(
            self.builder.literal(value)
            for value in (
                b'\\"',
                b"\\\\",
                b"\\/",
                b"\\b",
                b"\\f",
                b"\\n",
                b"\\r",
                b"\\t",
            )
        )
        hex_digit = list(range(ord("0"), ord("9") + 1))
        hex_digit += list(range(ord("a"), ord("f") + 1))
        hex_digit += list(range(ord("A"), ord("F") + 1))
        unicode_escape = self.builder.concat(
            [
                self.builder.literal(b"\\u"),
                *(self.builder.charset(hex_digit) for _ in range(4)),
            ]
        )
        return self.builder.alternate(
            [
                self.builder.charset(safe_ascii),
                simple_escape,
                unicode_escape,
            ]
        )

    def _compile_integer(self, schema: Mapping[str, Any]) -> Fragment:
        choices = self._bounded_integer_choices(schema)
        if choices is not None:
            return self.builder.alternate(
                self.builder.literal(str(value).encode("ascii")) for value in choices
            )
        return self._integer_grammar()

    def _compile_number(self, schema: Mapping[str, Any]) -> Fragment:
        if any(
            key in schema
            for key in (
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "multipleOf",
            )
        ):
            choices = self._bounded_integer_choices(schema)
            if not choices:
                raise UnsupportedSchemaError("cannot derive a numeric witness")
            return self.builder.alternate(
                self.builder.literal(str(value).encode("ascii")) for value in choices
            )
        return self._number_grammar()

    def _bounded_integer_choices(
        self,
        schema: Mapping[str, Any],
    ) -> list[int] | None:
        bound_keys = {
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
        }
        if not bound_keys.intersection(schema):
            return None

        lower = -10**9
        upper = 10**9
        if "minimum" in schema:
            lower = max(lower, math.ceil(float(schema["minimum"])))
        if "exclusiveMinimum" in schema:
            lower = max(lower, math.floor(float(schema["exclusiveMinimum"])) + 1)
        if "maximum" in schema:
            upper = min(upper, math.floor(float(schema["maximum"])))
        if "exclusiveMaximum" in schema:
            upper = min(upper, math.ceil(float(schema["exclusiveMaximum"])) - 1)
        if lower > upper:
            raise UnsupportedSchemaError("numeric bounds are unsatisfiable")

        multiple = schema.get("multipleOf", 1)
        if not isinstance(multiple, int) or multiple <= 0:
            raise UnsupportedSchemaError("only positive integer multipleOf is supported")

        first = math.ceil(lower / multiple) * multiple
        if first > upper:
            raise UnsupportedSchemaError("numeric bounds have no multipleOf witness")
        count = ((upper - first) // multiple) + 1
        if count <= self.options.max_integer_choices:
            return [first + multiple * index for index in range(count)]

        if lower <= 0 <= upper and 0 % multiple == 0:
            return [0]
        return [first]

    def _integer_grammar(self) -> Fragment:
        def digit():
            return self.builder.charset(range(ord("0"), ord("9") + 1))

        def nonzero():
            return self.builder.charset(range(ord("1"), ord("9") + 1))

        magnitude = self.builder.alternate(
            [
                self.builder.literal(b"0"),
                self.builder.concat([nonzero(), self.builder.star(digit())]),
            ]
        )
        return self.builder.concat(
            [
                self.builder.optional(self.builder.literal(b"-")),
                magnitude,
            ]
        )

    def _number_grammar(self) -> Fragment:
        def digit():
            return self.builder.charset(range(ord("0"), ord("9") + 1))

        def nonzero():
            return self.builder.charset(range(ord("1"), ord("9") + 1))

        integer = self.builder.alternate(
            [
                self.builder.literal(b"0"),
                self.builder.concat([nonzero(), self.builder.star(digit())]),
            ]
        )
        fraction = self.builder.concat(
            [
                self.builder.literal(b"."),
                digit(),
                self.builder.star(digit()),
            ]
        )
        exponent = self.builder.concat(
            [
                self.builder.charset([ord("e"), ord("E")]),
                self.builder.optional(
                    self.builder.charset([ord("+"), ord("-")])
                ),
                digit(),
                self.builder.star(digit()),
            ]
        )
        return self.builder.concat(
            [
                self.builder.optional(self.builder.literal(b"-")),
                integer,
                self.builder.optional(fraction),
                self.builder.optional(exponent),
            ]
        )
