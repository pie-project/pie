import unittest

from gpu_lr1.schema import CanonicalJSONSchemaCompiler, UnsupportedSchemaError


class SchemaCompilerTest(unittest.TestCase):
    def test_compiles_nested_object_and_array(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "active": {"type": "boolean"},
                "mode": {"enum": ["fast", "safe"]},
                "values": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                    },
                    "minItems": 1,
                    "maxItems": 2,
                },
            },
            "required": ["active", "mode", "values"],
            "additionalProperties": False,
        }
        dfa = CanonicalJSONSchemaCompiler(schema).compile()

        self.assertTrue(
            dfa.accepts('{"active":true,"mode":"fast","values":[0]}')
        )
        self.assertTrue(
            dfa.accepts('{"active":false,"mode":"safe","values":[1,2]}')
        )
        self.assertFalse(dfa.accepts("{}"))
        self.assertFalse(
            dfa.accepts('{"active":true,"mode":"invalid","values":[0]}')
        )

    def test_string_escaping_and_bounds(self) -> None:
        dfa = CanonicalJSONSchemaCompiler(
            {"type": "string", "minLength": 1, "maxLength": 2}
        ).compile()
        self.assertTrue(dfa.accepts('"a"'))
        self.assertTrue(dfa.accepts('"a\\n"'))
        self.assertFalse(dfa.accepts('""'))
        self.assertFalse(dfa.accepts('"abc"'))
        self.assertFalse(dfa.accepts('"unterminated'))

    def test_local_ref(self) -> None:
        schema = {
            "$defs": {"status": {"enum": ["ok", "error"]}},
            "$ref": "#/$defs/status",
        }
        dfa = CanonicalJSONSchemaCompiler(schema).compile()
        self.assertTrue(dfa.accepts('"ok"'))
        self.assertFalse(dfa.accepts('"unknown"'))

    def test_recursive_ref_is_explicitly_rejected(self) -> None:
        schema = {
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/$defs/node"}},
                }
            },
            "$ref": "#/$defs/node",
        }
        with self.assertRaisesRegex(UnsupportedSchemaError, "recursive"):
            CanonicalJSONSchemaCompiler(schema).compile()

    def test_one_of_is_rejected_instead_of_approximated(self) -> None:
        schema = {
            "oneOf": [
                {"type": "integer"},
                {"type": "number"},
            ]
        }
        with self.assertRaisesRegex(UnsupportedSchemaError, "oneOf"):
            CanonicalJSONSchemaCompiler(schema).compile()

    def test_optional_properties_respect_max_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "required": {"type": "boolean"},
                "optional_a": {"type": "boolean"},
                "optional_b": {"type": "boolean"},
            },
            "required": ["required"],
            "minProperties": 1,
            "maxProperties": 2,
        }
        dfa = CanonicalJSONSchemaCompiler(schema).compile()
        self.assertTrue(
            dfa.accepts('{"optional_a":true,"required":false}')
        )
        self.assertFalse(
            dfa.accepts(
                '{"optional_a":true,"optional_b":true,"required":false}'
            )
        )

    def test_enum_with_sibling_bounds_is_rejected(self) -> None:
        schema = {
            "type": "string",
            "minLength": 10,
            "enum": ["short"],
        }
        with self.assertRaisesRegex(UnsupportedSchemaError, "normalization"):
            CanonicalJSONSchemaCompiler(schema).compile()

    def test_enum_values_are_filtered_by_declared_type(self) -> None:
        dfa = CanonicalJSONSchemaCompiler(
            {
                "type": "integer",
                "enum": [1, "wrong", True],
            }
        ).compile()
        self.assertTrue(dfa.accepts("1"))
        self.assertFalse(dfa.accepts('"wrong"'))
        self.assertFalse(dfa.accepts("true"))

    def test_items_false_with_positive_min_items_is_rejected(self) -> None:
        with self.assertRaisesRegex(UnsupportedSchemaError, "minItems"):
            CanonicalJSONSchemaCompiler(
                {
                    "type": "array",
                    "items": False,
                    "minItems": 1,
                }
            ).compile()

    def test_prefix_items_with_reachable_additional_items_is_rejected(self) -> None:
        with self.assertRaisesRegex(UnsupportedSchemaError, "additional items"):
            CanonicalJSONSchemaCompiler(
                {
                    "type": "array",
                    "prefixItems": [{"type": "boolean"}],
                    "items": {"type": "integer"},
                    "maxItems": 2,
                }
            ).compile()


if __name__ == "__main__":
    unittest.main()
