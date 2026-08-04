"""Parsing `model/config.hpp` into a field list.

One parser, imported by every emitter. There used to be two copies — the C++
one here and the Rust one under `runtime/model-config/` — and the same bug had
to be found and fixed in both, twice: declarations whose line ends in a `//`
comment (24 fields, `head_dim` and `use_qk_norm` among them), then a field that
brace-initializes without `=` (`upsampling_ratios{8, 6, 5, 4}`). Both times the
generated output looked fine, because a field missing from an emitter is a
field missing silently.

So the parse lives once, and it refuses to return a partial answer.
"""
import collections
import re

#: Types every emitter knows how to handle without further mapping.
SIMPLE = {
    "int",
    "float",
    "bool",
    "std::string",
    "std::vector<int>",
    "std::vector<float>",
    "std::vector<std::string>",
}

_FIELD = re.compile(
    r"^\s{4}((?:const\s+)?[\w:<>,\s]*?)\s+(\w+)\s*(?:=\s*([^;]+)|(\{[^}]*\}))?;\s*(?://.*)?$",
    re.M,
)
_DECL = re.compile(r"^(?:const\s+)?[\w:<>,\s]+?\s+(\w+)\s*(?:=[^;]*|\{[^}]*\})?;$")
_STRUCT = re.compile(r"struct (\w+)\s*\{(.*?)\n\};", re.S)


#: One declared member: its C++ type, its name, and its initializer.
Field = collections.namedtuple("Field", "type name init")


def parse(source):
    """Every field of every struct in `source`, in declaration order.

    Returns `(structs, order)` where `structs` maps a struct name to its
    `Field` list and `order` is the order they were declared in.

    Raises when a line inside a struct body looks like a declaration and was
    not captured: a silently dropped field is the one failure this must not
    have.
    """
    structs, order = {}, []
    for match in _STRUCT.finditer(source):
        name, body = match.group(1), match.group(2)
        fields = [
            Field(type_.strip(), field, (eq or brace or None))
            for type_, field, eq, brace in _FIELD.findall(body)
            if type_ and not type_.startswith(("//", "enum")) and "return" not in type_
        ]
        _assert_complete(name, body, {f.name for f in fields})
        structs[name] = fields
        order.append(name)
    return structs, order


def _assert_complete(struct, body, captured):
    for line in body.splitlines():
        stripped = re.sub(r"//.*$", "", line).strip()
        if not stripped or stripped.startswith(("//", "/*", "*", "enum", "}", "{")):
            continue
        if not stripped.endswith(";"):
            continue
        decl = _DECL.match(stripped)
        if decl and decl.group(1) not in captured:
            raise SystemExit(
                f"{struct}: declaration not captured, so it would be missing "
                f"from the generated code: {line.strip()!r}"
            )
