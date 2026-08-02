"""What the old public surface added over the device layer, for the tests.

`engrain.Engine` and `engrain.Slots` are the surface now, and they are covered
by `test_api.py`. The tests that predate them are about what the *engine* must
refuse - a grammar from the wrong tokenizer, a token past the vocabulary, a
pool that outgrew a live batch - and those behaviours did not move. Rather than
lose them, they keep talking to the layer underneath through this, which is the
three lines they would otherwise each repeat.
"""

from __future__ import annotations

from engrain.internals import Compiler, DeviceGrammar

__all__ = ["Batch", "Compiler", "DeviceGrammar", "Engine"]


class Engine:
    """Compile, admit, hand out batches - the old surface's whole job.

    Kept here rather than in the library because it is not a layer, it is the
    three lines these tests would otherwise repeat.
    """

    def __init__(self, vocabulary, **options):
        self.compiler = Compiler(list(vocabulary))
        self._pool = DeviceGrammar(**options)
        # Admission is idempotent in the library because two requests must not
        # take two slots for one grammar; keep that here or these tests measure
        # the shim rather than the engine.
        self._ids = {}
        self._held = {}

    def compile_json_schema(self, schema, **options):
        grammar = self.compiler.compile_json_schema(schema, **options)
        self.admit(grammar)
        return grammar

    def compile_regex(self, pattern, **options):
        grammar = self.compiler.compile_regex(pattern, **options)
        self.admit(grammar)
        return grammar

    def admit(self, grammar):
        key = id(grammar)
        held = self._ids.get(key)
        if held is not None and self._pool.holds(*held):
            return held[0]
        identifier = self._pool.admit(grammar)
        self._ids[key] = (identifier, self._pool.generation(identifier))
        self._held[key] = grammar
        return identifier

    def batch(self, size, *, rollback=0):
        return Batch(self._pool.new_batch(size, rollback=rollback), self)

    @property
    def mask_words(self):
        return self._pool.mask_words

    @property
    def resident_bytes(self):
        return self._pool.resident_bytes()

    @property
    def pool(self):
        return self._pool


class Batch:
    """`DeviceBatch` under the names these tests were written against."""

    def __init__(self, batch, engine):
        self.raw = batch
        self._engine = engine

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def set_grammars(self, grammars):
        self.raw.set_grammars(
            [g if isinstance(g, int) else self._engine.admit(g) for g in grammars]
        )

    def allowed(self, capacity=4096):
        ids, counts, _ = self.raw.compact(capacity)
        return ids, counts

    def shortlist(self, capacity=8192):
        return self.raw.compact(capacity, both=True)

    def step(self, tokens):
        return self.raw.advance_and_fill(tokens)

    def capture(self):
        self.raw.capture()
        self.raw.capture_advance()
        self.raw.capture_step()

    @property
    def size(self):
        return self.raw.batch
