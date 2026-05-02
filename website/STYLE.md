# Pie documentation style guide

This file governs every page under `website/docs/`. When in doubt, optimize for a reader who is stuck and trying to get unstuck.

## Audience

ML systems engineers and applied AI engineers. They read carefully, write code, and dislike fluff. Assume they know Python and either Rust or another systems language. Do not assume familiarity with Pie's internals or with WebAssembly tooling.

## Voice

1. Second person, present tense, active. Write "You write an inferlet that loads a model", not "We will see how one writes an inferlet".
2. Confident, not hedged. "Pie schedules the request" beats "Pie will typically schedule the request".
3. **Short sentences. Default to one idea per sentence.** If a sentence has more than one comma or two clauses, consider splitting it. Long sentences reduce readability and look like generated prose.
4. **Be formal and respectful.** Describe behavior in neutral, factual terms. Avoid casual phrasings ("hacks around", "becomes first-class", "obvious way") and editorializing.
5. First paragraph of every page answers two questions: what is this page for, and who should read it. If a reader cannot tell in ten seconds, the intro has failed.
6. Acknowledge the research-prototype status where relevant (unstable APIs, rough edges). Do not apologize on every page.

## Words and phrases to avoid

These are the tells of generated prose. Strip them.

### Punctuation
- **Em-dashes (`—`)**. Use a colon, a semicolon, parentheses, or two sentences instead.
- **Triple-clause sentences with semicolons strung together.** Break them up.

### Rhetorical patterns
- "Not just X, but Y." Pick one and say it directly.
- "It's not X. It's Y." Same problem.
- "While X, Y." If both clauses matter, give them their own sentences.
- "X is more than Y." If Y is wrong, say what is right.
- The rule-of-three list where two items would do.

### Filler openers and connectives
- "It's worth noting that"
- "Let's dive in" / "Let's explore"
- "In essence" / "Ultimately" / "Crucially" / "Indeed"
- "Moreover" / "Furthermore" (one per page maximum, and only when adding genuinely new information)
- "At its core"
- "Under the hood" (acceptable when actually describing internals; avoid as a rhetorical flourish)

### Marketing vocabulary
- "Powerful", "robust", "comprehensive", "seamless", "cutting-edge", "blazing", "delightful"
- "Leverage" (use "use"), "utilize" (use "use"), "facilitate" (use "let" or "help")
- "Empowers developers to", "unlocks the ability to"

### Reader-condescending words
- "Simply", "just", "easy", "obviously", "of course"
- "As you can see" (the reader can or cannot; the phrase adds nothing)

### Casual or dismissive phrasings
- "Hacks around X", "the obvious way", "X is broken", "ugly workarounds". Describe what something does in neutral, factual terms.
- "Become first-class", "supercharge", "out of the box", "next-generation". Say what the thing actually does.

### Sycophancy and filler endings
- "I hope this helps"
- "Feel free to"
- Apologetic preambles before code

## Structure

1. Lead with a runnable code block when the page is a how-to. Explain afterward.
2. One concept per page. Cross-link instead of recapping.
3. Headings every thirty lines or so. Code-heavy is fine.
4. Short paragraphs of two to four sentences. Long paragraphs hide the answer.
5. End how-to pages with a "next steps" section that links to two or three related pages by name, not "see also".

## Vocabulary (consistency matters more than choice)

- **Pie** is capitalized.
- **inferlet** is lowercase, always, including at the start of a sentence when possible to rephrase. Never InferLet, Inferlet, or INFERLET.
- **KV cache** (two words, both capitalized as shown). Not kv-cache, kvcache, or KV-Cache.
- **forward pass**, **token stream**, **tool call** are two words.
- **WebAssembly** in prose. `wasm` is acceptable in code, file names, and shell commands.
- Use **process** for a running inferlet instance. Do not also call it "request" or "job". Pick one and stay with it across the docs.
- Use **the engine** or **Pie** for the running server. Avoid "the system", "the platform", "the framework".

## Code

1. Every snippet is runnable as written. No `// ...`, no `your_code_here()`, no pseudocode unless the page is explicitly about pseudocode.
2. Include imports. A reader should not have to grep for where `Sampler` comes from.
3. If you must elide, use a comment that names what was cut: `// elided: input validation` and link to the full example in `examples/`.
4. Show output when it clarifies. Do not make the reader guess what they will see.
5. Comments explain why, not what. Identifier names already say what.
6. Compile and run every snippet before merging. Broken code in docs is worse than no code.

## Multi-language tabs

Build pages (the how-tos) present each example in three tabs: Rust, Python, JavaScript. In that order.

- Rust is the canonical version. Write it first, get it right, then translate.
- The other two tabs may carry a small note "translated from the Rust example" if the language's API differs in shape.
- Learn pages are Rust-only. The narrative reads cleaner without tab switching, and a first-time reader benefits from a single consistent language.
- Reference pages are already split per language and do not use tabs.

Use the Docusaurus `@theme/Tabs` component. Default tab is Rust.

Import block at the top of every page that uses tabs (after the frontmatter):

```mdx
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
```

Wrap each multi-language code block:

````mdx
<Tabs groupId="lang" queryString>
<TabItem value="rust" label="Rust" default>

```rust
// Rust example
```

</TabItem>
<TabItem value="python" label="Python">

```python
# Python example
```

</TabItem>
<TabItem value="js" label="JavaScript">

```typescript
// JavaScript / TypeScript example
```

</TabItem>
</Tabs>
````

`groupId="lang"` plus `queryString` sync the language choice across all tab blocks on the site. A reader who picks Python on one page sees Python everywhere.

## Links

- Link by the destination page's title, not "click here" or "this page".
- Cross-link liberally. Better to over-link than to make a reader go back to the index.
- Internal links use relative paths so they survive moves. External links open in the same tab unless they go to a paper or a third-party tool.

## Diagrams and images

- SVG when possible. Raster only for screenshots.
- Every image has alt text that describes the content, not the file.
- Diagrams of architecture live next to the page that introduces them, not in a global `assets/` folder.

## The read-it-aloud test

Before merging a page, read one paragraph out loud. If it sounds like a vendor pitch, rewrite. If it sounds like a colleague explaining over coffee, ship it.
