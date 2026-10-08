// Rewrite an ES module's top-level import/export declarations so its body
// can run as the body of `async function (__pie_require, __pie_exports)`:
// StarlingMonkey evaluates scripts at run time but cannot load a module
// from a string. Imports become bindings from the module table
// `__pie_require` resolves, exports become assignments on `__pie_exports`,
// and everything else is left byte for byte where it was.

import * as acorn from 'acorn';

/** Names a binding pattern introduces (`const {a, b: [c]} = ...`). */
function patternNames(node, out) {
  switch (node.type) {
    case 'Identifier': out.push(node.name); break;
    case 'ObjectPattern': for (const p of node.properties) patternNames(p.type === 'RestElement' ? p.argument : p.value, out); break;
    case 'ArrayPattern': for (const e of node.elements) if (e) patternNames(e, out); break;
    case 'RestElement': patternNames(node.argument, out); break;
    case 'AssignmentPattern': patternNames(node.left, out); break;
    default: break;
  }
  return out;
}

/**
 * Rewrite a module's top-level import/export declarations so the body can
 * run as the body of `async function (__pie_require, __pie_exports)`.
 */
export function transformModule(source, file) {
  const ast = acorn.parse(source, { ecmaVersion: 'latest', sourceType: 'module', allowHashBang: true });
  const edits = [];
  const exportLines = [];
  let modules = 0;

  for (const node of ast.body) {
    if (node.type === 'ImportDeclaration') {
      const m = `__pie_m${modules++}`;
      const lines = [`const ${m} = __pie_require(${JSON.stringify(node.source.value)});`];
      for (const s of node.specifiers) {
        if (s.type === 'ImportDefaultSpecifier') lines.push(`const ${s.local.name} = ${m}.default ?? ${m};`);
        else if (s.type === 'ImportNamespaceSpecifier') lines.push(`const ${s.local.name} = ${m};`);
        else {
          const imported = s.imported.type === 'Identifier' ? s.imported.name : s.imported.value;
          lines.push(`const ${s.local.name} = ${m}[${JSON.stringify(imported)}];`);
        }
      }
      edits.push([node.start, node.end, lines.join(' ')]);
    } else if (node.type === 'ExportNamedDeclaration') {
      if (node.declaration) {
        const d = node.declaration;
        const names = [];
        if (d.type === 'VariableDeclaration') for (const decl of d.declarations) patternNames(decl.id, names);
        else if (d.id) names.push(d.id.name);
        for (const n of names) exportLines.push(`__pie_exports[${JSON.stringify(n)}] = ${n};`);
        edits.push([node.start, d.start, '']);
      } else {
        let from = null;
        if (node.source) {
          from = `__pie_m${modules++}`;
          edits.push([node.start, node.end, `const ${from} = __pie_require(${JSON.stringify(node.source.value)});`]);
        } else {
          edits.push([node.start, node.end, '']);
        }
        for (const s of node.specifiers) {
          const local = s.local.type === 'Identifier' ? s.local.name : s.local.value;
          const exported = s.exported.type === 'Identifier' ? s.exported.name : s.exported.value;
          const value = from ? `${from}[${JSON.stringify(local)}]` : local;
          exportLines.push(`__pie_exports[${JSON.stringify(exported)}] = ${value};`);
        }
      }
    } else if (node.type === 'ExportDefaultDeclaration') {
      const d = node.declaration;
      if ((d.type === 'FunctionDeclaration' || d.type === 'ClassDeclaration') && d.id) {
        edits.push([node.start, d.start, '']);
        exportLines.push(`__pie_exports.default = ${d.id.name};`);
      } else {
        edits.push([node.start, d.start, '__pie_exports.default = ']);
        if (source[node.end - 1] !== ';') edits.push([node.end, node.end, ';']);
      }
    } else if (node.type === 'ExportAllDeclaration') {
      const line = node.exported
        ? `__pie_exports[${JSON.stringify(node.exported.name ?? node.exported.value)}] = __pie_require(${JSON.stringify(node.source.value)});`
        : `Object.assign(__pie_exports, __pie_require(${JSON.stringify(node.source.value)}));`;
      edits.push([node.start, node.end, line]);
    }
  }

  let body = source;
  for (const [start, end, text] of edits.sort((a, b) => b[0] - a[0])) {
    body = body.slice(0, start) + text + body.slice(end);
  }
  // The wrapper opens on the program's first line so every line number in
  // a stack trace is the program's own; the file name comes from the
  // `sourceURL` directive.
  return `(async (__pie_require, __pie_exports) => {"use strict";${body}\n${exportLines.join('\n')}\nreturn __pie_exports;\n})\n//# sourceURL=${file}\n`;
}

/**
 * Evaluate a program's module and return its exports. `require` resolves
 * an import specifier to a module object.
 */
export async function evaluateModule(source, file, require) {
  const factory = (0, eval)(transformModule(source, file));
  return await factory(require, Object.create(null));
}

/** An error as one string carrying its message and its stack, whichever
 * engine formatted it (SpiderMonkey's `stack` omits the message). */
export function describeError(e) {
  if (typeof e === 'string') return e;
  const head = e?.name && e?.message !== undefined ? `${e.name}: ${e.message}` : String(e);
  const stack = typeof e?.stack === 'string' ? e.stack.trim() : '';
  if (!stack) return head;
  return stack.startsWith(head) ? stack : `${head}\n${stack}`;
}

