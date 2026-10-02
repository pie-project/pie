// The JavaScript language component: the one wasm every JavaScript
// inferlet runs in.
//
// A JavaScript inferlet is its source, an ES module that imports
// `@pie-project/inferlet` (or the host's `pie:inferlet/*` interfaces) and
//
//      "input": "<the caller's input, verbatim>"}
//
// StarlingMonkey evaluates scripts at run time but has no way to load an ES
// module from a string, so the source is parsed here (acorn) and its
// top-level `import`/`export` declarations are rewritten into plain
// bindings against a module table -- the inferlet library this component bundles and
// the host interfaces it links -- and the body runs as one async function.
// Everything else in the program is untouched, and a `//# sourceURL` keeps
// the program's own file name in stack traces.

import * as inferlet from '@pie-project/inferlet';
import { describeError, evaluateModule } from './transform.js';

import * as witModel from 'pie:inferlet/model@0.3.0';
import * as witTokenizer from 'pie:inferlet/tokenizer@0.3.0';
import * as witPipeline from 'pie:inferlet/pipeline@0.3.0';
import * as witWorkingSet from 'pie:inferlet/working-set@0.3.0';
import * as witChannel from 'pie:inferlet/channel@0.3.0';
import * as witForward from 'pie:inferlet/forward@0.3.0';
import * as witForwardRecurrent from 'pie:inferlet/forward-recurrent@0.3.0';
import * as witForwardHybrid from 'pie:inferlet/forward-hybrid@0.3.0';
import * as witForwardDiffusion from 'pie:inferlet/forward-diffusion@0.3.0';
import * as witGrammar from 'pie:inferlet/grammar@0.3.0';
import * as witChat from 'pie:inferlet/chat@0.3.0';
import * as witTools from 'pie:inferlet/tools@0.3.0';
import * as witReasoning from 'pie:inferlet/reasoning@0.3.0';
import * as witMedia from 'pie:inferlet/media@0.3.0';
import * as witSession from 'pie:inferlet/session@0.3.0';
import * as witSystem from 'pie:inferlet/system@0.3.0';

// Intl polyfill for the wasm engine: minimal DateTimeFormat for the chat
// template library.
if (typeof globalThis.Intl === 'undefined') {
  const MONTHS_LONG = ['January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'];
  const MONTHS_SHORT = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  globalThis.Intl = {
    DateTimeFormat(locale, options) {
      return {
        format(date) {
          if (options && options.month === 'long') return MONTHS_LONG[date.getMonth()];
          if (options && options.month === 'short') return MONTHS_SHORT[date.getMonth()];
          return date.toISOString();
        },
      };
    },
  };
}

// A function sent from a client (`inferlet(fn)` in @pie-project/client) is
// one function's text, with no module top to import from: it reaches the
// library through this global (`pie.model.encode(...)`).
globalThis.pie = inferlet;

const ENVELOPE_KEY = '__pie_script__';

const HOST = {
  model: witModel,
  tokenizer: witTokenizer,
  pipeline: witPipeline,
  'working-set': witWorkingSet,
  channel: witChannel,
  forward: witForward,
  'forward-recurrent': witForwardRecurrent,
  'forward-hybrid': witForwardHybrid,
  'forward-diffusion': witForwardDiffusion,
  grammar: witGrammar,
  chat: witChat,
  tools: witTools,
  reasoning: witReasoning,
  media: witMedia,
  session: witSession,
  system: witSystem,
};

/** The module a program's `import` resolves to. */
function resolveModule(specifier) {
  if (specifier === '@pie-project/inferlet' || specifier === 'inferlet') return inferlet;
  const host = /^pie:inferlet\/([a-z-]+)(?:@[\d.]+)?$/.exec(specifier);
  if (host && HOST[host[1]]) return HOST[host[1]];
  throw new Error(
    `cannot import '${specifier}': a JavaScript inferlet imports '@pie-project/inferlet' ` +
    `and the host's pie:inferlet/* interfaces; nothing else is bundled`,
  );
}

function parseInput(raw) {
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return { input: raw };
  }
}

function encode(result) {
  if (result == null) return '';
  if (typeof result === 'string') return result;
  return JSON.stringify(result);
}

// WIT export: pie:inferlet/run
export const run = {
  async run(input) {
    let outer = null;
    try {
      outer = input ? JSON.parse(input) : {};
    } catch {
      outer = null;
    }
    if (!outer || typeof outer !== 'object' || !(ENVELOPE_KEY in outer)) {
      // componentize-js maps a thrown *string* to the WIT `result<_, string>`
      // Err arm; an Error object would trap.
      throw `the JavaScript language component was launched without a program: the launch input carries no '${ENVELOPE_KEY}' envelope`;
    }
    const script = outer[ENVELOPE_KEY];
    const file = script.file || 'index.js';
    const inputData = parseInput(outer.input ?? '');
    try {
      const exports = await evaluateModule(script.source, file, resolveModule);
      const fn = exports.main ?? exports.default;
      if (typeof fn !== 'function') {
        throw new Error(
          `${script.name ?? 'the program'} exports no \`main\`; a JavaScript inferlet is a module ` +
          'that exports `function main(input)` (or exports it as its default)',
        );
      }
      return encode(await fn(inputData));
    } catch (e) {
      throw describeError(e);
    }
  },
};
