# Equivalence: did an IR change move anything it should not have?

A rename, a fold of ops into attributes, or a move of code between a kernel
crate and its engine should leave every plan the catalog produces as it was.
These scripts check that on every deployment, without a device:

| What | How it is compared |
|---|---|
| shard plans, tp 1/2/4/8 (`poem-compiler` example `shard_dump`) | `diff` |
| XLA programs (`engine-xla` example `coverage --dump`) | bytes, then `canon.py`: equal dataflow up to emission order and SSA names |
| Cerebras programs, phase, table and whole-fire modes (`engine-cerebras` example `coverage --dump`) | bytes |
| each coverage run's per-deployment lines | `diff` |

## Procedure

```sh
git worktree add ../base <the commit before the change>
(cd ../base && scripts/equivalence/dump.sh /big/disk/base)
scripts/equivalence/dump.sh /big/disk/cand
scripts/equivalence/compare.sh /big/disk/base /big/disk/cand
```

Run `dump.sh` from the tree to dump; `slangc` must be reachable
(`PIE_SLANGC`) since a workspace build compiles kernels-slang. A filter
argument (`dump.sh OUT qwen3`) keeps to the deployments whose names hold it.

Allowed differences, each to be read rather than assumed:

- a refusal message or a coverage line naming an op that was renamed;
- an XLA `loc` scope naming a renamed op (canon.py hashes `loc`; strip the
  old name from the base before comparing if that is the only difference);
- a change the commit means, such as more deployments sharding, said in its
  message with the count.

## What these do not cover

Kernels and engines that run on a device: the GPU suites (engine-cuda,
engine-metal, engine-vulkan, engine-wgpu) and `scripts/ci/lavapipe.sh`,
whose corner texts and token ids are compared byte for byte against the base.
