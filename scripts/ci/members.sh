#!/usr/bin/env bash
# `-p <crate>` for every workspace member, minus the names given.
set -euo pipefail
cargo metadata --no-deps --format-version 1 \
  | python3 -c "
import json, sys
skip = set(sys.argv[1:])
print(' '.join('-p ' + p['name'] for p in json.load(sys.stdin)['packages'] if p['name'] not in skip))
" "$@"
