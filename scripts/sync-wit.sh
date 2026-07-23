#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sync_tree() {
    local source="$1"
    local destination="$2"

    mkdir -p "$destination"
    find "$destination" -type f -name '*.wit' -delete
    cp -R "$source"/. "$destination"/
}

sync_tree "$root/interface/inferlet" "$root/sdk/rust/inferlet/wit"
sync_tree "$root/interface/inferlet" "$root/sdk/tools/bakery/src/bakery/wit"
sync_tree "$root/interface/plex/wit-0.6" "$root/sdk/rust/plex/wit"
