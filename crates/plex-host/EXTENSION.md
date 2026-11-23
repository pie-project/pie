# pie's extension of PLEX 1.1: cross-tenant share grants

PLEX 1.1 has no column for sharing KV across tenants. It leaves
cross-class acceptance out of scope (spec section 10), and its working-set
tree is read-only to policies (section 5). pie needs one: an inferlet may
ask to share its KV under a named group, and KV must match across tenants
only where an operator's policy granted it.

The extension is one column in its own package, `pie:plex-ext@0.1.0`
(`wit-ext/share.wit`). No frozen 1.1 file is edited, and a 1.1 policy
that does not import it is unaffected. A policy that imports it loads only
on pie: another host refuses the load, because it cannot enact the column.

## The column

```wit
package pie:plex-ext@0.1.0;

interface share-groups {
    use plex:core/types@1.1.0.{error-code};
    type group = string;
    type tenant = string;

    get: func(group: group) -> option<list<tenant>>;
    set: func(group: group, value: list<tenant>) -> result<_, error-code>;
    set-list: func(entries: list<tuple<group, list<tenant>>>) -> result<_, error-code>;
    delete: func(group: group);
    set-default: func(value: list<tenant>) -> result<_, error-code>;
    clear: func();
    get-refused: func() -> list<tuple<group, error-code>>;
}

world cluster {
    include plex:policy/cluster@1.1.0;
    import share-groups;
}
```

It has the same seven functions as every 1.1 column (section 4).

| | |
|---|---|
| Scope | cluster: one table for the deployment. `pie serve` always runs a controller, so a single node has one too. |
| Key | a group name |
| Value | the tenants that may share KV under the group, as `snapshot.process.tenant` names them: authenticated, never a hint |
| Default | no tenant. With no row, no group crosses tenants. |
| Writes | An empty group or tenant name is `invalid-argument`. Groups and tenants have no lifetime, so a row is never stale. |
| `get-refused` | always empty: a grant is enacted wherever it is read |

## What it means

An inferlet asks `share(scope)` with `scope` one of `session`, `tenant` or
`group(name)`. That verb is in pie's working-set WIT, not in PLEX.
- **Session and tenant scopes** never cross tenants and need no row.
- **A group scope** crosses tenants only among the tenants the group's row
  names (else the default's). A tenant not named neither reads KV under the
  group nor writes KV into it. Its ask shares within its own tenant
  instead, and `share` reports which scope took effect, so nothing changes
  silently.

The isolation is in the key. The KV hash domain is:

| Ask | Domain |
|---|---|
| `group(g)`, granted | model identity ‖ `group` ‖ `g` ‖ grant id |
| `group(g)`, not granted | model identity ‖ tenant ‖ `group` ‖ `g` (tenant-local) |
| `tenant` | model identity ‖ tenant |
| `session` | model identity ‖ tenant ‖ session |

The tenant is always the authenticated one, at every keying site: the chain
root, fresh page keys, directory and pool pushes.

**A grant's id** changes with every change to its group's row (the
default's with the default). KV written under one grant therefore does not
match under the next. After a revoke or a re-grant the old pages are
unreachable to every tenant, and they age out as any cached KV does.

## Failing closed

Every way a grant can be lost is no grant, and the ask is tenant-local:
- no table yet;
- the policy unloaded, or its rows withdrawn by its lease (`lease-calls`
  failures in a row, section 7);
- a worker with no gateway link standing. What it holds may be stale, so it
  holds nothing until a link pushes the table again.

Nothing already cached is lost: the pages stay, unmatched.

## In pie

- **Host** (`crates/plex-host`): the column in `tables.rs` (`SHARE_GROUPS`)
  and `runtime.rs` (bindgen over `wit/` and `wit-ext/`), and in `poll_delta`.
- **Controller** (`controller/src/cluster/port.rs`): enacts the rows into a
  versioned table and gives each grant its id.
  `GatewayDecisions.share` carries it to every gateway.
- **Gateway** (`gateway/src/policy/service.rs`): sends the table whole
  (`worker-api share_groups`) to each linked worker, on a change and as a
  worker links.
- **Worker** (`runtime/src/policy/share.rs`): `granted(group, tenant) ->
  Option<GrantId>`, one hash lookup; a newer version replaces the table, an
  older is ignored. The keying sites call it per pass.
- **SDK**: `plex::ext::share_groups`.
- **Probe**: the builtin `test-share`.

## Why a column and not a hint or a note

Hints are client-said and untrusted (section 3), and a tenant must not grant
itself access to another tenant's KV. Notes are trusted only as far as the
operator trusts the writing policy, and they have no lease. A column has
exactly one writer and is withdrawn with its policy, so a gone policy
cannot leave a grant in force.
