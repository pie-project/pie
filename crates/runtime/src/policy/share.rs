//! Which cache a working set's keys belong to: the scope its inferlet asked
//! (`working-set.share`), under the tenant the gateway authenticated.
//!
//! A content key chains from a domain ([`state::key::chain_slots`]), so two
//! sets whose domains differ can never name the same page: no lookup,
//! coalesced claim, keep, directory entry, fetch or pool push crosses them.
//! [`scope_domain`] is the one place a domain is made from a scope.
//!
//! **The grant seam.** A `group` scope crosses tenants only where a grant
//! covers the asking tenant; the cross-tenant grant table is the PLEX port's.
//! It answers [`granted`]; until it lands, [`NoGrants`] answers `None` for
//! every group, so a group is tenant-local. A grant is resolved when a set
//! asks and again each time a pass computes keys, so a revoked or reissued
//! grant (a new [`GrantId`]) is seen on the next pass.

use crate::store::kv::hash::{Hash256, cache_domain};

/// A tenant as the gateway authenticated it ([`crate::policy::Arrival`]),
/// never as an inferlet names one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TenantId(pub String);

/// One version of one cross-tenant grant: reissuing or revoking it changes
/// the id, and with it every key derived under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GrantId(pub u64);

/// What `working-set.share` names.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShareScope {
    Session,
    Tenant,
    Group(String),
}

/// Who asked: the process's authenticated tenant, and its session (the
/// one its launch named, else the process itself).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Asker {
    pub tenant: TenantId,
    pub session: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SessionId {
    /// The session the launch named, within its tenant.
    Named(String),
    /// A launch that named none is a session of its own.
    Process(uuid::Uuid),
}

/// What a set shares as: a scope, as its process's asker; or, for a pass a
/// peer offloaded here, the asking set's own domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShareAs {
    Scope(ShareScope, Asker),
    OnBehalf(Hash256),
}

/// The grants a group scope may cross tenants under.
pub trait GrantLookup {
    /// The grant covering `tenant` for `group`, if one does. O(1): it runs
    /// under the kv store's lock at each keyed pass.
    fn granted(&self, group: &str, tenant: &TenantId) -> Option<GrantId>;
}

/// No grant table: every group is tenant-local. The PLEX port replaces it.
pub struct NoGrants;

impl GrantLookup for NoGrants {
    fn granted(&self, _group: &str, _tenant: &TenantId) -> Option<GrantId> {
        None
    }
}

/// The grant covering `tenant` for `group`.
pub fn granted(group: &str, tenant: &TenantId) -> Option<GrantId> {
    NoGrants.granted(group, tenant)
}

/// The domain a set asking `scope` keys its pages under, within the model
/// whose domain is `model` (its [`state::identity::ModelIdentity::domain`]).
/// `grant` is [`granted`]'s answer for a group, and is ignored otherwise.
///
/// - session: model ‖ tenant ‖ "session" ‖ session
/// - tenant: model ‖ tenant
/// - group, granted: model ‖ "group" ‖ group ‖ grant
/// - group, not granted: model ‖ tenant ‖ "group" ‖ group
///
/// Each is led by its own tag and every part is length-prefixed, so no two
/// scopes, and no two parts' splits, give the same bytes.
pub fn scope_domain(
    model: &Hash256,
    scope: &ShareScope,
    asker: &Asker,
    grant: Option<GrantId>,
) -> Hash256 {
    let tenant = asker.tenant.0.as_bytes();
    let session;
    let grant_bytes;
    let parts: Vec<&[u8]> = match (scope, grant) {
        (ShareScope::Session, _) => {
            session = match &asker.session {
                SessionId::Named(name) => [b"named:".as_slice(), name.as_bytes()].concat(),
                SessionId::Process(pid) => [b"process:".as_slice(), pid.as_bytes()].concat(),
            };
            vec![b"scope:session", tenant, b"session", &session]
        }
        (ShareScope::Tenant, _) => vec![b"scope:tenant", tenant],
        (ShareScope::Group(group), Some(grant)) => {
            grant_bytes = grant.0.to_le_bytes();
            vec![b"scope:group-granted", b"group", group.as_bytes(), &grant_bytes]
        }
        (ShareScope::Group(group), None) => {
            vec![b"scope:group-local", tenant, b"group", group.as_bytes()]
        }
    };
    domain_of_parts(model, &parts)
}

/// The domain of a set that asked nothing: its own, from `seed` (the
/// store's, random each run) and `nonce` (one per set the store makes).
pub fn private_domain(model: &Hash256, seed: &Hash256, nonce: u64) -> Hash256 {
    domain_of_parts(model, &[b"scope:private", seed, &nonce.to_le_bytes()])
}

fn domain_of_parts(model: &Hash256, parts: &[&[u8]]) -> Hash256 {
    let mut seed = Vec::with_capacity(32 + parts.iter().map(|p| 8 + p.len()).sum::<usize>());
    seed.extend_from_slice(model);
    for part in parts {
        seed.extend_from_slice(&(part.len() as u64).to_le_bytes());
        seed.extend_from_slice(part);
    }
    cache_domain(&seed)
}

/// [`scope_domain`] with the grant resolved now; an offloaded pass's
/// domain as it came.
pub fn domain_now(model: &Hash256, ask: &ShareAs) -> Hash256 {
    match ask {
        ShareAs::Scope(scope, asker) => {
            let grant = match scope {
                ShareScope::Group(group) => granted(group, &asker.tenant),
                _ => None,
            };
            scope_domain(model, scope, asker, grant)
        }
        ShareAs::OnBehalf(domain) => *domain,
    }
}

/// A test's sets share within one tenant, as a tenant-scoped inferlet's.
#[cfg(test)]
pub fn test_tenant() -> ShareAs {
    tenant_of("test")
}

/// Tenant `tenant`'s tenant scope, under test.
#[cfg(test)]
pub fn tenant_of(tenant: &str) -> ShareAs {
    ShareAs::Scope(
        ShareScope::Tenant,
        Asker {
            tenant: TenantId(tenant.into()),
            session: SessionId::Named("test".into()),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asker(tenant: &str, session: &str) -> Asker {
        Asker {
            tenant: TenantId(tenant.into()),
            session: SessionId::Named(session.into()),
        }
    }

    /// Every scope, tenant, session, group and grant moves the domain, and
    /// no tenant or group name can spell another scope's bytes.
    #[test]
    fn every_scope_and_part_moves_the_domain() {
        let model = [7; 32];
        let a = asker("a", "s");
        let d = |scope: &ShareScope, asker: &Asker, grant| scope_domain(&model, scope, asker, grant);
        let group = ShareScope::Group("g".into());
        let all = [
            d(&ShareScope::Tenant, &a, None),
            d(&ShareScope::Tenant, &asker("b", "s"), None),
            d(&ShareScope::Session, &a, None),
            d(&ShareScope::Session, &asker("a", "t"), None),
            d(&ShareScope::Session, &asker("b", "s"), None),
            d(&group, &a, None),
            d(&group, &asker("b", "s"), None),
            d(&group, &a, Some(GrantId(1))),
            d(&group, &a, Some(GrantId(2))),
            d(&ShareScope::Group("h".into()), &a, None),
            // A tenant named like a tag spells no other scope.
            d(&ShareScope::Tenant, &asker("group", "s"), None),
            d(&ShareScope::Session, &asker("group", "session"), None),
            private_domain(&model, &[1; 32], 0),
            private_domain(&model, &[1; 32], 1),
            scope_domain(&[8; 32], &ShareScope::Tenant, &a, None),
        ];
        for (i, x) in all.iter().enumerate() {
            for y in &all[i + 1..] {
                assert_ne!(x, y);
            }
        }
        // A grant covers tenants alike; a tenant-scoped ask ignores one.
        assert_eq!(
            d(&group, &a, Some(GrantId(1))),
            d(&group, &asker("b", "x"), Some(GrantId(1)))
        );
        assert_eq!(
            d(&ShareScope::Tenant, &a, Some(GrantId(1))),
            d(&ShareScope::Tenant, &asker("a", "other"), None)
        );
    }

    #[test]
    fn the_stub_grants_nothing() {
        assert_eq!(granted("g", &TenantId("a".into())), None);
    }
}
