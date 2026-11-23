//! A cluster probe of pie's PLEX extension (`pie:plex-ext/share-groups`):
//! grants the group `team` to the tenants `acme` and `beta`, and the
//! default (every other group) to `acme` alone, on every call. Unloaded, its
//! rows are withdrawn and no group crosses tenants.

use plex::ext::share_groups;
use plex::prelude::*;

#[derive(Default)]
struct Share;

impl Policy for Share {
    fn plan(&mut self, _now: Mark) {
        share_groups::set("team", &["acme".to_string(), "beta".to_string()]).ok();
        share_groups::set_default(&["acme".to_string()]).ok();
        // An empty name is refused at the write.
        assert!(share_groups::set("", &[]).is_err());
    }
}

plex::export!(Share);
