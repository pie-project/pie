use plex::{Document, Host, Policy, State};

struct Spin;

impl Policy for Spin {
    fn route(_ctx: &Document, _state: &mut State, _host: &Host) -> Result<Document, String> {
        loop {
            core::hint::spin_loop();
        }
    }
}

plex::export_policy!(Spin);
