use plex::{Document, Policy};

struct Spin;

impl Policy for Spin {
    fn route(_input: &mut Document) -> Result<Document, String> {
        loop {
            core::hint::spin_loop();
        }
    }
}

plex::export_policy!(Spin);
