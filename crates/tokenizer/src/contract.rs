use crate::Tokenizer;

#[derive(Clone, Copy, Debug)]
pub struct Contract {
    pub markers: &'static [&'static [&'static str]],
    pub pinned: &'static [(&'static str, u32)],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    Missing {
        marker: &'static str,
    },
    Displaced {
        marker: &'static str,
        want: u32,
        found: u32,
    },
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { marker } => write!(
                f,
                "the serving row reads a `{marker}` token and this artifact's \
                 tokenizer spells no such token; the row and the checkpoint \
                 are not each other's"
            ),
            Self::Displaced {
                marker,
                want,
                found,
            } => write!(
                f,
                "the serving row pins `{marker}` at id {want} and this \
                 artifact's tokenizer holds it at {found}; a pinned marker \
                 selects a reading of the artifact, and this artifact answers \
                 a different one"
            ),
        }
    }
}

impl std::error::Error for Fault {}

impl Contract {
    pub fn verify(&self, tokenizer: &Tokenizer) -> Result<(), Fault> {
        for group in self.markers {
            for &marker in *group {
                if tokenizer.token_to_id(marker).is_none() {
                    return Err(Fault::Missing { marker });
                }
            }
        }
        for &(marker, want) in self.pinned {
            match tokenizer.token_to_id(marker) {
                None => return Err(Fault::Missing { marker }),
                Some(found) if found != want => {
                    return Err(Fault::Displaced {
                        marker,
                        want,
                        found,
                    });
                }
                Some(_) => {}
            }
        }
        Ok(())
    }
}
