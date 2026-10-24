//! The collectives of an expert-parallel group whose ranks are threads of
//! one process: each rank posts what it sends, the ranks meet, and each
//! reads every rank's post. A post is read before any rank posts the next
//! one (the ranks meet again after reading), so one exchange carries every
//! collective of every step, in the order the ranks fire them.

use std::sync::{Arc, Barrier, Mutex};

/// What a rank brings to one collective.
#[derive(Clone, Debug)]
pub enum Post {
    /// A dispatch's tokens: `rows` of them, each token's logical routes
    /// and their weights (empty when the experts read none).
    Tokens {
        rows: usize,
        x: Vec<f32>,
        routes: Vec<i32>,
        weights: Vec<f32>,
    },
    /// A combine's expert outputs, a row per landed route, with the
    /// landing's origins and local slots.
    Outputs {
        y: Vec<f32>,
        origin: Vec<i32>,
        local: Vec<i32>,
    },
}

/// The meeting point of `world` ranks.
pub struct Exchange {
    world: usize,
    meet: Barrier,
    posts: Mutex<Vec<Option<Arc<Post>>>>,
}

impl Exchange {
    #[must_use]
    pub fn new(world: usize) -> Arc<Exchange> {
        Arc::new(Exchange {
            world,
            meet: Barrier::new(world),
            posts: Mutex::new(vec![None; world]),
        })
    }

    #[must_use]
    pub fn world(&self) -> usize {
        self.world
    }

    /// Posts `post` as `rank`'s and returns every rank's, in rank order,
    /// once every rank has posted. Every rank calls it once a collective.
    pub fn all(&self, rank: usize, post: Post) -> Vec<Arc<Post>> {
        self.posts.lock().expect("the exchange is poisoned")[rank] = Some(Arc::new(post));
        self.meet.wait();
        let all = (self.posts.lock().expect("the exchange is poisoned").iter())
            .map(|p| p.clone().expect("every rank posted before the ranks met"))
            .collect();
        self.meet.wait();
        all
    }
}
