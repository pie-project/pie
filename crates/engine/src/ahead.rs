//! How many more fires a lane's pipeline will send after a given one, when
//! that is known: once the inferlet has closed the pipeline, every fire it
//! will post is posted. An engine that runs ahead of the fires it is given
//! (a device token loop) asks so it stops where the pipeline will.
//!
//! The runtime's pipeline keeps its [`Posts`] (the frames it posted, whether
//! it closed) and every fire it hands an engine carries an [`Ahead`] on its
//! lanes ([`crate::fire::Lane::ahead`]): a share of those posts and the
//! fire's place. The answer is worked out when the engine asks, not when the
//! fire was handed over: a fire is often handed over before its pipeline
//! has closed. Nothing is global; an engine in another process sees `None`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// A pipeline's posted frames and whether it has closed: the runtime's
/// pipeline holds it and every fire it hands an engine carries a share of
/// it ([`Ahead`], on the fire's lanes), so the answer comes from the fire
/// itself and reflects a close that happens after the hand-over.
#[derive(Debug, Default)]
pub struct Posts {
    inner: Mutex<PostsInner>,
}

#[derive(Debug, Default)]
struct PostsInner {
    /// Fires of each frame, by its sequence number.
    frames: BTreeMap<u64, u32>,
    closed: bool,
}

impl Posts {
    fn with<T>(&self, f: impl FnOnce(&mut PostsInner) -> T) -> T {
        f(&mut self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
    }

    /// Frame `seq` of `fires` fires was posted.
    pub fn posted(&self, seq: u64, fires: u32) {
        self.with(|p| {
            p.frames.insert(seq, fires);
        });
    }

    /// The pipeline posts no more.
    pub fn closed(&self) {
        self.with(|p| p.closed = true);
    }
}

/// One fire's place in its pipeline, on each of its lanes
/// ([`crate::fire::Lane::ahead`]): the pipeline's [`Posts`], its frame's
/// sequence number, its slot in the frame and the frame's fires. Not sent
/// across a process boundary (an engine there sees `None`).
#[derive(Clone, Debug)]
pub struct Ahead {
    pub posts: Arc<Posts>,
    pub seq: u64,
    pub slot: u32,
    pub fires: u32,
}

impl PartialEq for Ahead {
    fn eq(&self, other: &Ahead) -> bool {
        Arc::ptr_eq(&self.posts, &other.posts)
            && (self.seq, self.slot, self.fires) == (other.seq, other.slot, other.fires)
    }
}

impl Eq for Ahead {}

impl Ahead {
    /// The fires known to follow this one, once its pipeline has closed
    /// (until then, `None`): the rest of its frame and every later frame.
    #[must_use]
    pub fn fires_left(&self) -> Option<u32> {
        self.posts.with(|p| {
            // Frames before this one have been run.
            p.frames = p.frames.split_off(&self.seq);
            if !p.closed {
                return None;
            }
            let later: u32 = p.frames.range(self.seq + 1..).map(|(_, &fires)| fires).sum();
            Some(later + self.fires.saturating_sub(self.slot + 1))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_left_counts_once_the_pipeline_closes() {
        let posts = Arc::new(Posts::default());
        posts.posted(0, 2);
        let a = Ahead { posts: posts.clone(), seq: 0, slot: 0, fires: 2 };
        assert_eq!(a.fires_left(), None);
        posts.posted(1, 3);
        posts.closed();
        assert_eq!(a.fires_left(), Some(1 + 3));
        let b = Ahead { posts: posts.clone(), seq: 1, slot: 2, fires: 3 };
        assert_eq!(b.fires_left(), Some(0));
        assert_eq!(a.clone(), a);
        assert_ne!(a, Ahead { posts: Arc::new(Posts::default()), seq: 0, slot: 0, fires: 2 });
    }
}
