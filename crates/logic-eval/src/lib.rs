#![doc = include_str!("../README.md")]

mod parse;
mod prove;

// === Re-exports ===

pub use logic_eval_util::str::Str;
pub use parse::{
    inner::VAR_PREFIX,
    inner::{Parse, parse_str},
    repr::{Clause, ClauseDataset, Expr, Predicate, Term},
    text::Name,
};
pub use prove::{
    db::{ClauseRef, ClauseIter, Database},
    prover::ProveCx,
};

// === Hash map and set used within this crate ===

#[cfg(not(test))]
pub(crate) type Map<K, V> = std::collections::HashMap<K, V, ahash::RandomState>;

#[cfg(test)]
pub(crate) type Map<K, V> = std::collections::HashMap<K, V, FixedState>;

#[cfg(test)]
#[derive(Default, Clone, Copy)]
struct FixedState;

#[cfg(test)]
impl std::hash::BuildHasher for FixedState {
    type Hasher = std::hash::DefaultHasher;
    fn build_hasher(&self) -> Self::Hasher {
        std::hash::DefaultHasher::new()
    }
}

#[derive(Default, Clone, Copy)]
struct PassThroughHasher {
    hash: u64,
}

impl std::hash::Hasher for PassThroughHasher {
    fn write(&mut self, _bytes: &[u8]) {
        panic!("u64 is only allowed");
    }

    fn write_u64(&mut self, i: u64) {
        self.hash = i;
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

pub(crate) type NoHashState = std::hash::BuildHasherDefault<PassThroughHasher>;

pub(crate) type Result<T> = std::result::Result<T, Error>;
pub(crate) type Error = Box<dyn std::error::Error + Send + Sync>;
