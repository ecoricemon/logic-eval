#![doc = include_str!("../README.md")]

mod parse;
mod prove;

// === Re-exports ===

pub use parse::{
    inner::VAR_PREFIX,
    inner::{Parse, parse_str},
    repr::{Clause, ClauseDataset, Expr, Predicate, Term},
    text::Name,
};
pub use prove::{
    db::{ClauseIter, ClauseRef, Database},
    prover::ProveCx,
};

pub mod intern {
    pub use any_intern::*;
}

// === Hash map and set used within this crate ===

pub(crate) type Map<K, V> = std::collections::HashMap<K, V, fxhash::FxBuildHasher>;

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

pub(crate) type PassThroughState = std::hash::BuildHasherDefault<PassThroughHasher>;

// === Result/Error used within this crate ===

pub(crate) type Result<T> = std::result::Result<T, Error>;
pub(crate) type Error = Box<dyn std::error::Error + Send + Sync>;
