#![doc = include_str!("../README.md")]

mod parse;
mod prove;

// === Re-exports ===

pub use parse::{
    common::{Intern, InternedStr, StrInterner},
    inner::VAR_PREFIX,
    inner::{parse_str, Parse},
    repr::{Clause, ClauseDataset, Expr, Predicate, Term},
    text::Name,
};
pub use prove::{
    common::Atom,
    db::{ClauseIter, ClauseRef, Database},
    prover::ProveCx,
};

/// Re-exports of the interning types used by this crate.
pub mod intern {
    pub use any_intern::*;
}

// === Re-exports for this crate ===

pub(crate) use intern_alias::*;
#[allow(unused)]
mod intern_alias {
    use crate::{parse, prove, Intern};
    pub(crate) type NameIn<'int, Int> = parse::text::Name<<Int as Intern>::Interned<'int>>;
    pub(crate) type TermIn<'int, Int> = parse::repr::Term<NameIn<'int, Int>>;
    pub(crate) type ExprIn<'int, Int> = parse::repr::Expr<NameIn<'int, Int>>;
    pub(crate) type ClauseIn<'int, Int> = parse::repr::Clause<NameIn<'int, Int>>;
    pub(crate) type UniqueTermArrayIn<'int, Int> = prove::repr::UniqueTermArray<NameIn<'int, Int>>;
    pub(crate) type TermStorageIn<'int, Int> = prove::repr::TermStorage<NameIn<'int, Int>>;
    pub(crate) type ClauseDatasetIn<'int, Int> = parse::repr::ClauseDataset<NameIn<'int, Int>>;
}

pub(crate) type Map<K, V> = fxhash::FxHashMap<K, V>;
pub(crate) type IndexMap<K, V> = indexmap::IndexMap<K, V, fxhash::FxBuildHasher>;
pub(crate) type IndexSet<T> = indexmap::IndexSet<T, fxhash::FxBuildHasher>;
pub(crate) type PassThroughIndexMap<K, V> = indexmap::IndexMap<K, V, PassThroughState>;

use std::{
    error::Error as StdError,
    hash::{BuildHasherDefault, Hasher},
    result::Result as StdResult,
};

#[derive(Default, Clone, Copy)]
struct PassThroughHasher {
    hash: u64,
}

impl Hasher for PassThroughHasher {
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

pub(crate) type PassThroughState = BuildHasherDefault<PassThroughHasher>;

// === Result/Error used within this crate ===

pub(crate) type Result<T> = StdResult<T, Error>;
pub(crate) type Error = Box<dyn StdError + Send + Sync>;
