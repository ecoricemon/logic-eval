#![doc = include_str!("../README.md")]

mod parse;
mod prove;

// === Re-exports ===

pub use parse::{
    inner::VAR_PREFIX,
    inner::{parse_str, Parse},
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

use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    error::Error as StdError,
    fmt::{self, Debug, Display},
    hash::{BuildHasherDefault, Hash, Hasher},
    result::Result as StdResult,
    sync::{Arc, Mutex},
};

// === Hash map and set used within this crate ===

pub(crate) type Map<K, V> = HashMap<K, V, fxhash::FxBuildHasher>;

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

// === Interning dependency ===

pub trait Intern {
    type InternedStr<'a>: AsRef<str> + Borrow<str> + Clone + Eq + Ord + Hash + Debug + Display
    where
        Self: 'a;

    fn intern_formatted_str<T: Display + ?Sized>(
        &self,
        value: &T,
        upper_size: usize,
    ) -> StdResult<Self::InternedStr<'_>, fmt::Error>;

    fn intern_str(&self, text: &str) -> Self::InternedStr<'_> {
        self.intern_formatted_str(text, text.len()).unwrap()
    }
}

type DefaultInternerInner = HashSet<Arc<str>, fxhash::FxBuildHasher>;
pub struct DefaultInterner(Mutex<DefaultInternerInner>);

impl Default for DefaultInterner {
    fn default() -> Self {
        let set = HashSet::default();
        Self(Mutex::new(set))
    }
}

impl Intern for DefaultInterner {
    type InternedStr<'a>
        = Arc<str>
    where
        Self: 'a;

    fn intern_formatted_str<T: Display + ?Sized>(
        &self,
        value: &T,
        _: usize,
    ) -> StdResult<Self::InternedStr<'_>, fmt::Error> {
        let mut set = self.0.lock().unwrap();

        let value = value.to_string();
        if let Some(existing_value) = set.get(&*value) {
            Ok(existing_value.clone())
        } else {
            let value: Arc<str> = value.into();
            set.insert(value.clone());
            Ok(value)
        }
    }
}

impl Intern for any_intern::DroplessInterner {
    type InternedStr<'a>
        = any_intern::Interned<'a, str>
    where
        Self: 'a;

    fn intern_formatted_str<T: Display + ?Sized>(
        &self,
        value: &T,
        upper_size: usize,
    ) -> StdResult<Self::InternedStr<'_>, fmt::Error> {
        self.intern_formatted_str(value, upper_size)
    }

    fn intern_str(&self, text: &str) -> Self::InternedStr<'_> {
        self.intern(text)
    }
}

impl Intern for any_intern::Interner {
    type InternedStr<'a>
        = any_intern::Interned<'a, str>
    where
        Self: 'a;

    fn intern_formatted_str<T: Display + ?Sized>(
        &self,
        value: &T,
        upper_size: usize,
    ) -> StdResult<Self::InternedStr<'_>, fmt::Error> {
        self.intern_formatted_str(value, upper_size)
    }

    fn intern_str(&self, text: &str) -> Self::InternedStr<'_> {
        self.intern_dropless(text)
    }
}

// === Type aliases for complex `Intern` related types ===

pub(crate) use intern_alias::*;
#[allow(unused)]
mod intern_alias {
    use super::{parse, prove, Intern};
    pub(crate) type NameIn<'int, Int> = parse::text::Name<<Int as Intern>::InternedStr<'int>>;
    pub(crate) type TermIn<'int, Int> = parse::repr::Term<NameIn<'int, Int>>;
    pub(crate) type ExprIn<'int, Int> = parse::repr::Expr<NameIn<'int, Int>>;
    pub(crate) type ClauseIn<'int, Int> = parse::repr::Clause<NameIn<'int, Int>>;
    pub(crate) type UniqueTermArrayIn<'int, Int> = prove::repr::UniqueTermArray<NameIn<'int, Int>>;
    pub(crate) type TermStorageIn<'int, Int> = prove::repr::TermStorage<NameIn<'int, Int>>;
    pub(crate) type ClauseDatasetIn<'int, Int> = parse::repr::ClauseDataset<NameIn<'int, Int>>;
    pub(crate) type Name2Int<'int, Int> =
        prove::prover::IdxMap<'int, NameIn<'int, Int>, prove::prover::Integer, Int>;
    pub(crate) type Int2Name<'int, Int> =
        prove::prover::IdxMap<'int, prove::prover::Integer, NameIn<'int, Int>, Int>;
}
