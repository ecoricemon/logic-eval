use crate::{Name, VAR_PREFIX};
use core::hash::Hash;

/// Atom type used in parsed and proved terms.
pub trait Atom: Clone + Eq + Hash {
    /// Returns `true` when this atom represents a variable.
    fn is_variable(&self) -> bool;
}

impl<'int> Atom for any_intern::Interned<'int, str> {
    fn is_variable(&self) -> bool {
        self.starts_with(VAR_PREFIX)
    }
}

impl<'int> Atom for Name<any_intern::Interned<'int, str>> {
    fn is_variable(&self) -> bool {
        self.starts_with(VAR_PREFIX)
    }
}
