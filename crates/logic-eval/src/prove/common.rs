use crate::{Name, VAR_PREFIX};
use core::hash::Hash;

pub trait Atom: Clone + Eq + Hash {
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
