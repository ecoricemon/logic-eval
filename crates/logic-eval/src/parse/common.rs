use crate::Atom;
use core::fmt::{self, Display};

pub trait Intern {
    type Interned<'a>: Atom
    where
        Self: 'a;

    fn intern_str(&self, s: &str) -> Self::Interned<'_>;

    fn intern_formatted_str<T: Display + ?Sized>(
        &self,
        value: &T,
        upper_size: usize,
    ) -> Result<Self::Interned<'_>, fmt::Error>;
}

impl Intern for any_intern::DroplessInterner {
    type Interned<'a>
        = any_intern::Interned<'a, str>
    where
        Self: 'a;

    fn intern_str(&self, s: &str) -> Self::Interned<'_> {
        self.intern(s)
    }

    fn intern_formatted_str<T: Display + ?Sized>(
        &self,
        value: &T,
        upper_size: usize,
    ) -> Result<Self::Interned<'_>, fmt::Error> {
        <Self>::intern_formatted_str(self, value, upper_size)
    }
}

impl Intern for any_intern::Interner {
    type Interned<'a>
        = any_intern::Interned<'a, str>
    where
        Self: 'a;

    fn intern_str(&self, s: &str) -> Self::Interned<'_> {
        self.intern_dropless(s)
    }

    fn intern_formatted_str<T: Display + ?Sized>(
        &self,
        value: &T,
        upper_size: usize,
    ) -> Result<Self::Interned<'_>, fmt::Error> {
        <Self>::intern_formatted_str(self, value, upper_size)
    }
}

pub type StrInterner = any_intern::DroplessInterner;
pub type InternedStr<'int> = any_intern::Interned<'int, str>;
