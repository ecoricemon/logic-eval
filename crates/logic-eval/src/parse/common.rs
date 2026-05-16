use crate::Atom;
use core::fmt::{self, Display};

/// Interns parser strings and returns atom-compatible references.
pub trait Intern {
    /// The interned string reference type produced by this interner.
    type Interned<'a>: Atom
    where
        Self: 'a;

    /// Interns a string slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{Intern, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let name = interner.intern_str("sunny");
    ///
    /// assert_eq!(&*name, "sunny");
    /// ```
    fn intern_str(&self, s: &str) -> Self::Interned<'_>;

    /// Formats a value into an interned string using at most `upper_size` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{Intern, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let name = interner.intern_formatted_str(&42, 2).unwrap();
    ///
    /// assert_eq!(&*name, "42");
    /// ```
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

/// Default string interner for parsed text.
pub type StrInterner = any_intern::DroplessInterner;
/// Interned string type produced by [`StrInterner`].
pub type InternedStr<'int> = any_intern::Interned<'int, str>;
