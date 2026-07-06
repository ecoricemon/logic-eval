use crate::{Error, Intern, Result};

/// Parses `text` as `T` using the provided interner.
///
/// # Examples
///
/// ```
/// use logic_eval::{parse_str, Clause, StrInterner};
///
/// let interner = StrInterner::new();
/// let clause: Clause<_> = parse_str("parent(alice, bob).", &interner).unwrap();
///
/// assert_eq!(clause.to_string(), "parent(alice, bob).");
/// ```
pub fn parse_str<'int, Int, T>(text: &str, interner: &'int Int) -> Result<T>
where
    Int: Intern,
    T: Parse<'int, Int>,
{
    let mut buf = ParseBuffer::new(text);
    T::parse(&mut buf, interner)
}

/// A parser for values that may borrow interned strings.
pub trait Parse<'int, Int: Intern>: Sized + 'int {
    /// Tries to parse a value without constructing an error on mismatch.
    ///
    /// Implementations must leave `buf` unchanged when returning `None`.
    fn try_parse(_buf: &mut ParseBuffer<'_>, _interner: &'int Int) -> Option<Self>;

    /// Creates the error reported by [`Self::parse`] when parsing fails.
    fn expected_error(buf: &ParseBuffer<'_>) -> Error {
        format!("failed to parse from {}", buf.cur_text()).into()
    }

    /// Parses a value from `buf` using `interner`.
    fn parse(buf: &mut ParseBuffer<'_>, interner: &'int Int) -> Result<Self> {
        Self::try_parse(buf, interner).ok_or_else(|| Self::expected_error(buf))
    }
}

/// A cursor over a bounded slice of parser input.
#[derive(Clone, Copy)]
pub struct ParseBuffer<'a> {
    pub(crate) text: &'a str,
    /// Inclusive
    pub(crate) start: usize,
    /// Exclusive
    pub(crate) end: usize,
}

impl<'a> ParseBuffer<'a> {
    /// Creates a buffer that spans the whole input string.
    pub(crate) const fn new(text: &'a str) -> Self {
        Self {
            text,
            start: 0,
            end: text.len(),
        }
    }

    pub(crate) fn cur_text(&self) -> &str {
        &self.text[self.start..self.end]
    }

    pub(crate) fn parse<'int, Int, T>(&mut self, interner: &'int Int) -> Result<T>
    where
        Int: Intern,
        T: Parse<'int, Int>,
    {
        T::parse(self, interner)
    }

    pub(crate) fn peek_parse<'int, Int, T>(&self, interner: &'int Int) -> Option<(T, Self)>
    where
        Int: Intern,
        T: Parse<'int, Int>,
    {
        let mut peek = *self;
        T::try_parse(&mut peek, interner).map(|t| (t, peek))
    }
}

pub(crate) struct Ident(Location);

impl Ident {
    pub(crate) fn to_text<'a>(&self, whole_text: &'a str) -> &'a str {
        &whole_text[self.0.left..self.0.right]
    }
}

impl<Int: Intern> Parse<'_, Int> for Ident {
    fn try_parse(buf: &mut ParseBuffer<'_>, _: &'_ Int) -> Option<Self> {
        fn is_allowed_first(c: char) -> bool {
            c.is_alphabetic() || !(c.is_whitespace() || RESERVED.contains(&c))
        }

        fn is_allowed_rest(c: char) -> bool {
            c.is_alphanumeric() || !(c.is_whitespace() || RESERVED.contains(&c) || c == VAR_PREFIX)
        }

        let s = buf.cur_text();

        let l = s.find(|c: char| !c.is_whitespace())?;

        let mut r = l;

        let first = s[l..].chars().next().unwrap();
        if is_allowed_first(first) {
            r += first.len_utf8();
        } else {
            return None;
        }

        for rest in s[l..].chars().skip(1) {
            if is_allowed_rest(rest) {
                r += rest.len_utf8();
            } else {
                break;
            }
        }

        let loc = Location {
            left: buf.start + l,
            right: buf.start + r,
        };
        buf.start += r;
        Some(Ident(loc))
    }

    fn expected_error(buf: &ParseBuffer<'_>) -> Error {
        let s = buf.cur_text();
        if s.chars().all(char::is_whitespace) {
            "expected an ident, but input is empty".into()
        } else {
            format!("expected an ident from {s}").into()
        }
    }
}

macro_rules! impl_parse_for_string {
    ($str:literal, $ty:ident) => {
        impl<Int: Intern> Parse<'_, Int> for $ty {
            fn try_parse(buf: &mut ParseBuffer<'_>, _: &Int) -> Option<Self> {
                let s = buf.cur_text();

                let l = s.find(|c: char| !c.is_whitespace())?;
                let r = l + $str.len();

                let substr = s.get(l..r)?;

                if substr == $str {
                    let loc = Location {
                        left: buf.start + l,
                        right: buf.start + r,
                    };
                    buf.start += r;
                    Some($ty { _loc: loc })
                } else {
                    None
                }
            }

            fn expected_error(buf: &ParseBuffer<'_>) -> Error {
                format!("expected `{}` from `{}`", $str, buf.cur_text()).into()
            }
        }
    };
}

/// Prefix used to mark variables in parsed terms.
pub const VAR_PREFIX: char = '$';

pub(crate) const RESERVED: &[char] = &[
    ',',  // And
    ';',  // Or
    '.',  // End of a clause
    ':',  // Part of a 'is implied by'
    '-',  // Part of a 'is implied by'
    '\\', // Part of a not
    '+',  // Part of a not
    '(',  // Grouping terms
    ')',  // Grouping terms
    '[',  // List
    ']',  // List
    '|',  // Seperating head and tail in a list
    '+',  // Arithmetic operator
    '-',  // Arithmetic operator
    '*',  // Arithmetic operator
    '/',  // Arithmetic operator
];

pub(crate) struct CommaToken {
    pub(crate) _loc: Location,
}
impl_parse_for_string!(",", CommaToken);

pub(crate) struct DotToken {
    pub(crate) _loc: Location,
}
impl_parse_for_string!(".", DotToken);

pub(crate) struct HornToken {
    pub(crate) _loc: Location,
}
impl_parse_for_string!(":-", HornToken);

pub(crate) struct NegationToken {
    pub(crate) _loc: Location,
}
impl_parse_for_string!("\\+", NegationToken);

pub(crate) struct OpenParenToken {
    pub(crate) _loc: Location,
}
impl_parse_for_string!("(", OpenParenToken);

pub(crate) struct CloseParenToken {
    pub(crate) _loc: Location,
}
impl_parse_for_string!(")", CloseParenToken);

#[derive(Debug, Clone, Copy)]
pub(crate) struct Location {
    /// Inclusive
    left: usize,
    /// Exclusive
    right: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse::repr::Clause, ClauseIn, TermIn};

    type Interner = any_intern::DroplessInterner;

    #[test]
    fn test_parse() {
        fn assert(text: &str, interner: &Interner) {
            let clause: Clause<_> = parse_str(text, interner).unwrap();
            assert_eq!(text, clause.to_string());
        }

        let interner = Interner::new();

        assert("f.", &interner);
        assert("f(a, b).", &interner);
        assert("f(a, b) :- f.", &interner);
        assert("f(a, b) :- f(a).", &interner);
        assert("f(a, b) :- f(a), f(b).", &interner);
        assert("f(a, b) :- f(a); f(b).", &interner);
        assert("f(a, b) :- f(a), (f(b); f(c)).", &interner);
    }

    #[test]
    fn clause_requires_trailing_dot() {
        let interner = Interner::new();
        let mut buf = ParseBuffer::new("f");
        let original = buf;

        assert!(ClauseIn::<'_, Interner>::try_parse(&mut buf, &interner).is_none());
        assert_eq!(buf.start, original.start);
        assert_eq!(buf.end, original.end);
    }

    #[test]
    fn try_parse_failure_preserves_buffer() {
        fn assert_unchanged<'int, T>(text: &str, interner: &'int Interner)
        where
            T: Parse<'int, Interner>,
        {
            let mut buf = ParseBuffer::new(text);
            let original = buf;
            assert!(T::try_parse(&mut buf, interner).is_none());
            assert_eq!(buf.start, original.start);
            assert_eq!(buf.end, original.end);
        }

        let interner = Interner::new();

        assert_unchanged::<CommaToken>("foo", &interner);
        assert_unchanged::<Ident>(",", &interner);
        assert_unchanged::<TermIn<'_, Interner>>("foo(", &interner);
        assert_unchanged::<ClauseIn<'_, Interner>>("foo(", &interner);
    }
}
