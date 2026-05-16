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
    /// Parses a value from `buf` using `interner`.
    fn parse(buf: &mut ParseBuffer<'_>, interner: &'int Int) -> Result<Self>;
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
        // FIXME: No need to create error messages, which may cause performance issue.
        T::parse(&mut peek, interner).ok().map(|t| (t, peek))
    }
}

pub(crate) struct Ident(Location);

impl Ident {
    pub(crate) fn to_text<'a>(&self, whole_text: &'a str) -> &'a str {
        &whole_text[self.0.left..self.0.right]
    }
}

impl<Int: Intern> Parse<'_, Int> for Ident {
    fn parse(buf: &mut ParseBuffer<'_>, _: &'_ Int) -> Result<Self> {
        fn is_allowed_first(c: char) -> bool {
            c.is_alphabetic() || !(c.is_whitespace() || RESERVED.contains(&c))
        }

        fn is_allowed_rest(c: char) -> bool {
            c.is_alphanumeric() || !(c.is_whitespace() || RESERVED.contains(&c) || c == VAR_PREFIX)
        }

        let s = buf.cur_text();

        let Some(l) = s.find(|c: char| !c.is_whitespace()) else {
            return Err("expected an ident, but input is empty".into());
        };

        let mut r = l;

        let first = s[l..].chars().next().unwrap();
        if is_allowed_first(first) {
            r += first.len_utf8();
        } else {
            return Err(format!("expected an ident from {}", s).into());
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
        Ok(Ident(loc))
    }
}

macro_rules! impl_parse_for_string {
    ($str:literal, $ty:ident) => {
        impl<Int: Intern> Parse<'_, Int> for $ty {
            fn parse(buf: &mut ParseBuffer<'_>, _: &Int) -> Result<Self> {
                let s = buf.cur_text();

                let Some(l) = s.find(|c: char| !c.is_whitespace()) else {
                    return Err(format!("expected `{}` from `{}`", $str, s).into());
                };
                let r = l + $str.len();

                let substr = s
                    .get(l..r)
                    .ok_or(Error::from(format!("expected `{}` from `{s}`", $str)))?;

                if substr == $str {
                    let loc = Location {
                        left: buf.start + l,
                        right: buf.start + r,
                    };
                    buf.start += r;
                    Ok($ty { _loc: loc })
                } else {
                    Err(format!("expected `{}` from `{s}`", $str).into())
                }
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
    use crate::parse::repr::Clause;

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
}
