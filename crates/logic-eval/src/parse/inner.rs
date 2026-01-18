use crate::{Error, Result};
use any_intern::{DroplessInterner, Interned};

pub fn parse_str<'cx, T: Parse<'cx>>(gcx: &GlobalCx<'cx>, text: &str) -> Result<T> {
    let mut buf = ParseBuffer::new(text);
    T::parse(&mut buf, gcx)
}

pub trait Parse<'cx>: Sized + 'cx {
    fn parse(buf: &mut ParseBuffer<'_>, gcx: &GlobalCx<'cx>) -> Result<Self>;
}

#[derive(Debug, Clone, Copy)]
pub struct GlobalCx<'a> {
    // By using concrete type here, this crate is strongly coupled with the interner crate. A new
    // trait would good to abstract that, but it makes more complexity over the entire crate.
    // Consider bring the approach when concrete type is not suitable anymore.
    pub interner: &'a DroplessInterner,
}

impl<'a> GlobalCx<'a> {
    pub fn intern_str(&self, s: &str) -> Interned<'a, str> {
        self.interner.intern(s)
    }
}

#[derive(Clone, Copy)]
pub struct ParseBuffer<'a> {
    pub(crate) text: &'a str,
    /// Inclusive
    pub(crate) start: usize,
    /// Exclusive
    pub(crate) end: usize,
}

impl<'a> ParseBuffer<'a> {
    pub const fn new(text: &'a str) -> Self {
        Self {
            text,
            start: 0,
            end: text.len(),
        }
    }

    pub(crate) fn cur_text(&self) -> &str {
        &self.text[self.start..self.end]
    }

    pub(crate) fn parse<'cx, T: Parse<'cx>>(&mut self, gcx: &GlobalCx<'cx>) -> Result<T> {
        T::parse(self, gcx)
    }

    pub(crate) fn peek_parse<'cx, T: Parse<'cx>>(&self, gcx: &GlobalCx<'cx>) -> Option<(T, Self)> {
        let mut peek = *self;
        // FIXME: No need to create error messages, which may cause performance
        // issue.
        T::parse(&mut peek, gcx).ok().map(|t| (t, peek))
    }
}

pub(crate) struct Ident(Location);

impl Ident {
    pub(crate) fn to_text<'a>(&self, whole_text: &'a str) -> &'a str {
        &whole_text[self.0.left..self.0.right]
    }
}

impl Parse<'_> for Ident {
    fn parse(buf: &mut ParseBuffer<'_>, _: &GlobalCx<'_>) -> Result<Self> {
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
        impl<'cx> Parse<'cx> for $ty {
            fn parse(buf: &mut ParseBuffer<'_>, _: &GlobalCx<'cx>) -> Result<Self> {
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
