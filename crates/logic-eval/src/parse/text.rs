use super::{
    CloseParenToken, CommaToken, DotToken, HornToken, Ident, NegationToken, OpenParenToken, Parse,
    ParseBuffer, VAR_PREFIX,
    repr::{Clause, ClauseDataset, Expr, Term},
};
use crate::{Error, Result, Str};
use std::{borrow, fmt, ops};

impl Parse for ClauseDataset<Name> {
    fn parse(buf: &mut ParseBuffer<'_>) -> Result<Self> {
        let clauses = buf.parse()?;
        Ok(Self(clauses))
    }
}

impl Parse for Vec<Clause<Name>> {
    fn parse(buf: &mut ParseBuffer<'_>) -> Result<Self> {
        let mut v = Vec::new();
        while let Some((clause, moved_buf)) = buf.peek_parse::<Clause<Name>>() {
            v.push(clause);
            *buf = moved_buf;
        }
        Ok(v)
    }
}

impl Parse for Clause<Name> {
    fn parse(buf: &mut ParseBuffer<'_>) -> Result<Self> {
        let head = buf.parse::<Term<Name>>()?;

        let body = if let Some((_, mut body_buf)) = buf.peek_parse::<HornToken>() {
            let dot = body_buf
                .cur_text()
                .find('.')
                .ok_or(Error::from("clause must end with `.`"))?;
            body_buf.end = body_buf.start + dot;
            let body = body_buf.parse::<Expr<Name>>()?;

            buf.start = body_buf.end + 1; // Next to the dot
            Some(body)
        } else {
            let _ = buf.parse::<DotToken>()?;
            None
        };

        Ok(Self { head, body })
    }
}

// Precedence: `Paren ()` -> `Not \+` -> `And ,` -> `Or ;`
impl Expr<Name> {
    fn parse_or(buf: ParseBuffer<'_>) -> Result<Self> {
        Self::parse_by_delimiter(buf, ';', Self::parse_and, Self::Or)
    }

    fn parse_and(buf: ParseBuffer<'_>) -> Result<Self> {
        Self::parse_by_delimiter(buf, ',', Self::parse_not, Self::And)
    }

    fn parse_not(buf: ParseBuffer<'_>) -> Result<Self> {
        if let Some((_, moved_buf)) = buf.peek_parse::<NegationToken>() {
            let inner = Self::parse_not(moved_buf)?;
            Ok(Self::Not(Box::new(inner)))
        } else {
            Self::parse_paren(buf)
        }
    }

    fn parse_paren(buf: ParseBuffer<'_>) -> Result<Self> {
        if let Some((_, mut moved_buf)) = buf.peek_parse::<OpenParenToken>() {
            let r = moved_buf.cur_text().rfind(')').unwrap();
            moved_buf.end = moved_buf.start + r;
            moved_buf.parse::<Self>()
        } else {
            Self::parse_term(buf)
        }
    }

    fn parse_term(mut buf: ParseBuffer<'_>) -> Result<Self> {
        if buf.cur_text().chars().all(|c: char| c.is_whitespace()) {
            Err("expected a non-empty term expression".into())
        } else {
            buf.parse::<Term<Name>>().map(Self::Term)
        }
    }

    fn parse_by_delimiter(
        buf: ParseBuffer<'_>,
        del: char,
        parse_partial: fn(ParseBuffer<'_>) -> Result<Self>,
        wrap_vec: fn(Vec<Self>) -> Self,
    ) -> Result<Self> {
        let mut v = Vec::new();
        let (mut l, mut r) = (buf.start, buf.start);
        let mut open = 0;

        for c in buf.cur_text().chars() {
            if c == del && open == 0 {
                let partial = parse_partial(ParseBuffer {
                    text: buf.text,
                    start: l,
                    end: r,
                })?;
                v.push(partial);
                l = r + c.len_utf8();
            } else if c == '(' {
                open += 1;
            } else if c == ')' {
                open -= 1;
            }
            r += c.len_utf8();
        }

        let last = parse_partial(ParseBuffer {
            text: buf.text,
            start: l,
            end: r,
        })?;

        if !v.is_empty() {
            v.push(last);
            Ok(wrap_vec(v))
        } else {
            Ok(last)
        }
    }
}

impl Parse for Expr<Name> {
    fn parse(buf: &mut ParseBuffer<'_>) -> Result<Self> {
        // The buffer range is not being moved by this call.
        Self::parse_or(*buf)
    }
}

impl Term<Name> {
    fn parse_args(buf: &mut ParseBuffer<'_>) -> Result<Box<[Term<Name>]>> {
        let mut open = 0;
        while let Some((_, moved_buf)) = buf.peek_parse::<OpenParenToken>() {
            *buf = moved_buf;
            open += 1;
        }

        if open == 0 {
            return Ok([].into());
        }

        let mut args = Vec::new();
        let mut well_seperated = true;
        loop {
            if let Some((_, moved_buf)) = buf.peek_parse::<CloseParenToken>() {
                *buf = moved_buf;
                open -= 1;
                break;
            }

            if !well_seperated {
                return Err(format!("expected `,` from {}", buf.cur_text()).into());
            }

            let arg = buf.parse::<Term<Name>>()?;
            args.push(arg);

            let comma = buf.parse::<CommaToken>();
            well_seperated = comma.is_ok();
        }

        for _ in 0..open {
            buf.parse::<CloseParenToken>()?;
        }

        Ok(args.into())
    }
}

impl Parse for Term<Name> {
    fn parse(buf: &mut ParseBuffer<'_>) -> Result<Self> {
        let functor = Name::parse(buf)?;
        let args = Self::parse_args(buf)?;
        Ok(Self { functor, args })
    }
}

#[derive(Default, Clone, PartialEq, Eq, Hash)]
pub struct Name(Str /* Non-empty string */);

impl Name {
    pub(crate) fn is_variable(&self) -> bool {
        let first = self.0.chars().next().unwrap();
        first == VAR_PREFIX // btw, prolog style is `is_uppercase() or '_'`
    }
}

impl Parse for Name {
    fn parse(buf: &mut ParseBuffer<'_>) -> Result<Self> {
        let ident = buf.parse::<Ident>()?;
        let s: Str = ident.to_text(buf.text).into();
        debug_assert!(!s.is_empty());
        Ok(Self(s))
    }
}

impl<T: AsRef<str> + ?Sized> PartialEq<T> for Name {
    fn eq(&self, other: &T) -> bool {
        (**self).eq(other.as_ref())
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl ops::Deref for Name {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl borrow::Borrow<str> for Name {
    fn borrow(&self) -> &str {
        self
    }
}

impl<T: Into<Str>> From<T> for Name {
    fn from(value: T) -> Self {
        Self(value.into())
    }
}
