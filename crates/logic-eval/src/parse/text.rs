use super::{
    CloseParenToken, CommaToken, DotToken, GlobalCx, HornToken, Ident, NegationToken,
    OpenParenToken, Parse, ParseBuffer, VAR_PREFIX,
    repr::{Clause, ClauseDataset, Expr, Term},
};
use crate::{Error, Result};
use any_intern::Interned;
use std::{borrow, fmt, ops};

impl<'cx> Parse<'cx> for ClauseDataset<Name<'cx>> {
    fn parse(buf: &mut ParseBuffer<'_>, gcx: &GlobalCx<'cx>) -> Result<Self> {
        let clauses = buf.parse(gcx)?;
        Ok(Self(clauses))
    }
}

impl<'cx> Parse<'cx> for Vec<Clause<Name<'cx>>> {
    fn parse(buf: &mut ParseBuffer<'_>, gcx: &GlobalCx<'cx>) -> Result<Self> {
        let mut v = Vec::new();
        while let Some((clause, moved_buf)) = buf.peek_parse::<Clause<Name>>(gcx) {
            v.push(clause);
            *buf = moved_buf;
        }
        Ok(v)
    }
}

impl<'cx> Parse<'cx> for Clause<Name<'cx>> {
    fn parse(buf: &mut ParseBuffer<'_>, gcx: &GlobalCx<'cx>) -> Result<Self> {
        let head = buf.parse::<Term<Name>>(gcx)?;

        let body = if let Some((_, mut body_buf)) = buf.peek_parse::<HornToken>(gcx) {
            let dot = body_buf
                .cur_text()
                .find('.')
                .ok_or(Error::from("clause must end with `.`"))?;
            body_buf.end = body_buf.start + dot;
            let body = body_buf.parse::<Expr<Name>>(gcx)?;

            buf.start = body_buf.end + 1; // Next to the dot
            Some(body)
        } else {
            let _ = buf.parse::<DotToken>(gcx)?;
            None
        };

        Ok(Self { head, body })
    }
}

// Precedence: `Paren ()` -> `Not \+` -> `And ,` -> `Or ;`
impl<'cx> Expr<Name<'cx>> {
    fn parse_or(buf: ParseBuffer<'_>, gcx: &GlobalCx<'cx>) -> Result<Self> {
        Self::parse_by_delimiter(buf, gcx, ';', Self::parse_and, Self::Or)
    }

    fn parse_and(buf: ParseBuffer<'_>, gcx: &GlobalCx<'cx>) -> Result<Self> {
        Self::parse_by_delimiter(buf, gcx, ',', Self::parse_not, Self::And)
    }

    fn parse_not(buf: ParseBuffer<'_>, gcx: &GlobalCx<'cx>) -> Result<Self> {
        if let Some((_, moved_buf)) = buf.peek_parse::<NegationToken>(gcx) {
            let inner = Self::parse_not(moved_buf, gcx)?;
            Ok(Self::Not(Box::new(inner)))
        } else {
            Self::parse_paren(buf, gcx)
        }
    }

    fn parse_paren(buf: ParseBuffer<'_>, gcx: &GlobalCx<'cx>) -> Result<Self> {
        if let Some((_, mut moved_buf)) = buf.peek_parse::<OpenParenToken>(gcx) {
            let r = moved_buf.cur_text().rfind(')').unwrap();
            moved_buf.end = moved_buf.start + r;
            moved_buf.parse::<Self>(gcx)
        } else {
            Self::parse_term(buf, gcx)
        }
    }

    fn parse_term(mut buf: ParseBuffer<'_>, gcx: &GlobalCx<'cx>) -> Result<Self> {
        if buf.cur_text().chars().all(|c: char| c.is_whitespace()) {
            Err("expected a non-empty term expression".into())
        } else {
            buf.parse::<Term<Name>>(gcx).map(Self::Term)
        }
    }

    fn parse_by_delimiter(
        buf: ParseBuffer<'_>,
        gcx: &GlobalCx<'cx>,
        del: char,
        parse_partial: fn(ParseBuffer<'_>, &GlobalCx<'cx>) -> Result<Self>,
        wrap_vec: fn(Vec<Self>) -> Self,
    ) -> Result<Self> {
        let mut v = Vec::new();
        let (mut l, mut r) = (buf.start, buf.start);
        let mut open = 0;

        for c in buf.cur_text().chars() {
            if c == del && open == 0 {
                let buf = ParseBuffer {
                    text: buf.text,
                    start: l,
                    end: r,
                };
                let partial = parse_partial(buf, gcx)?;
                v.push(partial);
                l = r + c.len_utf8();
            } else if c == '(' {
                open += 1;
            } else if c == ')' {
                open -= 1;
            }
            r += c.len_utf8();
        }

        let buf = ParseBuffer {
            text: buf.text,
            start: l,
            end: r,
        };
        let last = parse_partial(buf, gcx)?;

        if !v.is_empty() {
            v.push(last);
            Ok(wrap_vec(v))
        } else {
            Ok(last)
        }
    }
}

impl<'cx> Parse<'cx> for Expr<Name<'cx>> {
    fn parse(buf: &mut ParseBuffer<'_>, gcx: &GlobalCx<'cx>) -> Result<Self> {
        // The buffer range is not being moved by this call.
        Self::parse_or(*buf, gcx)
    }
}

impl<'cx> Term<Name<'cx>> {
    fn parse_args(
        buf: &mut ParseBuffer<'_>,
        gcx: &GlobalCx<'cx>,
    ) -> Result<Box<[Term<Name<'cx>>]>> {
        let mut open = 0;
        while let Some((_, moved_buf)) = buf.peek_parse::<OpenParenToken>(gcx) {
            *buf = moved_buf;
            open += 1;
        }

        if open == 0 {
            return Ok([].into());
        }

        let mut args = Vec::new();
        let mut well_seperated = true;
        loop {
            if let Some((_, moved_buf)) = buf.peek_parse::<CloseParenToken>(gcx) {
                *buf = moved_buf;
                open -= 1;
                break;
            }

            if !well_seperated {
                return Err(format!("expected `,` from {}", buf.cur_text()).into());
            }

            let arg = buf.parse::<Term<Name>>(gcx)?;
            args.push(arg);

            let comma = buf.parse::<CommaToken>(gcx);
            well_seperated = comma.is_ok();
        }

        for _ in 0..open {
            buf.parse::<CloseParenToken>(gcx)?;
        }

        Ok(args.into())
    }
}

impl<'cx> Parse<'cx> for Term<Name<'cx>> {
    fn parse(buf: &mut ParseBuffer<'_>, gcx: &GlobalCx<'cx>) -> Result<Self> {
        let functor = Name::parse(buf, gcx)?;
        let args = Self::parse_args(buf, gcx)?;
        Ok(Self { functor, args })
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Name<'cx>(Interned<'cx, str>); // Non-empty string

impl<'cx> Name<'cx> {
    pub(crate) fn create(gcx: &GlobalCx<'cx>, s: &str) -> Self {
        debug_assert!(!s.is_empty());

        let interned = gcx.intern_str(s);
        Self(interned)
    }

    pub(crate) fn is_variable(&self) -> bool {
        let first = self.0.chars().next().unwrap();
        first == VAR_PREFIX // btw, prolog style is `is_uppercase() or '_'`
    }
}

impl<'cx> Parse<'cx> for Name<'cx> {
    fn parse(buf: &mut ParseBuffer<'_>, gcx: &GlobalCx<'cx>) -> Result<Self> {
        let ident = buf.parse::<Ident>(gcx)?;
        let interned = gcx.intern_str(ident.to_text(buf.text));
        debug_assert!(!interned.is_empty());
        Ok(Self(interned))
    }
}

impl fmt::Display for Name<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl fmt::Debug for Name<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<'cx> ops::Deref for Name<'cx> {
    type Target = Interned<'cx, str>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<str> for Name<'_> {
    fn as_ref(&self) -> &str {
        self
    }
}

impl borrow::Borrow<str> for Name<'_> {
    fn borrow(&self) -> &str {
        self
    }
}
