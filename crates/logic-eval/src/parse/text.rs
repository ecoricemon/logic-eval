use super::{
    repr::{Clause, Expr, Term},
    CloseParenToken, CommaToken, DotToken, HornToken, Ident, NegationToken, OpenParenToken, Parse,
    ParseBuffer, VAR_PREFIX,
};
use crate::{ClauseDatasetIn, ClauseIn, Error, ExprIn, Intern, NameIn, Result, TermIn};
use std::{
    borrow::Borrow,
    fmt::{self, Debug, Display},
    hash::Hash,
    ops,
};

impl<'int, Int: Intern> Parse<'int, Int> for ClauseDatasetIn<'int, Int> {
    fn parse(buf: &mut ParseBuffer<'_>, interner: &'int Int) -> Result<Self> {
        let clauses = buf.parse(interner)?;
        Ok(Self(clauses))
    }
}

impl<'int, Int: Intern> Parse<'int, Int> for Vec<ClauseIn<'int, Int>> {
    fn parse(buf: &mut ParseBuffer<'_>, interner: &'int Int) -> Result<Self> {
        let mut v = Vec::new();
        while let Some((clause, moved_buf)) = buf.peek_parse::<Int, Clause<Name<_>>>(interner) {
            v.push(clause);
            *buf = moved_buf;
        }
        Ok(v)
    }
}

impl<'int, Int: Intern> Parse<'int, Int> for ClauseIn<'int, Int> {
    fn parse(buf: &mut ParseBuffer<'_>, interner: &'int Int) -> Result<Self> {
        let head = buf.parse::<Int, Term<Name<_>>>(interner)?;

        let body = if let Some((_, mut body_buf)) = buf.peek_parse::<Int, HornToken>(interner) {
            let dot = body_buf
                .cur_text()
                .find('.')
                .ok_or(Error::from("clause must end with `.`"))?;
            body_buf.end = body_buf.start + dot;
            let body = body_buf.parse::<Int, Expr<Name<_>>>(interner)?;

            buf.start = body_buf.end + 1; // Next to the dot
            Some(body)
        } else {
            let _ = buf.parse::<Int, DotToken>(interner)?;
            None
        };

        Ok(Self { head, body })
    }
}

// Precedence: `Paren ()` -> `Not \+` -> `And ,` -> `Or ;`
impl Expr<Name<()>> {
    fn parse_or<'int, Int: Intern>(
        buf: ParseBuffer<'_>,
        interner: &'int Int,
    ) -> Result<ExprIn<'int, Int>> {
        Self::parse_by_delimiter(buf, interner, ';', Self::parse_and, Expr::Or)
    }

    fn parse_and<'int, Int: Intern>(
        buf: ParseBuffer<'_>,
        interner: &'int Int,
    ) -> Result<ExprIn<'int, Int>> {
        Self::parse_by_delimiter(buf, interner, ',', Self::parse_not, Expr::And)
    }

    fn parse_not<'int, Int: Intern>(
        buf: ParseBuffer<'_>,
        interner: &'int Int,
    ) -> Result<ExprIn<'int, Int>> {
        if let Some((_, moved_buf)) = buf.peek_parse::<Int, NegationToken>(interner) {
            let inner = Self::parse_not(moved_buf, interner)?;
            Ok(Expr::Not(Box::new(inner)))
        } else {
            Self::parse_paren(buf, interner)
        }
    }

    fn parse_paren<'int, Int: Intern>(
        buf: ParseBuffer<'_>,
        interner: &'int Int,
    ) -> Result<ExprIn<'int, Int>> {
        if let Some((_, mut moved_buf)) = buf.peek_parse::<Int, OpenParenToken>(interner) {
            let r = moved_buf.cur_text().rfind(')').unwrap();
            moved_buf.end = moved_buf.start + r;
            moved_buf.parse::<Int, Expr<Name<_>>>(interner)
        } else {
            Self::parse_term(buf, interner)
        }
    }

    fn parse_term<'int, Int: Intern>(
        mut buf: ParseBuffer<'_>,
        interner: &'int Int,
    ) -> Result<ExprIn<'int, Int>> {
        if buf.cur_text().chars().all(|c: char| c.is_whitespace()) {
            Err("expected a non-empty term expression".into())
        } else {
            buf.parse::<Int, Term<Name<_>>>(interner).map(Expr::Term)
        }
    }

    fn parse_by_delimiter<'int, Int: Intern>(
        buf: ParseBuffer<'_>,
        interner: &'int Int,
        del: char,
        parse_partial: fn(ParseBuffer<'_>, &'int Int) -> Result<ExprIn<'int, Int>>,
        wrap_vec: fn(Vec<ExprIn<'int, Int>>) -> ExprIn<'int, Int>,
    ) -> Result<ExprIn<'int, Int>> {
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
                let partial = parse_partial(buf, interner)?;
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
        let last = parse_partial(buf, interner)?;

        if !v.is_empty() {
            v.push(last);
            Ok(wrap_vec(v))
        } else {
            Ok(last)
        }
    }
}

impl<'int, Int: Intern> Parse<'int, Int> for ExprIn<'int, Int> {
    fn parse(buf: &mut ParseBuffer<'_>, interner: &'int Int) -> Result<Self> {
        // The buffer range is not being moved by this call.
        Expr::parse_or(*buf, interner)
    }
}

impl Term<Name<()>> {
    fn parse_args<'int, Int: Intern>(
        buf: &mut ParseBuffer<'_>,
        interner: &'int Int,
    ) -> Result<Vec<TermIn<'int, Int>>> {
        let mut open = 0;
        while let Some((_, moved_buf)) = buf.peek_parse::<Int, OpenParenToken>(interner) {
            *buf = moved_buf;
            open += 1;
        }

        if open == 0 {
            return Ok([].into());
        }

        let mut args = Vec::new();
        let mut well_seperated = true;
        loop {
            if let Some((_, moved_buf)) = buf.peek_parse::<Int, CloseParenToken>(interner) {
                *buf = moved_buf;
                open -= 1;
                break;
            }

            if !well_seperated {
                return Err(format!("expected `,` from {}", buf.cur_text()).into());
            }

            let arg = buf.parse::<Int, Term<Name<_>>>(interner)?;
            args.push(arg);

            let comma = buf.parse::<Int, CommaToken>(interner);
            well_seperated = comma.is_ok();
        }

        for _ in 0..open {
            buf.parse::<Int, CloseParenToken>(interner)?;
        }

        Ok(args)
    }
}

impl<'int, Int: Intern> Parse<'int, Int> for TermIn<'int, Int> {
    fn parse(buf: &mut ParseBuffer<'_>, interner: &'int Int) -> Result<Self> {
        let functor = Name::parse(buf, interner)?;
        let args = Term::parse_args(buf, interner)?;
        Ok(Self { functor, args })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Name<T>(pub T); // Non-empty string

impl Name<()> {
    pub fn with_intern<'int, Int: Intern>(s: &str, interner: &'int Int) -> NameIn<'int, Int> {
        debug_assert!(!s.is_empty());

        let interned = interner.intern_str(s);
        Name(interned)
    }
}

impl<T: AsRef<str>> Name<T> {
    pub(crate) fn is_variable(&self) -> bool {
        let first = self.0.as_ref().chars().next().unwrap();
        first == VAR_PREFIX // btw, prolog style is `is_uppercase() or '_'`
    }
}

impl<'int, Int: Intern> Parse<'int, Int> for NameIn<'int, Int> {
    fn parse(buf: &mut ParseBuffer<'_>, interner: &'int Int) -> Result<Self> {
        let ident = buf.parse::<Int, Ident>(interner)?;
        let interned = interner.intern_str(ident.to_text(buf.text));
        debug_assert!(!interned.as_ref().is_empty());
        Ok(Self(interned))
    }
}

impl<T> Borrow<T> for Name<T> {
    fn borrow(&self) -> &T {
        &self.0
    }
}

impl<T: Borrow<str>> Borrow<str> for Name<T> {
    fn borrow(&self) -> &str {
        self.0.borrow()
    }
}

impl<T: AsRef<str>> AsRef<str> for Name<T> {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

impl<T: PartialEq<str>> PartialEq<str> for Name<T> {
    fn eq(&self, other: &str) -> bool {
        self.0.eq(other)
    }
}

impl<T: PartialOrd<str>> PartialOrd<str> for Name<T> {
    fn partial_cmp(&self, other: &str) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

impl<T> ops::Deref for Name<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Display> Display for Name<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: Debug> Debug for Name<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
