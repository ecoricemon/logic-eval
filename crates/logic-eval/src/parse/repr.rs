use super::text::Name;
use std::{
    fmt::{self, Write},
    ops,
    vec::IntoIter,
};

#[derive(Clone)]
pub struct ClauseDataset<T>(pub Vec<Clause<T>>);

impl<T> IntoIterator for ClauseDataset<T> {
    type Item = Clause<T>;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T> ops::Deref for ClauseDataset<T> {
    type Target = Vec<Clause<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Clause<T> {
    pub head: Term<T>,
    pub body: Option<Expr<T>>,
}

impl<T> Clause<T> {
    pub(crate) fn map<U, F: FnMut(T) -> U>(self, f: &mut F) -> Clause<U> {
        Clause {
            head: self.head.map(f),
            body: self.body.map(|expr| expr.map(f)),
        }
    }

    pub(crate) fn replace_term<F>(&mut self, f: &mut F)
    where
        F: FnMut(&Term<T>) -> Option<Term<T>>,
    {
        self.head.replace_all(f);
        if let Some(body) = &mut self.body {
            body.replace_term(f);
        }
    }
}

impl<T: fmt::Display> fmt::Display for Clause<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.head.fmt(f)?;
        if let Some(body) = &self.body {
            f.write_str(" :- ")?;
            body.fmt(f)?;
        }
        f.write_char('.')
    }
}

#[derive(Default, Clone, PartialEq, Eq)]
pub struct Term<T> {
    pub functor: T,
    pub args: Box<[Term<T>]>,
}

impl<T> Term<T> {
    pub(crate) fn map<U, F: FnMut(T) -> U>(self, f: &mut F) -> Term<U> {
        Term {
            functor: f(self.functor),
            args: self.args.into_iter().map(|arg| arg.map(f)).collect(),
        }
    }

    pub fn replace_all<F>(&mut self, f: &mut F) -> bool
    where
        F: FnMut(&Term<T>) -> Option<Term<T>>,
    {
        if let Some(new) = f(self) {
            *self = new;
            true
        } else {
            let mut replaced = false;
            for arg in &mut self.args {
                replaced |= arg.replace_all(f);
            }
            replaced
        }
    }
}

impl<T: Clone> Term<T> {
    pub fn predicate(&self) -> Predicate<T> {
        Predicate {
            functor: self.functor.clone(),
            arity: self.args.len() as u32,
        }
    }
}

impl Term<Name<'_>> {
    pub fn is_variable(&self) -> bool {
        let is_variable = self.functor.is_variable();

        #[cfg(debug_assertions)]
        if is_variable {
            assert!(self.args.is_empty());
        }

        is_variable
    }

    pub fn contains_variable(&self) -> bool {
        if self.is_variable() {
            return true;
        }

        self.args.iter().any(|arg| arg.contains_variable())
    }
}

impl<T: fmt::Display> fmt::Display for Term<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.functor.fmt(f)?;
        if !self.args.is_empty() {
            f.write_char('(')?;
            for (i, arg) in self.args.iter().enumerate() {
                arg.fmt(f)?;
                if i + 1 < self.args.len() {
                    f.write_str(", ")?;
                }
            }
            f.write_char(')')?;
        }
        Ok(())
    }
}

impl fmt::Debug for Term<Name<'_>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            self.functor.fmt(f)
        } else {
            let mut d = f.debug_tuple(&self.functor);
            for arg in &self.args {
                d.field(&arg);
            }
            d.finish()
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum Expr<T> {
    Term(Term<T>),
    Not(Box<Expr<T>>),
    And(Vec<Expr<T>>),
    Or(Vec<Expr<T>>),
}

impl<T> Expr<T> {
    pub(crate) fn map<U, F: FnMut(T) -> U>(self, f: &mut F) -> Expr<U> {
        match self {
            Self::Term(v) => Expr::Term(v.map(f)),
            Self::Not(v) => Expr::Not(Box::new(v.map(f))),
            Self::And(v) => Expr::And(v.into_iter().map(|expr| expr.map(f)).collect()),
            Self::Or(v) => Expr::Or(v.into_iter().map(|expr| expr.map(f)).collect()),
        }
    }

    pub(crate) fn replace_term<F>(&mut self, f: &mut F)
    where
        F: FnMut(&Term<T>) -> Option<Term<T>>,
    {
        match self {
            Self::Term(term) => {
                term.replace_all(f);
            }
            Self::Not(inner) => inner.replace_term(f),
            Self::And(args) | Self::Or(args) => {
                for arg in args {
                    arg.replace_term(f);
                }
            }
        }
    }
}

impl<T: fmt::Display> fmt::Display for Expr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Term(term) => term.fmt(f)?,
            Self::Not(inner) => {
                f.write_str("\\+ ")?;
                if matches!(**inner, Self::And(_) | Self::Or(_)) {
                    f.write_char('(')?;
                    inner.fmt(f)?;
                    f.write_char(')')?;
                } else {
                    inner.fmt(f)?;
                }
            }
            Self::And(args) => {
                for (i, arg) in args.iter().enumerate() {
                    if matches!(arg, Self::Or(_)) {
                        f.write_char('(')?;
                        arg.fmt(f)?;
                        f.write_char(')')?;
                    } else {
                        arg.fmt(f)?;
                    }
                    if i + 1 < args.len() {
                        f.write_str(", ")?;
                    }
                }
            }
            Self::Or(args) => {
                for (i, arg) in args.iter().enumerate() {
                    arg.fmt(f)?;
                    if i + 1 < args.len() {
                        f.write_str("; ")?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl fmt::Debug for Expr<Name<'_>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Term(term) => fmt::Debug::fmt(term, f),
            Self::Not(inner) => f.debug_tuple("Not").field(inner).finish(),
            Self::And(args) => {
                let mut d = f.debug_tuple("And");
                for arg in args {
                    d.field(arg);
                }
                d.finish()
            }
            Self::Or(args) => {
                let mut d = f.debug_tuple("Or");
                for arg in args {
                    d.field(arg);
                }
                d.finish()
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Predicate<T> {
    pub functor: T,
    pub arity: u32,
}
