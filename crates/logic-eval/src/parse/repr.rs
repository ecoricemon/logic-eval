use super::text::Name;
use crate::Atom;
use std::{
    fmt::{self, Debug, Display, Write},
    ops,
    vec::IntoIter,
};

#[derive(Clone, Debug, Hash)]
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Clause<T> {
    pub head: Term<T>,
    pub body: Option<Expr<T>>,
}

impl<T> Clause<T> {
    pub fn fact(head: Term<T>) -> Self {
        Self { head, body: None }
    }

    pub fn rule(head: Term<T>, body: Expr<T>) -> Self {
        Self {
            head,
            body: Some(body),
        }
    }

    pub fn map<U, F: FnMut(T) -> U>(self, f: &mut F) -> Clause<U> {
        Clause {
            head: self.head.map(f),
            body: self.body.map(|expr| expr.map(f)),
        }
    }

    pub fn replace_term<F>(&mut self, f: &mut F)
    where
        F: FnMut(&Term<T>) -> Option<Term<T>>,
    {
        self.head.replace_all(f);
        if let Some(body) = &mut self.body {
            body.replace_term(f);
        }
    }
}

impl<T: Display> Display for Clause<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.head.fmt(f)?;
        if let Some(body) = &self.body {
            f.write_str(" :- ")?;
            body.fmt(f)?;
        }
        f.write_char('.')
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Term<T> {
    pub functor: T,
    pub args: Vec<Term<T>>,
}

impl<T> Term<T> {
    pub fn atom(functor: T) -> Self {
        Term {
            functor,
            args: vec![],
        }
    }

    pub fn compound<I: IntoIterator<Item = Term<T>>>(functor: T, args: I) -> Self {
        Term {
            functor,
            args: args.into_iter().collect(),
        }
    }

    pub fn map<U, F: FnMut(T) -> U>(self, f: &mut F) -> Term<U> {
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

impl<T: Atom> Term<Name<T>> {
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

impl<T: Display> Display for Term<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.functor, f)?;
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expr<T> {
    Term(Term<T>),
    Not(Box<Expr<T>>),
    And(Vec<Expr<T>>),
    Or(Vec<Expr<T>>),
}

impl<T> Expr<T> {
    pub fn term(term: Term<T>) -> Self {
        Self::Term(term)
    }

    pub fn term_atom(functor: T) -> Self {
        Self::Term(Term::atom(functor))
    }

    pub fn term_compound<I: IntoIterator<Item = Term<T>>>(functor: T, args: I) -> Self {
        Self::Term(Term::compound(functor, args))
    }

    pub fn expr_not(expr: Expr<T>) -> Self {
        Self::Not(Box::new(expr))
    }

    pub fn expr_and<I: IntoIterator<Item = Expr<T>>>(elems: I) -> Self {
        Self::And(elems.into_iter().collect())
    }

    pub fn expr_or<I: IntoIterator<Item = Expr<T>>>(elems: I) -> Self {
        Self::Or(elems.into_iter().collect())
    }

    pub fn map<U, F: FnMut(T) -> U>(self, f: &mut F) -> Expr<U> {
        match self {
            Self::Term(v) => Expr::Term(v.map(f)),
            Self::Not(v) => Expr::Not(Box::new(v.map(f))),
            Self::And(v) => Expr::And(v.into_iter().map(|expr| expr.map(f)).collect()),
            Self::Or(v) => Expr::Or(v.into_iter().map(|expr| expr.map(f)).collect()),
        }
    }

    pub fn replace_term<F>(&mut self, f: &mut F)
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

impl<T: Display> Display for Expr<T> {
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

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Predicate<T> {
    pub functor: T,
    pub arity: u32,
}
