use crate::{
    prove::{canonical as canon, prover::Integer},
    Atom,
};
use std::{
    fmt::{self, Debug, Display, Write},
    ops,
    vec::IntoIter,
};

/// A collection of parsed clauses.
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

/// A fact or rule clause.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Clause<T> {
    /// The clause head.
    pub head: Term<T>,
    /// The optional rule body.
    pub body: Option<Expr<T>>,
}

impl<T> Clause<T> {
    /// Creates a fact clause.
    pub fn fact(head: Term<T>) -> Self {
        Self { head, body: None }
    }

    /// Creates a rule clause with a head and body.
    pub fn rule(head: Term<T>, body: Expr<T>) -> Self {
        Self {
            head,
            body: Some(body),
        }
    }

    /// Maps every atom in the clause to another type.
    pub fn map<U, F: FnMut(T) -> U>(self, f: &mut F) -> Clause<U> {
        Clause {
            head: self.head.map(f),
            body: self.body.map(|expr| expr.map(f)),
        }
    }

    /// Replaces every matching term in the clause.
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

impl Clause<Integer> {
    /// Returns `true` if the clause needs SLG resolution (tabling).
    ///
    /// If a clause has left or mid recursion, it must be handled by tabling.
    ///
    /// # Examples
    /// foo(X, Y) :- foo(A, B) ...     // left recursion
    /// foo(X, Y) :- ... foo(A, B) ... // mid recursion
    pub fn needs_tabling(&self) -> bool {
        return if let Some(body) = &self.body {
            let mut head = self.head.clone();
            let mut body = body.clone();
            canon::canonicalize_term(&mut head);
            canon::canonicalize_expr_on_term(&mut body);
            helper(&body.distribute_not(), &head)
        } else {
            false
        };

        // === Internal helper functions ===

        fn helper(expr: &Expr<Integer>, head: &Term<Integer>) -> bool {
            match expr {
                Expr::Term(term) => term == head,
                Expr::Not(arg) => helper(arg, head),
                Expr::And(args) => {
                    if let Some((last, first)) = args.split_last() {
                        first.iter().any(|arg| helper(arg, head)) || helper(last, head)
                    } else {
                        false
                    }
                }
                Expr::Or(args) => args.iter().any(|arg| helper(arg, head)),
            }
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

/// A logic term with a functor and zero or more arguments.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Term<T> {
    /// The term functor.
    pub functor: T,
    /// The term arguments.
    pub args: Vec<Term<T>>,
}

impl<T> Term<T> {
    /// Creates an atom term with no arguments.
    pub fn atom(functor: T) -> Self {
        Term {
            functor,
            args: vec![],
        }
    }

    /// Creates a compound term from a functor and arguments.
    pub fn compound<I: IntoIterator<Item = Term<T>>>(functor: T, args: I) -> Self {
        Term {
            functor,
            args: args.into_iter().collect(),
        }
    }

    /// Maps every atom in the term to another type.
    pub fn map<U, F: FnMut(T) -> U>(self, f: &mut F) -> Term<U> {
        Term {
            functor: f(self.functor),
            args: self.args.into_iter().map(|arg| arg.map(f)).collect(),
        }
    }

    /// Replaces this term and all nested terms that match `f`.
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
    /// Returns this term's predicate.
    pub fn predicate(&self) -> Predicate<T> {
        Predicate {
            functor: self.functor.clone(),
            arity: self.args.len() as u32,
        }
    }
}

impl<T: Atom> Term<T> {
    /// Returns `true` if this term is a variable.
    pub fn is_variable(&self) -> bool {
        let is_variable = self.functor.is_variable();

        #[cfg(debug_assertions)]
        if is_variable {
            assert!(self.args.is_empty());
        }

        is_variable
    }

    /// Returns `true` if this term contains a variable.
    pub fn contains_variable(&self) -> bool {
        if self.is_variable() {
            return true;
        }

        self.args.iter().any(|arg| arg.contains_variable())
    }

    /// Applies `f` to every variable functor in the term.
    pub fn replace_variables<F: FnMut(&mut T)>(&mut self, mut f: F) {
        fn helper<T, F>(term: &mut Term<T>, f: &mut F)
        where
            T: Atom,
            F: FnMut(&mut T),
        {
            if term.is_variable() {
                f(&mut term.functor);
            } else {
                for arg in &mut term.args {
                    helper(arg, f);
                }
            }
        }
        helper(self, &mut f)
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

/// A logic expression.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expr<T> {
    /// A term expression.
    Term(Term<T>),
    /// Logical negation.
    Not(Box<Expr<T>>),
    /// Logical conjunction.
    And(Vec<Expr<T>>),
    /// Logical disjunction.
    Or(Vec<Expr<T>>),
}

impl<T> Expr<T> {
    /// Creates an expression from a term.
    pub fn term(term: Term<T>) -> Self {
        Self::Term(term)
    }

    /// Creates an atom term expression.
    pub fn term_atom(functor: T) -> Self {
        Self::Term(Term::atom(functor))
    }

    /// Creates a compound term expression.
    pub fn term_compound<I: IntoIterator<Item = Term<T>>>(functor: T, args: I) -> Self {
        Self::Term(Term::compound(functor, args))
    }

    /// Creates a negated expression.
    pub fn expr_not(expr: Expr<T>) -> Self {
        Self::Not(Box::new(expr))
    }

    /// Creates a conjunction expression.
    pub fn expr_and<I: IntoIterator<Item = Expr<T>>>(args: I) -> Self {
        Self::And(args.into_iter().collect())
    }

    /// Creates a disjunction expression.
    pub fn expr_or<I: IntoIterator<Item = Expr<T>>>(args: I) -> Self {
        Self::Or(args.into_iter().collect())
    }

    /// Maps every atom in the expression to another type.
    pub fn map<U, F: FnMut(T) -> U>(self, f: &mut F) -> Expr<U> {
        match self {
            Self::Term(term) => Expr::Term(term.map(f)),
            Self::Not(arg) => Expr::Not(Box::new(arg.map(f))),
            Self::And(args) => Expr::And(args.into_iter().map(|arg| arg.map(f)).collect()),
            Self::Or(args) => Expr::Or(args.into_iter().map(|arg| arg.map(f)).collect()),
        }
    }

    /// Replaces every matching term in the expression.
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

impl<T: PartialEq> Expr<T> {
    /// Returns `true` if this expression contains `term`.
    pub fn contains_term(&self, term: &Term<T>) -> bool {
        match self {
            Self::Term(t) => t == term,
            Self::Not(arg) => arg.contains_term(term),
            Self::And(args) | Self::Or(args) => args.iter().any(|arg| arg.contains_term(term)),
        }
    }

    /// e.g. ¬(A ∧ (B ∨ C)) -> ¬A ∨ (¬B ∧ ¬C)
    pub fn distribute_not(self) -> Self {
        match self {
            Self::Term(term) => Self::Term(term),
            Self::Not(expr) => match *expr {
                Self::Term(term) => Self::Not(Box::new(Self::Term(term))),
                Self::Not(inner) => inner.distribute_not(),
                Self::And(args) => Self::Or(
                    args.into_iter()
                        .map(|arg| Self::Not(Box::new(arg)).distribute_not())
                        .collect(),
                ),
                Self::Or(args) => Self::And(
                    args.into_iter()
                        .map(|arg| Self::Not(Box::new(arg)).distribute_not())
                        .collect(),
                ),
            },
            Self::And(args) => Self::And(args.into_iter().map(Self::distribute_not).collect()),
            Self::Or(args) => Self::Or(args.into_iter().map(Self::distribute_not).collect()),
        }
    }
}

impl<T: Display> Display for Expr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Term(term) => term.fmt(f)?,
            Self::Not(arg) => {
                f.write_str("\\+ ")?;
                if matches!(**arg, Self::And(_) | Self::Or(_)) {
                    f.write_char('(')?;
                    arg.fmt(f)?;
                    f.write_char(')')?;
                } else {
                    arg.fmt(f)?;
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

/// Predicate identity, represented by functor and arity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Predicate<T> {
    /// Predicate functor.
    pub functor: T,
    /// Predicate arity.
    pub arity: u32,
}

#[cfg(test)]
mod tests {
    use super::{Expr, Term};

    #[test]
    fn distribute_not_applies_de_morgan() {
        let expr = Expr::expr_not(Expr::expr_and([
            Expr::term_atom("a"),
            Expr::expr_or([Expr::term_atom("b"), Expr::term_atom("c")]),
        ]));

        let expected = Expr::expr_or([
            Expr::expr_not(Expr::term_atom("a")),
            Expr::expr_and([
                Expr::expr_not(Expr::term_atom("b")),
                Expr::expr_not(Expr::term_atom("c")),
            ]),
        ]);

        assert_eq!(expr.distribute_not(), expected);
    }

    #[test]
    fn distribute_not_removes_double_negation() {
        let expr = Expr::expr_not(Expr::expr_not(Expr::term(Term::compound(
            "f",
            [Term::atom("x")],
        ))));

        assert_eq!(
            expr.distribute_not(),
            Expr::term(Term::compound("f", [Term::atom("x")]))
        );
    }
}
