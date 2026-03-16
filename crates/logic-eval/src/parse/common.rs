use crate::{Atom, Clause, Expr, Term};
use core::fmt::{self, Display};
use smallvec::SmallVec;

pub trait Intern {
    type Interned<'a>: Atom
    where
        Self: 'a;

    fn intern_str(&self, s: &str) -> Self::Interned<'_>;

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

pub type StrInterner = any_intern::DroplessInterner;
pub type InternedStr<'int> = any_intern::Interned<'int, str>;

pub struct StrCanonicalizer<'int> {
    interner: &'int StrInterner,
}

impl<'int> StrCanonicalizer<'int> {
    pub fn new(interner: &'int StrInterner) -> Self {
        Self { interner }
    }

    pub fn canonicalize(&self, clause: Clause<InternedStr<'int>>) -> Clause<InternedStr<'int>> {
        let mut vars = SmallVec::new();
        find_var_in_clause(&clause, &mut vars);

        let mut clause = clause;
        clause.replace_term(&mut |term| {
            if !term.args.is_empty() {
                return None;
            }
            vars.iter().enumerate().find_map(|(i, var)| {
                if &term.functor == var {
                    Some(Term {
                        functor: self.interner.intern_formatted_str(&i, i % 10 + 1).unwrap(),
                        args: Vec::new(),
                    })
                } else {
                    None
                }
            })
        });

        return clause;

        // === Internal helper functions ===

        fn find_var_in_clause<T: Atom>(clause: &Clause<T>, vars: &mut SmallVec<[T; 4]>) {
            find_var_in_term(&clause.head, vars);
            if let Some(body) = &clause.body {
                find_var_in_expr(body, vars);
            }
        }

        fn find_var_in_expr<T: Atom>(expr: &Expr<T>, vars: &mut SmallVec<[T; 4]>) {
            match expr {
                Expr::Term(term) => find_var_in_term(term, vars),
                Expr::Not(expr) => find_var_in_expr(expr, vars),
                Expr::And(expr) | Expr::Or(expr) => {
                    for inner_expr in expr.iter() {
                        find_var_in_expr(inner_expr, vars);
                    }
                }
            }
        }

        fn find_var_in_term<T: Atom>(term: &Term<T>, vars: &mut SmallVec<[T; 4]>) {
            if term.functor.is_variable() {
                vars.push(term.functor.clone());
            } else {
                for arg in &term.args {
                    find_var_in_term(arg, vars);
                }
            }
        }
    }
}
