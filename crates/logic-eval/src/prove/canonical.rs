use crate::{
    prove::{
        prover::Integer,
        repr::{TermId, TermStorage, TermViewMut},
    },
    Atom, Expr, Map, Term,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CanonicalTermId(TermId);

pub(crate) fn canonicalize_term_id(stor: &mut TermStorage<Integer>, id: TermId) -> CanonicalTermId {
    let mut view = stor.get_term_mut(id);
    canonicalize_term_view(&mut view);
    CanonicalTermId(view.id())
}

/// e.g. f(X, Y, X) -> f(0, 1, 0)
pub(crate) fn canonicalize_term(term: &mut Term<Integer>) {
    let mut c = canonicalizer();
    term.replace_variables(|functor| *functor = c(*functor));
}

/// Applies [`canonicalize_term`] on each term without crossing term boundaries.
///
/// e.g. f(X), g(Y, X) -> f(0), g(0, 1) (not f(0), g(1, 0))
pub(crate) fn canonicalize_expr_on_term(expr: &mut Expr<Integer>) {
    match expr {
        Expr::Term(term) => canonicalize_term(term),
        Expr::Not(arg) => canonicalize_expr_on_term(arg),
        Expr::And(args) | Expr::Or(args) => {
            for arg in args {
                canonicalize_expr_on_term(arg);
            }
        }
    }
}

pub(crate) fn canonicalize_term_view(view: &mut TermViewMut<'_, Integer>) {
    let mut c = canonicalizer();
    view.replace_with(|functor| {
        if functor.is_variable() {
            Some(c(*functor))
        } else {
            None
        }
    });
}

fn canonicalizer() -> impl FnMut(Integer) -> Integer {
    let mut map = Map::default();
    move |functor: Integer| {
        if functor.is_variable() {
            let next_int = map.len() as u32;
            *map.entry(functor).or_insert(Integer::variable(next_int))
        } else {
            functor
        }
    }
}
