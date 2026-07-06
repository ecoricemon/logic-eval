use super::{
    proof_engine::{
        format::{NamedExprView, NamedTermView},
        AtomId, NameInterner, ProofEngine, QueryCx,
    },
    repr::{buf_term_hash, structurally_eq_terms, ClauseId, TermId, TermStorage},
};
use crate::{
    parse::{
        repr::{Clause, ClauseDataset, Expr, Predicate, Term},
        VAR_PREFIX,
    },
    prove::repr::{ExprKind, ExprView, TermView, TermViewIter},
    Atom, IndexMap, IndexSet, Map, PassThroughMap,
};
use core::{
    fmt::{self, Debug, Display, Write},
    iter::FusedIterator,
};
use smallvec::SmallVec;

/// A clause database that can answer logic queries.
pub struct Database<T> {
    /// Clauses grouped by predicate.
    clauses: IndexMap<Predicate<AtomId>, PredicateClauses>,

    /// Predicates that should be handled by tabling.
    tabled_predicates: IndexSet<Predicate<AtomId>>,

    /// Term and expression storage.
    database_storage: TermStorage<AtomId>,

    /// Mappings between `T` and [`AtomId`].
    ///
    /// [`AtomId`] is used internally for fast comparison, but clients need values mapped back to
    /// `T`.
    name_interner: NameInterner<T>,

    /// We do not allow duplicate clauses in the dataset.
    dup_checker: DuplicateClauseChecker,
}

impl<T: Atom> Database<T> {
    /// Iterates over all terms stored in the database.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{parse_str, Clause, Database, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let clause: Clause<_> = parse_str("sunny.", &interner).unwrap();
    /// let mut db = Database::default();
    /// db.insert_clause(clause);
    ///
    /// let terms = db.terms().map(|term| term.to_string()).collect::<Vec<_>>();
    /// assert_eq!(terms, vec!["sunny"]);
    /// ```
    pub fn terms(&self) -> NamedTermViewIter<'_, T> {
        NamedTermViewIter {
            term_iter: self.database_storage.terms.terms(),
            name_interner: &self.name_interner,
        }
    }

    /// Iterates over all clauses stored in the database.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{parse_str, Clause, Database, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let clause: Clause<_> = parse_str("sunny.", &interner).unwrap();
    /// let mut db = Database::default();
    /// db.insert_clause(clause);
    ///
    /// let clauses = db.clauses().map(|clause| clause.to_string()).collect::<Vec<_>>();
    /// assert_eq!(clauses, vec!["sunny."]);
    /// ```
    pub fn clauses(&self) -> ClauseIter<'_, T> {
        ClauseIter {
            clauses: &self.clauses,
            database_storage: &self.database_storage,
            name_interner: &self.name_interner,
            i: 0,
            j: 0,
        }
    }

    /// Inserts every clause from `dataset`.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{parse_str, ClauseDataset, Database, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let dataset: ClauseDataset<_> = parse_str("sunny.\nwarm.", &interner).unwrap();
    /// let mut db = Database::default();
    ///
    /// db.insert_dataset(dataset);
    ///
    /// assert_eq!(db.clauses().count(), 2);
    /// ```
    pub fn insert_dataset(&mut self, dataset: ClauseDataset<T>) {
        for clause in dataset {
            self.insert_clause(clause);
        }
    }

    /// Inserts the given clause into the database.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{parse_str, Clause, Database, Expr, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let clause: Clause<_> = parse_str("sunny.", &interner).unwrap();
    /// let query: Expr<_> = parse_str("sunny", &interner).unwrap();
    /// let mut db = Database::default();
    ///
    /// db.insert_clause(clause);
    ///
    /// assert!(db.query(query).is_true());
    /// ```
    pub fn insert_clause(&mut self, clause: Clause<T>) {
        let clause = clause.map(&mut |t| self.name_interner.intern(t));

        // Records whether the clause needs tabling.
        if clause.needs_tabling() {
            self.tabled_predicates.insert(clause.head.predicate());
        }

        // If the DB already contains the given clause, then returns.
        if !self.dup_checker.insert(clause.clone()) {
            return;
        }

        let key = clause.head.predicate();
        let value = ClauseId {
            head: self.database_storage.insert_term(clause.head),
            body: clause
                .body
                .map(|expr| self.database_storage.insert_expr(expr)),
        };

        self.clauses
            .entry(key)
            .or_default()
            .insert(value, &self.database_storage);
    }

    /// Starts a query against the database.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{parse_str, ClauseDataset, Database, Expr, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let dataset: ClauseDataset<_> = parse_str("parent(alice, bob).", &interner).unwrap();
    /// let query: Expr<_> = parse_str("parent(alice, $Who)", &interner).unwrap();
    /// let mut db = Database::default();
    /// db.insert_dataset(dataset);
    ///
    /// let mut cx = db.query(query);
    /// let answer = cx.prove_next().unwrap().next().unwrap();
    ///
    /// assert_eq!(answer.get_lhs_variable().as_ref(), "$Who");
    /// assert_eq!(answer.rhs().to_string(), "bob");
    /// ```
    pub fn query(&self, expr: Expr<T>) -> QueryCx<'_, T> {
        ProofEngine::new().prove(
            expr,
            &self.clauses,
            &self.tabled_predicates,
            &self.database_storage,
            &self.name_interner,
        )
    }

    /// * sanitize - Removes unacceptable characters from prolog.
    ///
    /// Requires T to implement [`AsRef<str>`] so that functor names can be serialized into Prolog
    /// syntax.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{parse_str, ClauseDataset, Database, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let dataset: ClauseDataset<_> = parse_str("parent(Alice, Bob).", &interner).unwrap();
    /// let mut db = Database::default();
    /// db.insert_dataset(dataset);
    ///
    /// let prolog = db.to_prolog(|name| name);
    /// assert_eq!(prolog, "parent(alice, bob).\n");
    /// ```
    pub fn to_prolog<F: FnMut(&str) -> &str>(&self, sanitize: F) -> String
    where
        T: AsRef<str>,
    {
        let mut prolog_text = String::new();

        let mut conv_map = ConversionMap {
            atom_id_to_str: Map::default(),
            sanitized_to_suffix: Map::default(),
            name_interner: &self.name_interner,
            sanitizer: sanitize,
        };

        for clauses in self.clauses.values() {
            for clause in &clauses.all {
                let head = self.database_storage.get_term(clause.head);
                write_term(head, &mut conv_map, &mut prolog_text);

                if let Some(body) = clause.body {
                    prolog_text.push_str(" :- ");

                    let body = self.database_storage.get_expr(body);
                    write_expr(body, &mut conv_map, &mut prolog_text);
                }

                prolog_text.push_str(".\n");
            }
        }

        return prolog_text;

        // === Internal helper functions ===

        struct ConversionMap<'a, T, F> {
            atom_id_to_str: Map<AtomId, String>,
            // e.g. 0 -> No suffix, 1 -> _1, 2 -> _2, ...
            sanitized_to_suffix: Map<&'a str, u32>,
            name_interner: &'a NameInterner<T>,
            sanitizer: F,
        }

        impl<T, F> ConversionMap<'_, T, F>
        where
            T: AsRef<str>,
            F: FnMut(&str) -> &str,
        {
            fn atom_id_to_str(&mut self, atom_id: AtomId) -> &str {
                self.atom_id_to_str.entry(atom_id).or_insert_with(|| {
                    let name = self.name_interner.get_name(&atom_id).unwrap();
                    let name: &str = name.as_ref();

                    let mut is_var = false;

                    // Removes variable prefix.
                    let name = if name.starts_with(VAR_PREFIX) {
                        is_var = true;
                        &name[1..]
                    } else {
                        name
                    };

                    // Removes other user-defined characters.
                    let pure_name = (self.sanitizer)(name);

                    let suffix = self
                        .sanitized_to_suffix
                        .entry(pure_name)
                        .and_modify(|x| *x += 1)
                        .or_insert(0);

                    let mut buf = String::new();

                    if is_var {
                        let upper = pure_name.chars().next().unwrap().to_uppercase();
                        for c in upper {
                            buf.push(c);
                        }
                    } else {
                        let lower = pure_name.chars().next().unwrap().to_lowercase();
                        for c in lower {
                            buf.push(c);
                        }
                    };
                    buf.push_str(&pure_name[1..]);

                    if *suffix == 0 {
                        buf
                    } else {
                        write!(&mut buf, "_{suffix}").unwrap();
                        buf
                    }
                })
            }
        }

        fn write_term<T, F>(
            term: TermView<'_, AtomId>,
            conv_map: &mut ConversionMap<'_, T, F>,
            prolog_text: &mut String,
        ) where
            T: AsRef<str>,
            F: FnMut(&str) -> &str,
        {
            let functor = term.functor();
            let args = term.args();
            let num_args = args.len();

            let functor = conv_map.atom_id_to_str(*functor);
            prolog_text.push_str(functor);

            if num_args > 0 {
                prolog_text.push('(');
                for (i, arg) in args.enumerate() {
                    write_term(arg, conv_map, prolog_text);
                    if i + 1 < num_args {
                        prolog_text.push_str(", ");
                    }
                }
                prolog_text.push(')');
            }
        }

        fn write_expr<T, F>(
            expr: ExprView<'_, AtomId>,
            conv_map: &mut ConversionMap<'_, T, F>,
            prolog_text: &mut String,
        ) where
            T: AsRef<str>,
            F: FnMut(&str) -> &str,
        {
            match expr.as_kind() {
                ExprKind::Term(term) => {
                    write_term(term, conv_map, prolog_text);
                }
                ExprKind::Not(inner) => {
                    prolog_text.push_str("\\+ ");
                    if matches!(inner.as_kind(), ExprKind::And(_) | ExprKind::Or(_)) {
                        prolog_text.push('(');
                        write_expr(inner, conv_map, prolog_text);
                        prolog_text.push(')');
                    } else {
                        write_expr(inner, conv_map, prolog_text);
                    }
                }
                ExprKind::And(args) => {
                    let num_args = args.len();
                    for (i, arg) in args.enumerate() {
                        if matches!(arg.as_kind(), ExprKind::Or(_)) {
                            prolog_text.push('(');
                            write_expr(arg, conv_map, prolog_text);
                            prolog_text.push(')');
                        } else {
                            write_expr(arg, conv_map, prolog_text);
                        }
                        if i + 1 < num_args {
                            prolog_text.push_str(", ");
                        }
                    }
                }
                ExprKind::Or(args) => {
                    let num_args = args.len();
                    for (i, arg) in args.enumerate() {
                        write_expr(arg, conv_map, prolog_text);
                        if i + 1 < num_args {
                            prolog_text.push_str("; ");
                        }
                    }
                }
            }
        }
    }
}

impl<T> Default for Database<T> {
    fn default() -> Self {
        Self {
            clauses: IndexMap::default(),
            tabled_predicates: IndexSet::default(),
            database_storage: TermStorage::default(),
            name_interner: NameInterner::default(),
            dup_checker: DuplicateClauseChecker::default(),
        }
    }
}

impl<T: Debug> Debug for Database<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Database")
            .field("clauses", &self.clauses)
            .field("tabled_predicates", &self.tabled_predicates)
            .field("database_storage", &self.database_storage)
            .field("name_interner", &self.name_interner)
            .field("dup_checker", &self.dup_checker)
            .finish_non_exhaustive()
    }
}

/// Clauses for one predicate, grouped with small lookup buckets for query optimization.
///
/// The buckets let proof search skip clauses that are obviously incompatible with a ground query
/// argument before importing them into query-local storage. They are conservative: candidates that
/// remain here still go through full unification later, and candidate order is restored to clause
/// insertion order.
#[derive(Debug, Default)]
pub(crate) struct PredicateClauses {
    all: Vec<ClauseId>,
    positions: Map<ClauseId, usize>,
    args: Vec<ArgBuckets>,
}

impl PredicateClauses {
    fn insert(&mut self, clause: ClauseId, storage: &TermStorage<AtomId>) {
        if self.positions.contains_key(&clause) {
            if cfg!(debug_assertions) {
                panic!("duplicate clause inserted into PredicateClauses: {clause:?}")
            } else {
                return;
            }
        }

        let position = self.all.len();
        self.all.push(clause);
        self.positions.insert(clause, position);

        let head = storage.get_term(clause.head);
        for (arg_index, arg) in head.args().enumerate() {
            while self.args.len() <= arg_index {
                self.args.push(ArgBuckets::default());
            }
            let buckets = &mut self.args[arg_index];
            if arg.contains_variable() {
                buckets.variable.push(clause);
            } else {
                let hash = buf_term_hash(storage.terms.as_ref(), arg.id);
                buckets.ground.entry(hash).or_default().push(clause);
            }
        }
    }

    /// Returns clauses except those that are obviously not unifiable with `query_term`.
    ///
    /// For example, given these clauses:
    ///
    /// ```text
    /// value(candidate0, target, exact0).
    /// value(candidate1, other, exact1).
    /// value($Candidate, target, fallback).
    /// ```
    ///
    /// A query like `value($Candidate, target, $Value)` can use the second argument to skip the
    /// `other` clause, while the exact `target` clause and the variable-head clause remain.
    pub(crate) fn candidates_for<'a>(
        &'a self,
        database_storage: &'a TermStorage<AtomId>,
        query_storage: &'a TermStorage<AtomId>,
        query_term: TermView<'_, AtomId>,
    ) -> SmallVec<[ClauseId; 2]> {
        let Some((arg_index, query_arg, buckets, ground)) =
            self.best_ground_arg_bucket(query_storage, query_term)
        else {
            return self.all.iter().copied().collect();
        };

        let mut candidates = buckets
            .variable
            .iter()
            .copied()
            .chain(ground.iter().copied().filter(|clause| {
                let db_head = database_storage.get_term(clause.head);
                let Some(db_arg) = db_head.args().nth(arg_index) else {
                    return false;
                };
                structurally_eq_terms(
                    database_storage.terms.as_ref(),
                    db_arg.id,
                    query_storage.terms.as_ref(),
                    query_arg,
                )
            }))
            .collect::<SmallVec<[ClauseId; 2]>>();
        // Buckets group clauses by argument shape, so restore insertion order before proof search.
        candidates.sort_by_key(|clause| self.positions[clause]);
        candidates
    }

    /// Picks the ground query argument that is expected to produce the fewest candidates.
    ///
    /// For each ground query argument, this checks the same-position bucket and estimates its
    /// candidate count as `variable clauses + same-hash ground clauses`. The smallest estimate is
    /// used by `candidates_for`.
    fn best_ground_arg_bucket<'a>(
        &'a self,
        query_storage: &'a TermStorage<AtomId>,
        query_term: TermView<'_, AtomId>,
    ) -> Option<(usize, TermId, &'a ArgBuckets, &'a [ClauseId])> {
        query_term
            .args()
            .enumerate()
            .filter_map(|(arg_index, arg)| {
                if arg.contains_variable() {
                    return None;
                }
                let buckets = self.args.get(arg_index)?;
                let hash = buf_term_hash(query_storage.terms.as_ref(), arg.id);
                let bucket = buckets
                    .ground
                    .get(&hash)
                    .map_or([].as_slice(), Vec::as_slice);
                Some((arg_index, arg.id, buckets, bucket))
            })
            .min_by_key(|(_, _, buckets, bucket)| buckets.variable.len() + bucket.len())
    }
}

#[derive(Debug, Default)]
struct ArgBuckets {
    ground: PassThroughMap<Vec<ClauseId>>,
    variable: Vec<ClauseId>,
}

/// Tracks clauses already inserted into a database after normalizing variable names.
///
/// Clause variables are local to one clause, so `p($A).` and `p($B).` should count as the same
/// clause. The checker rewrites each source variable to a per-clause canonical variable while
/// preserving repeated-variable identity across the head and body. For example,
/// `p($A, $B) :- q($A, $B).` and `p($A, $B) :- q($B, $A).` remain distinct.
#[derive(Debug, Default)]
struct DuplicateClauseChecker {
    seen: IndexSet<Clause<AtomId>>,

    /// Temporary map from source variables to canonical [`AtomId`]s.
    vars: Map<AtomId, AtomId>,
}

impl DuplicateClauseChecker {
    /// Returns `true` if the given clause is new and has not been seen before.
    fn insert(&mut self, clause: Clause<AtomId>) -> bool {
        let canonical_clause = clause.map(&mut |t| {
            if !t.is_variable() {
                t
            } else {
                let next_id = self.vars.len() as u32;
                *self.vars.entry(t).or_insert(AtomId::variable(next_id))
            }
        });
        let is_new = self.seen.insert(canonical_clause);
        self.vars.clear();
        is_new
    }
}

/// Rewrites variables using the values produced by `canonical_var`.
///
/// Returns `None` if `canonical_var` is `None` (i.e. deduplication disabled).
fn _convert_var_into_num<T: Atom>(
    this: &Clause<T>,
    canonical_var: Option<&dyn Fn(usize) -> T>,
) -> Option<Clause<T>> {
    let canonical_var = canonical_var?;
    let mut cloned: Option<Clause<T>> = None;

    let mut i = 0;

    while let Some(from) = find_var_in_clause(cloned.as_ref().unwrap_or(this)) {
        let from = from.clone();
        let canonical_t = canonical_var(i);

        let mut convert = |term: &Term<T>| {
            (term.functor == from && term.args.is_empty()).then_some(Term {
                functor: canonical_t.clone(),
                args: vec![],
            })
        };

        if let Some(cloned) = &mut cloned {
            cloned.replace_term(&mut convert);
        } else {
            let mut this = this.clone();
            this.replace_term(&mut convert);
            cloned = Some(this);
        }

        i += 1;
    }

    return cloned;

    // === Internal helper functions ===

    fn find_var_in_clause<T: Atom>(clause: &Clause<T>) -> Option<T> {
        find_var_in_term(&clause.head).or_else(|| clause.body.as_ref().and_then(find_var_in_expr))
    }

    fn find_var_in_expr<T: Atom>(expr: &Expr<T>) -> Option<T> {
        match expr {
            Expr::Term(term) => find_var_in_term(term),
            Expr::Not(arg) => find_var_in_expr(arg),
            Expr::And(args) | Expr::Or(args) => args.iter().find_map(find_var_in_expr),
        }
    }

    fn find_var_in_term<T: Atom>(term: &Term<T>) -> Option<T> {
        if term.functor.is_variable() {
            Some(term.functor.clone())
        } else {
            term.args.iter().find_map(find_var_in_term)
        }
    }
}

/// Iterator over clauses in a [`Database`].
#[derive(Clone)]
pub struct ClauseIter<'a, T> {
    clauses: &'a IndexMap<Predicate<AtomId>, PredicateClauses>,
    database_storage: &'a TermStorage<AtomId>,
    name_interner: &'a NameInterner<T>,
    i: usize,
    j: usize,
}

impl<'a, T> Iterator for ClauseIter<'a, T> {
    type Item = ClauseRef<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let id = loop {
            let (_, group) = self.clauses.get_index(self.i)?;

            if let Some(id) = group.all.get(self.j) {
                self.j += 1;
                break *id;
            }

            self.i += 1;
            self.j = 0;
        };

        Some(ClauseRef {
            id,
            database_storage: self.database_storage,
            name_interner: self.name_interner,
        })
    }
}

impl<T> FusedIterator for ClauseIter<'_, T> {}

/// Borrowed view of a clause stored in a [`Database`].
pub struct ClauseRef<'a, T> {
    id: ClauseId,
    database_storage: &'a TermStorage<AtomId>,
    name_interner: &'a NameInterner<T>,
}

impl<'a, T: Atom> ClauseRef<'a, T> {
    /// Returns the clause head.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{parse_str, Clause, Database, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let clause: Clause<_> = parse_str("outdoors :- sunny.", &interner).unwrap();
    /// let mut db = Database::default();
    /// db.insert_clause(clause);
    ///
    /// let clause = db.clauses().next().unwrap();
    /// assert_eq!(clause.head().to_string(), "outdoors");
    /// ```
    pub fn head(&self) -> NamedTermView<'a, T> {
        let head = self.database_storage.get_term(self.id.head);
        NamedTermView::new(head, self.name_interner)
    }

    /// Returns the clause body, if this clause is a rule.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{parse_str, Clause, Database, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let clause: Clause<_> = parse_str("outdoors :- sunny.", &interner).unwrap();
    /// let mut db = Database::default();
    /// db.insert_clause(clause);
    ///
    /// let clause = db.clauses().next().unwrap();
    /// assert_eq!(clause.body().unwrap().to_string(), "sunny");
    /// ```
    pub fn body(&self) -> Option<NamedExprView<'a, T>> {
        self.id.body.map(|id| {
            let body = self.database_storage.get_expr(id);
            NamedExprView::new(body, self.name_interner)
        })
    }
}

impl<T: Atom + Display> Display for ClauseRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.head(), f)?;

        if let Some(body) = self.body() {
            f.write_str(" :- ")?;
            Display::fmt(&body, f)?
        }

        f.write_char('.')
    }
}

impl<T: Atom + Debug> Debug for ClauseRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("Clause");

        let head = self.database_storage.get_term(self.id.head);
        d.field("head", &NamedTermView::new(head, self.name_interner));

        if let Some(body) = self.id.body {
            let body = self.database_storage.get_expr(body);
            d.field("body", &NamedExprView::new(body, self.name_interner));
        }

        d.finish()
    }
}

pub struct NamedTermViewIter<'a, T> {
    term_iter: TermViewIter<'a, AtomId>,
    name_interner: &'a NameInterner<T>,
}

impl<'a, T: Atom> Iterator for NamedTermViewIter<'a, T> {
    type Item = NamedTermView<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.term_iter
            .next()
            .map(|view| NamedTermView::new(view, self.name_interner))
    }
}

impl<T: Atom> FusedIterator for NamedTermViewIter<'_, T> {}

#[cfg(test)]
mod tests {
    use crate::{parse, AnswerCardinalityError, NameIn};

    type Interner = any_intern::DroplessInterner;
    type Database<'int> = crate::Database<NameIn<'int, Interner>>;
    type QueryCx<'a, 'int> = crate::QueryCx<'a, NameIn<'int, Interner>>;
    type ClauseDataset<'int> = crate::ClauseDatasetIn<'int, Interner>;
    type Expr<'int> = crate::ExprIn<'int, Interner>;
    type Clause<'int> = crate::ClauseIn<'int, Interner>;

    #[test]
    fn test_serial_queries() {
        fn assert_query<'int>(db: &mut Database<'int>, interner: &'int Interner) {
            let query = "g($X).";
            let query: Expr<'int> = parse::parse_str(query, interner).unwrap();

            let cx = db.query(query);
            let answer = collect_answer(cx);
            let expected = [["$X = a"], ["$X = b"]];
            assert_eq!(answer, expected);
        }

        let mut db = Database::default();
        let interner = Interner::new();

        for _ in 0..2 {
            insert_dataset(
                &mut db,
                &interner,
                r"
                f(a).
                f(b).
                g($X) :- f($X).
                ",
            );
            let len = db.terms().count();
            assert_query(&mut db, &interner);
            assert_eq!(db.terms().count(), len);
        }
    }

    #[test]
    fn test_not_expression() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            g(a).
            f($X) :- \+ g($X).
            ",
        );

        let query: Expr<'_> = parse::parse_str("f(a).", &interner).unwrap();
        let answer = collect_answer(db.query(query));
        assert!(answer.is_empty());

        let query: Expr<'_> = parse::parse_str("f(b).", &interner).unwrap();
        let answer = collect_answer(db.query(query));
        assert_eq!(answer.len(), 1);
    }

    #[test]
    fn test_and_expression() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            g(a).
            g(b).
            h(b).
            f($X) :- g($X), h($X).
            ",
        );

        let query: Expr<'_> = parse::parse_str("f($X).", &interner).unwrap();
        let answer = collect_answer(db.query(query));
        let expected = [["$X = b"]];
        assert_eq!(answer, expected);
    }

    #[test]
    fn test_or_expression() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            g(a).
            h(b).
            f($X) :- g($X); h($X).
            ",
        );

        let query: Expr<'_> = parse::parse_str("f($X).", &interner).unwrap();
        let answer = collect_answer(db.query(query));
        let expected = [["$X = a"], ["$X = b"]];
        assert_eq!(answer, expected);
    }

    #[test]
    fn test_mixed_expression() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            g(b).
            g(c).

            h(b).

            i(a).
            i(b).
            i(c).

            f($X) :- (\+ g($X); h($X)), i($X).
            ",
        );

        let query: Expr<'_> = parse::parse_str("f($X).", &interner).unwrap();
        let answer = collect_answer(db.query(query));
        let expected = [["$X = b"]];
        assert_eq!(answer, expected);
    }

    #[test]
    fn test_and_has_higher_precedence_than_or() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            member(alice).
            member(bob).

            paid(bob).

            staff(carol).

            allowed($X) :- member($X), paid($X); staff($X).
            ",
        );

        let query: Expr<'_> = parse::parse_str("allowed(alice).", &interner).unwrap();
        let answer = collect_answer(db.query(query));
        assert!(answer.is_empty());

        let query: Expr<'_> = parse::parse_str("allowed(bob).", &interner).unwrap();
        let answer = collect_answer(db.query(query));
        assert_eq!(answer.len(), 1);

        let query: Expr<'_> = parse::parse_str("allowed(carol).", &interner).unwrap();
        let answer = collect_answer(db.query(query));
        assert_eq!(answer.len(), 1);
    }

    #[test]
    fn test_parentheses_override_and_or_precedence() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            member(alice).
            member(bob).

            paid(bob).

            staff(carol).

            allowed($X) :- member($X), (paid($X); staff($X)).
            ",
        );

        let query: Expr<'_> = parse::parse_str("allowed($X).", &interner).unwrap();
        let answer = collect_answer(db.query(query));
        let expected = [["$X = bob"]];
        assert_eq!(answer, expected);
    }

    #[test]
    fn test_independent_rule_body_variables_form_cartesian_product() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            left(a0).
            left(a1).
            right(b0).
            right(b1).

            foo($A, $B) :- left($A), right($B).
            ",
        );

        let query: Expr<'_> = parse::parse_str("foo($A, $B).", &interner).unwrap();
        let mut answer = collect_answer(db.query(query));
        let mut expected = [
            ["$A = a0", "$B = b0"],
            ["$A = a0", "$B = b1"],
            ["$A = a1", "$B = b0"],
            ["$A = a1", "$B = b1"],
        ];

        answer.sort_unstable();
        expected.sort_unstable();
        assert_eq!(answer, expected);
    }

    #[test]
    fn test_not_applies_to_parenthesized_or() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            candidate(alice).
            candidate(bob).
            candidate(carol).
            candidate(dana).

            blocked(bob).
            archived(carol).

            active($X) :- candidate($X), \+ (blocked($X); archived($X)).
            ",
        );

        let query: Expr<'_> = parse::parse_str("active($X).", &interner).unwrap();
        let mut answer = collect_answer(db.query(query));
        let mut expected = [vec!["$X = alice".to_owned()], vec!["$X = dana".to_owned()]];

        answer.sort_unstable();
        expected.sort_unstable();
        assert_eq!(answer, expected);
    }

    #[test]
    fn test_not_with_grouped_and_or_expression() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            task(cleanup).
            task(deploy).
            task(report).
            task(backup).

            urgent(deploy).
            urgent(report).

            owner(alice, cleanup).
            owner(alice, report).
            owner(bob, backup).

            blocked(report).

            todo_for_alice($Task) :-
                task($Task),
                (urgent($Task); owner(alice, $Task)),
                \+ blocked($Task).
            ",
        );

        let query: Expr<'_> = parse::parse_str("todo_for_alice($Task).", &interner).unwrap();
        let mut answer = collect_answer(db.query(query));
        let mut expected = [
            vec!["$Task = cleanup".to_owned()],
            vec!["$Task = deploy".to_owned()],
        ];

        answer.sort_unstable();
        expected.sort_unstable();
        assert_eq!(answer, expected);
    }

    #[test]
    fn test_simple_recursion() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            impl(Clone, a).
            impl(Clone, b).
            impl(Clone, c).
            impl(Clone, Vec($T)) :- impl(Clone, $T).
            ",
        );

        let query: Expr<'_> = parse::parse_str("impl(Clone, $T).", &interner).unwrap();
        let mut cx = db.query(query);

        let mut assert_next = |expected: &[&str]| {
            let eval = cx.prove_next().unwrap();
            let assignments = eval.map(|assign| assign.to_string()).collect::<Vec<_>>();
            assert_eq!(assignments, expected);
        };

        assert_next(&["$T = a"]);
        assert_next(&["$T = b"]);
        assert_next(&["$T = c"]);
        assert_next(&["$T = Vec(a)"]);
        assert_next(&["$T = Vec(b)"]);
        assert_next(&["$T = Vec(c)"]);
        assert_next(&["$T = Vec(Vec(a))"]);
        assert_next(&["$T = Vec(Vec(b))"]);
        assert_next(&["$T = Vec(Vec(c))"]);
    }

    #[test]
    fn test_right_recursion() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            child(a, b).
            child(b, c).
            child(c, d).
            descend($X, $Y) :- child($X, $Y).
            descend($X, $Z) :- child($X, $Y), descend($Y, $Z).
            ",
        );

        let query: Expr<'_> = parse::parse_str("descend($X, $Y).", &interner).unwrap();
        let mut answer = collect_answer(db.query(query));

        let mut expected = [
            ["$X = a", "$Y = b"],
            ["$X = a", "$Y = c"],
            ["$X = a", "$Y = d"],
            ["$X = b", "$Y = c"],
            ["$X = b", "$Y = d"],
            ["$X = c", "$Y = d"],
        ];

        answer.sort_unstable();
        expected.sort_unstable();
        assert_eq!(answer, expected);
    }

    // SLG resolution (tabling) is required to pass this test.
    #[test]
    fn test_mid_recursion() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            edge(a, b).
            edge(b, c).
            edge(c, a).
            path($X, $Y) :- edge($X, $Z), path($Z, $W), edge($W, $Y).
            path($X, $Y) :- edge($X, $Y).
            ",
        );

        let query: Expr<'_> = parse::parse_str("path($X, $Y).", &interner).unwrap();
        let mut answer = collect_answer(db.query(query));

        let mut expected = [
            ["$X = a", "$Y = a"],
            ["$X = a", "$Y = b"],
            ["$X = a", "$Y = c"],
            ["$X = b", "$Y = a"],
            ["$X = b", "$Y = b"],
            ["$X = b", "$Y = c"],
            ["$X = c", "$Y = a"],
            ["$X = c", "$Y = b"],
            ["$X = c", "$Y = c"],
        ];

        answer.sort_unstable();
        expected.sort_unstable();
        assert_eq!(answer, expected);
    }

    // SLG resolution (tabling) is required to pass this test.
    #[test]
    fn test_left_recursion() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            parent(a, b).
            parent(b, c).
            parent(c, d).
            ancestor($X, $Y) :- ancestor($X, $Z), parent($Z, $Y).
            ancestor($X, $Y) :- parent($X, $Y).
            ",
        );

        let query: Expr<'_> = parse::parse_str("ancestor($X, $Y).", &interner).unwrap();
        let mut answer = collect_answer(db.query(query));

        let mut expected = [
            ["$X = a", "$Y = b"],
            ["$X = a", "$Y = c"],
            ["$X = a", "$Y = d"],
            ["$X = b", "$Y = c"],
            ["$X = b", "$Y = d"],
            ["$X = c", "$Y = d"],
        ];

        answer.sort_unstable();
        expected.sort_unstable();
        assert_eq!(answer, expected);
    }

    #[test]
    fn test_inserted_clause_is_immediately_visible_to_query() {
        let mut db = Database::default();
        let interner = Interner::new();

        let clause: Clause<'_> = parse::parse_str("f(a).", &interner).unwrap();
        db.insert_clause(clause);

        let clause: Clause<'_> = parse::parse_str("f(b).", &interner).unwrap();
        db.insert_clause(clause);
        assert_eq!(db.clauses().count(), 2);

        let query: Expr<'_> = parse::parse_str("f($X).", &interner).unwrap();
        let mut answer = collect_answer(db.query(query));
        let mut expected = [["$X = a"], ["$X = b"]];
        answer.sort_unstable();
        expected.sort_unstable();
        assert_eq!(answer, expected);
    }

    #[test]
    fn test_database_and_query_context_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<crate::Database<String>>();
    }

    #[test]
    fn test_repeated_clause_variable_must_match_consistently() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            equal($A, $A).
            same_pair($A, $A).
            parent(alice, bob).
            same_parent($Child, $Child) :- parent(alice, $Child).
            candidate($A, $A).
            candidate(a, b).
            ",
        );

        for query in [
            "equal(a, a).",
            "same_pair(box(a), box(a)).",
            "same_parent(bob, bob).",
            "candidate(a, b).",
        ] {
            let query: Expr<'_> = parse::parse_str(query, &interner).unwrap();
            assert!(db.query(query).is_true());
        }

        for query in [
            "equal(a, b).",
            "same_pair(box(a), box(b)).",
            "same_parent(bob, carol).",
        ] {
            let query: Expr<'_> = parse::parse_str(query, &interner).unwrap();
            assert!(!db.query(query).is_true());
        }
    }

    #[test]
    fn test_repeated_query_variable_returns_one_binding() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            double(pair(a, a), pair(a, a)).
            ",
        );

        let query: Expr<'_> =
            parse::parse_str("double(pair(a, a), pair($A, $A)).", &interner).unwrap();
        let answer = collect_answer(db.query(query));
        let expected = [["$A = a"]];
        assert_eq!(answer, expected);
    }

    #[test]
    fn test_answer_view_get_returns_binding_by_query_variable() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            foo(a0, b0).
            ",
        );

        let query: Expr<'_> = parse::parse_str("foo($A, $B).", &interner).unwrap();
        let mut cx = db.query(query);
        let mut answer = cx.prove_next().unwrap();

        assert_eq!(answer.get("$A").unwrap().to_string(), "a0");
        assert_eq!(answer.get("$B").unwrap().to_string(), "b0");
        assert!(answer.get("$Missing").is_none());

        assert_eq!(answer.next().unwrap().to_string(), "$A = a0");
        assert_eq!(answer.get("$A").unwrap().to_string(), "a0");
        assert_eq!(answer.get("$B").unwrap().to_string(), "b0");
    }

    #[test]
    fn test_answer_view_get_preserves_repeated_query_variable_invariant() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            double(pair(a, a), pair(a, a)).
            ",
        );

        let query: Expr<'_> =
            parse::parse_str("double(pair(a, a), pair($A, $A)).", &interner).unwrap();
        let mut cx = db.query(query);
        let mut answer = cx.prove_next().unwrap();

        assert_eq!(answer.get("$A").unwrap().to_string(), "a");
        assert_eq!(answer.next().unwrap().to_string(), "$A = a");
        assert!(answer.next().is_none());
    }

    #[test]
    fn test_answer_view_materialize_copies_the_full_answer_row() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            foo(a0, b0).
            ",
        );

        let query: Expr<'_> = parse::parse_str("foo($A, $B).", &interner).unwrap();
        let mut cx = db.query(query);
        let mut answer = cx.prove_next().unwrap();

        assert_eq!(answer.next().unwrap().to_string(), "$A = a0");

        let answer = answer.materialize();
        assert_eq!(answer.get("$A").unwrap().to_string(), "a0");
        assert_eq!(answer.get("$B").unwrap().to_string(), "b0");
        assert_eq!(
            answer
                .bindings()
                .iter()
                .map(|(var, term)| format!("{var} = {term}"))
                .collect::<Vec<_>>(),
            ["$A = a0", "$B = b0"]
        );
    }

    #[test]
    fn test_prove_unique_returns_the_only_answer() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            foo(a).
            ",
        );

        let query: Expr<'_> = parse::parse_str("foo($X).", &interner).unwrap();
        let mut cx = db.query(query);
        let answer = cx.prove_unique().unwrap();

        assert_eq!(answer.get("$X").unwrap().to_string(), "a");
    }

    #[test]
    fn test_prove_unique_rejects_no_answer() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            foo(a).
            ",
        );

        let query: Expr<'_> = parse::parse_str("foo(b).", &interner).unwrap();
        let mut cx = db.query(query);

        assert!(matches!(
            cx.prove_unique(),
            Err(AnswerCardinalityError::NoAnswer)
        ));
    }

    #[test]
    fn test_prove_unique_rejects_multiple_answers() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            foo(a).
            foo(b).
            ",
        );

        let query: Expr<'_> = parse::parse_str("foo($X).", &interner).unwrap();
        let mut cx = db.query(query);

        assert!(matches!(
            cx.prove_unique(),
            Err(AnswerCardinalityError::MultipleAnswers)
        ));
    }

    #[test]
    fn test_prove_unique_consumes_query_while_checking_cardinality() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            foo(a).
            foo(b).
            ",
        );

        let query: Expr<'_> = parse::parse_str("foo($X).", &interner).unwrap();
        let mut cx = db.query(query);

        assert!(matches!(
            cx.prove_unique(),
            Err(AnswerCardinalityError::MultipleAnswers)
        ));

        assert!(cx.prove_next().is_none());
    }

    #[test]
    fn test_rule_body_tries_later_candidate_after_failed_candidates() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            p($X) :- r($X, c).
            r($A, $B) :- e($A, $B).
            r($A, $B) :- e($B, $A).

            e(c, b).
            ",
        );

        let query: Expr<'_> = parse::parse_str("p(b)", &interner).unwrap();

        let cx = db.query(query);
        assert!(cx.is_true());
    }

    #[test]
    fn test_candidate_filter_query_filters_candidate_fanout() {
        const REQUESTS: usize = 8;
        const NOISE_CANDIDATES_PER_REQUEST: usize = 4;

        fn nested_type(name: &str) -> String {
            format!("vec(vec({name}))")
        }

        let mut source = String::new();
        for i in 0..REQUESTS {
            let request = format!("req{i}");
            let candidate = format!("candidate{i}");
            let expected = nested_type(&format!("ty{i}"));

            source.push_str(&format!("expected_value({request}, {expected}).\n"));
            source.push_str(&format!("enabled_candidate({candidate}).\n"));
            source.push_str(&format!(
                "candidate_value({request}, {candidate}, {expected}).\n"
            ));

            for j in 0..NOISE_CANDIDATES_PER_REQUEST {
                let noise_candidate = format!("candidate_noise_{i}_{j}");
                let noise_ty = nested_type(&format!("ty_noise_{i}_{j}"));
                source.push_str(&format!(
                    "candidate_value({request}, {noise_candidate}, {noise_ty}).\n"
                ));
            }
        }

        source.push_str(
            r"
            same_shape($Value, $Value).
            same_shape(vec($A), vec($B)) :- same_shape($A, $B).
            candidate_match($Request, $Candidate) :-
                candidate_value($Request, $Candidate, $Value),
                expected_value($Request, $Expected),
                same_shape($Value, $Expected),
                enabled_candidate($Candidate).
            ",
        );

        let mut db = Database::default();
        let interner = Interner::new();
        insert_dataset(&mut db, &interner, &source);

        let query: Expr<'_> =
            parse::parse_str("candidate_match(req3, $Candidate).", &interner).unwrap();
        let answer = collect_answer(db.query(query));
        let expected = [["$Candidate = candidate3"]];
        assert_eq!(answer, expected);
    }

    #[test]
    fn test_candidate_filter_preserves_clause_order_between_exact_and_variable_heads() {
        let mut db = Database::default();
        let interner = Interner::new();
        insert_dataset(
            &mut db,
            &interner,
            r"
            choice(a, exact).
            choice($X, fallback).
            ",
        );

        let query: Expr<'_> = parse::parse_str("choice(a, $Which).", &interner).unwrap();
        let answer = collect_answer(db.query(query));
        let expected = [["$Which = exact"], ["$Which = fallback"]];
        assert_eq!(answer, expected);
    }

    #[test]
    fn test_candidate_filter_uses_later_ground_arguments() {
        let mut db = Database::default();
        let interner = Interner::new();
        insert_dataset(
            &mut db,
            &interner,
            r"
            value(candidate0, target, exact0).
            value(candidate1, other, noise1).
            value(candidate2, target, exact2).
            value($Candidate, target, fallback).
            ",
        );

        let query: Expr<'_> =
            parse::parse_str("value($Candidate, target, $Value).", &interner).unwrap();
        let answer = collect_answer(db.query(query));
        let expected = [
            ["$Candidate = candidate0", "$Value = exact0"],
            ["$Candidate = candidate2", "$Value = exact2"],
        ];
        assert_eq!(answer, expected);
    }

    #[test]
    fn test_query_from_multiple_threads() {
        let mut db = Database::default();
        let interner = Interner::new();

        insert_dataset(
            &mut db,
            &interner,
            r"
            parent(alice, bob).
            parent(alice, carol).
            parent(carol, dave).

            ancestor($X, $Y) :- parent($X, $Y).
            ancestor($X, $Z) :- parent($X, $Y), ancestor($Y, $Z).
            ",
        );

        let barrier = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..4 {
                let db = &db;
                let barrier = &barrier;
                handles.push(scope.spawn(|| {
                    barrier.wait();

                    let query: Expr<'_> =
                        parse::parse_str("ancestor(alice, $Who).", &interner).unwrap();

                    let mut answers = collect_answer(db.query(query))
                        .into_iter()
                        .collect::<Vec<_>>();
                    answers.sort_unstable();
                    answers
                }));
            }

            for handle in handles {
                assert_eq!(
                    handle.join().unwrap(),
                    [["$Who = bob"], ["$Who = carol"], ["$Who = dave"]]
                );
            }
        });
    }

    // === Test helper functions ===

    fn insert_dataset<'int>(db: &mut Database<'int>, interner: &'int Interner, text: &str) {
        let dataset: ClauseDataset<'int> = parse::parse_str(text, interner).unwrap();
        db.insert_dataset(dataset);
    }

    fn collect_answer(mut cx: QueryCx<'_, '_>) -> Vec<Vec<String>> {
        let mut v = Vec::new();
        while let Some(eval) = cx.prove_next() {
            let x = eval.map(|assign| assign.to_string()).collect::<Vec<_>>();
            v.push(x);
        }
        v
    }
}

#[cfg(test)]
mod custom_atom_tests {
    use crate::{Atom, Clause, Database, Expr, QueryCx, Term};

    #[test]
    fn test_custom_atom() {
        #[allow(non_camel_case_types)]
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        enum A {
            child,
            descend,
            a,
            b,
            c,
            d,
            X,
            Y,
            Z,
        }

        impl Atom for A {
            fn is_variable(&self) -> bool {
                matches!(self, A::X | A::Y | A::Z)
            }
        }

        let mut db = Database::default();

        let child_a_b = Clause::fact(Term::compound(
            A::child,
            [Term::atom(A::a), Term::atom(A::b)],
        ));
        let child_b_c = Clause::fact(Term::compound(
            A::child,
            [Term::atom(A::b), Term::atom(A::c)],
        ));
        let child_c_d = Clause::fact(Term::compound(
            A::child,
            [Term::atom(A::c), Term::atom(A::d)],
        ));
        let descend_x_y = Clause::rule(
            Term::compound(A::descend, [Term::atom(A::X), Term::atom(A::Y)]),
            Expr::term_compound(A::child, [Term::atom(A::X), Term::atom(A::Y)]),
        );
        let descend_x_z = Clause::rule(
            Term::compound(A::descend, [Term::atom(A::X), Term::atom(A::Z)]),
            Expr::expr_and([
                Expr::term_compound(A::child, [Term::atom(A::X), Term::atom(A::Y)]),
                Expr::term_compound(A::descend, [Term::atom(A::Y), Term::atom(A::Z)]),
            ]),
        );
        db.insert_dataset(crate::ClauseDataset(vec![
            child_a_b,
            child_b_c,
            child_c_d,
            descend_x_y,
            descend_x_z,
        ]));

        let query = Expr::term_compound(A::descend, [Term::atom(A::X), Term::atom(A::Y)]);
        let mut answer = collect_answer(db.query(query));

        let mut expected = [
            [
                (Term::atom(A::X), Term::atom(A::a)),
                (Term::atom(A::Y), Term::atom(A::b)),
            ],
            [
                (Term::atom(A::X), Term::atom(A::a)),
                (Term::atom(A::Y), Term::atom(A::c)),
            ],
            [
                (Term::atom(A::X), Term::atom(A::a)),
                (Term::atom(A::Y), Term::atom(A::d)),
            ],
            [
                (Term::atom(A::X), Term::atom(A::b)),
                (Term::atom(A::Y), Term::atom(A::c)),
            ],
            [
                (Term::atom(A::X), Term::atom(A::b)),
                (Term::atom(A::Y), Term::atom(A::d)),
            ],
            [
                (Term::atom(A::X), Term::atom(A::c)),
                (Term::atom(A::Y), Term::atom(A::d)),
            ],
        ];

        answer.sort_unstable();
        expected.sort_unstable();
        assert_eq!(answer, expected);
    }

    // === Test helper functions ===

    fn collect_answer<T: Atom>(mut cx: QueryCx<'_, T>) -> Vec<Vec<(Term<T>, Term<T>)>> {
        let mut v = Vec::new();
        while let Some(eval) = cx.prove_next() {
            let pairs = eval.map(|assign| (assign.lhs(), assign.rhs())).collect();
            v.push(pairs);
        }
        v
    }
}
