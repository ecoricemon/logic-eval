use super::{
    prover::{
        format::{NamedExprView, NamedTermView},
        Integer, NameIntMap, NameIntMapState, ProveCx, Prover,
    },
    repr::{ClauseId, TermStorage, TermStorageLen},
};
use crate::{
    parse::{
        repr::{Clause, ClauseDataset, Expr, Predicate, Term},
        VAR_PREFIX,
    },
    prove::repr::{ExprKind, ExprView, TermView, TermViewIter},
    Atom, IndexMap, IndexSet, Map,
};
use core::{
    fmt::{self, Debug, Display, Write},
    iter::FusedIterator,
};

pub struct Database<T> {
    /// Clause id dataset.
    clauses: IndexMap<Predicate<Integer>, Vec<ClauseId>>,

    /// Clauses that should be handled by tabling.
    table_clauses: IndexSet<Predicate<Integer>>,

    /// We do not allow duplicate clauses in the dataset.
    dup_checker: DuplicateClauseChecker,

    /// Term and expression storage.
    stor: TermStorage<Integer>,

    /// Proof search engine.
    prover: Prover,

    /// Mappings between T and Integer.
    ///
    /// Integer is internally used for fast comparison, but we need to get it back to T for the
    /// clients.
    nimap: NameIntMap<T>,

    /// States of DB's fields.
    ///
    /// This is used when we discard some changes on the DB.
    revert_point: Option<DatabaseState>,
}

impl<T: Atom> Database<T> {
    pub fn new() -> Self {
        Self {
            clauses: IndexMap::default(),
            table_clauses: IndexSet::default(),
            dup_checker: DuplicateClauseChecker::default(),
            stor: TermStorage::new(),
            prover: Prover::new(),
            nimap: NameIntMap::new(),
            revert_point: None,
        }
    }

    pub fn terms(&self) -> NamedTermViewIter<'_, T> {
        NamedTermViewIter {
            term_iter: self.stor.terms.terms(),
            nimap: &self.nimap,
        }
    }

    pub fn clauses(&self) -> ClauseIter<'_, T> {
        ClauseIter {
            clauses: &self.clauses,
            stor: &self.stor,
            nimap: &self.nimap,
            i: 0,
            j: 0,
        }
    }

    pub fn insert_dataset(&mut self, dataset: ClauseDataset<T>) {
        for clause in dataset {
            self.insert_clause(clause);
        }
    }

    /// Inserts the given clause to the DB.
    pub fn insert_clause(&mut self, clause: Clause<T>) {
        // Saves current state. We will revert DB when the change is not committed.
        if self.revert_point.is_none() {
            self.revert_point = Some(self.state());
        }

        let clause = clause.map(&mut |t| self.nimap.name_to_int(t));

        // Records whether the clause needs tabling.
        if clause.needs_tabling() {
            self.table_clauses.insert(clause.head.predicate());
        }

        // If the DB already contains the given clause, then returns.
        if !self.dup_checker.insert(clause.clone()) {
            return;
        }

        let key = clause.head.predicate();
        let value = ClauseId {
            head: self.stor.insert_term(clause.head),
            body: clause.body.map(|expr| self.stor.insert_expr(expr)),
        };

        self.clauses
            .entry(key)
            .and_modify(|similar_clauses| {
                if similar_clauses.iter().all(|clause| clause != &value) {
                    similar_clauses.push(value);
                }
            })
            .or_insert(vec![value]);
    }

    pub fn query(&mut self, expr: Expr<T>) -> ProveCx<'_, T> {
        // Discards uncommitted changes.
        if let Some(revert_point) = self.revert_point.take() {
            self.revert(revert_point);
        }

        self.prover.prove(
            expr,
            &self.clauses,
            &self.table_clauses,
            &mut self.stor,
            &mut self.nimap,
        )
    }

    pub fn commit(&mut self) {
        self.revert_point.take();
    }

    /// * sanitize - Removes unacceptable characters from prolog.
    ///
    /// Requires T to implement [`AsRef<str>`] so that functor names can be serialized into Prolog
    /// syntax.
    pub fn to_prolog<F: FnMut(&str) -> &str>(&self, sanitize: F) -> String
    where
        T: AsRef<str>,
    {
        let mut prolog_text = String::new();

        let mut conv_map = ConversionMap {
            int_to_str: Map::default(),
            sanitized_to_suffix: Map::default(),
            nimap: &self.nimap,
            sanitizer: sanitize,
        };

        for clauses in self.clauses.values() {
            for clause in clauses {
                let head = self.stor.get_term(clause.head);
                write_term(head, &mut conv_map, &mut prolog_text);

                if let Some(body) = clause.body {
                    prolog_text.push_str(" :- ");

                    let body = self.stor.get_expr(body);
                    write_expr(body, &mut conv_map, &mut prolog_text);
                }

                prolog_text.push_str(".\n");
            }
        }

        return prolog_text;

        // === Internal helper functions ===

        struct ConversionMap<'a, T, F> {
            int_to_str: Map<Integer, String>,
            // e.g. 0 -> No suffix, 1 -> _1, 2 -> _2, ...
            sanitized_to_suffix: Map<&'a str, u32>,
            nimap: &'a NameIntMap<T>,
            sanitizer: F,
        }

        impl<T, F> ConversionMap<'_, T, F>
        where
            T: AsRef<str>,
            F: FnMut(&str) -> &str,
        {
            fn int_to_str(&mut self, int: Integer) -> &str {
                self.int_to_str.entry(int).or_insert_with(|| {
                    let name = self.nimap.get_name(&int).unwrap();
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
            term: TermView<'_, Integer>,
            conv_map: &mut ConversionMap<'_, T, F>,
            prolog_text: &mut String,
        ) where
            T: AsRef<str>,
            F: FnMut(&str) -> &str,
        {
            let functor = term.functor();
            let args = term.args();
            let num_args = args.len();

            let functor = conv_map.int_to_str(*functor);
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
            expr: ExprView<'_, Integer>,
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

    fn revert(
        &mut self,
        DatabaseState {
            clauses_len,
            clause_set_len,
            stor_len,
            nimap_state,
        }: DatabaseState,
    ) {
        self.clauses.truncate(clauses_len.len());
        for (i, len) in clauses_len.into_iter().enumerate() {
            self.clauses[i].truncate(len);
        }
        self.dup_checker.truncate(clause_set_len);
        self.stor.truncate(stor_len);
        self.nimap.revert(nimap_state);
        // `self.prover: Prover` does not store any persistent data.
    }

    fn state(&self) -> DatabaseState {
        DatabaseState {
            clauses_len: self.clauses.values().map(|v| v.len()).collect(),
            clause_set_len: self.dup_checker.len(),
            stor_len: self.stor.len(),
            nimap_state: self.nimap.state(),
        }
    }
}

impl<T: Atom> Default for Database<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Debug> Debug for Database<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Database")
            .field("clauses", &self.clauses)
            .field("dup_checker", &self.dup_checker)
            .field("stor", &self.stor)
            .field("nimap", &self.nimap)
            .field("revert_point", &self.revert_point)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, PartialEq, Eq)]
struct DatabaseState {
    clauses_len: Vec<usize>,
    clause_set_len: usize,
    stor_len: TermStorageLen,
    nimap_state: NameIntMapState,
}

#[derive(Debug, Default)]
struct DuplicateClauseChecker {
    seen: IndexSet<Clause<Integer>>,

    /// Temporary buffer for granting [`Integer`] to variables.
    vars: Vec<Integer>,
}

impl DuplicateClauseChecker {
    /// Returns true if the given clause is new, has not been seen before.
    fn insert(&mut self, clause: Clause<Integer>) -> bool {
        let canonical_clause = clause.map(&mut |t| {
            if !t.is_variable() {
                t
            } else if let Some(found) = self.vars.iter().find(|&&var| var == t) {
                *found
            } else {
                let next_int = self.vars.len() as u32;
                let int = Integer::variable(next_int);
                self.vars.push(int);
                int
            }
        });
        let is_new = self.seen.insert(canonical_clause);
        self.vars.clear();
        is_new
    }

    fn len(&self) -> usize {
        self.seen.len()
    }

    fn truncate(&mut self, len: usize) {
        self.seen.truncate(len);
    }
}

/// Turns variables into `_$0`, `_$1`, and so on using the given canonical_var function.
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

#[derive(Clone)]
pub struct ClauseIter<'a, T> {
    clauses: &'a IndexMap<Predicate<Integer>, Vec<ClauseId>>,
    stor: &'a TermStorage<Integer>,
    nimap: &'a NameIntMap<T>,
    i: usize,
    j: usize,
}

impl<'a, T> Iterator for ClauseIter<'a, T> {
    type Item = ClauseRef<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let id = loop {
            let (_, group) = self.clauses.get_index(self.i)?;

            if let Some(id) = group.get(self.j) {
                self.j += 1;
                break *id;
            }

            self.i += 1;
            self.j = 0;
        };

        Some(ClauseRef {
            id,
            stor: self.stor,
            nimap: self.nimap,
        })
    }
}

impl<T> FusedIterator for ClauseIter<'_, T> {}

pub struct ClauseRef<'a, T> {
    id: ClauseId,
    stor: &'a TermStorage<Integer>,
    nimap: &'a NameIntMap<T>,
}

impl<'a, T: Atom> ClauseRef<'a, T> {
    pub fn head(&self) -> NamedTermView<'a, T> {
        let head = self.stor.get_term(self.id.head);
        NamedTermView::new(head, self.nimap)
    }

    pub fn body(&self) -> Option<NamedExprView<'a, T>> {
        self.id.body.map(|id| {
            let body = self.stor.get_expr(id);
            NamedExprView::new(body, self.nimap)
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

        let head = self.stor.get_term(self.id.head);
        d.field("head", &NamedTermView::new(head, self.nimap));

        if let Some(body) = self.id.body {
            let body = self.stor.get_expr(body);
            d.field("body", &NamedExprView::new(body, self.nimap));
        }

        d.finish()
    }
}

pub struct NamedTermViewIter<'a, T> {
    term_iter: TermViewIter<'a, Integer>,
    nimap: &'a NameIntMap<T>,
}

impl<'a, T: Atom> Iterator for NamedTermViewIter<'a, T> {
    type Item = NamedTermView<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.term_iter
            .next()
            .map(|view| NamedTermView::new(view, self.nimap))
    }
}

impl<T: Atom> FusedIterator for NamedTermViewIter<'_, T> {}

#[cfg(test)]
mod str_atom_tests {
    use crate::{parse, NameIn};

    type Interner = any_intern::DroplessInterner;
    type Database<'int> = crate::Database<NameIn<'int, Interner>>;
    type ProveCx<'a, 'int> = crate::ProveCx<'a, NameIn<'int, Interner>>;
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

        let mut db = Database::new();
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
            let len = db.stor.len();
            assert_query(&mut db, &interner);
            assert_eq!(db.stor.len(), len);
        }
    }

    #[test]
    fn test_not_expression() {
        let mut db = Database::new();
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
        let mut db = Database::new();
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
        let mut db = Database::new();
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
        let mut db = Database::new();
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
    fn test_simple_recursion() {
        let mut db = Database::new();
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
        let mut db = Database::new();
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
        let mut db = Database::new();
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
        let mut db = Database::new();
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
    fn test_discarding_uncomitted_change() {
        let mut db = Database::new();
        let interner = Interner::new();

        let clause: Clause<'_> = parse::parse_str("f(a).", &interner).unwrap();
        db.insert_clause(clause);
        let fa_state = db.state();
        db.commit();

        let clause: Clause<'_> = parse::parse_str("f(b).", &interner).unwrap();
        db.insert_clause(clause);

        let query: Expr<'_> = parse::parse_str("f($X).", &interner).unwrap();
        let answer = collect_answer(db.query(query));

        // `f(b).` was discarded.
        let expected = [["$X = a"]];
        assert_eq!(answer, expected);
        assert_eq!(db.state(), fa_state);
    }

    // === Test helper functions ===

    fn insert_dataset<'int>(db: &mut Database<'int>, interner: &'int Interner, text: &str) {
        let dataset: ClauseDataset<'int> = parse::parse_str(text, interner).unwrap();
        db.insert_dataset(dataset);
        db.commit();
    }

    fn collect_answer(mut cx: ProveCx<'_, '_>) -> Vec<Vec<String>> {
        let mut v = Vec::new();
        while let Some(eval) = cx.prove_next() {
            let x = eval.map(|assign| assign.to_string()).collect::<Vec<_>>();
            v.push(x);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use crate::{Atom, Clause, ClauseDataset, Database, Expr, ProveCx, Term};

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

        let mut db = Database::new();

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
        insert_dataset(
            &mut db,
            crate::ClauseDataset(vec![
                child_a_b,
                child_b_c,
                child_c_d,
                descend_x_y,
                descend_x_z,
            ]),
        );

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

    fn insert_dataset<T: Atom>(db: &mut Database<T>, dataset: ClauseDataset<T>) {
        db.insert_dataset(dataset);
        db.commit();
    }

    fn collect_answer<T: Atom>(mut cx: ProveCx<'_, T>) -> Vec<Vec<(Term<T>, Term<T>)>> {
        let mut v = Vec::new();
        while let Some(eval) = cx.prove_next() {
            let pairs = eval.map(|assign| (assign.lhs(), assign.rhs())).collect();
            v.push(pairs);
        }
        v
    }
}
