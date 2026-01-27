use super::{
    prover::{
        Integer, NameIntMap, NameIntMapState, ProveCx, Prover,
        format::{NamedExprView, NamedTermView},
    },
    repr::{ClauseId, TermStorage, TermStorageLen},
};
use crate::{
    ClauseDatasetIn, ClauseIn, DefaultInterner, ExprIn, Int2Name, Intern, Map,
    parse::{
        VAR_PREFIX,
        repr::{Clause, Expr, Predicate, Term},
        text::Name,
    },
    prove::repr::{ExprKind, ExprView, TermView, TermViewIter},
};
use indexmap::{IndexMap, IndexSet};
use logic_eval_util::reference::Ref;
use std::{
    fmt::{self, Write},
    iter,
};

#[derive(Debug)]
pub struct Database<'int, Int: Intern = DefaultInterner> {
    /// Clause id dataset.
    clauses: IndexMap<Predicate<Integer>, Vec<ClauseId>>,

    /// We do not allow duplicated clauses in the dataset.
    clause_texts: IndexSet<String>,

    /// Term and expression storage.
    stor: TermStorage<Integer>,

    /// Proof search engine.
    prover: Prover,

    /// Mappings between [`Name`] and [`Int`].
    ///
    /// [`Int`] is internally used for fast comparison, but we need to get it
    /// back to [`Name`] for the clients.
    nimap: NameIntMap<'int, Int>,

    /// States of DB's fields.
    ///
    /// This is used when we discard some changes on the DB.
    revert_point: Option<DatabaseState>,

    interner: Ref<'int, Int>,
}

impl Database<'static> {
    /// Creates a database with default string interner.
    ///
    /// Because creating database through this method makes a leaked memory, you should not forget
    /// to call [`dealloc`](Self::dealloc).
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let leaked = Box::leak(Box::new(DefaultInterner::default()));
        let interner = Ref::from_mut(leaked);

        Self {
            clauses: IndexMap::default(),
            clause_texts: IndexSet::default(),
            stor: TermStorage::new(),
            prover: Prover::new(),
            nimap: NameIntMap::new(interner),
            revert_point: None,
            interner,
        }
    }

    #[rustfmt::skip]
    pub fn dealloc(self) {
        // Prevents us from accidently implementing Clone for the type. Deallocation relies on the
        // fact that the type cannot be cloned, so that we have only one instance.
        #[allow(dead_code)]
        {
            struct ImplDetector<T>(std::marker::PhantomData<T>);
            trait NotClone { const IS_CLONE: bool = false; }
            impl<T> NotClone for ImplDetector<T> {}
            impl<T: Clone> ImplDetector<T> { const IS_CLONE: bool = true; }
            const _: () = const { assert!(!ImplDetector::<Database<'static>>::IS_CLONE) };
        }

        // Safety:
        // * There's only one instance and we got the ownership of the instance.
        let _ = unsafe { Box::from_raw(self.interner.as_ptr()) };
    }
}

impl<'int, Int: Intern> Database<'int, Int> {
    pub fn with_interner(interner: &'int Int) -> Self {
        let interner = Ref::from_ref(interner);

        Self {
            clauses: IndexMap::default(),
            clause_texts: IndexSet::default(),
            stor: TermStorage::new(),
            prover: Prover::new(),
            nimap: NameIntMap::new(interner),
            revert_point: None,
            interner,
        }
    }

    pub fn interner(&self) -> &'int Int {
        self.interner.as_ref()
    }

    pub fn terms(&self) -> NamedTermViewIter<'_, 'int, Int> {
        NamedTermViewIter {
            term_iter: self.stor.terms.terms(),
            int2name: &self.nimap.int2name,
        }
    }

    pub fn clauses(&self) -> ClauseIter<'_, 'int, Int> {
        ClauseIter {
            clauses: &self.clauses,
            stor: &self.stor,
            int2name: &self.nimap.int2name,
            i: 0,
            j: 0,
        }
    }

    pub fn insert_dataset(&mut self, dataset: ClauseDatasetIn<'int, Int>) {
        for clause in dataset {
            self.insert_clause(clause);
        }
    }

    pub fn insert_clause(&mut self, clause: ClauseIn<'int, Int>) {
        // Saves current state. We will revert DB when the change is not
        // committed.
        if self.revert_point.is_none() {
            self.revert_point = Some(self.state());
        }

        // If this DB contains the given clause, then returns.
        let serialized = if let Some(converted) =
            Clause::convert_var_into_num(&clause, self.interner.as_ref())
        {
            converted.to_string()
        } else {
            clause.to_string()
        };
        if !self.clause_texts.insert(serialized) {
            return;
        }

        let clause = clause.map(&mut |name| self.nimap.name_to_int(name));

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

    pub fn query(&mut self, expr: ExprIn<'int, Int>) -> ProveCx<'_, 'int, Int> {
        // Discards uncomitted changes.
        if let Some(revert_point) = self.revert_point.take() {
            self.revert(revert_point);
        }

        self.prover
            .prove(expr, &self.clauses, &mut self.stor, &mut self.nimap)
    }

    pub fn commit(&mut self) {
        self.revert_point.take();
    }

    /// * sanitize - Removes unacceptable characters from prolog.
    pub fn to_prolog<F: FnMut(&str) -> &str>(&self, sanitize: F) -> String {
        let mut prolog_text = String::new();

        let mut conv_map = ConversionMap {
            int_to_str: Map::default(),
            sanitized_to_suffix: Map::default(),
            int2name: &self.nimap.int2name,
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

        struct ConversionMap<'a, 'int, Int: Intern, F> {
            int_to_str: Map<Integer, String>,
            // e.g. 0 -> No suffix, 1 -> _1, 2 -> _2, ...
            sanitized_to_suffix: Map<&'a str, u32>,
            int2name: &'a Int2Name<'int, Int>,
            sanitizer: F,
        }

        impl<Int, F> ConversionMap<'_, '_, Int, F>
        where
            Int: Intern,
            F: FnMut(&str) -> &str,
        {
            fn int_to_str(&mut self, int: Integer) -> &str {
                self.int_to_str.entry(int).or_insert_with(|| {
                    let name = self.int2name.get(&int).unwrap();
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

        fn write_term<Int, F>(
            term: TermView<'_, Integer>,
            conv_map: &mut ConversionMap<'_, '_, Int, F>,
            prolog_text: &mut String,
        ) where
            Int: Intern,
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

        fn write_expr<Int, F>(
            expr: ExprView<'_, Integer>,
            conv_map: &mut ConversionMap<'_, '_, Int, F>,
            prolog_text: &mut String,
        ) where
            Int: Intern,
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
            clause_texts_len,
            stor_len,
            nimap_state,
        }: DatabaseState,
    ) {
        self.clauses.truncate(clauses_len.len());
        for (i, len) in clauses_len.into_iter().enumerate() {
            self.clauses[i].truncate(len);
        }
        self.clause_texts.truncate(clause_texts_len);
        self.stor.truncate(stor_len);
        self.nimap.revert(nimap_state);
        // `self.prover: Prover` does not store any persistent data.
    }

    fn state(&self) -> DatabaseState {
        DatabaseState {
            clauses_len: self.clauses.values().map(|v| v.len()).collect(),
            clause_texts_len: self.clause_texts.len(),
            stor_len: self.stor.len(),
            nimap_state: self.nimap.state(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct DatabaseState {
    clauses_len: Vec<usize>,
    clause_texts_len: usize,
    stor_len: TermStorageLen,
    nimap_state: NameIntMapState,
}

#[derive(Clone)]
pub struct ClauseIter<'a, 'int, Int: Intern> {
    clauses: &'a IndexMap<Predicate<Integer>, Vec<ClauseId>>,
    stor: &'a TermStorage<Integer>,
    int2name: &'a Int2Name<'int, Int>,
    i: usize,
    j: usize,
}

impl<'a, 'int, Int: Intern> Iterator for ClauseIter<'a, 'int, Int> {
    type Item = ClauseRef<'a, 'int, Int>;

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
            int2name: self.int2name,
        })
    }
}

impl<Int: Intern> iter::FusedIterator for ClauseIter<'_, '_, Int> {}

pub struct ClauseRef<'a, 'int, Int: Intern> {
    id: ClauseId,
    stor: &'a TermStorage<Integer>,
    int2name: &'a Int2Name<'int, Int>,
}

impl<'a, 'int, Int: Intern> ClauseRef<'a, 'int, Int> {
    pub fn head(&self) -> NamedTermView<'a, 'int, Int> {
        let head = self.stor.get_term(self.id.head);
        NamedTermView::new(head, self.int2name)
    }

    pub fn body(&self) -> Option<NamedExprView<'a, 'int, Int>> {
        self.id.body.map(|id| {
            let body = self.stor.get_expr(id);
            NamedExprView::new(body, self.int2name)
        })
    }
}

impl<Int: Intern> fmt::Display for ClauseRef<'_, '_, Int> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.head(), f)?;

        if let Some(body) = self.body() {
            f.write_str(" :- ")?;
            fmt::Display::fmt(&body, f)?
        }

        f.write_char('.')
    }
}

impl<Int: Intern> fmt::Debug for ClauseRef<'_, '_, Int> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("Clause");

        let head = self.stor.get_term(self.id.head);
        d.field("head", &NamedTermView::new(head, self.int2name));

        if let Some(body) = self.id.body {
            let body = self.stor.get_expr(body);
            d.field("body", &NamedExprView::new(body, self.int2name));
        }

        d.finish()
    }
}

impl Clause<Name<()>> {
    /// Turns variables into `_$0`, `_$1`, and so on.
    pub(crate) fn convert_var_into_num<'int, Int: Intern>(
        this: &ClauseIn<'int, Int>,
        interner: &'int Int,
    ) -> Option<ClauseIn<'int, Int>> {
        let mut cloned: Option<ClauseIn<'int, Int>> = None;

        let mut i = 0;

        while let Some(var) = find_var_in_clause(cloned.as_ref().unwrap_or(this)) {
            let from = var.clone();

            let mut convert = |term: &Term<Name<_>>| {
                (term == &from).then_some(Term {
                    functor: Name::with_intern(&format!("_{VAR_PREFIX}{i}"), interner),
                    args: [].into(),
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

        fn find_var_in_clause<T: AsRef<str>>(clause: &Clause<Name<T>>) -> Option<&Term<Name<T>>> {
            let var = find_var_in_term(&clause.head);
            if var.is_some() {
                return var;
            }
            find_var_in_expr(clause.body.as_ref()?)
        }

        fn find_var_in_expr<T: AsRef<str>>(expr: &Expr<Name<T>>) -> Option<&Term<Name<T>>> {
            match expr {
                Expr::Term(term) => find_var_in_term(term),
                Expr::Not(inner) => find_var_in_expr(inner),
                Expr::And(args) | Expr::Or(args) => args.iter().find_map(find_var_in_expr),
            }
        }

        fn find_var_in_term<T: AsRef<str>>(term: &Term<Name<T>>) -> Option<&Term<Name<T>>> {
            const _: () = assert!(VAR_PREFIX == '$');

            if term.is_variable() && !term.functor.as_ref().starts_with("_$") {
                Some(term)
            } else {
                term.args.iter().find_map(find_var_in_term)
            }
        }
    }
}

pub struct NamedTermViewIter<'a, 'int, Int: Intern> {
    term_iter: TermViewIter<'a, Integer>,
    int2name: &'a Int2Name<'int, Int>,
}

impl<'a, 'int, Int: Intern> Iterator for NamedTermViewIter<'a, 'int, Int> {
    type Item = NamedTermView<'a, 'int, Int>;

    fn next(&mut self) -> Option<Self::Item> {
        self.term_iter
            .next()
            .map(|view| NamedTermView::new(view, self.int2name))
    }
}

impl<Int: Intern> iter::FusedIterator for NamedTermViewIter<'_, '_, Int> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{
        self,
        repr::{Clause, ClauseDataset, Expr},
    };
    use any_intern::DroplessInterner;

    #[test]
    fn test_parse() {
        fn assert(text: &str, interner: &impl Intern) {
            let clause: Clause<Name<_>> = parse::parse_str(text, interner).unwrap();
            assert_eq!(text, clause.to_string());
        }

        let interner = DroplessInterner::default();

        assert("f.", &interner);
        assert("f(a, b).", &interner);
        assert("f(a, b) :- f.", &interner);
        assert("f(a, b) :- f(a).", &interner);
        assert("f(a, b) :- f(a), f(b).", &interner);
        assert("f(a, b) :- f(a); f(b).", &interner);
        assert("f(a, b) :- f(a), (f(b); f(c)).", &interner);
    }

    #[test]
    fn test_serial_queries() {
        fn insert<Int: Intern>(db: &mut Database<'_, Int>) {
            insert_dataset(
                db,
                r"
                f(a).
                f(b).
                g($X) :- f($X).
                ",
            );
        }

        fn query<Int: Intern>(db: &mut Database<'_, Int>) {
            let query = "g($X).";
            let query: Expr<Name<_>> = parse::parse_str(query, db.interner()).unwrap();
            let answer = collect_answer(db.query(query));

            let expected = [["$X = a"], ["$X = b"]];

            assert_eq!(answer, expected);
        }

        let mut db = Database::new();

        insert(&mut db);
        let org_stor_len = db.stor.len();
        query(&mut db);
        debug_assert_eq!(org_stor_len, db.stor.len());

        insert(&mut db);
        debug_assert_eq!(org_stor_len, db.stor.len());
        query(&mut db);
        debug_assert_eq!(org_stor_len, db.stor.len());

        db.dealloc();
    }

    #[test]
    fn test_various_expressions() {
        test_not_expression();
        test_and_expression();
        test_or_expression();
        test_mixed_expression();
    }

    fn test_not_expression() {
        let mut db = Database::new();

        insert_dataset(
            &mut db,
            r"
            g(a).
            f($X) :- \+ g($X).
            ",
        );

        let query = "f(a).";
        let query: Expr<Name<_>> = parse::parse_str(query, db.interner()).unwrap();
        let answer = collect_answer(db.query(query));
        assert!(answer.is_empty());

        let query = "f(b).";
        let query: Expr<Name<_>> = parse::parse_str(query, db.interner()).unwrap();
        let answer = collect_answer(db.query(query));
        assert_eq!(answer.len(), 1);

        db.dealloc();
    }

    fn test_and_expression() {
        let mut db = Database::new();

        insert_dataset(
            &mut db,
            r"
            g(a).
            g(b).
            h(b).
            f($X) :- g($X), h($X).
            ",
        );

        let query = "f($X).";
        let query: Expr<Name<_>> = parse::parse_str(query, db.interner()).unwrap();
        let answer = collect_answer(db.query(query));

        let expected = [["$X = b"]];
        assert_eq!(answer, expected);

        db.dealloc();
    }

    fn test_or_expression() {
        let mut db = Database::new();

        insert_dataset(
            &mut db,
            r"
            g(a).
            h(b).
            f($X) :- g($X); h($X).
            ",
        );

        let query = "f($X).";
        let query: Expr<Name<_>> = parse::parse_str(query, db.interner()).unwrap();
        let answer = collect_answer(db.query(query));

        let expected = [["$X = a"], ["$X = b"]];
        assert_eq!(answer, expected);

        db.dealloc();
    }

    fn test_mixed_expression() {
        let mut db = Database::new();

        insert_dataset(
            &mut db,
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

        let query = "f($X).";
        let query: Expr<Name<_>> = parse::parse_str(query, db.interner()).unwrap();
        let answer = collect_answer(db.query(query));

        let expected = [["$X = b"]];
        assert_eq!(answer, expected);

        db.dealloc();
    }

    #[test]
    fn test_recursion() {
        test_simple_recursion();
        test_right_recursion();
    }

    fn test_simple_recursion() {
        let mut db = Database::new();

        insert_dataset(
            &mut db,
            r"
            impl(Clone, a).
            impl(Clone, b).
            impl(Clone, c).
            impl(Clone, Vec($T)) :- impl(Clone, $T).
            ",
        );

        let query = "impl(Clone, $T).";
        let query: Expr<Name<_>> = parse::parse_str(query, db.interner()).unwrap();
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

        drop(cx);
        db.dealloc();
    }

    fn test_right_recursion() {
        let mut db = Database::new();

        insert_dataset(
            &mut db,
            r"
            child(a, b).
            child(b, c).
            child(c, d).
            descend($X, $Y) :- child($X, $Y).
            descend($X, $Z) :- child($X, $Y), descend($Y, $Z).
            ",
        );

        let query = "descend($X, $Y).";
        let query: Expr<Name<_>> = parse::parse_str(query, db.interner()).unwrap();
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

        db.dealloc();
    }

    #[test]
    fn test_discarding_uncomitted_change() {
        let mut db = Database::new();

        let text = "f(a).";
        let clause = parse::parse_str(text, db.interner()).unwrap();
        db.insert_clause(clause);
        let fa_state = db.state();
        db.commit();

        let text = "f(b).";
        let clause = parse::parse_str(text, db.interner()).unwrap();
        db.insert_clause(clause);

        let query = "f($X).";
        let query: Expr<Name<_>> = parse::parse_str(query, db.interner()).unwrap();
        let answer = collect_answer(db.query(query));

        // `f(b).` was discarded.
        let expected = [["$X = a"]];
        assert_eq!(answer, expected);
        assert_eq!(db.state(), fa_state);

        db.dealloc();
    }

    fn insert_dataset<Int: Intern>(db: &mut Database<'_, Int>, text: &str) {
        let dataset: ClauseDataset<Name<_>> = parse::parse_str(text, db.interner()).unwrap();
        db.insert_dataset(dataset);
        db.commit();
    }

    fn collect_answer<Int: Intern>(mut cx: ProveCx<'_, '_, Int>) -> Vec<Vec<String>> {
        let mut v = Vec::new();
        while let Some(eval) = cx.prove_next() {
            let x = eval.map(|assign| assign.to_string()).collect::<Vec<_>>();
            v.push(x);
        }
        v
    }
}
