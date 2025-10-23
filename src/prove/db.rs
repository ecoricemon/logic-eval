use super::{
    prover::{
        Int, NameIntMap, NameIntMapState, ProveCx, Prover,
        format::{NamedExprView, NamedTermView},
    },
    repr::{ClauseId, TermStorage, TermStorageLen},
};
use crate::{
    Map,
    parse::{
        VAR_PREFIX,
        repr::{Clause, ClauseDataset, Expr, Predicate, Term},
        text::Name,
    },
    prove::repr::{ExprKind, ExprView, TermView, TermViewIter},
};
use indexmap::{IndexMap, IndexSet};
use std::{
    fmt::{self, Write},
    iter,
};

#[derive(Debug)]
pub struct Database {
    /// Clause id dataset.
    clauses: IndexMap<Predicate<Int>, Vec<ClauseId>>,

    /// We do not allow duplicated clauses in the dataset.
    clause_texts: IndexSet<String>,

    /// Term and expression storage.
    stor: TermStorage<Int>,

    /// Proof search engine.
    prover: Prover,

    /// Mappings between [`Name`] and [`Int`].
    ///
    /// [`Int`] is internally used for fast comparison, but we need to get it
    /// back to [`Name`] for the clients.
    nimap: NameIntMap,

    /// States of DB's fields.
    ///
    /// This is used when we discard some changes on the DB.
    revert_point: Option<DatabaseState>,
}

impl Database {
    pub fn new() -> Self {
        Self {
            clauses: IndexMap::default(),
            clause_texts: IndexSet::default(),
            stor: TermStorage::new(),
            prover: Prover::new(),
            nimap: NameIntMap::new(),
            revert_point: None,
        }
    }

    pub fn terms(&self) -> NamedTermViewIter<'_> {
        NamedTermViewIter {
            term_iter: self.stor.terms.terms(),
            int2name: &self.nimap.int2name,
        }
    }

    pub fn clauses(&self) -> impl iter::FusedIterator<Item = ClauseRef<'_>> {
        self.clauses
            .values()
            .flat_map(|group| group.iter().cloned())
            .map(|id| ClauseRef {
                id,
                stor: &self.stor,
                int2name: &self.nimap.int2name,
            })
    }

    pub fn insert_dataset(&mut self, dataset: ClauseDataset<Name>) {
        for clause in dataset {
            self.insert_clause(clause);
        }
    }

    pub fn insert_clause(&mut self, clause: Clause<Name>) {
        // Saves current state. We will revert DB when the change is not
        // committed.
        if self.revert_point.is_none() {
            self.revert_point = Some(self.state());
        }

        // If this DB contains the given clause, then returns.
        let serialized = if let Some(converted) = clause.convert_var_into_num() {
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

    pub fn query(&mut self, expr: Expr<Name>) -> ProveCx<'_> {
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

        struct ConversionMap<'a, F> {
            int_to_str: Map<Int, String>,
            // e.g. 0 -> No suffix, 1 -> _1, 2 -> _2, ...
            sanitized_to_suffix: Map<&'a str, u32>,
            int2name: &'a IndexMap<Int, Name>,
            sanitizer: F,
        }

        impl<F: FnMut(&str) -> &str> ConversionMap<'_, F> {
            fn int_to_str(&mut self, int: Int) -> &str {
                self.int_to_str.entry(int).or_insert_with(|| {
                    let name = self.int2name.get(&int).unwrap();

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

        fn write_term<F: FnMut(&str) -> &str>(
            term: TermView<'_, Int>,
            conv_map: &mut ConversionMap<'_, F>,
            prolog_text: &mut String,
        ) {
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

        fn write_expr<F: FnMut(&str) -> &str>(
            expr: ExprView<'_, Int>,
            conv_map: &mut ConversionMap<'_, F>,
            prolog_text: &mut String,
        ) {
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

impl Default for Database {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, PartialEq, Eq)]
struct DatabaseState {
    clauses_len: Vec<usize>,
    clause_texts_len: usize,
    stor_len: TermStorageLen,
    nimap_state: NameIntMapState,
}

pub struct ClauseRef<'a> {
    id: ClauseId,
    stor: &'a TermStorage<Int>,
    int2name: &'a IndexMap<Int, Name>,
}

impl fmt::Display for ClauseRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let head = self.stor.get_term(self.id.head);
        fmt::Display::fmt(&NamedTermView::new(head, self.int2name), f)?;

        if let Some(body) = self.id.body {
            let body = self.stor.get_expr(body);
            f.write_str(" :- ")?;
            fmt::Display::fmt(&NamedExprView::new(body, self.int2name), f)?
        }
        f.write_char('.')
    }
}

impl fmt::Debug for ClauseRef<'_> {
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

impl Clause<Name> {
    /// Turns variables into `_$0`, `_$1`, and so on.
    pub(crate) fn convert_var_into_num(&self) -> Option<Self> {
        let mut cloned: Option<Self> = None;

        let mut i = 0;

        while let Some(var) = find_var_in_clause(cloned.as_ref().unwrap_or(self)) {
            let from = var.clone();

            let mut convert = |term: &Term<Name>| {
                (term == &from).then_some(Term {
                    functor: format!("_{VAR_PREFIX}{i}").into(),
                    args: [].into(),
                })
            };

            if let Some(cloned) = &mut cloned {
                cloned.replace_term(&mut convert);
            } else {
                let mut this = self.clone();
                this.replace_term(&mut convert);
                cloned = Some(this);
            }

            i += 1;
        }

        return cloned;

        // === Internal helper functions ===

        fn find_var_in_clause(clause: &Clause<Name>) -> Option<&Term<Name>> {
            let var = find_var_in_term(&clause.head);
            if var.is_some() {
                return var;
            }
            find_var_in_expr(clause.body.as_ref()?)
        }

        fn find_var_in_expr(expr: &Expr<Name>) -> Option<&Term<Name>> {
            match expr {
                Expr::Term(term) => find_var_in_term(term),
                Expr::Not(inner) => find_var_in_expr(inner),
                Expr::And(args) | Expr::Or(args) => args.iter().find_map(find_var_in_expr),
            }
        }

        fn find_var_in_term(term: &Term<Name>) -> Option<&Term<Name>> {
            const _: () = assert!(VAR_PREFIX == '$');

            if term.is_variable() && !term.functor.starts_with("_$") {
                Some(term)
            } else {
                term.args.iter().find_map(find_var_in_term)
            }
        }
    }
}

pub struct NamedTermViewIter<'a> {
    term_iter: TermViewIter<'a, Int>,
    int2name: &'a IndexMap<Int, Name>,
}

impl<'a> Iterator for NamedTermViewIter<'a> {
    type Item = NamedTermView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.term_iter
            .next()
            .map(|view| NamedTermView::new(view, self.int2name))
    }
}

impl iter::FusedIterator for NamedTermViewIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{
        self,
        repr::{Clause, Expr},
    };

    #[test]
    fn test_parse() {
        fn assert(text: &str) {
            let clause: Clause<Name> = parse::parse_str(text).unwrap();
            assert_eq!(text, clause.to_string());
        }

        assert("f.");
        assert("f(a, b).");
        assert("f(a, b) :- f.");
        assert("f(a, b) :- f(a).");
        assert("f(a, b) :- f(a), f(b).");
        assert("f(a, b) :- f(a); f(b).");
        assert("f(a, b) :- f(a), (f(b); f(c)).");
    }

    #[test]
    fn test_serial_queries() {
        let mut db = Database::new();

        fn insert(db: &mut Database) {
            insert_dataset(
                db,
                r"
                f(a).
                f(b).
                g($X) :- f($X).
                ",
            );
        }

        fn query(db: &mut Database) {
            let query = "g($X).";
            let query: Expr<Name> = parse::parse_str(query).unwrap();
            let answer = collect_answer(db.query(query));

            let expected = [["$X = a"], ["$X = b"]];

            assert_eq!(answer, expected);
        }

        insert(&mut db);
        let org_stor_len = db.stor.len();
        query(&mut db);
        debug_assert_eq!(org_stor_len, db.stor.len());

        insert(&mut db);
        debug_assert_eq!(org_stor_len, db.stor.len());
        query(&mut db);
        debug_assert_eq!(org_stor_len, db.stor.len());
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
        let query: Expr<Name> = parse::parse_str(query).unwrap();
        let answer = collect_answer(db.query(query));
        assert!(answer.is_empty());

        let query = "f(b).";
        let query: Expr<Name> = parse::parse_str(query).unwrap();
        let answer = collect_answer(db.query(query));
        assert_eq!(answer.len(), 1);
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
        let query: Expr<Name> = parse::parse_str(query).unwrap();
        let answer = collect_answer(db.query(query));

        let expected = [["$X = b"]];

        assert_eq!(answer, expected);
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
        let query: Expr<Name> = parse::parse_str(query).unwrap();
        let answer = collect_answer(db.query(query));

        let expected = [["$X = a"], ["$X = b"]];

        assert_eq!(answer, expected);
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
        let query: Expr<Name> = parse::parse_str(query).unwrap();
        let answer = collect_answer(db.query(query));

        let expected = [["$X = b"]];

        assert_eq!(answer, expected);
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
        let query: Expr<Name> = parse::parse_str(query).unwrap();
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
        let query: Expr<Name> = parse::parse_str(query).unwrap();
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

        let text = "f(a).";
        let clause = parse::parse_str(text).unwrap();
        db.insert_clause(clause);
        let fa_state = db.state();
        db.commit();

        let text = "f(b).";
        let clause = parse::parse_str(text).unwrap();
        db.insert_clause(clause);

        let query = "f($X).";
        let query: Expr<Name> = parse::parse_str(query).unwrap();
        let answer = collect_answer(db.query(query));

        // `f(b).` was discarded.
        let expected = [["$X = a"]];
        assert_eq!(answer, expected);
        assert_eq!(db.state(), fa_state);
    }

    fn insert_dataset(db: &mut Database, text: &str) {
        let dataset: ClauseDataset<Name> = parse::parse_str(text).unwrap();
        db.insert_dataset(dataset);
        db.commit();
    }

    fn collect_answer(mut cx: ProveCx<'_>) -> Vec<Vec<String>> {
        let mut v = Vec::new();
        while let Some(eval) = cx.prove_next() {
            let x = eval.map(|assign| assign.to_string()).collect::<Vec<_>>();
            v.push(x);
        }
        v
    }
}
