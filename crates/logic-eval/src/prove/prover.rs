use super::repr::{
    ApplyResult, ClauseId, ExprId, ExprKind, ExprView, TermDeepView, TermElem, TermId, TermStorage,
    TermStorageLen, TermView, TermViewMut, UniqueTermArray,
};
use crate::{
    Map,
    parse::{
        VAR_PREFIX,
        repr::{Expr, Predicate, Term},
        text::Name,
    },
};
use indexmap::IndexMap;
use std::{
    collections::VecDeque,
    fmt::{self, Write},
    iter,
    ops::{self, Range},
};

pub(crate) type ClauseMap = IndexMap<Predicate<Int>, Vec<ClauseId>>;

#[derive(Debug)]
pub(crate) struct Prover {
    uni_op: UnificationOperator,

    /// Nodes created during proof search.
    nodes: Vec<Node>,

    /// Variable assignments.
    ///
    /// For example, `assignment[X] = a` means that `X(term id)` is assigned to
    /// `a(term id)`. If a value is identical to its index, it means the term is
    /// not assigned to anything.
    assignments: Vec<usize>,

    /// A given query.
    query: ExprId,

    /// Variables in the root node(query).
    ///
    /// This could be used to find what terms these variables are assigned to.
    query_vars: Vec<TermId>,

    /// Task queue containing node index.
    queue: VecDeque<usize>,

    /// A buffer containing mapping between variables and temporary variables.
    ///
    /// This buffer is used when we convert variables into temporary variables
    /// for a clause. After the conversion, this buffer get empty.
    temp_var_buf: Map<TermId, TermId>,

    /// A monotonically increasing integer that is used for generating
    /// temporary variables.
    temp_var_int: u32,
}

impl Prover {
    pub(crate) fn new() -> Self {
        Self {
            uni_op: UnificationOperator::new(),
            nodes: Vec::new(),
            assignments: Vec::new(),
            query: ExprId(0),
            query_vars: Vec::new(),
            queue: VecDeque::new(),
            temp_var_buf: Map::default(),
            temp_var_int: 0,
        }
    }

    fn clear(&mut self) {
        self.uni_op.clear();
        self.nodes.clear();
        self.assignments.clear();
        self.query_vars.clear();
        self.queue.clear();
    }

    pub(crate) fn prove<'a>(
        &'a mut self,
        query: Expr<Name>,
        clause_map: &'a ClauseMap,
        stor: &'a mut TermStorage<Int>,
        nimap: &'a mut NameIntMap,
    ) -> ProveCx<'a> {
        self.clear();

        let old_nimap_state = nimap.state();
        let query = query.map(&mut |name| nimap.name_to_int(name));

        let old_stor_len = stor.len();
        self.query = stor.insert_expr(query);

        stor.get_expr(self.query)
            .with_term(&mut |term: TermView<'_, Int>| {
                term.with_variable(&mut |term| self.query_vars.push(term.id));
            });

        self.nodes.push(Node {
            kind: NodeKind::Expr(self.query),
            uni_delta: 0..0,
            parent: self.nodes.len(),
        });
        self.queue.push_back(0);

        ProveCx::new(self, clause_map, stor, nimap, old_stor_len, old_nimap_state)
    }

    /// Evaluates the given node with all possible clauses in the clause
    /// dataset, then returns whether a proof search path is complete or not.
    ///
    /// If it reached an end of paths, it returns proof search result within
    /// `Some`. The proof search result is either true or false, which means
    /// the expression in the given node is true or not.
    fn evaluate_node(
        &mut self,
        node_index: usize,
        clause_map: &ClauseMap,
        stor: &mut TermStorage<Int>,
    ) -> Option<bool> {
        let node = self.nodes[node_index].clone();
        let node_expr = match node.kind {
            NodeKind::Expr(expr_id) => expr_id,
            NodeKind::Leaf(eval) => {
                self.find_assignment(node_index);
                return Some(eval);
            }
        };

        let predicate = stor.get_expr(node_expr).leftmost_term().predicate();
        let similar_clauses = if let Some(v) = clause_map.get(&predicate) {
            v.as_slice()
        } else {
            &[]
        };

        let old_len = self.nodes.len();
        for clause in similar_clauses {
            let head = stor.get_term(clause.head);

            if !stor.get_expr(node_expr).is_unifiable(head) {
                continue;
            }

            let clause = Self::convert_var_into_temp(
                *clause,
                stor,
                &mut self.temp_var_buf,
                &mut self.temp_var_int,
            );
            if let Some(new_node) = self.unify_node_with_clause(node_index, clause, stor) {
                self.nodes.push(new_node);
                self.queue.push_back(self.nodes.len() - 1);
            }
        }

        // We may need to apply true or false to the leftmost term of the node
        // expression due to unification failure or exhaustive search.
        // - Unification failure means the leftmost term should be false.
        // - But we need to consider exhaustive search possibility at the same
        //   time.

        let expr = stor.get_expr(node_expr);
        let eval = self.nodes.len() > old_len;
        let mut need_apply = None;

        let lost_possibility = match assume_leftmost_term(expr, eval) {
            AssumeResult::Incomplete { lost } => lost,
            AssumeResult::Complete { lost, .. } => lost,
        };
        if lost_possibility {
            need_apply = Some(!eval);
        } else if !eval {
            need_apply = Some(false);
        }

        if let Some(to) = need_apply {
            let mut expr = stor.get_expr_mut(node_expr);
            let kind = match expr.apply_to_leftmost_term(to) {
                ApplyResult::Expr => NodeKind::Expr(expr.id()),
                ApplyResult::Complete(eval) => NodeKind::Leaf(eval),
            };
            self.nodes.push(Node {
                kind,
                uni_delta: 0..0,
                parent: node_index,
            });
            self.queue.push_back(self.nodes.len() - 1);
        }

        return None;

        // === Internal helper functions ===

        enum AssumeResult {
            /// The whole expression could not completely evaluated even though
            /// the assumption is realized.
            Incomplete {
                /// Whether or not the assumption will make us lose some search
                /// possibilities.
                lost: bool,
            },

            /// The whole expression will be completely evaluated if the
            /// assumption is realized.
            Complete {
                /// Evalauted as true or false.
                eval: bool,
                lost: bool,
            },
        }

        fn assume_leftmost_term(expr: ExprView<'_, Int>, to: bool) -> AssumeResult {
            match expr.as_kind() {
                ExprKind::Term(_) => AssumeResult::Complete {
                    eval: to,
                    lost: false,
                },
                ExprKind::Not(inner) => match assume_leftmost_term(inner, to) {
                    res @ AssumeResult::Incomplete { .. } => res,
                    AssumeResult::Complete { eval, lost } => {
                        AssumeResult::Complete { eval: !eval, lost }
                    }
                },
                ExprKind::And(mut args) => {
                    // Unlike 'Or', even if 'And' contains variables and the
                    // whole expression will be evaluated false, those variables
                    // must be ignored. They don't belong to 'lost'.
                    match assume_leftmost_term(args.next().unwrap(), to) {
                        res @ AssumeResult::Incomplete { .. } => res,
                        AssumeResult::Complete { eval, lost } => {
                            if !eval {
                                AssumeResult::Complete { eval: false, lost }
                            } else {
                                AssumeResult::Incomplete { lost }
                            }
                        }
                    }
                }
                ExprKind::Or(mut args) => {
                    // The whole 'Or' is true if any argument is true. But we
                    // will lose possible search paths if we ignore right
                    // variables.
                    match assume_leftmost_term(args.next().unwrap(), to) {
                        res @ AssumeResult::Incomplete { .. } => res,
                        AssumeResult::Complete { eval, lost } => {
                            if eval {
                                let right_var = args.any(|arg| arg.contains_variable());
                                AssumeResult::Complete {
                                    eval: true,
                                    lost: lost | right_var,
                                }
                            } else {
                                AssumeResult::Incomplete { lost }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Replaces variables in a clause with other temporary variables.
    ///
    // Why we replace variables with temporary variables in clauses before
    // unifying?
    // 1. That's because variables in different clauses are actually different
    // from each other even they have the same identity. Variable's identity
    // is valid only in the one clause where they belong.
    // 2. Also, we apply this method whenever unification happens because one
    // clause can be used mupltiple times in a single proof search path. Then
    // it is considered as a different clause.
    fn convert_var_into_temp(
        mut clause_id: ClauseId,
        stor: &mut TermStorage<Int>,
        temp_var_buf: &mut Map<TermId, TermId>,
        temp_var_int: &mut u32,
    ) -> ClauseId {
        debug_assert!(temp_var_buf.is_empty());

        let mut f = |terms: &mut UniqueTermArray<Int>, term_id: TermId| {
            let term = terms.get_mut(term_id);
            if term.is_variable() {
                let src = term.id();

                temp_var_buf.entry(src).or_insert_with(|| {
                    let temp_term = Term {
                        functor: Int::temporary(*temp_var_int),
                        args: [].into(),
                    };
                    *temp_var_int += 1;
                    terms.insert(temp_term)
                });
            }
        };

        stor.get_term_mut(clause_id.head).with_terminal(&mut f);

        if let Some(body) = clause_id.body {
            stor.get_expr_mut(body).with_terminal(&mut f);
        }

        for (src, dst) in temp_var_buf.drain() {
            let mut head = stor.get_term_mut(clause_id.head);
            head.replace(src, dst);
            clause_id.head = head.id();

            if let Some(body) = clause_id.body {
                let mut body = stor.get_expr_mut(body);
                body.replace_term(src, dst);
                clause_id.body = Some(body.id());
            }
        }

        clause_id
    }

    fn unify_node_with_clause(
        &mut self,
        node_index: usize,
        clause: ClauseId,
        stor: &mut TermStorage<Int>,
    ) -> Option<Node> {
        debug_assert!(self.uni_op.ops.is_empty());

        let NodeKind::Expr(node_expr) = self.nodes[node_index].kind else {
            unreachable!()
        };

        if !stor
            .get_expr(node_expr)
            .leftmost_term()
            .unify(stor.get_term(clause.head), &mut |op| {
                self.uni_op.push_op(op)
            })
        {
            return None;
        }
        let (node_expr, clause, uni_delta) = self.uni_op.consume_ops(stor, node_expr, clause);

        if let Some(body) = clause.body {
            let mut lhs = stor.get_expr_mut(node_expr);
            lhs.replace_leftmost_term(body);
            return Some(Node {
                kind: NodeKind::Expr(lhs.id()),
                uni_delta,
                parent: node_index,
            });
        }

        let mut lhs = stor.get_expr_mut(node_expr);
        let kind = match lhs.apply_to_leftmost_term(true) {
            ApplyResult::Expr => NodeKind::Expr(lhs.id()),
            ApplyResult::Complete(eval) => NodeKind::Leaf(eval),
        };
        Some(Node {
            kind,
            uni_delta,
            parent: node_index,
        })
    }

    /// Finds all assignments from the given node to the root node.
    ///
    /// Then, the assignment information is stored at [`Self::assignments`].
    fn find_assignment(&mut self, node_index: usize) {
        // Collects unification records.
        self.assignments.clear();

        let mut cur_index = node_index;
        loop {
            let node = &self.nodes[cur_index];
            let range = node.uni_delta.clone();

            for (from, to) in self.uni_op.get_record(range).iter().cloned() {
                let (from, to) = (from.0, to.0);

                for i in self.assignments.len()..=from.max(to) {
                    self.assignments.push(i);
                }

                let root_from = find(&mut self.assignments, from);
                let root_to = find(&mut self.assignments, to);
                self.assignments[root_from] = root_to;
            }

            if node.parent == cur_index {
                break;
            }
            cur_index = node.parent;
        }

        return;

        // === Internal helper functions ===

        fn find(buf: &mut [usize], i: usize) -> usize {
            if buf[i] == i {
                i
            } else {
                let root = find(buf, buf[i]);
                buf[i] = root;
                root
            }
        }
    }
}

#[derive(Debug)]
struct UnificationOperator {
    ops: Vec<UnifyOp>,

    /// History of unification.
    ///
    /// A pair of term ids means that `pair.0` is assiend to `pair.1`. For
    /// example, `(X, a)` means `X` is assigned to `a`.
    record: Vec<(TermId, TermId)>,
}

impl UnificationOperator {
    const fn new() -> Self {
        Self {
            ops: Vec::new(),
            record: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.ops.clear();
        self.record.clear();
    }

    fn push_op(&mut self, op: UnifyOp) {
        self.ops.push(op);
    }

    #[must_use]
    fn consume_ops(
        &mut self,
        stor: &mut TermStorage<Int>,
        mut left: ExprId,
        mut right: ClauseId,
    ) -> (ExprId, ClauseId, Range<usize>) {
        let record_start = self.record.len();

        for op in self.ops.drain(..) {
            match op {
                UnifyOp::Left { from, to } => {
                    let mut expr = stor.get_expr_mut(left);
                    expr.replace_term(from, to);
                    left = expr.id();

                    self.record.push((from, to));
                }
                UnifyOp::Right { from, to } => {
                    if let Some(right_body) = right.body {
                        let mut expr = stor.get_expr_mut(right_body);
                        expr.replace_term(from, to);
                        right.body = Some(expr.id());

                        self.record.push((from, to));
                    }
                }
            }
        }

        (left, right, record_start..self.record.len())
    }

    fn get_record(&self, range: Range<usize>) -> &[(TermId, TermId)] {
        &self.record[range]
    }
}

#[derive(Debug, Clone)]
struct Node {
    kind: NodeKind,
    uni_delta: Range<usize>,
    parent: usize,
}

#[derive(Debug, Clone, Copy)]
enum NodeKind {
    /// A non-terminal node containig an expression id that needs to be
    /// evaluated.
    Expr(ExprId),

    /// A terminal node containing whether a proof path ends with true or false.
    Leaf(bool),
}

#[derive(Debug)]
enum UnifyOp {
    Left { from: TermId, to: TermId },
    Right { from: TermId, to: TermId },
}

pub struct ProveCx<'a> {
    prover: &'a mut Prover,
    clause_map: &'a ClauseMap,
    stor: &'a mut TermStorage<Int>,
    nimap: &'a mut NameIntMap,
    old_stor_len: TermStorageLen,
    old_nimap_state: NameIntMapState,
}

impl<'a> ProveCx<'a> {
    fn new(
        prover: &'a mut Prover,
        clause_map: &'a ClauseMap,
        stor: &'a mut TermStorage<Int>,
        nimap: &'a mut NameIntMap,
        old_stor_len: TermStorageLen,
        old_nimap_state: NameIntMapState,
    ) -> Self {
        Self {
            prover,
            clause_map,
            stor,
            nimap,
            old_stor_len,
            old_nimap_state,
        }
    }

    pub fn prove_next(&mut self) -> Option<EvalView<'_>> {
        while let Some(node_index) = self.prover.queue.pop_front() {
            if let Some(proof_result) =
                self.prover
                    .evaluate_node(node_index, self.clause_map, self.stor)
            {
                // Returns Some(EvalView) only if the result is TRUE.
                if proof_result {
                    return Some(EvalView {
                        query_vars: &self.prover.query_vars,
                        terms: &self.stor.terms.buf,
                        assignments: &self.prover.assignments,
                        int2name: &self.nimap.int2name,
                        start: 0,
                        end: self.prover.query_vars.len(),
                    });
                }
            }
        }
        None
    }

    pub fn is_true(mut self) -> bool {
        self.prove_next().is_some()
    }
}

impl Drop for ProveCx<'_> {
    fn drop(&mut self) {
        self.stor.truncate(self.old_stor_len.clone());
        self.nimap.revert(self.old_nimap_state.clone());
    }
}

pub struct EvalView<'a> {
    query_vars: &'a [TermId],
    terms: &'a [TermElem<Int>],
    assignments: &'a [usize],
    int2name: &'a IndexMap<Int, Name>,
    /// Inclusive
    start: usize,
    /// Exclusive
    end: usize,
}

impl EvalView<'_> {
    const fn len(&self) -> usize {
        self.end - self.start
    }
}

impl<'a> Iterator for EvalView<'a> {
    type Item = Assignment<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let from = self.query_vars[self.start];
            self.start += 1;

            Some(Assignment {
                buf: self.terms,
                from,
                assignments: self.assignments,
                int2name: self.int2name,
            })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = <Self>::len(self);
        (len, Some(len))
    }
}

impl ExactSizeIterator for EvalView<'_> {
    fn len(&self) -> usize {
        <Self>::len(self)
    }
}

impl iter::FusedIterator for EvalView<'_> {}

pub struct Assignment<'a> {
    buf: &'a [TermElem<Int>],
    from: TermId,
    assignments: &'a [usize],
    int2name: &'a IndexMap<Int, Name>,
}

impl<'a> Assignment<'a> {
    /// Creates left hand side term of the assignment.
    ///
    /// To create a term, this method could allocate memory for the term.
    pub fn lhs(&self) -> Term<Name> {
        Self::term_view_to_term(self.lhs_view(), self.int2name)
    }

    /// Creates right hand side term of the assignment.
    ///
    /// To create a term, this method could allocate memory for the term.
    pub fn rhs(&self) -> Term<Name> {
        Self::term_deep_view_to_term(self.rhs_view(), self.int2name)
    }

    /// Returns left hand side variable name of the assignment.
    ///
    /// Note that assignment's left hand side is always variable.
    pub fn get_lhs_variable(&self) -> &Name {
        let int = self.lhs_view().get_contained_variable().unwrap();
        self.int2name.get(&int).unwrap()
    }

    fn term_view_to_term(view: TermView<'_, Int>, int2name: &IndexMap<Int, Name>) -> Term<Name> {
        let functor = view.functor();
        let args = view.args();

        let functor = if let Some(name) = int2name.get(functor) {
            name.clone()
        } else {
            let mut debug_string = String::new();
            write!(&mut debug_string, "{:?}", functor).unwrap();
            debug_string.into()
        };

        let args = args
            .into_iter()
            .map(|arg| Self::term_view_to_term(arg, int2name))
            .collect();

        Term { functor, args }
    }

    fn term_deep_view_to_term(
        view: TermDeepView<'_, Int>,
        int2name: &IndexMap<Int, Name>,
    ) -> Term<Name> {
        let functor = view.functor();
        let args = view.args();

        let functor = if let Some(name) = int2name.get(functor) {
            name.clone()
        } else {
            let mut debug_string = String::new();
            write!(&mut debug_string, "{:?}", functor).unwrap();
            debug_string.into()
        };

        let args = args
            .into_iter()
            .map(|arg| Self::term_deep_view_to_term(arg, int2name))
            .collect();

        Term { functor, args }
    }

    const fn lhs_view(&self) -> TermView<'_, Int> {
        TermView {
            buf: self.buf,
            id: self.from,
        }
    }

    const fn rhs_view(&self) -> TermDeepView<'_, Int> {
        TermDeepView {
            buf: self.buf,
            links: self.assignments,
            id: self.from,
        }
    }
}

impl fmt::Display for Assignment<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let view = format::NamedTermView::new(self.lhs_view(), self.int2name);
        fmt::Display::fmt(&view, f)?;

        f.write_str(" = ")?;

        let view = format::NamedTermDeepView::new(self.rhs_view(), self.int2name);
        fmt::Display::fmt(&view, f)
    }
}

impl fmt::Debug for Assignment<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lhs = format::NamedTermView::new(self.lhs_view(), self.int2name);
        let rhs = format::NamedTermDeepView::new(self.rhs_view(), self.int2name);

        f.debug_struct("Assignment")
            .field("lhs", &lhs)
            .field("rhs", &rhs)
            .finish()
    }
}

impl ExprView<'_, Int> {
    fn is_unifiable(&self, other: TermView<'_, Int>) -> bool {
        match self.as_kind() {
            ExprKind::Term(term) => term.is_unifiable(other),
            ExprKind::Not(inner) => inner.is_unifiable(other),
            ExprKind::And(mut args) | ExprKind::Or(mut args) => {
                args.next().unwrap().is_unifiable(other)
            }
        }
    }

    fn contains_variable(&self) -> bool {
        match self.as_kind() {
            ExprKind::Term(term) => term.contains_variable(),
            ExprKind::Not(inner) => inner.contains_variable(),
            ExprKind::And(mut args) | ExprKind::Or(mut args) => {
                args.any(|arg| arg.contains_variable())
            }
        }
    }
}

impl TermView<'_, Int> {
    fn unify<F: FnMut(UnifyOp)>(self, other: Self, f: &mut F) -> bool {
        if self.is_variable() {
            f(UnifyOp::Left {
                from: self.id,
                to: other.id,
            });
            true
        } else if other.is_variable() {
            f(UnifyOp::Right {
                from: other.id,
                to: self.id,
            });
            true
        } else if self.functor() == other.functor() {
            let zip = self.args().zip(other.args());
            // Unifies only if all arguments are unifiable.
            if self.arity() == other.arity() && zip.clone().all(|(l, r)| l.is_unifiable(r)) {
                for (l, r) in zip {
                    l.unify(r, f);
                }
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    fn is_unifiable(&self, other: Self) -> bool {
        if self.is_variable() || other.is_variable() {
            true
        } else if self.functor() == other.functor() {
            if self.arity() == other.arity() {
                self.args()
                    .zip(other.args())
                    .all(|(l, r)| l.is_unifiable(r))
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Returns true if this term is a variable.
    ///
    /// e.g. Terms like `X`, `Y` will return true.
    fn is_variable(&self) -> bool {
        self.arity() == 0 && self.functor().is_variable()
    }

    /// Returns true if this term is a variable or contains variable in it.
    ///
    /// e.g. Terms like `X` of `f(X)` will return true.
    fn contains_variable(&self) -> bool {
        self.is_variable() || self.args().any(|arg| arg.contains_variable())
    }

    fn get_contained_variable(&self) -> Option<Int> {
        if self.is_variable() {
            Some(*self.functor())
        } else {
            self.args().find_map(|arg| arg.get_contained_variable())
        }
    }

    fn with_variable<F: FnMut(&Self)>(&self, f: &mut F) {
        if self.is_variable() {
            f(self);
        } else {
            for arg in self.args() {
                arg.with_variable(f);
            }
        }
    }
}

impl TermViewMut<'_, Int> {
    fn is_variable(&self) -> bool {
        self.arity() == 0 && self.functor().is_variable()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Int(u32);

impl Int {
    const VAR_FLAG: u32 = 0x1 << 31;
    const TEMPORARY_FLAG: u32 = 0x1 << 30;

    pub(crate) fn from_text(s: &Name, mut index: u32) -> Self {
        if s.is_variable() {
            index |= Self::VAR_FLAG;
        }
        Self(index)
    }

    pub(crate) const fn temporary(int: u32) -> Self {
        Self(int | Self::VAR_FLAG | Self::TEMPORARY_FLAG)
    }

    pub(crate) const fn is_variable(self) -> bool {
        (Self::VAR_FLAG & self.0) == Self::VAR_FLAG
    }

    pub(crate) const fn is_temporary_variable(self) -> bool {
        let mask: u32 = Self::VAR_FLAG | Self::TEMPORARY_FLAG;
        (mask & self.0) == mask
    }
}

impl fmt::Debug for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mask: u32 = Self::VAR_FLAG | Self::TEMPORARY_FLAG;
        let index = !mask & self.0;

        if self.is_variable() {
            f.write_char(VAR_PREFIX)?;
        }
        if self.is_temporary_variable() {
            f.write_char('#')?;
        }
        index.fmt(f)
    }
}

impl ops::AddAssign<u32> for Int {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs;
    }
}

/// Only mapping of user-input clauses and queries are stored in this map.
/// Auto-generated variables or something like that are not stored here.
#[derive(Debug)]
pub(crate) struct NameIntMap {
    pub(crate) name2int: IndexMap<Name, Int>,
    pub(crate) int2name: IndexMap<Int, Name>,
    next_int: u32,
}

impl NameIntMap {
    pub(crate) fn new() -> Self {
        Self {
            name2int: IndexMap::default(),
            int2name: IndexMap::default(),
            next_int: 0,
        }
    }

    pub(crate) fn name_to_int(&mut self, name: Name) -> Int {
        if let Some(int) = self.name2int.get(&name) {
            *int
        } else {
            let int = Int::from_text(&name, self.next_int);

            self.name2int.insert(name.clone(), int);
            self.int2name.insert(int, name);

            self.next_int += 1;
            int
        }
    }

    pub(crate) fn state(&self) -> NameIntMapState {
        NameIntMapState {
            name2int_len: self.name2int.len(),
            int2name_len: self.int2name.len(),
            next_int: self.next_int,
        }
    }

    pub(crate) fn revert(
        &mut self,
        NameIntMapState {
            name2int_len,
            int2name_len,
            next_int,
        }: NameIntMapState,
    ) {
        self.name2int.truncate(name2int_len);
        self.int2name.truncate(int2name_len);
        self.next_int = next_int;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NameIntMapState {
    name2int_len: usize,
    int2name_len: usize,
    next_int: u32,
}

pub(crate) mod format {
    use super::*;

    pub struct NamedTermView<'a> {
        view: TermView<'a, Int>,
        int2name: &'a IndexMap<Int, Name>,
    }

    impl<'a> NamedTermView<'a> {
        pub(crate) const fn new(
            view: TermView<'a, Int>,
            int2name: &'a IndexMap<Int, Name>,
        ) -> Self {
            Self { view, int2name }
        }

        pub fn is(&self, term: &Term<Name>) -> bool {
            let functor = self.view.functor();
            let Some(functor) = self.int2name.get(functor) else {
                return false;
            };

            if functor != &term.functor {
                return false;
            }

            self.args().zip(&term.args).all(|(l, r)| l.is(r))
        }

        pub fn contains(&self, term: &Term<Name>) -> bool {
            if self.is(term) {
                return true;
            }

            self.args().any(|arg| arg.contains(term))
        }

        fn args(&self) -> impl Iterator<Item = Self> {
            self.view.args().map(|arg| Self {
                view: arg,
                int2name: self.int2name,
            })
        }
    }

    impl fmt::Display for NamedTermView<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let Self { view, int2name } = self;

            let functor = view.functor();
            let args = view.args();
            let num_args = args.len();

            write_int(functor, int2name, f)?;

            if num_args > 0 {
                f.write_char('(')?;
                for (i, arg) in args.enumerate() {
                    fmt::Display::fmt(&Self::new(arg, int2name), f)?;
                    if i + 1 < num_args {
                        f.write_str(", ")?;
                    }
                }
                f.write_char(')')?;
            }
            Ok(())
        }
    }

    impl fmt::Debug for NamedTermView<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let Self { view, int2name } = self;

            let functor = view.functor();
            let args = view.args();
            let num_args = args.len();

            if num_args == 0 {
                write_int(functor, int2name, f)
            } else {
                let mut d = if let Some(name) = int2name.get(functor) {
                    f.debug_tuple(name)
                } else {
                    let mut debug_string = String::new();
                    write!(&mut debug_string, "{:?}", functor)?;
                    f.debug_tuple(&debug_string)
                };

                for arg in args {
                    d.field(&Self::new(arg, int2name));
                }
                d.finish()
            }
        }
    }

    pub(crate) struct NamedTermDeepView<'a> {
        view: TermDeepView<'a, Int>,
        int2name: &'a IndexMap<Int, Name>,
    }

    impl<'a> NamedTermDeepView<'a> {
        pub(crate) const fn new(
            view: TermDeepView<'a, Int>,
            int2name: &'a IndexMap<Int, Name>,
        ) -> Self {
            Self { view, int2name }
        }
    }

    impl fmt::Display for NamedTermDeepView<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let Self { view, int2name } = self;

            let functor = view.functor();
            let args = view.args();
            let num_args = args.len();

            write_int(functor, int2name, f)?;

            if num_args > 0 {
                f.write_char('(')?;
                for (i, arg) in args.enumerate() {
                    fmt::Display::fmt(&Self::new(arg, int2name), f)?;
                    if i + 1 < num_args {
                        f.write_str(", ")?;
                    }
                }
                f.write_char(')')?;
            }
            Ok(())
        }
    }

    impl fmt::Debug for NamedTermDeepView<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let Self { view, int2name } = self;

            let functor = view.functor();
            let args = view.args();
            let num_args = args.len();

            if num_args == 0 {
                write_int(functor, int2name, f)
            } else {
                let mut d = if let Some(name) = int2name.get(functor) {
                    f.debug_tuple(name)
                } else {
                    let mut debug_string = String::new();
                    write!(&mut debug_string, "{:?}", functor)?;
                    f.debug_tuple(&debug_string)
                };

                for arg in args {
                    d.field(&Self::new(arg, int2name));
                }
                d.finish()
            }
        }
    }

    pub struct NamedExprView<'a> {
        view: ExprView<'a, Int>,
        int2name: &'a IndexMap<Int, Name>,
    }

    impl<'a> NamedExprView<'a> {
        pub(crate) const fn new(
            view: ExprView<'a, Int>,
            int2name: &'a IndexMap<Int, Name>,
        ) -> Self {
            Self { view, int2name }
        }

        pub fn contains_term(&self, term: &Term<Name>) -> bool {
            match self.view.as_kind() {
                ExprKind::Term(view) => NamedTermView {
                    view,
                    int2name: self.int2name,
                }
                .contains(term),
                ExprKind::Not(view) => NamedExprView {
                    view,
                    int2name: self.int2name,
                }
                .contains_term(term),
                ExprKind::And(args) | ExprKind::Or(args) => args.into_iter().any(|view| {
                    NamedExprView {
                        view,
                        int2name: self.int2name,
                    }
                    .contains_term(term)
                }),
            }
        }
    }

    impl fmt::Display for NamedExprView<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let Self { view, int2name } = self;

            match view.as_kind() {
                ExprKind::Term(term) => fmt::Display::fmt(
                    &NamedTermView {
                        view: term,
                        int2name,
                    },
                    f,
                )?,
                ExprKind::Not(inner) => {
                    f.write_str("\\+ ")?;
                    if matches!(inner.as_kind(), ExprKind::And(_) | ExprKind::Or(_)) {
                        f.write_char('(')?;
                        fmt::Display::fmt(&Self::new(inner, int2name), f)?;
                        f.write_char(')')?;
                    } else {
                        fmt::Display::fmt(&Self::new(inner, int2name), f)?;
                    }
                }
                ExprKind::And(args) => {
                    let num_args = args.len();
                    for (i, arg) in args.enumerate() {
                        if matches!(arg.as_kind(), ExprKind::Or(_)) {
                            f.write_char('(')?;
                            fmt::Display::fmt(&Self::new(arg, int2name), f)?;
                            f.write_char(')')?;
                        } else {
                            fmt::Display::fmt(&Self::new(arg, int2name), f)?;
                        }
                        if i + 1 < num_args {
                            f.write_str(", ")?;
                        }
                    }
                }
                ExprKind::Or(args) => {
                    let num_args = args.len();
                    for (i, arg) in args.enumerate() {
                        fmt::Display::fmt(&Self::new(arg, int2name), f)?;
                        if i + 1 < num_args {
                            f.write_str("; ")?;
                        }
                    }
                }
            }
            Ok(())
        }
    }

    impl fmt::Debug for NamedExprView<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let Self { view, int2name } = self;

            match view.as_kind() {
                ExprKind::Term(term) => fmt::Debug::fmt(&NamedTermView::new(term, int2name), f),
                ExprKind::Not(inner) => f
                    .debug_tuple("Not")
                    .field(&NamedExprView::new(inner, int2name))
                    .finish(),
                ExprKind::And(args) => {
                    let mut d = f.debug_tuple("And");
                    for arg in args {
                        d.field(&NamedExprView::new(arg, int2name));
                    }
                    d.finish()
                }
                ExprKind::Or(args) => {
                    let mut d = f.debug_tuple("Or");
                    for arg in args {
                        d.field(&NamedExprView::new(arg, int2name));
                    }
                    d.finish()
                }
            }
        }
    }

    fn write_int(int: &Int, map: &IndexMap<Int, Name>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = map.get(int) {
            f.write_str(name)
        } else {
            fmt::Debug::fmt(int, f)
        }
    }
}
