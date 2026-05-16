use super::{
    canonical,
    repr::{
        ApplyResult, ClauseId, ExprId, ExprKind, ExprView, TermDeepView, TermElem, TermId,
        TermStorage, TermStorageLen, TermView, TermViewMut, UniqueTermArray,
    },
    table::Table,
};
use crate::{
    parse::repr::{Expr, Predicate, Term},
    prove::table::{TableEntry, TableIndex},
    Atom, IndexMap, IndexSet, Map, VAR_PREFIX,
};
use core::{
    fmt::{self, Debug, Display, Write},
    hash::Hash,
    iter,
    ops::{self, Range},
};
use smallvec::SmallVec;
use std::collections::VecDeque;

#[derive(Debug)]
pub(crate) struct Prover {
    uni_op: UnificationOperator,

    /// Nodes created during proof search.
    nodes: Vec<Node>,

    /// Variable assignments (e.g. X = a, Y = z)
    term_assigns: TermAssignments,

    /// A given query.
    query: ExprId,

    /// Variables in the root node query.
    ///
    /// Used to find which terms these variables are assigned to.
    query_vars: Vec<TermId>,

    /// Previously returned ground query answers.
    query_answers: Vec<Vec<TermId>>,

    /// Task queue containing node index.
    queue: NodeQueue,

    /// A buffer mapping variables to temporary variables.
    ///
    /// This buffer is used when we convert variables into temporary variables for a clause. It is
    /// empty after each conversion.
    temp_var_buf: Map<TermId, TermId>,

    /// A monotonically increasing integer used to generate temporary variables.
    temp_var_int: u32,

    /// SLG resolution.
    table: Table,
}

impl Prover {
    pub(crate) fn new() -> Self {
        Self {
            uni_op: UnificationOperator::new(),
            nodes: Vec::new(),
            term_assigns: TermAssignments::default(),
            query: ExprId(0),
            query_vars: Vec::new(),
            query_answers: Vec::new(),
            queue: NodeQueue::default(),
            temp_var_buf: Map::default(),
            temp_var_int: 0,
            table: Table::default(),
        }
    }

    fn clear(&mut self) {
        self.uni_op.clear();
        self.nodes.clear();
        self.term_assigns.clear();
        self.query_vars.clear();
        self.query_answers.clear();
        self.queue.clear();
        self.table.clear();
    }

    pub(crate) fn prove<'a, T: Atom>(
        &'a mut self,
        query: Expr<T>,
        clauses: &'a IndexMap<Predicate<Integer>, Vec<ClauseId>>,
        table_clauses: &'a IndexSet<Predicate<Integer>>,
        stor: &'a mut TermStorage<Integer>,
        nimap: &'a mut NameIntMap<T>,
    ) -> ProveCx<'a, T> {
        self.clear();

        let old_nimap_state = nimap.state();
        let query = query.map(&mut |name| nimap.name_to_int(name));

        let old_stor_len = stor.len();
        self.query = stor.insert_expr(query);

        stor.get_expr(self.query)
            .with_term(&mut |term: TermView<'_, Integer>| {
                term.with_variable(|term| self.query_vars.push(term.id));
            });

        let node_kind = NodeKind::Expr(self.query);
        let node_parent = self.nodes.len();
        self.nodes.push(Node::new(node_kind, node_parent));
        self.queue.push(0);

        ProveCx {
            prover: self,
            clauses,
            table_clauses,
            stor,
            nimap,
            old_stor_len,
            old_nimap_state,
        }
    }

    /// Evaluates the given node against matching clauses or table answers, then returns whether a
    /// proof-search path is complete.
    ///
    /// If it reaches the end of a path, it returns the proof-search result in `Some`. The result is
    /// true or false, matching the evaluated value of the expression in the given node.
    fn evaluate_node(
        &mut self,
        node_index: usize,
        clauses: &IndexMap<Predicate<Integer>, Vec<ClauseId>>,
        table_clauses: &IndexSet<Predicate<Integer>>,
        stor: &mut TermStorage<Integer>,
    ) -> Option<bool> {
        let node_expr = match self.nodes[node_index].kind {
            NodeKind::Expr(expr_id) => expr_id,
            NodeKind::Leaf(eval) => {
                self.find_assignments(node_index);
                // On a successful proof, records the answer in the nearest ancestor-owned SLG
                // table entry, then notifies all waiting consumers.
                if eval {
                    self.update_answer_and_notify(node_index, stor);
                }
                return Some(eval);
            }
        };

        let node_leftmost = stor.get_expr(node_expr).leftmost_term().id;
        let node_leftmost_pred = stor.get_term(node_leftmost).predicate();
        let mut similar_clauses = &[][..];
        let mut clause_buf: SmallVec<[ClauseId; 1]> = SmallVec::new();

        // === SLG path ===
        // * Table entry - Created from non-canonical leftmost term of the node. In tabling,
        //   we use canonical variables for table keys only.

        if table_clauses.contains(&node_leftmost_pred) {
            let key = canonical::canonicalize_term_id(stor, node_leftmost);
            if let Some((_, entry)) = self.table.get_mut(&key) {
                entry.register_consumer(node_index);

                // No answers yet? The node may be woken up by notification.
                let answer_offset = self.nodes[node_index].table_answer_offset;
                let answers = entry.answers(answer_offset);
                if answers.is_empty() {
                    return None;
                }
                let next_offset = answer_offset + 1;
                self.nodes[node_index].table_answer_offset = next_offset;

                // Synthesize an answer clause, then unify it with the current node.
                let mut term = stor.get_term_mut(node_leftmost);
                let vars = term.as_view().collect_variables();
                for (var, answer) in vars.into_iter().zip(answers) {
                    term.replace(var, *answer);
                }
                clause_buf.push(ClauseId {
                    head: term.id(),
                    body: None,
                });
                similar_clauses = &clause_buf[..];

                // More answers? We'll handle them next time.
                if !entry.answers(next_offset).is_empty() {
                    self.queue.push(node_index);
                }
            } else {
                // First encounter: create a table entry, then proceed with SLD.
                if let Some(entry) = TableEntry::from_term_view(&stor.get_term(node_leftmost)) {
                    let index = self.table.register(key, entry);
                    self.nodes[node_index].table_owner = Some(index);
                }
            }
        }

        // === BFS based SLD path ===

        if similar_clauses.is_empty() {
            if let Some(v) = clauses.get(&node_leftmost_pred) {
                similar_clauses = v.as_slice()
            }
        }

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
                self.queue.push(self.nodes.len() - 1);
            }
        }

        // We may need to apply true or false to the leftmost term of the node expression due to
        // unification failure or exhaustive search.
        // - Unification failure means the leftmost term should be false.
        // - But we need to consider exhaustive search at the same time.

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
            let node_kind = match expr.apply_to_leftmost_term(to) {
                ApplyResult::Expr => NodeKind::Expr(expr.id()),
                ApplyResult::Complete(eval) => NodeKind::Leaf(eval),
            };
            let node_parent = node_index;
            self.nodes.push(Node::new(node_kind, node_parent));
            self.queue.push(self.nodes.len() - 1);
        }

        return None;

        // === Internal helper functions ===

        enum AssumeResult {
            /// The whole expression could not be completely evaluated even though the assumption is
            /// realized.
            Incomplete {
                /// Whether or not the assumption will make us lose some search possibilities.
                lost: bool,
            },

            /// The whole expression will be completely evaluated if the assumption is realized.
            Complete {
                /// Evaluated as true or false.
                eval: bool,
                lost: bool,
            },
        }

        fn assume_leftmost_term(expr: ExprView<'_, Integer>, to: bool) -> AssumeResult {
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
                    // Unlike 'Or', even if 'And' contains variables and the whole expression will
                    // be evaluated false, those variables must be ignored. They don't belong to
                    // 'lost'.
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
                    // The whole 'Or' is true if any argument is true. But we will lose possible
                    // search paths if we ignore right variables.
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

    /// Finds the nearest ancestor node that owns SLG table entry, then updates the entry and
    /// notifies all waiting consumers.
    fn update_answer_and_notify(&mut self, node_index: usize, stor: &TermStorage<Integer>) {
        let tabled_ancestor = {
            let mut cur = node_index;
            loop {
                if self.nodes[cur].table_owner.is_some() {
                    break Some(cur);
                }
                let parent = self.nodes[cur].parent;
                if parent == cur {
                    break None;
                }
                cur = parent;
            }
        };

        if let Some(ancestor) = tabled_ancestor {
            let table_index = self.nodes[ancestor].table_owner.unwrap();
            let entry = &mut self.table[table_index];
            let all_answers_concrete = entry.variables().iter().all(|&var| {
                if let Some(answer) = self.term_assigns.find(var) {
                    !stor.get_term(answer).contains_variable()
                } else {
                    false
                }
            });

            if all_answers_concrete && !entry.has_answer(&self.term_assigns) {
                entry.update_answer(&self.term_assigns);
                for i in entry.consumer_nodes() {
                    if i != node_index {
                        self.queue.push(i);
                    }
                }
            }
        }
    }

    /// Replaces variables in a clause with other temporary variables.
    //
    // Why replace variables with temporary variables in clauses before unifying?
    // 1. Variables in different clauses are distinct from each other, even when they have the same
    //    identity. A variable identity is valid only within the clause it belongs to.
    // 2. We apply this method whenever unification happens because one clause can be used multiple
    //    times in a single proof-search path. Each use is considered a distinct clause.
    fn convert_var_into_temp(
        mut clause_id: ClauseId,
        stor: &mut TermStorage<Integer>,
        temp_var_buf: &mut Map<TermId, TermId>,
        temp_var_int: &mut u32,
    ) -> ClauseId {
        debug_assert!(temp_var_buf.is_empty());

        let mut f = |terms: &mut UniqueTermArray<Integer>, term_id: TermId| {
            let term = terms.get_mut(term_id);
            if term.is_variable() {
                let src = term.id();

                temp_var_buf.entry(src).or_insert_with(|| {
                    let temp_term = Term {
                        functor: Integer::temporary(*temp_var_int),
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
        stor: &mut TermStorage<Integer>,
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
        let (node_expr, clause, uni_history) = self.uni_op.consume_ops(stor, node_expr, clause);

        if let Some(body) = clause.body {
            let mut lhs = stor.get_expr_mut(node_expr);
            lhs.replace_leftmost_term(body);
            let node_kind = NodeKind::Expr(lhs.id());
            let node_parent = node_index;
            let node = Node::new(node_kind, node_parent).with_unification_history(uni_history);
            return Some(node);
        }

        let mut lhs = stor.get_expr_mut(node_expr);
        let node_kind = match lhs.apply_to_leftmost_term(true) {
            ApplyResult::Expr => NodeKind::Expr(lhs.id()),
            ApplyResult::Complete(eval) => NodeKind::Leaf(eval),
        };
        let node_parent = node_index;
        let node = Node::new(node_kind, node_parent).with_unification_history(uni_history);
        Some(node)
    }

    /// Finds all from/to relations while traversing from the given node to the root, then adds the
    /// relations to [`TermAssignments`].
    fn find_assignments(&mut self, node_index: usize) {
        self.term_assigns.clear();

        let mut cur_index = node_index;
        loop {
            let node = &self.nodes[cur_index];
            let range = node.uni_history.clone();

            for (from, to) in self.uni_op.get_record(range).iter().cloned() {
                self.term_assigns.add(from, to);
            }

            if node.parent == cur_index {
                break;
            }
            cur_index = node.parent;
        }
    }

    /// Records the current proof result as a query answer if it is ground and not duplicated,
    /// then returns whether a new answer was recorded.
    fn record_query_answer(&mut self, stor: &mut TermStorage<Integer>) -> bool {
        let mut answer = Vec::with_capacity(self.query_vars.len());
        for &var in &self.query_vars {
            let Some(resolved) = self.materialize_assigned_term(var, stor) else {
                return false;
            };
            answer.push(resolved);
        }

        // No query vars -> empty iter -> all() returns true.
        if self.query_answers.iter().all(|seen| seen != &answer) {
            self.query_answers.push(answer);
            true
        } else {
            false
        }
    }

    /// Builds a fully substituted term for a query-side term from `term_assigns`.
    ///
    /// Examples:
    ///
    /// | assignments         |  input   |  output  |
    /// | ------------------- | :------: | :------: |
    /// | `T = Vec(a)`        |   `T`    | `Vec(a)` |
    /// | `T = a`             | `Vec(T)` | `Vec(a)` |
    /// | `T = Vec(U), U = a` |   `T`    | `Vec(a)` |
    /// | `T = Vec(U)`        |   `T`    |  `None`  |
    ///
    /// This must materialize the whole term tree, not just rewrite functors in place. The returned
    /// `TermId` always points to a ground term inserted into `stor`.
    fn materialize_assigned_term(
        &self,
        term_id: TermId,
        stor: &mut TermStorage<Integer>,
    ) -> Option<TermId> {
        let term = stor.get_term(term_id);
        if term.is_variable() {
            let resolved = self.term_assigns.find(term_id)?;
            if resolved == term_id {
                return None;
            }
            return self.materialize_assigned_term(resolved, stor);
        }

        let functor = *term.functor();
        let arg_ids = term.args().map(|arg| arg.id).collect::<Vec<_>>();
        let args = arg_ids
            .into_iter()
            .map(|arg_id| {
                self.materialize_assigned_term(arg_id, stor)
                    .map(|id| stor.get_term(id).deserialize())
            })
            .collect::<Option<Vec<_>>>()?;

        let materialized = Term { functor, args };
        Some(stor.insert_term(materialized))
    }
}

/// Manages batched unification updates between the current goal and a clause.
///
/// Unifying the leftmost term of the goal with the clause head produces [`UnifyOp`]s. Append the
/// operations in order, then apply them to the whole goal and clause at once with [`consume_ops`].
///
/// [`consume_ops`]: Self::consume_ops
#[derive(Debug)]
struct UnificationOperator {
    /// Buffered unification operations.
    ops: Vec<UnifyOp>,

    /// Unification history.
    ///
    /// This is a record of `(from, to)` pairs. Each pair means unification substituted `from` with
    /// `to`. For example, `(X, a)` means the variable `X` was substituted with `a`.
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

    /// Returns
    /// * `ExprId` - `left` after applying the operations
    /// * `ClauseId` - `right` after applying the operations
    /// * `Range<usize>` - A range of unification history (from/to pairs). You can retrieve the
    ///   from/to pairs via [`get_record`].
    ///
    /// [`get_record`]: Self::get_record
    #[must_use]
    fn consume_ops(
        &mut self,
        stor: &mut TermStorage<Integer>,
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

#[derive(Debug, Default)]
struct NodeQueue {
    inner: VecDeque<usize>,
}

impl NodeQueue {
    fn clear(&mut self) {
        self.inner.clear();
    }

    fn contains(&self, node_index: &usize) -> bool {
        self.inner.contains(node_index)
    }

    fn push(&mut self, node_index: usize) {
        if !self.contains(&node_index) {
            self.inner.push_back(node_index);
        }
    }

    fn pop(&mut self) -> Option<usize> {
        self.inner.pop_front()
    }
}

#[derive(Debug, Clone)]
struct Node {
    kind: NodeKind,
    parent: usize,

    /// A range of unification history applied to prove this node:
    /// pairs of from ([`TermId`]) -> to ([`TermId`]).
    ///
    /// You can retrieve the from/to pairs via [`UnificationOperator::get_record`].
    uni_history: Range<usize>,

    /// Table entry owned by this node, if this node is the producer of a tabled subgoal.
    table_owner: Option<TableIndex>,

    /// Number of answers already consumed from a table entry.
    table_answer_offset: usize,
}

impl Node {
    fn new(kind: NodeKind, parent: usize) -> Self {
        Self {
            kind,
            parent,
            uni_history: 0..0,
            table_owner: None,
            table_answer_offset: 0,
        }
    }

    fn with_unification_history(mut self, uni_history: Range<usize>) -> Self {
        self.uni_history = uni_history;
        self
    }
}

#[derive(Debug, Clone, Copy)]
enum NodeKind {
    /// A non-terminal node containing an expression ID that needs to be evaluated.
    Expr(ExprId),

    /// A terminal node containing whether a proof path ends with true or false.
    Leaf(bool),
}

#[derive(Debug, Default)]
pub(crate) struct TermAssignments {
    /// Union-find from/to relations.
    ///
    /// # Examples
    /// `roots[a]: a` means TermId(a) is not unified with anything.
    /// `roots[v]: w` means TermId(v) is a variable and it is unified with TermId(w).
    relations: Vec<TermId>,
}

impl TermAssignments {
    pub(crate) fn find(&self, from: TermId) -> Option<TermId> {
        let to = *self.relations.get(from.0)?;
        if from == to {
            Some(to)
        } else {
            self.find(to)
        }
    }

    pub(crate) fn find_optimize(&mut self, from: TermId) -> TermId {
        let new_len = from.0 + 1;
        for i in self.len()..new_len {
            self.relations.push(TermId(i));
        }

        let to = self.relations[from.0];
        if from == to {
            to
        } else {
            let root = self.find_optimize(to);
            self.relations[from.0] = root;
            root
        }
    }

    fn len(&self) -> usize {
        self.relations.len()
    }

    fn clear(&mut self) {
        self.relations.clear();
    }

    fn add(&mut self, from: TermId, to: TermId) {
        let root_from = self.find_optimize(from);
        let root_to = self.find_optimize(to);
        self.relations[root_from.0] = root_to;
    }
}

/// Unification operation induced by unifying the current goal term with a clause head.
#[derive(Debug)]
enum UnifyOp {
    /// Unification operation that rewrites the goal expression on the query side.
    ///
    /// Substitutes all `from`s in the goal expression with `to`.
    Left { from: TermId, to: TermId },

    /// Unification operation that rewrites the clause body on the clause side.
    ///
    /// Substitutes all `from`s in the clause's body with `to`.
    Right { from: TermId, to: TermId },
}

/// Proof-search context for a query.
pub struct ProveCx<'a, T: Atom> {
    prover: &'a mut Prover,
    clauses: &'a IndexMap<Predicate<Integer>, Vec<ClauseId>>,
    table_clauses: &'a IndexSet<Predicate<Integer>>,
    stor: &'a mut TermStorage<Integer>,
    nimap: &'a mut NameIntMap<T>,
    old_stor_len: TermStorageLen,
    old_nimap_state: NameIntMapState,
}

impl<'a, T: Atom> ProveCx<'a, T> {
    /// Returns the next proof result, if one is available.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{parse_str, ClauseDataset, Database, Expr, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let dataset: ClauseDataset<_> =
    ///     parse_str("parent(alice, bob). parent(alice, carol).", &interner).unwrap();
    /// let query: Expr<_> = parse_str("parent(alice, $Who)", &interner).unwrap();
    /// let mut db = Database::new();
    /// db.insert_dataset(dataset);
    /// db.commit();
    ///
    /// let mut cx = db.query(query);
    /// let mut answers = Vec::new();
    /// while let Some(answer) = cx.prove_next() {
    ///     for assignment in answer {
    ///         answers.push(assignment.rhs().to_string());
    ///     }
    /// }
    /// answers.sort_unstable();
    /// assert_eq!(answers, vec!["bob", "carol"]);
    /// ```
    pub fn prove_next(&mut self) -> Option<EvalView<'_, T>> {
        while let Some(node_index) = self.prover.queue.pop() {
            if let Some(proof_result) =
                self.prover
                    .evaluate_node(node_index, self.clauses, self.table_clauses, self.stor)
            {
                // Return Some(EvalView) only if the result is TRUE and yielded a new ground
                // query answer.
                if proof_result && self.prover.record_query_answer(self.stor) {
                    return Some(EvalView {
                        query_vars: &self.prover.query_vars,
                        terms: &self.stor.terms.buf,
                        term_assigns: &self.prover.term_assigns,
                        nimap: self.nimap,
                        start: 0,
                        end: self.prover.query_vars.len(),
                    });
                }
            }
        }
        None
    }

    /// Returns `true` if the query has at least one proof.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{parse_str, ClauseDataset, Database, Expr, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let dataset: ClauseDataset<_> = parse_str("sunny.", &interner).unwrap();
    /// let query: Expr<_> = parse_str("sunny", &interner).unwrap();
    /// let mut db = Database::new();
    /// db.insert_dataset(dataset);
    /// db.commit();
    ///
    /// assert!(db.query(query).is_true());
    /// ```
    pub fn is_true(mut self) -> bool {
        self.prove_next().is_some()
    }
}

impl<T: Atom> Drop for ProveCx<'_, T> {
    fn drop(&mut self) {
        self.stor.truncate(self.old_stor_len.clone());
        self.nimap.revert(self.old_nimap_state.clone());
    }
}

/// View over the assignments produced by one proof result.
pub struct EvalView<'a, T> {
    query_vars: &'a [TermId],
    terms: &'a [TermElem<Integer>],
    term_assigns: &'a TermAssignments,
    nimap: &'a NameIntMap<T>,
    /// Inclusive
    start: usize,
    /// Exclusive
    end: usize,
}

impl<T> EvalView<'_, T> {
    const fn len(&self) -> usize {
        self.end - self.start
    }
}

impl<'a, T> Iterator for EvalView<'a, T> {
    type Item = Assignment<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let from = self.query_vars[self.start];
            self.start += 1;

            Some(Assignment {
                buf: self.terms,
                from,
                term_assigns: self.term_assigns,
                nimap: self.nimap,
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

impl<T> ExactSizeIterator for EvalView<'_, T> {
    fn len(&self) -> usize {
        <Self>::len(self)
    }
}

impl<T> iter::FusedIterator for EvalView<'_, T> {}

/// A single variable assignment from a proof result.
pub struct Assignment<'a, T> {
    buf: &'a [TermElem<Integer>],
    from: TermId,
    term_assigns: &'a TermAssignments,
    nimap: &'a NameIntMap<T>,
}

impl<'a, T: 'a> Assignment<'a, T> {
    /// Returns the left-hand-side variable name of the assignment.
    ///
    /// Note that the assignment's left-hand side is always a variable.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{parse_str, ClauseDataset, Database, Expr, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let dataset: ClauseDataset<_> = parse_str("parent(alice, bob).", &interner).unwrap();
    /// let query: Expr<_> = parse_str("parent(alice, $Who)", &interner).unwrap();
    /// let mut db = Database::new();
    /// db.insert_dataset(dataset);
    /// db.commit();
    ///
    /// let mut cx = db.query(query);
    /// let assignment = cx.prove_next().unwrap().next().unwrap();
    ///
    /// assert_eq!(assignment.get_lhs_variable().as_ref(), "$Who");
    /// ```
    pub fn get_lhs_variable(&self) -> &T {
        let int = self.lhs_view().find_variable().unwrap();
        self.nimap.get_name(&int).unwrap()
    }

    const fn lhs_view(&self) -> TermView<'_, Integer> {
        TermView {
            buf: self.buf,
            id: self.from,
        }
    }

    const fn rhs_view(&self) -> TermDeepView<'_, Integer> {
        TermDeepView {
            buf: self.buf,
            term_assigns: self.term_assigns,
            id: self.from,
        }
    }
}

impl<'a, T: Atom + 'a> Assignment<'a, T> {
    /// Creates the left-hand-side term of the assignment.
    ///
    /// Creating a term may allocate memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{parse_str, ClauseDataset, Database, Expr, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let dataset: ClauseDataset<_> = parse_str("parent(alice, bob).", &interner).unwrap();
    /// let query: Expr<_> = parse_str("parent(alice, $Who)", &interner).unwrap();
    /// let mut db = Database::new();
    /// db.insert_dataset(dataset);
    /// db.commit();
    ///
    /// let mut cx = db.query(query);
    /// let assignment = cx.prove_next().unwrap().next().unwrap();
    ///
    /// assert_eq!(assignment.lhs().to_string(), "$Who");
    /// ```
    pub fn lhs(&self) -> Term<T> {
        Self::term_view_to_term(self.lhs_view(), self.nimap)
    }

    /// Creates the right-hand-side term of the assignment.
    ///
    /// Creating a term may allocate memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval::{parse_str, ClauseDataset, Database, Expr, StrInterner};
    ///
    /// let interner = StrInterner::new();
    /// let dataset: ClauseDataset<_> = parse_str("parent(alice, bob).", &interner).unwrap();
    /// let query: Expr<_> = parse_str("parent(alice, $Who)", &interner).unwrap();
    /// let mut db = Database::new();
    /// db.insert_dataset(dataset);
    /// db.commit();
    ///
    /// let mut cx = db.query(query);
    /// let assignment = cx.prove_next().unwrap().next().unwrap();
    ///
    /// assert_eq!(assignment.rhs().to_string(), "bob");
    /// ```
    pub fn rhs(&self) -> Term<T> {
        Self::term_deep_view_to_term(self.rhs_view(), self.nimap)
    }

    fn term_view_to_term(view: TermView<'_, Integer>, nimap: &NameIntMap<T>) -> Term<T> {
        let functor = view.functor();
        let args = view.args();

        let functor = if let Some(name) = nimap.get_name(functor) {
            name.clone()
        } else {
            unreachable!("integer {:?} has no name mapping", functor)
        };

        let args = args
            .into_iter()
            .map(|arg| Self::term_view_to_term(arg, nimap))
            .collect();

        Term { functor, args }
    }

    fn term_deep_view_to_term(view: TermDeepView<'_, Integer>, nimap: &NameIntMap<T>) -> Term<T> {
        let functor = view.functor();
        let args = view.args();

        let functor = if let Some(name) = nimap.get_name(functor) {
            name.clone()
        } else {
            unreachable!("integer {:?} has no name mapping", functor)
        };

        let args = args
            .into_iter()
            .map(|arg| Self::term_deep_view_to_term(arg, nimap))
            .collect();

        Term { functor, args }
    }
}

impl<T: Atom + Display> Display for Assignment<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let view = format::NamedTermView::new(self.lhs_view(), self.nimap);
        Display::fmt(&view, f)?;

        f.write_str(" = ")?;

        let view = format::NamedTermDeepView::new(self.rhs_view(), self.nimap);
        Display::fmt(&view, f)
    }
}

impl<T: Atom + Debug> Debug for Assignment<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lhs = format::NamedTermView::new(self.lhs_view(), self.nimap);
        let rhs = format::NamedTermDeepView::new(self.rhs_view(), self.nimap);

        f.debug_struct("Assignment")
            .field("lhs", &lhs)
            .field("rhs", &rhs)
            .finish()
    }
}

impl ExprView<'_, Integer> {
    fn is_unifiable(&self, other: TermView<'_, Integer>) -> bool {
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

impl TermView<'_, Integer> {
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
}

impl TermViewMut<'_, Integer> {
    fn is_variable(&self) -> bool {
        self.arity() == 0 && self.functor().is_variable()
    }
}

/// Internal integer representation of an atom.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Integer(u32);

impl Integer {
    const VAR_FLAG: u32 = 0x1 << 31;
    const TEMPORARY_FLAG: u32 = 0x1 << 30;

    pub(crate) fn from_value<T: Atom>(s: &T, mut index: u32) -> Self {
        if s.is_variable() {
            index |= Self::VAR_FLAG;
        }
        Self(index)
    }

    pub(crate) fn variable(int: u32) -> Self {
        let mask = Self::VAR_FLAG;
        debug_assert_eq!(int & mask, 0);
        Self(int | mask)
    }

    pub(crate) fn temporary(int: u32) -> Self {
        let mask = Self::VAR_FLAG | Self::TEMPORARY_FLAG;
        debug_assert_eq!(int & mask, 0);
        Self(int | mask)
    }

    pub(crate) const fn is_temporary_variable(self) -> bool {
        let mask = Self::VAR_FLAG | Self::TEMPORARY_FLAG;
        (mask & self.0) == mask
    }
}

impl Atom for Integer {
    fn is_variable(&self) -> bool {
        (Self::VAR_FLAG & self.0) == Self::VAR_FLAG
    }
}

impl Debug for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mask: u32 = Self::VAR_FLAG | Self::TEMPORARY_FLAG;
        let index = !mask & self.0;

        if self.is_variable() {
            f.write_char(VAR_PREFIX)?;
        }
        if self.is_temporary_variable() {
            f.write_char('#')?;
        }
        Debug::fmt(&index, f)
    }
}

impl ops::AddAssign<u32> for Integer {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs;
    }
}

/// Stores mappings only for user-provided clause/query names.
///
/// Auto-generated names such as temporary variables are not stored here.
#[derive(Debug)]
pub(crate) struct NameIntMap<T> {
    name2int: IndexMap<T, Integer>,
    int2name: IndexMap<Integer, T>,
    next_int: u32,
}

impl<T> NameIntMap<T> {
    pub(crate) fn new() -> Self {
        Self {
            name2int: IndexMap::default(),
            int2name: IndexMap::default(),
            next_int: 0,
        }
    }

    pub(crate) fn get_name(&self, int: &Integer) -> Option<&T> {
        self.int2name.get(int)
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

impl<T: Atom> NameIntMap<T> {
    pub(crate) fn name_to_int(&mut self, name: T) -> Integer {
        if let Some(int) = self.name2int.get(&name) {
            *int
        } else {
            let int = Integer::from_value(&name, self.next_int);

            self.name2int.insert(name.clone(), int);
            self.int2name.insert(int, name);

            self.next_int += 1;
            int
        }
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

    pub struct NamedTermView<'a, T> {
        view: TermView<'a, Integer>,
        nimap: &'a NameIntMap<T>,
    }

    impl<'a, T> NamedTermView<'a, T> {
        pub(crate) const fn new(view: TermView<'a, Integer>, nimap: &'a NameIntMap<T>) -> Self {
            Self { view, nimap }
        }

        fn args<'s>(&'s self) -> impl Iterator<Item = NamedTermView<'a, T>> + 's {
            self.view.args().map(|arg| Self {
                view: arg,
                nimap: self.nimap,
            })
        }
    }

    impl<'a, T: Atom> NamedTermView<'a, T> {
        /// Returns `true` if this view is equal to `term`.
        ///
        /// # Examples
        ///
        /// ```
        /// use logic_eval::{parse_str, Clause, Database, StrInterner, Term};
        ///
        /// let interner = StrInterner::new();
        /// let clause: Clause<_> = parse_str("parent(alice, bob).", &interner).unwrap();
        /// let expected: Term<_> = parse_str("parent(alice, bob)", &interner).unwrap();
        /// let mut db = Database::new();
        /// db.insert_clause(clause);
        /// db.commit();
        ///
        /// let term = db.terms().next().unwrap();
        /// assert!(term.is(&expected));
        /// ```
        pub fn is(&self, term: &Term<T>) -> bool {
            let functor = self.view.functor();
            let Some(functor) = self.nimap.get_name(functor) else {
                return false;
            };

            if functor != &term.functor {
                return false;
            }

            self.args().zip(&term.args).all(|(l, r)| l.is(r))
        }

        /// Returns `true` if this view contains `term` as itself or a nested argument.
        ///
        /// # Examples
        ///
        /// ```
        /// use logic_eval::{parse_str, Clause, Database, StrInterner, Term};
        ///
        /// let interner = StrInterner::new();
        /// let clause: Clause<_> = parse_str("parent(alice, bob).", &interner).unwrap();
        /// let expected: Term<_> = parse_str("bob", &interner).unwrap();
        /// let mut db = Database::new();
        /// db.insert_clause(clause);
        /// db.commit();
        ///
        /// let term = db.terms().next().unwrap();
        /// assert!(term.contains(&expected));
        /// ```
        pub fn contains(&self, term: &Term<T>) -> bool {
            if self.is(term) {
                return true;
            }

            self.args().any(|arg| arg.contains(term))
        }
    }

    impl<'a, T: Display> Display for NamedTermView<'a, T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let Self { view, nimap } = self;

            let functor = view.functor();
            let args = view.args();
            let num_args = args.len();

            write_int(functor, nimap, f)?;

            if num_args > 0 {
                f.write_char('(')?;
                for (i, arg) in args.enumerate() {
                    fmt::Display::fmt(&Self::new(arg, nimap), f)?;
                    if i + 1 < num_args {
                        f.write_str(", ")?;
                    }
                }
                f.write_char(')')?;
            }
            Ok(())
        }
    }

    impl<'a, T: Debug> Debug for NamedTermView<'a, T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let Self { view, nimap } = self;

            let functor = view.functor();
            let args = view.args();
            let num_args = args.len();

            if num_args == 0 {
                if let Some(name) = nimap.get_name(functor) {
                    fmt::Debug::fmt(name, f)
                } else {
                    fmt::Debug::fmt(functor, f)
                }
            } else {
                let name_str = if let Some(name) = nimap.get_name(functor) {
                    format!("{:?}", name)
                } else {
                    format!("{:?}", functor)
                };
                let mut d = f.debug_tuple(&name_str);

                for arg in args {
                    d.field(&Self::new(arg, nimap));
                }
                d.finish()
            }
        }
    }

    pub(crate) struct NamedTermDeepView<'a, T> {
        view: TermDeepView<'a, Integer>,
        nimap: &'a NameIntMap<T>,
    }

    impl<'a, T> NamedTermDeepView<'a, T> {
        pub(crate) const fn new(view: TermDeepView<'a, Integer>, nimap: &'a NameIntMap<T>) -> Self {
            Self { view, nimap }
        }
    }

    impl<'a, T: Display> Display for NamedTermDeepView<'a, T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let Self { view, nimap } = self;

            let functor = view.functor();
            let args = view.args();
            let num_args = args.len();

            write_int(functor, nimap, f)?;

            if num_args > 0 {
                f.write_char('(')?;
                for (i, arg) in args.enumerate() {
                    fmt::Display::fmt(&Self::new(arg, nimap), f)?;
                    if i + 1 < num_args {
                        f.write_str(", ")?;
                    }
                }
                f.write_char(')')?;
            }
            Ok(())
        }
    }

    impl<'a, T: Debug> Debug for NamedTermDeepView<'a, T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let Self { view, nimap } = self;

            let functor = view.functor();
            let args = view.args();
            let num_args = args.len();

            if num_args == 0 {
                if let Some(name) = nimap.get_name(functor) {
                    fmt::Debug::fmt(name, f)
                } else {
                    fmt::Debug::fmt(functor, f)
                }
            } else {
                let name_str = if let Some(name) = nimap.get_name(functor) {
                    format!("{:?}", name)
                } else {
                    format!("{:?}", functor)
                };
                let mut d = f.debug_tuple(&name_str);

                for arg in args {
                    d.field(&Self::new(arg, nimap));
                }
                d.finish()
            }
        }
    }

    pub struct NamedExprView<'a, T> {
        view: ExprView<'a, Integer>,
        nimap: &'a NameIntMap<T>,
    }

    impl<'a, T> NamedExprView<'a, T> {
        pub(crate) const fn new(view: ExprView<'a, Integer>, nimap: &'a NameIntMap<T>) -> Self {
            Self { view, nimap }
        }
    }

    impl<'a, T: Atom> NamedExprView<'a, T> {
        /// Returns `true` if this expression view contains `term`.
        ///
        /// # Examples
        ///
        /// ```
        /// use logic_eval::{parse_str, Clause, Database, StrInterner, Term};
        ///
        /// let interner = StrInterner::new();
        /// let clause: Clause<_> = parse_str("outdoors :- sunny, warm.", &interner).unwrap();
        /// let expected: Term<_> = parse_str("warm", &interner).unwrap();
        /// let mut db = Database::new();
        /// db.insert_clause(clause);
        /// db.commit();
        ///
        /// let body = db.clauses().next().unwrap().body().unwrap();
        /// assert!(body.contains_term(&expected));
        /// ```
        pub fn contains_term(&self, term: &Term<T>) -> bool {
            match self.view.as_kind() {
                ExprKind::Term(view) => NamedTermView {
                    view,
                    nimap: self.nimap,
                }
                .contains(term),
                ExprKind::Not(view) => NamedExprView {
                    view,
                    nimap: self.nimap,
                }
                .contains_term(term),
                ExprKind::And(args) | ExprKind::Or(args) => args.into_iter().any(|view| {
                    NamedExprView {
                        view,
                        nimap: self.nimap,
                    }
                    .contains_term(term)
                }),
            }
        }
    }

    impl<'a, T: Display> Display for NamedExprView<'a, T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let Self { view, nimap } = self;

            match view.as_kind() {
                ExprKind::Term(term) => fmt::Display::fmt(&NamedTermView { view: term, nimap }, f)?,
                ExprKind::Not(inner) => {
                    f.write_str("\\+ ")?;
                    if matches!(inner.as_kind(), ExprKind::And(_) | ExprKind::Or(_)) {
                        f.write_char('(')?;
                        fmt::Display::fmt(&Self::new(inner, nimap), f)?;
                        f.write_char(')')?;
                    } else {
                        fmt::Display::fmt(&Self::new(inner, nimap), f)?;
                    }
                }
                ExprKind::And(args) => {
                    let num_args = args.len();
                    for (i, arg) in args.enumerate() {
                        if matches!(arg.as_kind(), ExprKind::Or(_)) {
                            f.write_char('(')?;
                            fmt::Display::fmt(&Self::new(arg, nimap), f)?;
                            f.write_char(')')?;
                        } else {
                            fmt::Display::fmt(&Self::new(arg, nimap), f)?;
                        }
                        if i + 1 < num_args {
                            f.write_str(", ")?;
                        }
                    }
                }
                ExprKind::Or(args) => {
                    let num_args = args.len();
                    for (i, arg) in args.enumerate() {
                        fmt::Display::fmt(&Self::new(arg, nimap), f)?;
                        if i + 1 < num_args {
                            f.write_str("; ")?;
                        }
                    }
                }
            }
            Ok(())
        }
    }

    impl<'a, T: Debug> Debug for NamedExprView<'a, T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let Self { view, nimap } = self;

            match view.as_kind() {
                ExprKind::Term(term) => fmt::Debug::fmt(&NamedTermView::new(term, nimap), f),
                ExprKind::Not(inner) => f
                    .debug_tuple("Not")
                    .field(&NamedExprView::new(inner, nimap))
                    .finish(),
                ExprKind::And(args) => {
                    let mut d = f.debug_tuple("And");
                    for arg in args {
                        d.field(&NamedExprView::new(arg, nimap));
                    }
                    d.finish()
                }
                ExprKind::Or(args) => {
                    let mut d = f.debug_tuple("Or");
                    for arg in args {
                        d.field(&NamedExprView::new(arg, nimap));
                    }
                    d.finish()
                }
            }
        }
    }

    fn write_int<T: fmt::Display>(
        int: &Integer,
        nimap: &NameIntMap<T>,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if let Some(name) = nimap.get_name(int) {
            fmt::Display::fmt(name, f)
        } else {
            fmt::Debug::fmt(int, f)
        }
    }
}
