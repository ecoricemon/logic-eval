use crate::{prove::prover::TermAssignments, Atom, Expr, PassThroughIndexMap, Predicate, Term};
use fxhash::FxHasher;
use std::{
    hash::{Hash, Hasher},
    iter, ops,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ClauseId {
    pub(crate) head: TermId,
    pub(crate) body: Option<ExprId>,
}

#[derive(Debug)]
pub(crate) struct TermStorage<T> {
    pub(crate) exprs: ExprArray,
    pub(crate) terms: UniqueTermArray<T>,
}

impl<T> TermStorage<T> {
    pub(crate) fn new() -> Self {
        Self {
            exprs: ExprArray::new(),
            terms: UniqueTermArray::new(),
        }
    }

    pub(crate) fn len(&self) -> TermStorageLen {
        TermStorageLen {
            expr_len: self.exprs.len(),
            term_len: self.terms.len(),
        }
    }

    pub(crate) fn truncate(&mut self, len: TermStorageLen) {
        self.exprs.truncate(len.expr_len);
        self.terms.truncate(len.term_len);
    }

    pub(crate) fn get_expr(&self, id: ExprId) -> ExprView<'_, T> {
        self.exprs.get(id, &self.terms.buf)
    }

    pub(crate) fn get_expr_mut(&mut self, id: ExprId) -> ExprViewMut<'_, T> {
        self.exprs.get_mut(id, &mut self.terms)
    }

    pub(crate) fn get_term(&self, id: TermId) -> TermView<'_, T> {
        self.terms.get(id)
    }

    pub(crate) fn get_term_mut(&mut self, id: TermId) -> TermViewMut<'_, T> {
        self.terms.get_mut(id)
    }
}

impl<T: Clone + Eq + Hash> TermStorage<T> {
    pub(crate) fn insert_expr(&mut self, expr: Expr<T>) -> ExprId {
        self.exprs.insert(expr, &mut self.terms)
    }

    pub(crate) fn insert_term(&mut self, term: Term<T>) -> TermId {
        self.terms.insert(term)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TermStorageLen {
    expr_len: usize,
    term_len: TermArrayLen,
}

#[derive(Debug)]
pub(crate) struct ExprArray {
    buf: Vec<ExprElem>,
}

impl ExprArray {
    const fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn get<'a, T>(&'a self, id: ExprId, term_buf: &'a [TermElem<T>]) -> ExprView<'a, T> {
        ExprView {
            expr_buf: &self.buf,
            term_buf,
            id,
        }
    }

    fn get_mut<'a, T>(
        &'a mut self,
        id: ExprId,
        terms: &'a mut UniqueTermArray<T>,
    ) -> ExprViewMut<'a, T> {
        ExprViewMut {
            exprs: self,
            terms,
            id,
        }
    }

    fn insert<T: Clone + Eq + Hash>(
        &mut self,
        expr: Expr<T>,
        term_arr: &mut UniqueTermArray<T>,
    ) -> ExprId {
        match expr {
            Expr::Term(term) => {
                let tid = term_arr.insert(term);
                let elem = ExprElem::Term(tid);
                let id = ExprId(self.buf.len());
                self.buf.push(elem);
                id
            }
            Expr::Not(arg) => {
                let idx = self.reserve(1);
                let inner_id = self.insert(*arg, term_arr);
                self.buf[idx] = ExprElem::Not(inner_id);
                ExprId(idx)
            }
            Expr::And(args) => {
                let num_args = args.len();
                let idx = self.reserve(1 + num_args);

                self.buf[idx] = ExprElem::And { len: num_args };
                for (i, arg) in args.into_iter().enumerate() {
                    let arg_id = self.insert(arg, term_arr);
                    self.buf[idx + 1 + i] = ExprElem::Expr(arg_id);
                }

                ExprId(idx)
            }
            Expr::Or(args) => {
                let num_args = args.len();
                let idx = self.reserve(1 + num_args);

                self.buf[idx] = ExprElem::Or { len: num_args };
                for (i, arg) in args.into_iter().enumerate() {
                    let arg_id = self.insert(arg, term_arr);
                    self.buf[idx + 1 + i] = ExprElem::Expr(arg_id);
                }

                ExprId(idx)
            }
        }
    }

    fn push(&mut self, elem: ExprElem) {
        self.buf.push(elem);
    }

    fn pop(&mut self) -> Option<ExprElem> {
        self.buf.pop()
    }

    fn len(&self) -> usize {
        self.buf.len()
    }

    fn truncate(&mut self, len: usize) {
        self.buf.truncate(len);
    }

    fn reserve(&mut self, additional: usize) -> usize {
        let cur_len = self.buf.len();
        self.buf.resize_with(cur_len + additional, ExprElem::dummy);
        cur_len
    }

    fn copy_within(&mut self, src: ExprId, dst: ExprId, num: usize) {
        let src_range = src.0..(src.0 + num);
        let dst = dst.0;
        self.buf.copy_within(src_range, dst);
    }
}

impl AsRef<[ExprElem]> for ExprArray {
    fn as_ref(&self) -> &[ExprElem] {
        &self.buf
    }
}

impl ops::Index<ExprId> for ExprArray {
    type Output = ExprElem;

    fn index(&self, index: ExprId) -> &Self::Output {
        &self.buf[index.0]
    }
}

impl ops::Index<usize> for ExprArray {
    type Output = ExprElem;

    fn index(&self, index: usize) -> &Self::Output {
        &self.buf[index]
    }
}

impl ops::IndexMut<ExprId> for ExprArray {
    fn index_mut(&mut self, index: ExprId) -> &mut Self::Output {
        &mut self.buf[index.0]
    }
}

impl ops::IndexMut<usize> for ExprArray {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.buf[index]
    }
}

impl ops::Index<ExprId> for [ExprElem] {
    type Output = ExprElem;

    fn index(&self, index: ExprId) -> &Self::Output {
        &self[index.0]
    }
}

impl ops::IndexMut<ExprId> for [ExprElem] {
    fn index_mut(&mut self, index: ExprId) -> &mut Self::Output {
        &mut self[index.0]
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ExprView<'a, T> {
    expr_buf: &'a [ExprElem],
    term_buf: &'a [TermElem<T>],
    id: ExprId,
}

impl<'a, T> ExprView<'a, T> {
    pub(crate) fn as_kind(&self) -> ExprKind<'a, T> {
        match self.expr_buf[self.id] {
            ExprElem::Term(term_id) => ExprKind::Term(TermView {
                buf: self.term_buf,
                id: term_id,
            }),
            ExprElem::Not(inner_id) => ExprKind::Not(Self {
                expr_buf: self.expr_buf,
                term_buf: self.term_buf,
                id: inner_id,
            }),
            ExprElem::And { len } => ExprKind::And(ExprViewIter {
                expr_buf: self.expr_buf,
                term_buf: self.term_buf,
                start: self.id + 1,
                end: self.id + 1 + len,
            }),
            ExprElem::Or { len } => ExprKind::Or(ExprViewIter {
                expr_buf: self.expr_buf,
                term_buf: self.term_buf,
                start: self.id + 1,
                end: self.id + 1 + len,
            }),
            ExprElem::Expr(expr_id) => Self {
                expr_buf: self.expr_buf,
                term_buf: self.term_buf,
                id: expr_id,
            }
            .as_kind(),
        }
    }

    pub(crate) fn leftmost_term(self) -> TermView<'a, T> {
        match self.as_kind() {
            ExprKind::Term(term) => term,
            ExprKind::Not(expr) => expr.leftmost_term(),
            ExprKind::And(mut exprs) | ExprKind::Or(mut exprs) => {
                exprs.next().unwrap().leftmost_term()
            }
        }
    }

    /// The given function `f` is applied on the top terms of the expression, but not applied on
    /// inner terms of the top terms.
    ///
    /// Note that a term is also recursive.
    pub(crate) fn with_term<F: FnMut(TermView<'a, T>)>(&self, f: &mut F) {
        match self.as_kind() {
            ExprKind::Term(term) => f(term),
            ExprKind::Not(inner) => inner.with_term(f),
            ExprKind::And(args) | ExprKind::Or(args) => {
                args.into_iter().for_each(|arg| arg.with_term(f))
            }
        }
    }
}

pub(crate) enum ExprKind<'a, T> {
    Term(TermView<'a, T>),
    Not(ExprView<'a, T>),
    And(ExprViewIter<'a, T>),
    Or(ExprViewIter<'a, T>),
}

pub(crate) struct ExprViewIter<'a, T> {
    expr_buf: &'a [ExprElem],
    term_buf: &'a [TermElem<T>],
    /// Inclusive
    start: ExprId,
    /// Exclusive
    end: ExprId,
}

impl<'a, T> ExprViewIter<'a, T> {
    const fn len(&self) -> usize {
        self.end.0 - self.start.0
    }
}

impl<'a, T> Iterator for ExprViewIter<'a, T> {
    type Item = ExprView<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let ExprElem::Expr(expr_id) = self.expr_buf[self.start] else {
                unreachable!()
            };
            self.start += 1;
            Some(ExprView {
                expr_buf: self.expr_buf,
                term_buf: self.term_buf,
                id: expr_id,
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

impl<T> ExactSizeIterator for ExprViewIter<'_, T> {
    fn len(&self) -> usize {
        <Self>::len(self)
    }
}

impl<T> iter::FusedIterator for ExprViewIter<'_, T> {}

pub(crate) struct ExprViewMut<'a, T> {
    exprs: &'a mut ExprArray,
    terms: &'a mut UniqueTermArray<T>,
    id: ExprId,
}

impl<'a, T> ExprViewMut<'a, T> {
    pub(crate) fn id(&self) -> ExprId {
        self.id
    }

    pub(crate) fn with_terminal<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut UniqueTermArray<T>, TermId),
    {
        fn helper<'a, T, F>(this: &mut ExprViewMut<'a, T>, f: &mut F)
        where
            F: FnMut(&mut UniqueTermArray<T>, TermId),
        {
            this.find_then_move();

            match this.exprs[this.id] {
                ExprElem::Term(term) => {
                    TermViewMut {
                        arr: this.terms,
                        id: term,
                    }
                    .with_terminal(f);
                }
                ExprElem::Not(inner) => {
                    let org = this.id;
                    this.id = inner;
                    helper(this, f);
                    this.id = org;
                }
                ExprElem::And { len } | ExprElem::Or { len } => {
                    let org = this.id;
                    for _ in 0..len {
                        this.id += 1;
                        helper(this, f);
                    }
                    this.id = org;
                }
                ExprElem::Expr(_) => unreachable!(),
            }
        }
        helper(self, &mut f)
    }

    /// Finds the destination of an [`ExprElem::Expr`] chain, then moves this view to the final
    /// expression.
    fn find_then_move(&mut self) {
        self.id = self.find(self.id);
    }

    fn find(&mut self, src: ExprId) -> ExprId {
        if let ExprElem::Expr(next) = self.exprs[src] {
            let dst = self.find(next);
            self.exprs[src] = ExprElem::Expr(dst);
            dst
        } else {
            src
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ApplyResult {
    /// An expression still remains as an expression after application of a boolean value.
    Expr,

    /// An expression is completely evaluated after application of a boolean value.
    Complete(bool),
}

impl<'a, T: Clone> ExprViewMut<'a, T> {
    /// Sets the leftmost term to the given boolean value.
    ///
    /// This operation always clones the original expression then does the setting on that. Plus,
    /// this could unwrap AND or OR if they do not contain two or more arguments after evaluation.
    pub(crate) fn apply_to_leftmost_term(&mut self, to: bool) -> ApplyResult {
        // Clones this expression deeply.
        let mut handle_term = |this: &mut Self, term_id: TermId, _: ()| {
            let new_expr = ExprElem::Term(term_id);
            let new_id = ExprId(this.exprs.len());
            this.exprs.push(new_expr);
            Some(new_id)
        };
        self.id = self.clone_on_write((), &mut handle_term).unwrap();

        return apply(self.id, to, self.exprs);

        // === Internal helper functions ===

        /// Applies eval in place.
        fn apply(id: ExprId, to: bool, exprs: &mut ExprArray) -> ApplyResult {
            match exprs[id] {
                ExprElem::Term(_) => ApplyResult::Complete(to),
                ExprElem::Not(inner_id) => match apply(inner_id, to, exprs) {
                    ApplyResult::Expr => ApplyResult::Expr,
                    ApplyResult::Complete(eval) => ApplyResult::Complete(!eval),
                },
                ExprElem::And { len } => {
                    let eval = match apply(id + 1, to, exprs) {
                        ApplyResult::Expr => return ApplyResult::Expr,
                        ApplyResult::Complete(eval) => eval,
                    };

                    if !eval {
                        return ApplyResult::Complete(false);
                    }

                    match len {
                        3.. => {
                            let src = id + 2;
                            let dst = id + 1;
                            let num = len - 1;
                            exprs.copy_within(src, dst, num);
                            exprs[id] = ExprElem::And { len: len - 1 };
                            ApplyResult::Expr
                        }
                        2 => {
                            exprs[id] = ExprElem::Expr(id + 2);
                            ApplyResult::Expr
                        }
                        _ => unreachable!(),
                    }
                }
                ExprElem::Or { len } => {
                    let eval = match apply(id + 1, to, exprs) {
                        ApplyResult::Expr => return ApplyResult::Expr,
                        ApplyResult::Complete(eval) => eval,
                    };

                    if eval {
                        return ApplyResult::Complete(true);
                    }

                    match len {
                        3.. => {
                            let src = id + 2;
                            let dst = id + 1;
                            let num = len - 1;
                            exprs.copy_within(src, dst, num);
                            exprs[id] = ExprElem::Or { len: len - 1 };
                            ApplyResult::Expr
                        }
                        2 => {
                            exprs[id] = ExprElem::Expr(id + 2);
                            ApplyResult::Expr
                        }
                        _ => unreachable!(),
                    }
                }
                ExprElem::Expr(inner_id) => apply(inner_id, to, exprs),
            }
        }
    }

    /// Replaces the leftmost term with the given `to` in a clone-on-write way.
    ///
    /// If the replacement took place, then a new expression is created then this view becomes to
    /// point the new expression instead of modifying original expression directly.
    pub(crate) fn replace_leftmost_term(&mut self, to: ExprId) -> bool {
        let mut has_met_first_term = false;

        let mut handle_term = |_: &mut Self, _: TermId, input: ExprId| {
            if !has_met_first_term {
                has_met_first_term = true;
                Some(input)
            } else {
                None
            }
        };

        if let Some(new_id) = self.clone_on_write(to, &mut handle_term) {
            self.id = new_id;
            true
        } else {
            false
        }
    }

    fn clone_on_write<Input, HandleTerm>(
        &mut self,
        input: Input,
        handle_term: &mut HandleTerm,
    ) -> Option<ExprId>
    where
        Input: Clone,
        HandleTerm: FnMut(&mut Self, TermId, Input) -> Option<ExprId>,
    {
        self.find_then_move();

        match self.exprs[self.id] {
            ExprElem::Term(term_id) => handle_term(self, term_id, input),
            ExprElem::Not(inner_id) => {
                let cur = self.id;

                // Reserves buffer space for a new expression in advance.
                let new_id = self.exprs.len();
                self.exprs.push(ExprElem::dummy());

                self.id = inner_id;
                let res = if let Some(new_inner_id) = self.clone_on_write(input, handle_term) {
                    self.exprs[new_id] = ExprElem::Not(new_inner_id);
                    Some(ExprId(new_id))
                } else {
                    // Discards the buffer change.
                    let dead = self.exprs.pop();
                    debug_assert_eq!(dead, Some(ExprElem::dummy()));
                    None
                };

                self.id = cur;
                res
            }
            ExprElem::And { len } => self.clone_on_write_for_and_or(
                len,
                input,
                |this, input| this.clone_on_write(input, handle_term),
                |len| ExprElem::And { len },
            ),
            ExprElem::Or { len } => self.clone_on_write_for_and_or(
                len,
                input,
                |this, input| this.clone_on_write(input, handle_term),
                |len| ExprElem::Or { len },
            ),
            ExprElem::Expr(_) => unreachable!(),
        }
    }

    fn clone_on_write_for_and_or<Input, Cow, LenToElem>(
        &mut self,
        num_args: usize,
        input: Input,
        mut clone_on_write: Cow,
        len_to_elem: LenToElem,
    ) -> Option<ExprId>
    where
        Input: Clone,
        Cow: FnMut(&mut Self, Input) -> Option<ExprId>,
        LenToElem: FnOnce(usize) -> ExprElem,
    {
        // Reserves buffer space for a new expression in advance.
        let org_buf_len = self.exprs.len();
        let new = self.exprs.reserve(num_args + 1);

        // Tries to replace the term for the arguments.
        let cur = self.id.0;
        let mut has_written = false;
        for i in 0..num_args {
            let ExprElem::Expr(arg_id) = self.exprs[cur + 1 + i] else {
                unreachable!()
            };
            self.id = arg_id;
            self.exprs[new + 1 + i] = if let Some(new_arg_id) = clone_on_write(self, input.clone())
            {
                has_written = true;
                ExprElem::Expr(new_arg_id)
            } else {
                ExprElem::Expr(arg_id)
            };
        }
        self.id = ExprId(cur);

        if has_written {
            self.exprs[new] = len_to_elem(num_args);
            Some(ExprId(new))
        } else {
            // Discards the buffer change.
            self.exprs.truncate(org_buf_len);
            None
        }
    }
}

impl<'a, T: Atom> ExprViewMut<'a, T> {
    /// If this expression contains `from`, then replaces them to `to` in a clone-on-write way.
    ///
    /// If the replacement took place, then a new expression is created then this view becomes to
    /// point the new expression instead of modifying original expression directly.
    pub(crate) fn replace_term(&mut self, from: TermId, to: TermId) -> bool {
        let mut handle_term = |this: &mut Self, term_id: TermId, (from, to): (TermId, TermId)| {
            let mut term_view = TermViewMut {
                arr: this.terms,
                id: term_id,
            };
            if term_view.replace(from, to) {
                let new_term_id = term_view.id;
                let new_expr = ExprElem::Term(new_term_id);
                let new_id = ExprId(this.exprs.len());
                this.exprs.push(new_expr);
                Some(new_id)
            } else {
                None
            }
        };

        if let Some(new_id) = self.clone_on_write((from, to), &mut handle_term) {
            self.id = new_id;
            true
        } else {
            false
        }
    }
}

/// Element representing a part of a term expression in a unified buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExprElem {
    Term(TermId),
    Not(ExprId),
    /// Followed by [`ExprElem::Expr`] entries.
    And {
        len: usize,
    },
    /// Followed by [`ExprElem::Expr`] entries.
    Or {
        len: usize,
    },
    Expr(ExprId),
}

impl ExprElem {
    #[inline]
    const fn dummy() -> Self {
        Self::Term(TermId(0xDEADDEAD))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ExprId(pub(crate) usize);

impl ops::Add<usize> for ExprId {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl ops::AddAssign<usize> for ExprId {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs;
    }
}

#[derive(Debug)]
pub(crate) struct UniqueTermArray<T> {
    /// e.g. ... [Functor], [Len], [Arg0], [Arg1], ...
    pub(crate) buf: Vec<TermElem<T>>,

    /// Mapping between term's hash values and terms.
    ///
    /// This helps you find similar terms to keep the uniqueness. But for efficiency, the value,
    /// [`Vec<TermId>`], could contain stale data. For example, [`Self::buf`] could be shrunk by
    /// truncate method, but values of this map still would point to removed area of the buffer
    /// becuase values themselves are not shrunk.
    ///
    /// You are encouraged to call two methods below to access this field, [`Self::add_mapping`] and
    /// [`Self::get_similar`], which hide the problem.
    pub(crate) map: PassThroughIndexMap<u64, Vec<TermId>>,
}

impl<T> UniqueTermArray<T> {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            map: PassThroughIndexMap::default(),
        }
    }

    pub(crate) fn terms(&self) -> TermViewIter<'_, T> {
        TermViewIter {
            buf: &self.buf,
            cur: 0,
        }
    }

    pub(crate) fn get(&self, id: TermId) -> TermView<'_, T> {
        Self::_get(&self.buf, id)
    }

    const fn _get(buf: &[TermElem<T>], id: TermId) -> TermView<'_, T> {
        TermView { buf, id }
    }

    pub(crate) fn get_mut(&mut self, id: TermId) -> TermViewMut<'_, T> {
        TermViewMut { arr: self, id }
    }

    pub(crate) fn len(&self) -> TermArrayLen {
        TermArrayLen {
            buf_len: self.buf.len(),
            map_len: self.map.len(),
        }
    }

    pub(crate) fn truncate(&mut self, len: TermArrayLen) {
        self.buf.truncate(len.buf_len);
        self.map.truncate(len.map_len);
    }

    fn reserve(&mut self, additional: usize) -> usize {
        let cur_len = self.buf.len();
        self.buf
            .resize_with(cur_len + additional, || TermElem::dummy());
        cur_len
    }
}

impl<T: Clone + Eq + Hash + PartialEq> UniqueTermArray<T> {
    fn add_mapping(&mut self, hash: u64, id: TermId) {
        self.map
            .entry(hash)
            .and_modify(|similar| similar.push(id))
            .or_insert(vec![id]);
    }

    fn get_similar<'a: 'p, 'p>(
        ids: Option<&'a mut Vec<TermId>>,
        buf: &'a [TermElem<T>],
        hash: u64,
    ) -> SimilarTerms<'p, T> {
        SimilarTerms {
            ids,
            buf,
            hash,
            cur: 0,
        }
    }

    /// If there's no structurally identical term in the array, then returns Ok with the `new_id`.
    /// Otherwise, returns Err with the existing id.
    ///
    /// If result is Ok, then registers the `new_id` in the map.
    fn try_register_unique_term(&mut self, new_id: TermId) -> Result<TermId, TermId> {
        let hash = buf_term_hash(&self.buf, new_id);
        let dup = 'search: {
            for existing_id in Self::get_similar(self.map.get_mut(&hash), &self.buf, hash) {
                if existing_id != new_id && structually_eq(&self.buf, existing_id, new_id) {
                    break 'search Some(existing_id);
                }
            }
            None
        };
        if let Some(existing_id) = dup {
            return Err(existing_id);
        }

        self.add_mapping(hash, new_id);
        Ok(new_id)
    }

    pub(crate) fn insert(&mut self, term: Term<T>) -> TermId {
        // Checks existence.
        let hash = term_hash(&term);
        for id in Self::get_similar(self.map.get_mut(&hash), &self.buf, hash) {
            if is_arg_set(&self.buf, id) && is_same(&self.buf, id, &term) {
                return id;
            }
        }

        // Mapping
        let id = TermId(self.buf.len());
        self.add_mapping(hash, id);

        // Functor
        self.buf.push(TermElem::Functor(term.functor));

        // Arity
        self.buf.push(TermElem::Arity(term.args.len() as u32));

        // Reserves following slots in advance.
        let arg_idx = self.reserve(term.args.len());

        // Args
        for (i, arg) in term.args.into_iter().enumerate() {
            self.buf[arg_idx + i] = TermElem::Arg(self.insert(arg));
        }

        return id;

        // === Internal helper functions ===

        /// Checks whether arguments of the given term have been set or not by comparing with the
        /// predefined dummy value.
        fn is_arg_set<T: PartialEq>(buf: &[TermElem<T>], id: TermId) -> bool {
            let arity = UniqueTermArray::_get(buf, id).arity();
            (0..arity).all(|i| buf[id.0 + 2 + i as usize] != TermElem::dummy())
        }

        fn is_same<T: PartialEq>(buf: &[TermElem<T>], left: TermId, right: &Term<T>) -> bool {
            let left = UniqueTermArray::_get(buf, left);

            left.functor() == &right.functor
                && left.arity() as usize == right.args.len()
                && left
                    .args()
                    .zip(&right.args)
                    .all(|(arg_view, arg)| is_same(buf, arg_view.id, arg))
        }
    }
}

impl<T> AsRef<[TermElem<T>]> for UniqueTermArray<T> {
    fn as_ref(&self) -> &[TermElem<T>] {
        &self.buf
    }
}

impl<T> ops::Index<TermId> for UniqueTermArray<T> {
    type Output = TermElem<T>;

    fn index(&self, index: TermId) -> &Self::Output {
        &self.buf[index.0]
    }
}

impl<T> ops::Index<TermId> for [TermElem<T>] {
    type Output = TermElem<T>;

    fn index(&self, index: TermId) -> &Self::Output {
        &self[index.0]
    }
}

struct SimilarTerms<'a, T> {
    ids: Option<&'a mut Vec<TermId>>,
    buf: &'a [TermElem<T>],
    hash: u64,
    cur: usize,
}

impl<T: PartialEq + Hash> Iterator for SimilarTerms<'_, T> {
    type Item = TermId;

    fn next(&mut self) -> Option<Self::Item> {
        let ids = self.ids.as_mut()?;

        // `id` could be outdated, but we remove them here.
        while let Some(id) = ids.get(self.cur) {
            if !matches!(self.buf.get(id.0), Some(TermElem::Functor(_)))
                || buf_term_hash(self.buf, *id) != self.hash
            {
                ids.swap_remove(self.cur);
                continue;
            }

            self.cur += 1;
            return Some(*id);
        }

        None
    }
}

impl<T: PartialEq + Hash> iter::FusedIterator for SimilarTerms<'_, T> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TermArrayLen {
    buf_len: usize,
    map_len: usize,
}

pub(crate) struct TermViewIter<'a, T> {
    buf: &'a [TermElem<T>],
    cur: usize,
}

impl<'a, T> Iterator for TermViewIter<'a, T> {
    type Item = TermView<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.buf.len() {
            return None;
        }

        let view = TermView {
            buf: self.buf,
            id: TermId(self.cur),
        };

        self.cur += 2 + view.arity() as usize;

        Some(view)
    }
}

impl<T> iter::FusedIterator for TermViewIter<'_, T> {}

#[derive(Clone)]
pub struct TermView<'a, T> {
    pub(crate) buf: &'a [TermElem<T>],
    pub(crate) id: TermId,
}

impl<'a, T> TermView<'a, T> {
    pub(crate) fn functor(&self) -> &T {
        let TermElem::Functor(functor) = &self.buf[self.id] else {
            unreachable!()
        };
        functor
    }

    pub(crate) fn arity(&self) -> u32 {
        let TermElem::Arity(n) = self.buf[self.id + 1] else {
            unreachable!()
        };
        n
    }

    pub(crate) fn args(&self) -> TermViewArgs<'a, T> {
        let start = self.id + 2;
        let end = start + self.arity() as usize;
        TermViewArgs {
            buf: self.buf,
            start,
            end,
        }
    }
}

impl<T: Clone> TermView<'_, T> {
    pub(crate) fn predicate(&self) -> Predicate<T> {
        Predicate {
            functor: self.functor().clone(),
            arity: self.arity(),
        }
    }

    pub(crate) fn deserialize(self) -> Term<T> {
        Term {
            functor: self.functor().clone(),
            args: self.args().map(Self::deserialize).collect(),
        }
    }
}

impl<T: Atom> TermView<'_, T> {
    /// Returns true if this term is a variable.
    ///
    /// e.g. Terms like `X`, `Y` will return true.
    pub(crate) fn is_variable(&self) -> bool {
        let is_var = self.functor().is_variable();

        #[cfg(debug_assertions)]
        if is_var {
            assert_eq!(self.arity(), 0);
        }

        is_var
    }

    /// Returns true if this term is a variable or contains a variable in it.
    ///
    /// e.g. Terms like `X` or `f(X)` will return true.
    pub(crate) fn contains_variable(&self) -> bool {
        self.is_variable() || self.args().any(|arg| arg.contains_variable())
    }

    pub(crate) fn with_variable<F: FnMut(&Self)>(&self, mut f: F) {
        fn helper<'a, T, F>(view: &TermView<'a, T>, f: &mut F)
        where
            T: Atom,
            F: FnMut(&TermView<'a, T>),
        {
            if view.is_variable() {
                f(view);
            } else {
                for arg in view.args() {
                    helper(&arg, f);
                }
            }
        }
        helper(self, &mut f)
    }

    pub(crate) fn collect_variables(&self) -> Vec<TermId> {
        let mut vars = Vec::new();
        self.with_variable(|var| {
            if !vars.contains(&var.id) {
                vars.push(var.id);
            }
        });
        vars
    }
}

impl<T: Atom + Copy> TermView<'_, T> {
    /// Returns the first variable functor in this term.
    pub(crate) fn find_variable(&self) -> Option<T> {
        if self.is_variable() {
            Some(*self.functor())
        } else {
            self.args().find_map(|arg| arg.find_variable())
        }
    }
}

#[derive(Clone)]
pub(crate) struct TermViewArgs<'a, T> {
    buf: &'a [TermElem<T>],
    /// Inclusive
    start: TermId,
    /// Exclusive
    end: TermId,
}

impl<T> TermViewArgs<'_, T> {
    const fn len(&self) -> usize {
        self.end.0 - self.start.0
    }
}

impl<'a, T> Iterator for TermViewArgs<'a, T> {
    type Item = TermView<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let TermElem::Arg(id) = self.buf[self.start] else {
                unreachable!()
            };
            self.start += 1;
            Some(TermView { buf: self.buf, id })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = <Self>::len(self);
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for TermViewArgs<'_, T> {
    fn len(&self) -> usize {
        <Self>::len(self)
    }
}

impl<T> iter::FusedIterator for TermViewArgs<'_, T> {}

#[derive(Debug, Clone)]
pub struct TermDeepView<'a, T> {
    pub(crate) buf: &'a [TermElem<T>],
    pub(crate) term_assigns: &'a TermAssignments,
    pub(crate) id: TermId,
}

impl<'a, T> TermDeepView<'a, T> {
    pub(crate) fn functor(&self) -> &T {
        let view = self.jump();

        let TermElem::Functor(functor) = &view.buf[view.id] else {
            unreachable!()
        };
        functor
    }

    pub(crate) fn arity(&self) -> u32 {
        let view = self.jump();

        let TermElem::Arity(n) = view.buf[view.id + 1] else {
            unreachable!()
        };
        n
    }

    pub(crate) fn args(&self) -> TermDeepViewArgs<'a, T> {
        let view = self.jump();

        let start = view.id + 2;
        let end = start + view.arity() as usize;
        TermDeepViewArgs {
            buf: view.buf,
            term_assigns: view.term_assigns,
            start,
            end,
        }
    }

    pub(crate) fn jump(&self) -> Self {
        let root = self.term_assigns.find(self.id).unwrap_or(self.id);
        Self {
            buf: self.buf,
            term_assigns: self.term_assigns,
            id: root,
        }
    }
}

#[derive(Clone)]
pub(crate) struct TermDeepViewArgs<'a, T> {
    buf: &'a [TermElem<T>],
    term_assigns: &'a TermAssignments,
    /// Inclusive
    start: TermId,
    /// Exclusive
    end: TermId,
}

impl<T> TermDeepViewArgs<'_, T> {
    const fn len(&self) -> usize {
        self.end.0 - self.start.0
    }
}

impl<'a, T> Iterator for TermDeepViewArgs<'a, T> {
    type Item = TermDeepView<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let TermElem::Arg(id) = self.buf[self.start] else {
                unreachable!()
            };
            self.start += 1;
            Some(TermDeepView {
                buf: self.buf,
                term_assigns: self.term_assigns,
                id,
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

impl<T> ExactSizeIterator for TermDeepViewArgs<'_, T> {
    fn len(&self) -> usize {
        <Self>::len(self)
    }
}

impl<T> iter::FusedIterator for TermDeepViewArgs<'_, T> {}

pub(crate) struct TermViewMut<'a, T> {
    arr: &'a mut UniqueTermArray<T>,
    id: TermId,
}

impl<'a, T> TermViewMut<'a, T> {
    pub(crate) fn as_view(&self) -> TermView<'_, T> {
        TermView {
            buf: &self.arr.buf,
            id: self.id,
        }
    }

    pub(crate) const fn id(&self) -> TermId {
        self.id
    }

    pub(crate) fn functor(&self) -> &T {
        let TermElem::Functor(functor) = &self.arr.buf[self.id.0] else {
            unreachable!()
        };
        functor
    }

    pub(crate) fn arity(&self) -> u32 {
        let TermElem::Arity(n) = self.arr.buf[self.id.0 + 1] else {
            unreachable!()
        };
        n
    }

    pub(crate) fn with_terminal<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut UniqueTermArray<T>, TermId),
    {
        fn helper<'a, T, F>(this: &mut TermViewMut<'a, T>, f: &mut F)
        where
            F: FnMut(&mut UniqueTermArray<T>, TermId),
        {
            let arity = this.arity();
            if arity == 0 {
                f(this.arr, this.id);
            } else {
                let org = this.id.0;

                for i in 0..arity as usize {
                    let TermElem::Arg(arg_id) = this.arr.buf[org + 2 + i] else {
                        unreachable!()
                    };
                    this.id = arg_id;
                    helper(this, f);
                }

                this.id = TermId(org);
            }
        }
        helper(self, &mut f)
    }
}

impl<T: Atom> TermViewMut<'_, T> {
    /// Applies the given function to all functors in this term, then returns true if any functor
    /// has been replaced.
    ///
    /// The original term never changes. Instead this method makes a new term when required. If a
    /// new term is generated, this view becomes to point to it.
    pub(crate) fn replace_with<F: FnMut(&T) -> Option<T>>(&mut self, mut map_func: F) -> bool {
        let mut buf_off = self.arr.buf.len();
        let mut early_exit = |_: TermId| None;
        if let Some(new_id) = self.replace_inner(&mut buf_off, &mut map_func, &mut early_exit) {
            self.id = new_id;
            true
        } else {
            false
        }
    }

    /// If this term is `from` or contains `from`, then replaces them to `to` in a clone-on-write
    /// way.
    ///
    /// If the replacement took place, then a new term is created then this view becomes to point
    /// the new term instead of modifying original term directly.
    pub(crate) fn replace(&mut self, from: TermId, to: TermId) -> bool {
        if from == to {
            return false;
        }

        let mut buf_off = self.arr.buf.len();
        let mut map_func = |_: &T| None;
        let mut early_exit = |id: TermId| (id == from).then_some(to);
        if let Some(new_id) = self.replace_inner(&mut buf_off, &mut map_func, &mut early_exit) {
            self.id = new_id;
            true
        } else {
            false
        }
    }

    /// Shared clone-on-write implementation used by [`Self::replace`] and [`Self::replace_with`].
    ///
    /// * `map_func` - Optionally replaces the functor of the current term.
    /// * `early_exit` - Optionally replaces the current term by id, short-circuiting before any
    ///   cloning.
    fn replace_inner<MapFn, MapId>(
        &mut self,
        buf_off: &mut usize,
        map_func: &mut MapFn,
        early_exit: &mut MapId,
    ) -> Option<TermId>
    where
        MapFn: FnMut(&T) -> Option<T>,
        MapId: FnMut(TermId) -> Option<TermId>,
    {
        // Short-circuit: return a direct replacement for this term without cloning.
        if let Some(id) = early_exit(self.id) {
            return Some(id);
        }

        // Term id & term space for this view.
        let cur = self.id.0;
        // Term id & term space for a new term.
        let new = *buf_off;

        // Reserves buffer space for the size of this term if required.
        let (arity, org_buf_len, new_end) = self.ensure_space(new);

        // Moves the offset for nested inner terms.
        *buf_off = new_end;

        let new_functor = map_func(self.functor());
        let is_func_replaced = new_functor.is_some();

        // Tries to replace the arguments.
        let mut is_arg_replaced = false;
        for i in 0..arity as usize {
            let TermElem::Arg(arg_id) = self.arr.buf[cur + 2 + i] else {
                unreachable!()
            };

            self.id = arg_id;
            let new_arg_id = self.replace_inner(buf_off, map_func, early_exit);
            self.arr.buf[new + 2 + i] = if let Some(new_arg_id) = new_arg_id {
                is_arg_replaced = true;
                TermElem::Arg(new_arg_id)
            } else {
                TermElem::Arg(arg_id)
            };
        }
        self.id = TermId(cur);

        if is_func_replaced || is_arg_replaced {
            // Sets the functor/arity at the new space.
            self.arr.buf[new] = match new_functor {
                Some(f) => TermElem::Functor(f),
                None => TermElem::Functor(self.functor().clone()),
            };
            self.arr.buf[new + 1] = TermElem::Arity(arity);

            // Dedup: Reuse an existing structurally identical term if present, otherwise register
            // the new one.
            match self.arr.try_register_unique_term(TermId(new)) {
                Ok(id) => Some(id),
                Err(existing_id) => {
                    self.arr.buf.truncate(org_buf_len); // Discards the buffer change
                    Some(existing_id)
                }
            }
        } else {
            self.arr.buf.truncate(org_buf_len); // Discards the buffer change
            None
        }
    }

    /// Reserves enough term space for this term in the term array if required.
    ///
    /// Returns
    /// - arity
    /// - old term array length
    /// - `off + 2 + arity`, which means the end index of the reserved term space.
    fn ensure_space(&mut self, off: usize) -> (u32, usize, usize) {
        let cur = self.id.0;
        let TermElem::Arity(arity) = self.arr.buf[cur + 1] else {
            unreachable!()
        };

        let term_space = 2 /* functor + arity itself */ + arity as usize;
        let end = off + term_space;

        let old_len = self.arr.buf.len();
        if old_len < end {
            self.arr.buf.resize_with(end, || TermElem::dummy());
        }

        (arity, old_len, end)
    }
}

/// Element representing a part of a term in a unified buffer.
///
/// Allowed sequences are as follows.
/// * Functor, Arity(0)
/// * Functor, Arity(n), Arg, ...
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TermElem<T> {
    Functor(T),
    Arity(u32),
    Arg(TermId),
}

impl<T> TermElem<T> {
    #[inline]
    const fn dummy() -> Self {
        Self::Arity(0xDEADDEAD)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct TermId(pub(crate) usize);

impl ops::Add<usize> for TermId {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl ops::AddAssign<usize> for TermId {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs;
    }
}

/// Generates the same hash value as what [`buf_term_hash`] generates.
fn term_hash<T: Hash>(term: &Term<T>) -> u64 {
    // A hasher with fixed keys
    let mut hasher = FxHasher::default();
    write_term(&mut hasher, term);
    return hasher.finish();

    // === Internal helper functions ===

    /// Write functor(T) and arity(u32) in a DFS way.
    fn write_term<H: Hasher, T: Hash>(hasher: &mut H, term: &Term<T>) {
        term.functor.hash(hasher);
        hasher.write_u32(term.args.len() as u32);

        for arg in &term.args {
            write_term(hasher, arg);
        }
    }
}

/// Returns true if the two terms at `a` and `b` are structurally identical.
fn structually_eq<T: PartialEq>(buf: &[TermElem<T>], a: TermId, b: TermId) -> bool {
    if a == b {
        return true;
    }
    let TermElem::Functor(fa) = &buf[a.0] else {
        return false;
    };
    let TermElem::Functor(fb) = &buf[b.0] else {
        return false;
    };
    if fa != fb {
        return false;
    }
    let TermElem::Arity(na) = buf[a.0 + 1] else {
        return false;
    };
    let TermElem::Arity(nb) = buf[b.0 + 1] else {
        return false;
    };
    if na != nb {
        return false;
    }
    (0..na as usize).all(|i| {
        let TermElem::Arg(arg_a) = buf[a.0 + 2 + i] else {
            return false;
        };
        let TermElem::Arg(arg_b) = buf[b.0 + 2 + i] else {
            return false;
        };
        structually_eq(buf, arg_a, arg_b)
    })
}

/// Generates the same hash value as what [`term_hash`] generates.
fn buf_term_hash<T: Hash>(buf: &[TermElem<T>], id: TermId) -> u64 {
    // A hasher with fixed keys
    let mut hasher = FxHasher::default();
    write_term(&mut hasher, buf, id);
    return hasher.finish();

    // === Internal helper functions ===

    /// Write functor(T) and arity(u32) in a DFS way.
    fn write_term<H: Hasher, T: Hash>(hasher: &mut H, buf: &[TermElem<T>], id: TermId) {
        let i = id.0;
        let TermElem::Functor(functor) = &buf[i] else {
            unreachable!()
        };
        let TermElem::Arity(arity) = &buf[i + 1] else {
            unreachable!()
        };

        functor.hash(hasher); // Functor
        arity.hash(hasher); // Arity

        for j in 0..*arity as usize {
            let TermElem::Arg(arg) = &buf[i + 2 + j] else {
                unreachable!()
            };
            write_term(hasher, buf, *arg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, ExprIn, Intern, Name, TermIn, TermStorageIn, UniqueTermArrayIn};
    use any_intern::DroplessInterner;

    #[test]
    fn test_expr_array_replace_term() {
        let mut buf = TermStorage::new();
        let interner = DroplessInterner::default();

        let id_expr = insert_expr(&mut buf, &interner, "f(g(X)), (Y; Z), X");

        let old_len = buf.terms.buf.len();
        let id_term_f = insert_term(&mut buf.terms, &interner, "f(g(X))");
        let id_term_y = insert_term(&mut buf.terms, &interner, "Y");
        let id_term_z = insert_term(&mut buf.terms, &interner, "Z");
        let id_term_x = insert_term(&mut buf.terms, &interner, "X");
        // Inserted existing terms, so that the array must have not changed.
        assert_eq!(buf.terms.buf.len(), old_len);

        let mut expected_buf: Vec<ExprElem> = vec![
            /*  0 */ ExprElem::And { len: 3 },
            /*  1 */ ExprElem::Expr(ExprId(4)),
            /*  2 */ ExprElem::Expr(ExprId(5)),
            /*  3 */ ExprElem::Expr(ExprId(10)),
            /*  4 */ ExprElem::Term(id_term_f),
            /*  5 */ ExprElem::Or { len: 2 },
            /*  6 */ ExprElem::Expr(ExprId(8)),
            /*  7 */ ExprElem::Expr(ExprId(9)),
            /*  8 */ ExprElem::Term(id_term_y),
            /*  9 */ ExprElem::Term(id_term_z),
            /* 10 */ ExprElem::Term(id_term_x),
        ];

        assert_eq!(buf.exprs.buf, expected_buf);

        let id_term_a = insert_term(&mut buf.terms, &interner, "a");
        let replaced = buf.get_expr_mut(id_expr).replace_term(id_term_x, id_term_a);
        assert!(replaced);

        let old_len = buf.terms.buf.len();
        let id_term_fa = insert_term(&mut buf.terms, &interner, "f(g(a))");
        // The upper term has already been inserted to the array by the
        // replacement. So that the array must have not changed.
        assert_eq!(buf.terms.buf.len(), old_len);

        let clone_on_replace: Vec<ExprElem> = vec![
            /* 11 */ ExprElem::And { len: 3 },
            /* 12 */ ExprElem::Expr(ExprId(15)),
            /* 13 */ ExprElem::Expr(ExprId(5)),
            /* 14 */ ExprElem::Expr(ExprId(16)),
            /* 15 */ ExprElem::Term(id_term_fa),
            /* 16 */ ExprElem::Term(id_term_a),
        ];
        expected_buf.extend(clone_on_replace);

        assert_eq!(buf.exprs.buf, expected_buf);
    }

    #[test]
    fn test_expr_array_replace_expr() {
        let mut buf = TermStorage::new();
        let interner = DroplessInterner::default();

        let id_expr = insert_expr(&mut buf, &interner, "X, Y");
        let id_to = insert_expr(&mut buf, &interner, "f(X), g(X)");

        let old_len = buf.terms.buf.len();
        let id_term_x = insert_term(&mut buf.terms, &interner, "X");
        let id_term_y = insert_term(&mut buf.terms, &interner, "Y");
        let id_term_fx = insert_term(&mut buf.terms, &interner, "f(X)");
        let id_term_gx = insert_term(&mut buf.terms, &interner, "g(X)");
        // Inserted existing terms, so that the array must have not changed.
        assert_eq!(buf.terms.buf.len(), old_len);

        let mut expected_buf: Vec<ExprElem> = vec![
            /*  0 */ ExprElem::And { len: 2 },
            /*  1 */ ExprElem::Expr(ExprId(3)),
            /*  2 */ ExprElem::Expr(ExprId(4)),
            /*  3 */ ExprElem::Term(id_term_x),
            /*  4 */ ExprElem::Term(id_term_y),
            /*  5 */ ExprElem::And { len: 2 },
            /*  6 */ ExprElem::Expr(ExprId(8)),
            /*  7 */ ExprElem::Expr(ExprId(9)),
            /*  8 */ ExprElem::Term(id_term_fx),
            /*  9 */ ExprElem::Term(id_term_gx),
        ];

        assert_eq!(buf.exprs.buf, expected_buf);

        let mut view = buf.get_expr_mut(id_expr);
        let replaced = view.replace_leftmost_term(id_to);
        assert!(replaced);

        let clone_on_replace: Vec<ExprElem> = vec![
            /* 10 */ ExprElem::And { len: 2 },
            /* 11 */ ExprElem::Expr(ExprId(5)),
            /* 12 */ ExprElem::Expr(ExprId(4)),
        ];
        expected_buf.extend(clone_on_replace);

        assert_eq!(buf.exprs.buf, expected_buf);
    }

    #[test]
    fn test_term_array_replace() {
        let mut arr = UniqueTermArray::new();
        let interner = DroplessInterner::default();

        let id_x = insert_term(&mut arr, &interner, "X");
        let id_a = insert_term(&mut arr, &interner, "a");
        let id_f = insert_term(&mut arr, &interner, "f(g(X), h(X, Y))");

        let mut expected_buf: Vec<TermElem<Name<_>>> = vec![
            /*  0 */ TermElem::Functor(Name::with_intern("X", &interner)),
            /*  1 */ TermElem::Arity(0),
            /*  2 */ TermElem::Functor(Name::with_intern("a", &interner)),
            /*  3 */ TermElem::Arity(0),
            /*  4 */ TermElem::Functor(Name::with_intern("f", &interner)),
            /*  5 */ TermElem::Arity(2),
            /*  6 */ TermElem::Arg(TermId(8)),
            /*  7 */ TermElem::Arg(TermId(11)),
            /*  8 */ TermElem::Functor(Name::with_intern("g", &interner)),
            /*  9 */ TermElem::Arity(1),
            /* 10 */ TermElem::Arg(TermId(0)),
            /* 11 */ TermElem::Functor(Name::with_intern("h", &interner)),
            /* 12 */ TermElem::Arity(2),
            /* 13 */ TermElem::Arg(TermId(0)),
            /* 14 */ TermElem::Arg(TermId(15)),
            /* 15 */ TermElem::Functor(Name::with_intern("Y", &interner)),
            /* 16 */ TermElem::Arity(0),
        ];

        assert_eq!(arr.buf, expected_buf);
        assert_eq!(id_x, TermId(0));
        assert_eq!(id_a, TermId(2));
        assert_eq!(id_f, TermId(4));

        // === Replace ===

        let replaced = arr.get_mut(id_f).replace(id_x, id_a);
        assert!(replaced);

        let clone_on_replace: Vec<TermElem<Name<_>>> = vec![
            /* 17 */ TermElem::Functor(Name::with_intern("f", &interner)),
            /* 18 */ TermElem::Arity(2),
            /* 19 */ TermElem::Arg(TermId(21)),
            /* 20 */ TermElem::Arg(TermId(24)),
            /* 21 */ TermElem::Functor(Name::with_intern("g", &interner)),
            /* 22 */ TermElem::Arity(1),
            /* 23 */ TermElem::Arg(TermId(2)),
            /* 24 */ TermElem::Functor(Name::with_intern("h", &interner)),
            /* 25 */ TermElem::Arity(2),
            /* 26 */ TermElem::Arg(TermId(2)),
            /* 27 */ TermElem::Arg(TermId(15)),
        ];
        expected_buf.extend(clone_on_replace);

        assert_eq!(arr.buf, expected_buf);
    }

    #[test]
    #[rustfmt::skip]
    fn test_term_array_replace_with() {
        let mut arr = UniqueTermArray::new();
        let interner = DroplessInterner::default();

        let id_f = insert_term(&mut arr, &interner, "f($X, $Y, $X)");

        let mut expected_buf: Vec<TermElem<Name<_>>> = vec![
            /*  0 */ TermElem::Functor(Name::with_intern("f", &interner)),
            /*  1 */ TermElem::Arity(3),
            /*  2 */ TermElem::Arg(TermId(5)),
            /*  3 */ TermElem::Arg(TermId(7)),
            /*  4 */ TermElem::Arg(TermId(5)), // X appears twice, but stored once
            /*  5 */ TermElem::Functor(Name::with_intern("$X", &interner)),
            /*  6 */ TermElem::Arity(0),
            /*  7 */ TermElem::Functor(Name::with_intern("$Y", &interner)),
            /*  8 */ TermElem::Arity(0),
        ];

        assert_eq!(arr.buf, expected_buf);
        assert_eq!(id_f, TermId(0));

        // === Replace ===

        let mut view = arr.get_mut(id_f);
        let replaced = view.replace_with(|functor| {
            let interned = match functor.as_ref() {
                "$X" => interner.intern_formatted_str(&0, 1).unwrap(),
                "$Y" => interner.intern_formatted_str(&1, 1).unwrap(),
                _ => return None,
            };
            Some(Name::new(interned))
        });
        assert!(replaced);

        let clone_on_replace: Vec<TermElem<Name<_>>> = vec![
            /*  9 */ TermElem::Functor(Name::with_intern("f", &interner)),
            /* 10 */ TermElem::Arity(3),
            /* 11 */ TermElem::Arg(TermId(14)),
            /* 12 */ TermElem::Arg(TermId(16)),
            /* 13 */ TermElem::Arg(TermId(14)), // reuses first "0" instead of a duplicate
            /* 14 */ TermElem::Functor(Name::with_intern("0", &interner)),
            /* 15 */ TermElem::Arity(0),
            /* 16 */ TermElem::Functor(Name::with_intern("1", &interner)),
            /* 17 */ TermElem::Arity(0),
        ];
        expected_buf.extend(clone_on_replace);

        // The view now points to the newly created f(0, 1, 0).
        assert_eq!(view.id(), TermId(9));
        assert_eq!(arr.buf, expected_buf);
        // The original f(X, Y, X) at id=0 is untouched.
        assert_eq!(arr.buf[0], TermElem::Functor(Name::with_intern("f", &interner)));
        assert_eq!(arr.buf[2], TermElem::Arg(TermId(5)));
        assert_eq!(arr.buf[4], TermElem::Arg(TermId(5))); // both X args still point to original X
    }

    #[test]
    fn test_recursive_term() {
        let mut arr = UniqueTermArray::new();
        let interner = DroplessInterner::default();

        insert_term(&mut arr, &interner, "f(f(a))");

        let expected_buf: &[TermElem<Name<_>>] = &[
            /*  0 */ TermElem::Functor(Name::with_intern("f", &interner)),
            /*  1 */ TermElem::Arity(1),
            /*  2 */ TermElem::Arg(TermId(3)),
            /*  3 */ TermElem::Functor(Name::with_intern("f", &interner)),
            /*  4 */ TermElem::Arity(1),
            /*  5 */ TermElem::Arg(TermId(6)),
            /*  6 */ TermElem::Functor(Name::with_intern("a", &interner)),
            /*  7 */ TermElem::Arity(0),
        ];

        assert_eq!(arr.buf, expected_buf);
    }

    fn insert_expr<'int, Int: Intern>(
        buf: &mut TermStorageIn<'int, Int>,
        interner: &'int Int,
        text: &str,
    ) -> ExprId {
        let expr: ExprIn<'int, Int> = parse::parse_str(text, interner).unwrap();
        buf.insert_expr(expr)
    }

    fn insert_term<'int, Int: Intern>(
        arr: &mut UniqueTermArrayIn<'int, Int>,
        interner: &'int Int,
        text: &str,
    ) -> TermId {
        let term: TermIn<'int, Int> = parse::parse_str(text, interner).unwrap();
        arr.insert(term)
    }
}
