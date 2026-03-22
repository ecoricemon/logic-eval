use super::{
    canonical::CanonicalTermId,
    prover::Integer,
    repr::{TermId, TermView},
};
use crate::{prove::prover::TermAssignments, Map};
use core::ops::{Index, IndexMut};

#[derive(Debug, Default)]
pub(crate) struct Table {
    indices: Map<CanonicalTermId, usize>,
    entries: Vec<TableEntry>,
}

impl Table {
    pub(crate) fn clear(&mut self) {
        self.indices.clear();
        self.entries.clear();
    }

    pub(crate) fn get_mut(
        &mut self,
        key: &CanonicalTermId,
    ) -> Option<(TableIndex, &mut TableEntry)> {
        self.indices
            .get(key)
            .map(|&i| (TableIndex(i), &mut self.entries[i]))
    }

    pub(crate) fn register(&mut self, key: CanonicalTermId, entry: TableEntry) -> TableIndex {
        if let Some(table_index) = self.indices.get(&key) {
            return TableIndex(*table_index);
        }

        let table_index = self.entries.len();
        self.indices.insert(key, table_index);
        self.entries.push(entry);
        TableIndex(table_index)
    }
}

impl Index<TableIndex> for Table {
    type Output = TableEntry;

    fn index(&self, index: TableIndex) -> &Self::Output {
        &self.entries[index.0]
    }
}

impl IndexMut<TableIndex> for Table {
    fn index_mut(&mut self, index: TableIndex) -> &mut Self::Output {
        &mut self.entries[index.0]
    }
}

#[derive(Debug)]
pub(crate) struct TableEntry {
    /// Non-empty assignment record for variables in a node.
    ///
    /// All [`SeenAssignments`] have the same length of answers. Use the same index to get a set of
    /// answers.
    ///
    /// # Examples
    ///
    /// If we have 3 answers like below, the answers for (X, Y) are only (a, i), (b, j) or (c, k).
    /// Other combinations are invalid.
    /// X = a or b or c
    /// Y = i or j or k
    seen: AnswerMatrix,

    /// Consumers(nodes) will be notified when their entry has updated.
    consumers: Vec<Consumer>,
}

impl TableEntry {
    /// Making an entry can be rejected when
    /// - `view` is just a variable, which doesn't make sense for tabling. f(X) should be given for
    ///   example.
    /// - `view` doesn't contain any variables in it. It doesn't need the tabling.
    pub(crate) fn from_term_view(view: &TermView<'_, Integer>) -> Option<Self> {
        if view.is_variable() || !view.contains_variable() {
            return None;
        }

        let mut vars = Vec::new();
        for arg in view.args() {
            arg.with_variable(|var| {
                if !vars.contains(&var.id) {
                    vars.push(var.id);
                }
            });
        }

        Some(Self {
            seen: AnswerMatrix::with_variables(vars),
            consumers: Vec::new(),
        })
    }

    /// See [`AnswerMatrix::update`].
    pub(crate) fn update_answer(&mut self, term_assigns: &TermAssignments) {
        self.seen.update(term_assigns);
    }

    pub(crate) fn has_answer(&self, term_assigns: &TermAssignments) -> bool {
        self.seen.has_answer(term_assigns)
    }

    pub(crate) fn consumer_nodes(&self) -> impl Iterator<Item = usize> + '_ {
        self.consumers.iter().map(|c| c.node_index)
    }

    pub(crate) fn variables(&self) -> &[TermId] {
        self.seen.column(0)
    }

    /// An empty slice is returned when the `answer_index` is out of bounds.
    pub(crate) fn answers(&self, answer_index: usize) -> &[TermId] {
        let col = answer_index + 1;
        self.seen.column(col)
    }

    pub(crate) fn register_consumer(&mut self, node_index: usize) {
        if self
            .consumers
            .iter()
            .all(|consumer| consumer.node_index != node_index)
        {
            self.consumers.push(Consumer { node_index });
        }
    }
}

/// A non-empty variable-answer relations.
///
///  unique var  | answer1 | answer2 | assign3 |
/// :----------: | :-----: | :-----: | :-----: |
///      X       |    a    |    x    |    i    |
///      Y       |    b    |    y    |    j    |
///      W       |    c    |    z    |    k    |
///      Z       |    d    |    w    |    l    |
#[derive(Debug)]
pub(crate) struct AnswerMatrix {
    /// Column-wise elements, e.g. X, Y, W, Z, a, b, c, d, ...
    elems: Vec<TermId>,
    rows: usize,
    // For double check
    cols: usize,
}

impl AnswerMatrix {
    fn with_variables(vars: Vec<TermId>) -> Self {
        let rows = vars.len();
        debug_assert!(rows > 0);

        for i in 0..rows {
            for j in i + 1..rows {
                debug_assert_ne!(vars[i], vars[j]);
            }
        }

        Self {
            elems: vars,
            rows,
            cols: 1,
        }
    }

    /// An empty slice is returned when the `col` is out of bounds.
    fn column(&self, col: usize) -> &[TermId] {
        let start = col * self.rows;
        let end = start + self.rows;

        if end <= self.elems.len() {
            &self.elems[start..end]
        } else {
            &[]
        }
    }

    /// This method assumes that the `term_assigns` has concrete answers(atoms, not variables) for
    /// variables of this entry.
    fn update(&mut self, term_assigns: &TermAssignments) {
        self.elems.reserve_exact(self.rows);
        for r in 0..self.rows {
            let var = self.elems[r];
            let answer = term_assigns.find(var).unwrap();
            self.elems.push(answer);
        }
        self.cols += 1;
    }

    fn has_answer(&self, term_assigns: &TermAssignments) -> bool {
        let vars = self.column(0);
        for col_idx in 1..self.cols {
            let answers = self.column(col_idx);
            if vars
                .iter()
                .zip(answers)
                .all(|(var, answer)| term_assigns.find(*var) == Some(*answer))
            {
                return true;
            }
        }
        false
    }
}

#[derive(Debug)]
struct Consumer {
    node_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct TableIndex(usize);
