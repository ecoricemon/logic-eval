use crate::Map;
use std::{borrow::Borrow, hash::Hash};

/// A scoped symbol table with transparent and opaque block boundaries.
#[derive(Debug)]
pub struct SymbolTable<K, V> {
    map: Map<K, Vec<Symbol<V>>>,
}

impl<K, V> SymbolTable<K, V> {
    /// Removes every symbol and block marker.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::symbol::SymbolTable;
    ///
    /// let mut table = SymbolTable::default();
    /// table.push("x", 1);
    /// assert!(!table.is_empty());
    ///
    /// table.clear();
    /// assert!(table.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Returns `true` if the table contains no symbols or block markers.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::symbol::SymbolTable;
    ///
    /// let mut table = SymbolTable::default();
    /// assert!(table.is_empty());
    ///
    /// table.push("x", 1);
    /// assert!(!table.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Pushes a block boundary that allows lookup to continue into outer blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::symbol::SymbolTable;
    ///
    /// let mut table = SymbolTable::default();
    /// table.push("x", 1);
    /// table.push_transparent_block();
    ///
    /// assert_eq!(table.get("x"), Some(&1));
    /// ```
    pub fn push_transparent_block(&mut self) {
        for v in self.map.values_mut() {
            v.push(Symbol::Transparent);
        }
    }

    /// Pushes a block boundary that hides symbols in outer blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::symbol::SymbolTable;
    ///
    /// let mut table = SymbolTable::default();
    /// table.push("x", 1);
    /// table.push_opaque_block();
    ///
    /// assert_eq!(table.get("x"), None);
    /// ```
    pub fn push_opaque_block(&mut self) {
        for v in self.map.values_mut() {
            v.push(Symbol::Opaque);
        }
    }

    /// Pops symbols from the current block.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::symbol::SymbolTable;
    ///
    /// let mut table = SymbolTable::default();
    /// table.push("x", 1);
    /// table.push_transparent_block();
    /// table.push("x", 2);
    ///
    /// assert_eq!(table.get("x"), Some(&2));
    /// table.pop_block();
    /// assert_eq!(table.get("x"), Some(&1));
    /// ```
    pub fn pop_block(&mut self) {
        for v in self.map.values_mut() {
            while let Some(Symbol::Data(_)) = v.pop() {}
        }
        self.map.retain(|_, v| !v.is_empty());
    }
}

impl<K: Hash + Eq, V> SymbolTable<K, V> {
    /// Pushes a symbol binding for `name`.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::symbol::SymbolTable;
    ///
    /// let mut table = SymbolTable::default();
    /// table.push("x", 1);
    ///
    /// assert_eq!(table.get("x"), Some(&1));
    /// ```
    pub fn push(&mut self, name: K, symbol: V) {
        if let Some(v) = self.map.get_mut(&name) {
            v.push(Symbol::Data(symbol));
        } else {
            self.map.insert(name, vec![Symbol::Data(symbol)]);
        }
    }

    /// Pops the most recent symbol binding for `name`.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::symbol::SymbolTable;
    ///
    /// let mut table = SymbolTable::default();
    /// table.push("x", 1);
    /// table.push("x", 2);
    ///
    /// assert_eq!(table.pop("x"), Some(2));
    /// assert_eq!(table.get("x"), Some(&1));
    /// ```
    pub fn pop<Q>(&mut self, name: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match self.map.get_mut(name)?.pop()? {
            Symbol::Data(v) => Some(v),
            Symbol::Transparent | Symbol::Opaque => None,
        }
    }

    /// Returns the visible symbol binding for `name`.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::symbol::SymbolTable;
    ///
    /// let mut table = SymbolTable::default();
    /// table.push("x", 1);
    /// table.push_transparent_block();
    /// table.push("x", 2);
    ///
    /// assert_eq!(table.get("x"), Some(&2));
    /// assert_eq!(table.get("y"), None);
    /// ```
    pub fn get<Q>(&self, name: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get(name)?.iter().rev().find_map(|x| match x {
            Symbol::Data(v) => Some(Some(v)),
            Symbol::Transparent => None,  // Continue the iteration.
            Symbol::Opaque => Some(None), // Stop the iteration.
        })?
    }

    /// Returns the visible mutable symbol binding for `name`.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::symbol::SymbolTable;
    ///
    /// let mut table = SymbolTable::default();
    /// table.push("x", 1);
    ///
    /// *table.get_mut("x").unwrap() = 2;
    /// assert_eq!(table.get("x"), Some(&2));
    /// ```
    pub fn get_mut<Q>(&mut self, name: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map
            .get_mut(name)?
            .iter_mut()
            .rev()
            .find_map(|x| match x {
                Symbol::Data(v) => Some(Some(v)),
                Symbol::Transparent => None,  // Continue the iteration.
                Symbol::Opaque => Some(None), // Stop the iteration.
            })?
    }
}

impl<K, V> Default for SymbolTable<K, V> {
    fn default() -> Self {
        Self {
            map: Map::default(),
        }
    }
}

#[derive(Debug)]
enum Symbol<T> {
    Data(T),
    Transparent,
    Opaque,
}
