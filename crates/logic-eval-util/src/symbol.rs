use crate::Map;
use std::{borrow::Borrow, hash::Hash};

#[derive(Debug)]
pub struct SymbolTable<K, V> {
    map: Map<K, Vec<Symbol<V>>>,
}

impl<K, V> SymbolTable<K, V> {
    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn push_transparent_block(&mut self) {
        for v in self.map.values_mut() {
            v.push(Symbol::Transparent);
        }
    }

    pub fn push_opaque_block(&mut self) {
        for v in self.map.values_mut() {
            v.push(Symbol::Opaque);
        }
    }

    pub fn pop_block(&mut self) {
        for v in self.map.values_mut() {
            while let Some(Symbol::Data(_)) = v.pop() {}
        }
        self.map.retain(|_, v| !v.is_empty());
    }
}

impl<K: Hash + Eq, V> SymbolTable<K, V> {
    pub fn push(&mut self, name: K, symbol: V) {
        if let Some(v) = self.map.get_mut(&name) {
            v.push(Symbol::Data(symbol));
        } else {
            self.map.insert(name, vec![Symbol::Data(symbol)]);
        }
    }

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
