use super::str::Str;
use crate::Map;

#[derive(Debug)]
pub struct SymbolTable<T> {
    map: Map<Str, Vec<Symbol<T>>>,
}

impl<T> SymbolTable<T> {
    pub fn new() -> Self {
        Self {
            map: Map::default(),
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn push(&mut self, name: Str, symbol: T) {
        if let Some(v) = self.map.get_mut(&name) {
            v.push(Symbol::Data(symbol));
        } else {
            self.map.insert(name, vec![Symbol::Data(symbol)]);
        }
    }

    pub fn pop(&mut self, name: &str) -> Option<T> {
        match self.map.get_mut(name)?.pop()? {
            Symbol::Data(v) => Some(v),
            Symbol::Transparent | Symbol::Opaque => None,
        }
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

    pub fn get(&self, name: &str) -> Option<&T> {
        self.map.get(name)?.iter().rev().find_map(|x| match x {
            Symbol::Data(v) => Some(Some(v)),
            Symbol::Transparent => None,  // Continue the iteration.
            Symbol::Opaque => Some(None), // Stop the iteration.
        })?
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut T> {
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

impl<T> Default for SymbolTable<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
enum Symbol<T> {
    Data(T),
    Transparent,
    Opaque,
}
