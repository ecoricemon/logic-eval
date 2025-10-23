use crate::Map;
use smallvec::{SmallVec, smallvec};
use std::{
    cell::Cell,
    hash::{DefaultHasher, Hash, Hasher},
    iter, mem, ops,
};

/// - Each value is unique in this container. if multiple indices have to refer
///   to the same value, then only one index points to the real value and the
///   others point to indirect values, which are just jump indices to the real
///   value.
/// - You can get a value via its index.
/// - This container doesn't support remove methods except clear method, so that
///   index cannot be broken until you call the clear method.
#[derive(Debug, Clone)]
pub struct UniqueContainer<T> {
    values: Values<T>,

    /// Hash of a `T` -> Indices to `values`, but only for [`Value::Data`].
    map: Map<u64, SmallVec<[usize; 1]>>,
}

impl<T> UniqueContainer<T> {
    pub fn new() -> Self {
        Self {
            values: Values::new(),
            map: Map::default(),
        }
    }

    pub const fn len(&self) -> usize {
        self.values.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    pub fn iter(&self) -> PairIter<T> {
        self.values.iter()
    }

    pub fn values(&self) -> ValueIter<T> {
        self.values.values()
    }

    pub fn clear(&mut self) {
        self.values.clear();
        self.map.clear();
    }
}

impl<T> UniqueContainer<T>
where
    T: Hash + Eq + std::fmt::Debug, // TODO: remove debug
{
    /// Inserts the given value in the container.
    ///
    /// If the same value was found by [`PartialEq`], then the old value is
    /// replaced with the given new value.
    pub fn insert(&mut self, value: T) -> usize {
        let hash = Self::hash(&value);
        if let Some(index) = self._find(hash, &value) {
            self.values.replace(index, Value::Data(value));
            return index;
        }

        let index = self.values.len();

        Self::append_mapping(&mut self.map, hash, index);
        self.values.push(Value::Data(value));

        index
    }

    pub fn next_index<Q>(&mut self, value: &Q) -> usize
    where
        Q: Hash + PartialEq<T> + ?Sized,
    {
        self.find(value).unwrap_or(self.values.len())
    }

    pub fn replace<Q>(&mut self, old: &Q, new: T) -> bool
    where
        Q: Hash + PartialEq<T> + ?Sized,
    {
        if let Some(index) = self.find(old) {
            self.replace_at(index, new);
            true
        } else {
            false
        }
    }

    /// Replaces a value at the given index with the given value.
    ///
    /// Note that some other indices that point to the old value will point to
    /// the given new value after replacement.
    ///
    /// You may keep in mind that you should get an index using [`Self::find`].
    /// Because what you want would be to replace [`Value::Data`] which can
    /// be obtained by the find function. Otherwise, you may pick up an index
    /// to a [`Value::Indirect`] which ends up a wrong result.
    fn replace_at(&mut self, index: usize, value: T) {
        let hash = Self::hash(&value);

        if let Some(exist_idx) = self._find(hash, &value) {
            self.values
                .replace(index, Value::Indirect(Cell::new(exist_idx)));
            self.values.replace(exist_idx, Value::Data(value));
        } else {
            Self::append_mapping(&mut self.map, hash, index);
            let old = self.values.replace(index, Value::Data(value));
            if let Value::Data(old) = old {
                Self::remove_mapping(&mut self.map, Self::hash(&old), index);
            }
        }
    }

    pub fn find<Q>(&self, value: &Q) -> Option<usize>
    where
        Q: Hash + PartialEq<T> + ?Sized,
    {
        let hash = Self::hash(value);
        self._find(hash, value)
    }

    /// Returns an index to the given value in the container.
    ///
    /// The returned index points to a real data, which is [`Value::Data`] in
    /// other words.
    fn _find<Q>(&self, hash: u64, value: &Q) -> Option<usize>
    where
        Q: PartialEq<T> + ?Sized,
    {
        if let Some(idxs) = self.map.get(&hash) {
            for idx in idxs.iter() {
                if value == &self.values[*idx] {
                    return Some(*idx);
                }
            }
        }
        None
    }

    fn hash<Q>(value: &Q) -> u64
    where
        Q: Hash + PartialEq<T> + ?Sized,
    {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn append_mapping(map: &mut Map<u64, SmallVec<[usize; 1]>>, hash: u64, index: usize) {
        map.entry(hash)
            .and_modify(|indices| indices.push(index))
            .or_insert(smallvec![index]);
    }

    fn remove_mapping(map: &mut Map<u64, SmallVec<[usize; 1]>>, hash: u64, index: usize) {
        let Some(indices) = map.get_mut(&hash) else {
            return;
        };

        let Some((i, _)) = indices.iter().enumerate().find(|(_, ii)| **ii == index) else {
            return;
        };

        indices.swap_remove(i);

        if indices.is_empty() {
            map.remove(&hash);
        }
    }
}

impl<T> ops::Index<usize> for UniqueContainer<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<T> Default for UniqueContainer<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
struct Values<T>(Vec<Value<T>>);

impl<T> Values<T> {
    const fn new() -> Self {
        Self(Vec::new())
    }

    const fn len(&self) -> usize {
        self.0.len()
    }

    fn iter(&self) -> PairIter<T> {
        PairIter::new(self)
    }

    fn values(&self) -> ValueIter<T> {
        ValueIter::new(self)
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.get_with_opt(index).map(|(_idx, ty)| ty)
    }

    fn get_with_opt(&self, index: usize) -> Option<(usize, &T)> {
        match self.0.get(index)? {
            Value::Indirect(v) => {
                let (w, ty) = self.get_with_opt(v.get())?;
                v.set(w);
                Some((w, ty))
            }
            Value::Data(ty) => Some((index, ty)),
        }
    }

    fn replace(&mut self, index: usize, value: Value<T>) -> Value<T> {
        mem::replace(&mut self.0[index], value)
    }

    fn push(&mut self, value: Value<T>) {
        self.0.push(value);
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}

impl<T> ops::Index<usize> for Values<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index is out of bounds")
    }
}

#[derive(Debug, Clone)]
enum Value<T> {
    Indirect(Cell<usize>),
    Data(T),
}

/// The iterator skips indirect values.
#[derive(Clone)]
pub struct PairIter<'a, T> {
    values: &'a [Value<T>],
    cur: usize,
    remain: usize,
}

impl<'a, T> PairIter<'a, T> {
    fn new(values: &'a Values<T>) -> Self {
        Self {
            values: values.0.as_slice(),
            cur: 0,
            remain: values
                .0
                .iter()
                .filter(|value| matches!(value, Value::Data(_)))
                .count(),
        }
    }

    const fn len(&self) -> usize {
        self.remain
    }
}

impl<'a, T> Iterator for PairIter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remain == 0 {
            return None;
        }

        let value = loop {
            let value = &self.values[self.cur];
            self.cur += 1;
            match value {
                Value::Indirect(_) => continue,
                Value::Data(value) => break value,
            }
        };
        self.remain -= 1;
        Some((self.cur - 1, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = <Self>::len(self);
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for PairIter<'_, T> {
    fn len(&self) -> usize {
        <Self>::len(self)
    }
}

impl<T> iter::FusedIterator for PairIter<'_, T> {}

/// The iterator skips indirect values.
#[derive(Clone)]
pub struct ValueIter<'a, T> {
    rest: &'a [Value<T>],
    remain: usize,
}

impl<'a, T> ValueIter<'a, T> {
    fn new(values: &'a Values<T>) -> Self {
        Self {
            rest: values.0.as_slice(),
            remain: values
                .0
                .iter()
                .filter(|value| matches!(value, Value::Data(_)))
                .count(),
        }
    }

    const fn len(&self) -> usize {
        self.remain
    }
}

impl<'a, T> Iterator for ValueIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let value = loop {
            let (head, rest) = self.rest.split_first()?;
            self.rest = rest;
            match head {
                Value::Indirect(_) => continue,
                Value::Data(value) => break value,
            }
        };
        self.remain -= 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = <Self>::len(self);
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for ValueIter<'_, T> {
    fn len(&self) -> usize {
        <Self>::len(self)
    }
}

impl<T> iter::FusedIterator for ValueIter<'_, T> {}
