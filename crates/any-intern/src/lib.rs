#![doc = include_str!("../README.md")]

mod any;
mod common;
mod dropless;
mod typed;

// === Re-exports ===

pub use any::{AnyArena, AnyInternSet, AnyInterner};
pub use common::{Interned, UnsafeLock};
pub use dropless::{Dropless, DroplessInternSet, DroplessInterner};
pub use typed::TypedArena;

use std::{
    any::TypeId,
    borrow,
    collections::HashMap,
    hash::{BuildHasher, Hash},
};

/// A generic interner for storing and deduplicating values of various types.
///
/// The `Interner` provides a mechanism to store values in a way that ensures each unique value is
/// stored only once. It supports interning both static types and dropless types, allowing efficient
/// memory usage and fast lookups.
///
/// Interning is useful when you need to store many instances of the same value and want to avoid
/// duplication. Instead of storing multiple copies of the same value, the `Interner` ensures that
/// only one instance of each unique value exists, and all references point to that instance.
///
/// # Examples
///
/// ```
/// use any_intern::Interner;
///
/// #[derive(PartialEq, Eq, Hash, Debug)]
/// struct A(u32);
///
/// #[derive(PartialEq, Eq, Hash, Debug)]
/// struct B(String);
///
/// let interner = Interner::new();
///
/// // Interning integers
/// let int1 = interner.intern_static(42_u32);
/// let int2 = interner.intern_static(42_u32);
/// assert_eq!(int1, int2); // Same value, same reference
///
/// // Interning custom structs
/// let a1 = interner.intern_static(A(1));
/// let a2 = interner.intern_static(A(1));
/// assert_eq!(a1, a2); // Same value, same reference
///
/// // Interning strings
/// let b1 = interner.intern_dropless(&*String::from("hello"));
/// let b2 = interner.intern_dropless(&*String::from("hello"));
/// assert_eq!(b1, b2); // Same value, same reference
/// ```
pub struct Interner<S = fxhash::FxBuildHasher> {
    /// Intern storage for static types.
    pub anys: UnsafeLock<HashMap<TypeId, AnyInternSet, S>>,

    /// Intern storage for dropless types.
    pub dropless: DroplessInterner,
}

impl Interner {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<S: BuildHasher> Interner<S> {
    /// Stores a value in the interner, returning a reference to the interned value.
    ///
    /// This method inserts the given value into the interner if it does not already exist. If the
    /// value already exists, a reference to the existing value is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use any_intern::Interner;
    ///
    /// #[derive(PartialEq, Eq, Hash, Debug)]
    /// struct A(u32);
    ///
    /// #[derive(PartialEq, Eq, Hash, Debug)]
    /// struct B(String);
    ///
    /// let interner = Interner::new();
    ///
    /// // Interning integers
    /// let int1 = interner.intern_static(42_u32);
    /// let int2 = interner.intern_static(42_u32);
    /// assert_eq!(int1, int2); // Same value, same reference
    /// assert_eq!(int1.raw().as_ptr(), int2.raw().as_ptr());
    ///
    /// // Interning custom structs
    /// let a1 = interner.intern_static(A(1));
    /// let a2 = interner.intern_static(A(1));
    /// assert_eq!(a1, a2); // Same value, same reference
    /// assert_eq!(a1.raw().as_ptr(), a2.raw().as_ptr());
    ///
    /// // Interning strings
    /// let b1 = interner.intern_static(B("hello".to_string()));
    /// let b2 = interner.intern_static(B("hello".to_string()));
    /// assert_eq!(b1, b2); // Same value, same reference
    /// assert_eq!(b1.raw().as_ptr(), b2.raw().as_ptr());
    ///
    /// // Interning different values
    /// let b3 = interner.intern_static(B("world".to_string()));
    /// assert_ne!(b1, b3); // Different values, different references
    /// ```
    pub fn intern_static<K: Hash + Eq + 'static>(&self, value: K) -> Interned<'_, K> {
        self.with_any_set::<K, _, _>(|set| unsafe {
            // Safety: Type `K` is consistent and correct.
            set.intern(value)
        })
    }

    /// Stores a value in the interner, creating it only if it does not already exist.
    ///
    /// This method allows you to provide a key and a closure to generate the value. If the key
    /// already exists in the interner, the closure is not called, and a reference to the existing
    /// value is returned. If the key does not exist, the closure is called to create the value,
    /// which is then stored in the interner.
    ///
    /// This method is useful when the value is expensive to compute, as it avoids unnecessary
    /// computation if the value already exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use any_intern::Interner;
    ///
    /// #[derive(PartialEq, Eq, Hash, Debug)]
    /// struct A(u32);
    ///
    /// impl std::borrow::Borrow<u32> for A {
    ///     fn borrow(&self) -> &u32 {
    ///         &self.0
    ///     }
    /// }
    ///
    /// let interner = Interner::new();
    ///
    /// let a = interner.intern_static_with(&42, || A(42));
    /// assert_eq!(interner.len(), 1);
    /// assert_eq!(*a, &A(42));
    ///
    /// let b = interner.intern_static_with(&42, || A(99)); // Closure is not called
    /// assert_eq!(interner.len(), 1);
    /// assert_eq!(*b, &A(42));
    ///
    /// let c = interner.intern_static_with(&43, || A(43));
    /// assert_eq!(interner.len(), 2);
    /// assert_eq!(*c, &A(43));
    /// ```
    pub fn intern_static_with<'a, K, Q, F>(&'a self, key: &Q, make_value: F) -> Interned<'a, K>
    where
        K: borrow::Borrow<Q> + 'static,
        Q: Hash + Eq + ?Sized,
        F: FnOnce() -> K,
    {
        self.with_any_set::<K, _, _>(|set| unsafe {
            // Safety: Type `K` is consistent and correct.
            set.intern_with(key, make_value)
        })
    }

    /// Retrieves a reference to a value in the interner based on the provided key.
    ///
    /// This method checks if a value corresponding to the given key exists in the interner. If it
    /// exists, a reference to the interned value is returned. Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use any_intern::Interner;
    ///
    /// let interner = Interner::new();
    /// interner.intern_static(42_u32);
    ///
    /// assert_eq!(*interner.get::<u32, _>(&42_u32).unwrap(), &42);
    /// assert!(interner.get::<u32, _>(&99_u32).is_none());
    /// ```
    pub fn get<K, Q>(&self, key: &Q) -> Option<Interned<'_, K>>
    where
        K: borrow::Borrow<Q> + 'static,
        Q: Hash + Eq + ?Sized,
    {
        self.with_any_set::<K, _, _>(|set| unsafe {
            // Safety: Type `K` is consistent and correct.
            set.get(key)
        })
    }

    /// Stores the given dropless value in the interner then returns reference to the value if the
    /// interner doesn't contain the same value yet.
    ///
    /// If the same value exists in the interner, reference to the existing value is returned.
    ///
    /// This method does not take the value's ownership. Instead, it copies the value into the
    /// interner's memory, then returns reference to that.
    ///
    /// # Eaxmples
    ///
    /// ```
    /// use any_intern::Interner;
    ///
    /// let interner = Interner::new();
    /// let a = interner.intern_dropless("hello");
    /// let b = interner.intern_dropless(*Box::new("hello"));
    /// let c = interner.intern_dropless("hi");
    /// assert_eq!(a, b);
    /// assert_ne!(a, c);
    /// ```
    pub fn intern_dropless<K: Dropless + ?Sized>(&self, value: &K) -> Interned<'_, K> {
        self.dropless.intern(value)
    }

    /// Retrieves a reference to a value in the interner based on the provided key.
    ///
    /// This method checks if a value corresponding to the given key exists in the interner. If it
    /// exists, a reference to the interned value is returned. Otherwise, `None` is returned.
    ///
    /// # Eaxmples
    ///
    /// ```
    /// use any_intern::Interner;
    ///
    /// let interner = Interner::new();
    /// interner.intern_dropless("hello");
    ///
    /// assert_eq!(interner.get_dropless("hello").as_deref(), Some(&"hello"));
    /// assert!(interner.get_dropless("hi").is_none());
    /// ```
    pub fn get_dropless<K: Dropless + ?Sized>(&self, value: &K) -> Option<Interned<'_, K>> {
        self.dropless.get(value)
    }

    /// Returns number of values the interner contains.
    pub fn len(&self) -> usize {
        self.with_any_sets(|sets| sets.values().map(AnyInternSet::len).sum::<usize>())
            + self.dropless.len()
    }

    /// Returns true if the interner is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Removes all values in the interner.
    ///
    /// Although the interner support interior mutability, clear method requires mutable access
    /// to the interner to invalidate all [`Interned`]s referencing the interner.
    pub fn clear(&mut self) {
        self.with_any_sets(|sets| {
            for set in sets.values_mut() {
                set.clear();
            }
        });
        self.dropless.clear();
    }

    /// * f - Its argument is guaranteed to be a set for the type `K`.
    fn with_any_set<'this, K, F, R>(&'this self, f: F) -> R
    where
        K: 'static,
        F: FnOnce(&'this mut AnyInternSet) -> R,
        R: 'this,
    {
        self.with_any_sets(|sets| {
            let set = sets
                .entry(TypeId::of::<K>())
                .or_insert_with(|| AnyInternSet::of::<K>());
            f(set)
        })
    }

    fn with_any_sets<'this, F, R>(&self, f: F) -> R
    where
        F: FnOnce(&'this mut HashMap<TypeId, AnyInternSet, S>) -> R,
        R: 'this,
        S: 'this,
    {
        // Safety: Mutex unlocking is paired with the locking.
        unsafe {
            let sets = self.anys.lock().as_mut();
            let ret = f(sets);
            self.anys.unlock();
            ret
        }
    }
}

impl<S: Default> Default for Interner<S> {
    fn default() -> Self {
        // Safety: Only one instance
        let anys = unsafe { UnsafeLock::new(HashMap::default()) };
        Self {
            anys,
            dropless: DroplessInterner::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{self, RawInterned};

    #[test]
    #[rustfmt::skip]
    fn test_interner() {
        #[derive(PartialEq, Eq, Hash)]
        struct A(i32);
        #[derive(PartialEq, Eq, Hash)]
        struct B(i32);

        let interner = Interner::new();

        let groups: [&[RawInterned]; _] = [
            &[interner.intern_static(A(0)).raw(), interner.intern_static(A(0)).raw()],
            &[interner.intern_static(A(1)).raw()],
            &[interner.intern_static(B(0)).raw(), interner.intern_static(B(0)).raw()],
            &[interner.intern_static(B(1)).raw()],
        ];
        common::assert_group_addr_eq(&groups);
    }
}
