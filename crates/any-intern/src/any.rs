use super::common::{self, Interned, RawInterned, UnsafeLock};
use bumpalo::Bump;
use hashbrown::{HashTable, hash_table::Entry};
use std::{
    alloc::Layout,
    any::TypeId,
    borrow,
    cell::Cell,
    hash::{BuildHasher, Hash},
    mem, ptr,
};

/// A type-erased interner for storing and deduplicating values of a single type.
///
/// This interner is simply a wrapper of [`AnyInternSet`] with interior mutability. If you need a
/// collection of interners for various types like a hash map of interners, then consider using the
/// `AnyInterSet` with a container providing interior mutability such as [`ManualMutex`].
///
/// # Examples
///
/// ```
/// use any_intern::AnyInterner;
///
/// #[derive(PartialEq, Eq, Hash, Debug)]
/// struct A(u32);
///
/// let interner = AnyInterner::of::<A>();
///
/// unsafe {
///     let a1 = interner.intern(A(42));
///     let a2 = interner.intern(A(42));
///     assert_eq!(a1, a2); // Same value, same reference
///
///     let a3 = interner.intern(A(99));
///     assert_ne!(a1, a3); // Different values, different references
/// }
/// ```
///
/// # Safety
///
/// Many methods in `AnyInterner` are marked as `unsafe` because they rely on the caller to ensure
/// that the correct type is used when interacting with the interner. Using an incorrect type can
/// lead to undefined behavior.
pub struct AnyInterner<S = fxhash::FxBuildHasher> {
    inner: UnsafeLock<AnyInternSet<S>>,
}

impl AnyInterner {
    pub fn of<K: 'static>() -> Self {
        // Safety: Only one instance
        let inner = unsafe { UnsafeLock::new(AnyInternSet::of::<K>()) };
        Self { inner }
    }
}

impl<S: BuildHasher> AnyInterner<S> {
    pub fn with_hasher<K: 'static>(hash_builder: S) -> Self {
        // Safety: Only one instance
        let inner = unsafe { UnsafeLock::new(AnyInternSet::with_hasher::<K>(hash_builder)) };
        Self { inner }
    }

    /// Returns number of values the interner contains.
    pub fn len(&self) -> usize {
        self.with_inner(|set| set.len())
    }

    /// Returns true if the interner is empty.
    pub fn is_empty(&self) -> bool {
        self.with_inner(|set| set.is_empty())
    }

    /// Stores a value in the interner, returning a reference to the interned value.
    ///
    /// This method inserts the given value into the interner if it does not already exist. If the
    /// value already exists, a reference to the existing value is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use any_intern::AnyInterner;
    ///
    /// #[derive(PartialEq, Eq, Hash, Debug)]
    /// struct A(u32);
    ///
    /// let interner = AnyInterner::of::<A>();
    ///
    /// unsafe {
    ///     let a1 = interner.intern(A(42));
    ///     let a2 = interner.intern(A(42));
    ///     assert_eq!(a1, a2); // Same value, same reference
    ///     assert_eq!(a1.raw().as_ptr(), a2.raw().as_ptr());
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// Undefined behavior if incorrect type `K` is given.
    pub unsafe fn intern<K>(&self, value: K) -> Interned<'_, K>
    where
        K: Hash + Eq + 'static,
    {
        self.with_inner(|set| unsafe { set.intern(value) })
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
    /// use any_intern::AnyInterner;
    ///
    ///
    /// #[derive(PartialEq, Eq, Hash, Debug)]
    /// struct A(i32);
    ///
    /// impl std::borrow::Borrow<i32> for A {
    ///     fn borrow(&self) -> &i32 {
    ///         &self.0
    ///     }
    /// }
    ///
    /// let interner = AnyInterner::of::<A>();
    ///
    /// unsafe {
    ///     let a = interner.intern_with(&42, || A(42));
    ///     assert_eq!(interner.len(), 1);
    ///     assert_eq!(*a, &A(42));
    ///
    ///     let b = interner.intern_with(&42, || A(99)); // Closure is not called
    ///     assert_eq!(interner.len(), 1);
    ///     assert_eq!(*b, &A(42));
    ///
    ///     let c = interner.intern_with(&43, || A(43));
    ///     assert_eq!(interner.len(), 2);
    ///     assert_eq!(*c, &A(43));
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// Undefined behavior if incorrect type `K` is given.
    pub unsafe fn intern_with<K, Q, F>(&self, key: &Q, make_value: F) -> Interned<'_, K>
    where
        K: borrow::Borrow<Q> + 'static,
        Q: Hash + Eq + ?Sized,
        F: FnOnce() -> K,
    {
        self.with_inner(|set| unsafe { set.intern_with(key, make_value) })
    }

    /// Retrieves a reference to a value in the interner based on the provided key.
    ///
    /// This method checks if a value corresponding to the given key exists in the interner. If it
    /// exists, a reference to the interned value is returned. Otherwise, `None` is returned.
    ///
    /// # Eaxmples
    ///
    /// ```
    /// use any_intern::AnyInterner;
    ///
    /// let interner = AnyInterner::of::<i32>();
    /// unsafe {
    ///     interner.intern(42);
    ///     assert_eq!(*interner.get::<i32, _>(&42).unwrap(), &42);
    ///     assert!(interner.get::<i32, _>(&99).is_none());
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// Undefined behavior if incorrect type `K` is given.
    pub unsafe fn get<K, Q>(&self, key: &Q) -> Option<Interned<'_, K>>
    where
        K: borrow::Borrow<Q> + 'static,
        Q: Hash + Eq + ?Sized,
    {
        self.with_inner(|set| unsafe { set.get(key) })
    }

    /// Returns true if the interner contains values of the given type.
    pub fn is_type_of<K: 'static>(&self) -> bool {
        self.with_inner(|set| set.is_type_of::<K>())
    }

    /// Removes all items in the interner.
    ///
    /// Although the interner support interior mutability, clear method requires mutable access
    /// to the interner to invalidate all [`Interned`]s referencing the interner.
    pub fn clear(&mut self) {
        self.with_inner(|set| set.clear())
    }

    fn with_inner<'this, F, R>(&'this self, f: F) -> R
    where
        F: FnOnce(&'this mut AnyInternSet<S>) -> R,
        R: 'this,
    {
        // Safety: Mutex unlocking is paired with the locking.
        unsafe {
            let set = self.inner.lock().as_mut();
            let ret = f(set);
            self.inner.unlock();
            ret
        }
    }
}

/// A type-erased interning set for storing and deduplicating values of a single type without
/// interior mutability.
///
/// # Examples
///
/// ```
/// use any_intern::AnyInternSet;
///
/// #[derive(PartialEq, Eq, Hash, Debug)]
/// struct A(u32);
///
/// let mut set = AnyInternSet::of::<A>();
///
/// unsafe {
///     let a1 = set.intern(A(42)).raw();
///     let a2 = set.intern(A(42)).raw();
///     assert_eq!(a1, a2); // Same value, same reference
///
///     let a3 = set.intern(A(99)).raw();
///     assert_ne!(a1, a3); // Different values, different references
/// }
/// ```
///
/// # Safety
///
/// Many methods in `AnyInternSet` are marked as `unsafe` because they rely on the caller to ensure
/// that the correct type is used when interacting with the interner. Using an incorrect type can
/// lead to undefined behavior.
pub struct AnyInternSet<S = fxhash::FxBuildHasher> {
    arena: AnyArena,
    set: HashTable<RawInterned>,
    hash_builder: S,
}

impl AnyInternSet {
    pub fn of<K: 'static>() -> Self {
        Self {
            arena: AnyArena::of::<K>(),
            set: HashTable::new(),
            hash_builder: Default::default(),
        }
    }
}

impl<S: Default> AnyInternSet<S> {
    pub fn default_of<K: 'static>() -> Self {
        Self {
            arena: AnyArena::of::<K>(),
            set: HashTable::new(),
            hash_builder: Default::default(),
        }
    }
}

impl<S: BuildHasher> AnyInternSet<S> {
    pub fn with_hasher<K: 'static>(hash_builder: S) -> Self {
        Self {
            arena: AnyArena::of::<K>(),
            set: HashTable::new(),
            hash_builder,
        }
    }

    /// Returns number of values in the set.
    pub const fn len(&self) -> usize {
        self.arena.len()
    }

    /// Returns true if the set is empty.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Stores a value in the set, returning a reference to the value.
    ///
    /// This method inserts the given value into the set if it does not already exist. If the value
    /// already exists, a reference to the existing value is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use any_intern::AnyInternSet;
    ///
    /// #[derive(PartialEq, Eq, Hash, Debug)]
    /// struct A(u32);
    ///
    /// let mut set = AnyInternSet::of::<A>();
    ///
    /// unsafe {
    ///     let a1 = set.intern(A(42)).raw();
    ///     let a2 = set.intern(A(42)).raw();
    ///     assert_eq!(a1, a2); // Same value, same reference
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// Undefined behavior if incorrect type `K` is given.
    pub unsafe fn intern<K>(&mut self, value: K) -> Interned<'_, K>
    where
        K: Hash + Eq + 'static,
    {
        debug_assert!(self.is_type_of::<K>());

        unsafe {
            let hash = Self::hash(&self.hash_builder, &value);
            let eq = Self::table_eq::<K, K>(&value);
            let hasher = Self::table_hasher::<K, K>(&self.hash_builder);
            match self.set.entry(hash, eq, hasher) {
                Entry::Occupied(entry) => Interned::from_raw(*entry.get()),
                Entry::Vacant(entry) => {
                    let ref_ = self.arena.alloc(value);
                    let interned = Interned::unique(ref_);
                    let raw = interned.raw();
                    entry.insert(raw);
                    interned
                }
            }
        }
    }

    /// Stores a value in the set, creating it only if it does not already exist.
    ///
    /// This method allows you to provide a key and a closure to generate the value. If the key
    /// already exists in the set, the closure is not called, and a reference to the existing value
    /// is returned. If the key does not exist, the closure is called to create the value, which is
    /// then stored in the set.
    ///
    /// This method is useful when the value is expensive to compute, as it avoids unnecessary
    /// computation if the value already exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use any_intern::AnyInternSet;
    ///
    ///
    /// #[derive(PartialEq, Eq, Hash, Debug)]
    /// struct A(i32);
    ///
    /// impl std::borrow::Borrow<i32> for A {
    ///     fn borrow(&self) -> &i32 {
    ///         &self.0
    ///     }
    /// }
    ///
    /// let mut set = AnyInternSet::of::<A>();
    ///
    /// unsafe {
    ///     let a = set.intern_with(&42, || A(42));
    ///     assert_eq!(*a, &A(42));
    ///     assert_eq!(set.len(), 1);
    ///
    ///     let b = set.intern_with(&42, || A(99)); // Closure is not called
    ///     assert_eq!(*b, &A(42));
    ///     assert_eq!(set.len(), 1);
    ///
    ///     let c = set.intern_with(&43, || A(43));
    ///     assert_eq!(*c, &A(43));
    ///     assert_eq!(set.len(), 2);
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// Undefined behavior if incorrect type `K` is given.
    pub unsafe fn intern_with<K, Q, F>(&mut self, key: &Q, make_value: F) -> Interned<'_, K>
    where
        K: borrow::Borrow<Q> + 'static,
        Q: Hash + Eq + ?Sized,
        F: FnOnce() -> K,
    {
        debug_assert!(self.is_type_of::<K>());

        unsafe {
            let hash = Self::hash(&self.hash_builder, key);
            let eq = Self::table_eq::<K, Q>(key);
            let hasher = Self::table_hasher::<K, Q>(&self.hash_builder);
            match self.set.entry(hash, eq, hasher) {
                Entry::Occupied(entry) => Interned::from_raw(*entry.get()),
                Entry::Vacant(entry) => {
                    let value = make_value();
                    let ref_ = self.arena.alloc(value);
                    let interned = Interned::unique(ref_);
                    let raw = interned.raw();
                    entry.insert(raw);
                    interned
                }
            }
        }
    }

    /// Retrieves a reference to a value in the set based on the provided key.
    ///
    /// This method checks if a value corresponding to the given key exists in the set. If it
    /// exists, a reference to the value is returned. Otherwise, `None` is returned.
    ///
    /// # Eaxmples
    ///
    /// ```
    /// use any_intern::AnyInternSet;
    ///
    /// let mut set = AnyInternSet::of::<i32>();
    /// unsafe {
    ///     set.intern(42);
    ///     assert_eq!(*set.get::<i32, _>(&42).unwrap(), &42);
    ///     assert!(set.get::<i32, _>(&99).is_none());
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// Undefined behavior if incorrect type `K` is given.
    pub unsafe fn get<K, Q>(&self, key: &Q) -> Option<Interned<'_, K>>
    where
        K: borrow::Borrow<Q> + 'static,
        Q: Hash + Eq + ?Sized,
    {
        debug_assert!(self.is_type_of::<K>());

        unsafe {
            let hash = Self::hash(&self.hash_builder, key);
            let eq = Self::table_eq::<K, Q>(key);
            self.set.find(hash, eq).map(|raw| Interned::from_raw(*raw))
        }
    }

    /// Returns true if the set contains values of the given type.
    pub fn is_type_of<K: 'static>(&self) -> bool {
        self.arena.is_type_of::<K>()
    }

    /// Removes all items in the set.
    pub fn clear(&mut self) {
        self.arena.clear();
        self.set.clear();
    }

    /// Returns `eq` closure that is used for some methods on the [`HashTable`].
    ///
    /// # Safety
    ///
    /// Undefined behavior if incorrect type `K` is given.
    unsafe fn table_eq<K, Q>(key: &Q) -> impl FnMut(&RawInterned) -> bool
    where
        K: borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        move |entry: &RawInterned| unsafe {
            let value = entry.cast::<K>().as_ref();
            value.borrow() == key
        }
    }

    /// Returns `hasher` closure that is used for some methods on the [`HashTable`].
    ///
    /// # Safety
    ///
    /// Undefined behavior if incorrect type `K` is given.
    unsafe fn table_hasher<K, Q>(hash_builder: &S) -> impl Fn(&RawInterned) -> u64
    where
        K: borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        |entry: &RawInterned| unsafe {
            let value = entry.cast::<K>().as_ref();
            Self::hash(hash_builder, value.borrow())
        }
    }

    fn hash<K: Hash + ?Sized>(hash_builder: &S, value: &K) -> u64 {
        hash_builder.hash_one(value)
    }
}

pub struct AnyArena {
    bump: Bump,
    ty: TypeId,
    stride: usize,
    raw_drop_slice: Option<unsafe fn(*mut u8, usize)>,
    len: Cell<usize>,
}

impl AnyArena {
    pub fn of<T: 'static>() -> Self {
        Self {
            bump: Bump::new(),
            ty: TypeId::of::<T>(),
            stride: Layout::new::<T>().pad_to_align().size(),
            raw_drop_slice: if mem::needs_drop::<T>() {
                Some(common::cast_then_drop_slice::<T>)
            } else {
                None
            },
            len: Cell::new(0),
        }
    }

    pub fn is_type_of<T: 'static>(&self) -> bool {
        TypeId::of::<T>() == self.ty
    }

    /// Returns number of elements in this arena.
    pub const fn len(&self) -> usize {
        self.len.get()
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn alloc<T: 'static>(&self, value: T) -> &mut T {
        debug_assert!(self.is_type_of::<T>());

        self.len.set(self.len() + 1);
        self.bump.alloc(value)
    }

    pub fn clear(&mut self) {
        self.drop_all();
        self.bump.reset();
        self.len.set(0);
    }

    fn drop_all(&mut self) {
        if let Some(raw_drop_slice) = self.raw_drop_slice {
            if self.stride > 0 {
                for chunk in self.bump.iter_allocated_chunks() {
                    // Chunk would not be divisible by the `stride` especially when the stride is
                    // greater than 16. In that case, we should ignore the remainder.
                    let num_elems = chunk.len() / self.stride;
                    let ptr = chunk.as_ptr().cast::<u8>().cast_mut();
                    unsafe {
                        raw_drop_slice(ptr, num_elems);
                    }
                }
            } else {
                let ptr = ptr::dangling_mut::<u8>();
                unsafe {
                    raw_drop_slice(ptr, self.len());
                }
            }
        }
    }
}

impl Drop for AnyArena {
    fn drop(&mut self) {
        self.drop_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_any_interner() {
        #[derive(PartialEq, Eq, Hash, Debug)]
        struct A(i32);

        let interner = AnyInterner::of::<A>();

        unsafe {
            let a = interner.intern(A(0));
            let b = interner.intern(A(0));
            let c = interner.intern(A(1));
            assert_eq!(a, b);
            assert_ne!(a, c);
        }
    }

    #[test]
    fn test_arena() {
        test_arena_alloc();
        test_arena_drop();
    }

    fn test_arena_alloc() {
        const START: u32 = 0;
        const END: u32 = 100;
        const EXPECTED: u32 = (END + START) * (END - START + 1) / 2;

        let arena = AnyArena::of::<u32>();
        let mut refs = Vec::new();
        for i in START..=END {
            let ref_ = arena.alloc(i);
            refs.push(ref_);
        }
        let acc = refs.into_iter().map(|ref_| *ref_).sum::<u32>();
        assert_eq!(acc, EXPECTED);
    }

    fn test_arena_drop() {
        macro_rules! test {
            ($arr_len:literal, $align:literal) => {{
                thread_local! {
                    static SUM: Cell<u32> = Cell::new(0);
                    static CNT: Cell<u32> = Cell::new(0);
                }

                #[repr(align($align))]
                struct A([u8; $arr_len]);

                // Restricted by `u8` and `A::new()`.
                const _: () = const { assert!($arr_len < 256) };

                impl A {
                    fn new() -> Self {
                        Self(std::array::from_fn(|i| i as u8))
                    }

                    fn sum() -> u32 {
                        ($arr_len - 1) * $arr_len / 2
                    }
                }

                impl Drop for A {
                    fn drop(&mut self) {
                        let sum = self.0.iter().map(|n| *n as u32).sum::<u32>();
                        SUM.set(SUM.get() + sum);
                        CNT.set(CNT.get() + 1);
                    }
                }

                struct Zst;

                impl Drop for Zst {
                    fn drop(&mut self) {
                        CNT.set(CNT.get() + 1);
                    }
                }

                const REPEAT: u32 = 10;

                // === Non-ZST type ===

                let arena = AnyArena::of::<A>();
                for _ in 0..REPEAT {
                    arena.alloc(A::new());
                }
                drop(arena);

                assert_eq!(SUM.get(), A::sum() * REPEAT);
                assert_eq!(CNT.get(), REPEAT);
                SUM.set(0);
                CNT.set(0);

                // === ZST type ===

                let arena = AnyArena::of::<Zst>();
                for _ in 0..REPEAT {
                    arena.alloc(Zst);
                }
                drop(arena);

                assert_eq!(CNT.get(), REPEAT);
                CNT.set(0);
            }};
        }

        // Array len, align
        test!(1, 1);
        test!(1, 2);
        test!(1, 4);
        test!(1, 8);
        test!(1, 16);
        test!(1, 32);
        test!(1, 64);
        test!(1, 128);
        test!(1, 256);

        test!(100, 1);
        test!(100, 2);
        test!(100, 4);
        test!(100, 8);
        test!(100, 16);
        test!(100, 32);
        test!(100, 64);
        test!(100, 128);
        test!(100, 256);
    }
}
