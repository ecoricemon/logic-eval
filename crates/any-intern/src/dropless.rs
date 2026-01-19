use super::common::{Interned, RawInterned, UnsafeLock};
use bumpalo::Bump;
use hashbrown::{HashTable, hash_table::Entry};
use std::{
    alloc::Layout,
    hash::{BuildHasher, Hash},
    ptr::{self, NonNull},
    slice,
};

/// A trait for types that can be stored in a dropless interner.
///
/// The `Dropless` trait provides methods for working with types that can be stored in a
/// [`DroplessInterner`] or [`DroplessInternSet`]. It defines how instances of a type are converted
/// to and from raw byte representations, hashed, and compared for equality. This is useful for
/// interning values without requiring ownership.
///
/// # Safety
///
/// Implementing the `Dropless` trait requires careful attention to alignment and memory safety. The
/// methods in this trait are marked as `unsafe` because they rely on the caller to ensure that the
/// provided byte slices are properly aligned and valid for the type. Misuse of these methods can
/// lead to undefined behavior.
///
/// # Examples
///
/// ```
/// use any_intern::Dropless;
/// use std::alloc::Layout;
/// use std::ptr::NonNull;
///
/// #[derive(PartialEq, Eq, Hash, Debug)]
/// struct MyType(u32);
///
/// impl Dropless for MyType {
///     fn as_byte_ptr(&self) -> NonNull<u8> {
///         NonNull::from_ref(self).cast::<u8>()
///     }
///
///     fn layout(&self) -> Layout {
///         Layout::for_value(self)
///     }
///
///     unsafe fn from_bytes(bytes: &[u8]) -> &Self {
///         let ptr = bytes.as_ptr().cast::<Self>();
///         unsafe { ptr.as_ref().unwrap_unchecked() }
///     }
///
///     unsafe fn hash<S: std::hash::BuildHasher>(build_hasher: &S, bytes: &[u8]) -> u64 {
///         let this = unsafe { Self::from_bytes(bytes) };
///         build_hasher.hash_one(this)
///     }
///
///     unsafe fn eq(a: &[u8], b: &[u8]) -> bool {
///         unsafe { Self::from_bytes(a) == Self::from_bytes(b) }
///     }
/// }
/// ```
pub trait Dropless {
    /// Returns pointer to the instance.
    fn as_byte_ptr(&self) -> NonNull<u8> {
        NonNull::from_ref(self).cast::<u8>()
    }

    /// Returns layout of the type.
    fn layout(&self) -> Layout {
        Layout::for_value(self)
    }

    /// Converts a byte slice into a reference to the type.
    ///
    /// The byte slice is guaranteed to be well aligned and correct data for the type.
    ///
    /// # Safety
    ///
    /// Undefined behavior if any conditions below are not met.
    /// * Implementation should interpret the byte slice into the type correctly.
    /// * Caller should give well aligned data for the type.
    unsafe fn from_bytes(bytes: &[u8]) -> &Self;

    /// Computes a hash value for the type using the provided byte slice.
    ///
    /// The byte slice is guaranteed to be well aligned and correct data for the type.
    ///
    /// # Safety
    ///
    /// Undefined behavior if any conditions below are not met.
    /// * Implementation should interpret the byte slice into the type correctly.
    /// * Caller should give well aligned data for the type.
    unsafe fn hash<S: BuildHasher>(build_hasher: &S, bytes: &[u8]) -> u64;

    /// Compares two byte slices for equality as instances of the type.
    ///
    /// The byte slices are guaranteed to be well aligned and correct data for the type.
    ///
    /// # Safety
    ///
    /// Undefined behavior if any conditions below are not met.
    /// * Implementation should interpret the byte slice into the type correctly.
    /// * Caller should give well aligned data for the type.
    unsafe fn eq(a: &[u8], b: &[u8]) -> bool;
}

macro_rules! simple_impl_dropless_fn {
    (from_bytes) => {
        unsafe fn from_bytes(bytes: &[u8]) -> &Self {
            let ptr = bytes.as_ptr().cast::<Self>();
            unsafe { ptr.as_ref().unwrap_unchecked() }
        }
    };
    (hash) => {
        unsafe fn hash<S: BuildHasher>(build_hasher: &S, bytes: &[u8]) -> u64 {
            let this = unsafe { Self::from_bytes(bytes) };
            build_hasher.hash_one(this)
        }
    };
    (eq) => {
        unsafe fn eq(a: &[u8], b: &[u8]) -> bool {
            unsafe { Self::from_bytes(a) == Self::from_bytes(b) }
        }
    };
}

macro_rules! impl_dropless {
    ($($ty:ty)*) => {
        $(
            impl Dropless for $ty {
                simple_impl_dropless_fn!(from_bytes);
                simple_impl_dropless_fn!(hash);
                simple_impl_dropless_fn!(eq);
            }
        )*
    };
}

impl_dropless!(i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize bool char);

impl<T: Dropless + Hash + Eq> Dropless for [T] {
    unsafe fn from_bytes(bytes: &[u8]) -> &Self {
        let ptr = bytes.as_ptr().cast::<T>();
        let stride = Layout::new::<T>().pad_to_align().size();
        let num_elems = bytes.len() / stride;
        unsafe { slice::from_raw_parts(ptr, num_elems) }
    }
    simple_impl_dropless_fn!(hash);
    simple_impl_dropless_fn!(eq);
}

impl<T: Dropless + Hash + Eq, const N: usize> Dropless for [T; N] {
    simple_impl_dropless_fn!(from_bytes);
    simple_impl_dropless_fn!(hash);
    simple_impl_dropless_fn!(eq);
}

impl Dropless for str {
    unsafe fn from_bytes(bytes: &[u8]) -> &Self {
        unsafe { str::from_utf8_unchecked(bytes) }
    }
    simple_impl_dropless_fn!(hash);
    simple_impl_dropless_fn!(eq);
}

impl<T: Dropless + Hash + Eq> Dropless for Option<T> {
    simple_impl_dropless_fn!(from_bytes);
    simple_impl_dropless_fn!(hash);
    simple_impl_dropless_fn!(eq);
}

impl<T: Dropless + Hash + Eq, E: Dropless + Hash + Eq> Dropless for Result<T, E> {
    simple_impl_dropless_fn!(from_bytes);
    simple_impl_dropless_fn!(hash);
    simple_impl_dropless_fn!(eq);
}

/// An interner for storing and deduplicating values without requiring ownership.
///
/// The `DroplessInterner` is designed for interning values that implement the [`Dropless`] trait.
/// It allows efficient storage and retrieval of values by ensuring that each unique value is stored
/// only once. Interning with this type always copies the given value into an internal buffer,
/// making it suitable for use cases where ownership is not required.
///
/// # Examples
///
/// ```
/// use any_intern::DroplessInterner;
///
/// let mut interner = DroplessInterner::new();
///
/// // Interning strings
/// let hello = interner.intern("hello");
/// let world = interner.intern("world");
/// let another_hello = interner.intern("hello");
///
/// assert_eq!(hello, another_hello); // Same value, same reference
/// assert_ne!(hello, world); // Different values, different references
///
/// // Checking if a value exists
/// assert!(interner.get("hello").is_some());
/// assert!(interner.get("unknown").is_none());
///
/// // Clearing the interner
/// interner.clear();
/// assert!(interner.is_empty());
/// ```
///
/// # Safety
///
/// The `DroplessInterner` relies on the `Dropless` trait for converting values to and from raw
/// byte representations. It is the responsibility of the `Dropless` implementation to ensure
/// memory safety and alignment when interacting with the interner.
#[derive(Debug)]
pub struct DroplessInterner<S = fxhash::FxBuildHasher> {
    inner: UnsafeLock<DroplessInternSet<S>>,
}

impl DroplessInterner {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<S: BuildHasher> DroplessInterner<S> {
    pub fn with_hasher(hash_builder: S) -> Self {
        // Safety: Only one instance
        let inner = unsafe { UnsafeLock::new(DroplessInternSet::with_hasher(hash_builder)) };
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
    /// This method inserts the given value into the interner if it does not already exist.  If the
    /// value already exists, a reference to the existing value is returned. The value is copied
    /// into an internal buffer, making it suitable for use cases where ownership is not required.
    ///
    /// # Examples
    ///
    /// ```
    /// use any_intern::DroplessInterner;
    ///
    /// let interner = DroplessInterner::new();
    ///
    /// // Interning strings
    /// let hello = interner.intern("hello");
    /// let world = interner.intern("world");
    /// let another_hello = interner.intern("hello");
    ///
    /// assert_eq!(hello, another_hello); // Same value, same reference
    /// assert_ne!(hello, world); // Different values, different references
    ///
    /// // Interning arrays
    /// let array1 = interner.intern(&[1, 2, 3]);
    /// let array2 = interner.intern(&[1, 2, 3]);
    /// let array3 = interner.intern(&[4, 5, 6]);
    ///
    /// assert_eq!(array1, array2); // Same value, same reference
    /// assert_ne!(array1, array3); // Different values, different references
    /// ```
    pub fn intern<K: Dropless + ?Sized>(&self, value: &K) -> Interned<'_, K> {
        self.with_inner(|set| set.intern(value))
    }

    /// Retrieves a reference to a value in the interner based on the provided key.
    ///
    /// This method checks if a value corresponding to the given key exists in the interner. If it
    /// exists, a reference to the interned value is returned. Otherwise, `None` is returned.
    ///
    /// # Eaxmples
    ///
    /// ```
    /// use any_intern::DroplessInterner;
    ///
    /// let interner = DroplessInterner::new();
    ///
    /// // Interning strings
    /// let hello = interner.intern("hello");
    ///
    /// assert_eq!(*interner.get("hello").unwrap(), "hello");
    /// assert!(interner.get("world").is_none());
    /// ```
    pub fn get<K: Dropless + ?Sized>(&self, value: &K) -> Option<Interned<'_, K>> {
        self.with_inner(|set| set.get(value))
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
        F: FnOnce(&'this mut DroplessInternSet<S>) -> R,
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

impl<S: Default> Default for DroplessInterner<S> {
    fn default() -> Self {
        // Safety: Only one instance
        let inner = unsafe { UnsafeLock::new(DroplessInternSet::default()) };
        Self { inner }
    }
}

/// A dropless interner.
///
/// Interning on this type always copies the given value into internal buffer.
#[derive(Debug)]
pub struct DroplessInternSet<S = fxhash::FxBuildHasher> {
    bump: Bump,
    set: HashTable<DynInternEntry<S>>,
    hash_builder: S,
}

impl DroplessInternSet {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<S: BuildHasher> DroplessInternSet<S> {
    pub fn with_hasher(hash_builder: S) -> Self {
        Self {
            bump: Bump::new(),
            set: HashTable::new(),
            hash_builder,
        }
    }

    pub fn intern<K: Dropless + ?Sized>(&mut self, value: &K) -> Interned<'_, K> {
        let src = value.as_byte_ptr();
        let layout = value.layout();
        unsafe {
            let bytes = slice::from_raw_parts(src.as_ptr(), layout.size());
            let hash = <K as Dropless>::hash(&self.hash_builder, bytes);
            let eq = Self::table_eq::<K>(bytes, layout.align());
            let hasher = Self::table_hasher(&self.hash_builder);
            let ref_ = match self.set.entry(hash, eq, hasher) {
                Entry::Occupied(entry) => Self::ref_from_entry(entry.get()),
                Entry::Vacant(entry) => {
                    // New allocation & copy
                    let dst = self.bump.alloc_layout(layout);
                    ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr(), layout.size());

                    let occupied = entry.insert(DynInternEntry {
                        data: RawInterned(dst),
                        layout,
                        hash: <K as Dropless>::hash,
                    });
                    Self::ref_from_entry(occupied.get())
                }
            };
            Interned::unique(ref_)
        }
    }

    /// # Eaxmples
    ///
    /// ```
    /// use any_intern::DroplessInternSet;
    /// use std::hash::RandomState;
    ///
    /// let mut set = DroplessInternSet::with_hasher(RandomState::new());
    /// set.intern("hello");
    ///
    /// assert_eq!(set.get("hello").as_deref(), Some(&"hello"));
    /// assert!(set.get("hi").is_none());
    /// ```
    pub fn get<K: Dropless + ?Sized>(&self, value: &K) -> Option<Interned<'_, K>> {
        let ptr = value.as_byte_ptr();
        let layout = value.layout();
        unsafe {
            let bytes = slice::from_raw_parts(ptr.as_ptr(), layout.size());
            let hash = <K as Dropless>::hash(&self.hash_builder, bytes);
            let eq = Self::table_eq::<K>(bytes, layout.align());
            let entry = self.set.find(hash, eq)?;
            let ref_ = Self::ref_from_entry(entry);
            Some(Interned::unique(ref_))
        }
    }

    pub fn len(&self) -> usize {
        self.set.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.bump.reset();
        self.set.clear();
    }

    /// Returns `eq` closure that is used for some methods on the [`HashTable`].
    fn table_eq<K: Dropless + ?Sized>(
        key: &[u8],
        align: usize,
    ) -> impl FnMut(&DynInternEntry<S>) -> bool {
        move |entry: &DynInternEntry<S>| unsafe {
            if align == entry.layout.align() {
                let entry_bytes = slice::from_raw_parts(entry.data.as_ptr(), entry.layout.size());
                <K as Dropless>::eq(key, entry_bytes)
            } else {
                false
            }
        }
    }

    /// Returns `hasher` closure that is used for some methods on the [`HashTable`].
    fn table_hasher(hash_builder: &S) -> impl Fn(&DynInternEntry<S>) -> u64 {
        |entry: &DynInternEntry<S>| unsafe {
            let entry_bytes = slice::from_raw_parts(entry.data.as_ptr(), entry.layout.size());
            // Safety: `entry_bytes` is aligned for the entry hash function.
            (entry.hash)(hash_builder, entry_bytes)
        }
    }

    unsafe fn ref_from_entry<'a, K: Dropless + ?Sized>(entry: &DynInternEntry<S>) -> &'a K {
        unsafe {
            let bytes = slice::from_raw_parts(entry.data.as_ptr(), entry.layout.size());
            K::from_bytes(bytes)
        }
    }
}

impl<S: Default> Default for DroplessInternSet<S> {
    fn default() -> Self {
        Self {
            bump: Bump::new(),
            set: HashTable::new(),
            hash_builder: Default::default(),
        }
    }
}

#[derive(Debug)]
struct DynInternEntry<S> {
    data: RawInterned,
    layout: Layout,
    /// Input bytes must be well aligned.
    hash: unsafe fn(&S, &[u8]) -> u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common;

    #[test]
    fn test_dropless_interner() {
        test_dropless_interner_int();
        test_dropless_interner_str();
        test_dropless_interner_bytes();
        test_dropless_interner_mixed();
        test_dropless_interner_many();
        test_dropless_interner_alignment_handling();
    }

    fn test_dropless_interner_int() {
        let interner = DroplessInterner::new();

        let a = interner.intern(&0_u32).raw();
        let b = interner.intern(&0_u32).raw();
        let c = interner.intern(&1_u32).raw();
        // d has the same bytes and layout as c's, so memory will be shared.
        let d = interner.intern(&1_i32).raw();

        let groups: [&[RawInterned]; _] = [&[a, b], &[c, d]];
        common::assert_group_addr_eq(&groups);
    }

    fn test_dropless_interner_str() {
        let interner = DroplessInterner::new();

        let a = interner.intern("apple").raw();
        let b = interner.intern(*Box::new("apple")).raw();
        let c = interner.intern(&*String::from("apple")).raw();
        let d = interner.intern("banana").raw();

        let groups: [&[RawInterned]; _] = [&[a, b, c], &[d]];
        common::assert_group_addr_eq(&groups);
    }

    fn test_dropless_interner_bytes() {
        let interner = DroplessInterner::new();

        let a = interner.intern(&[0, 1]).raw();
        let boxed: Box<[i32]> = Box::new([0, 1]);
        let b = interner.intern(&*boxed).raw();
        let c = interner.intern(&*vec![0, 1]).raw();
        let d = interner.intern(&[2, 3]).raw();

        let groups: [&[RawInterned]; _] = [&[a, b, c], &[d]];
        common::assert_group_addr_eq(&groups);
    }

    fn test_dropless_interner_mixed() {
        let interner = DroplessInterner::new();

        interner.intern(&1_u32);
        interner.intern(&[2, 3]);
        interner.intern("apple");
        assert_eq!(interner.get("apple").as_deref(), Some(&"apple"));
        assert!(interner.get("banana").is_none());
        assert_eq!(interner.get(&1_u32).as_deref(), Some(&&1_u32));
        assert!(interner.get(&2_u32).is_none());
        assert_eq!(interner.get(&[2, 3]).as_deref(), Some(&&[2, 3]));
        assert!(interner.get(&[2]).is_none());
    }

    fn test_dropless_interner_many() {
        let interner = DroplessInterner::new();

        let mut interned_int = Vec::new();
        let mut interned_str = Vec::new();
        let mut interned_bytes = Vec::new();

        const N: usize = 1000;

        let strs = (0..N).map(|i| (i * 10_000).to_string()).collect::<Vec<_>>();

        // Interns lots of items.
        for i in 0..N {
            let int = &(i as u32); // 4 bytes
            let str_ = &*strs[i]; // greater than 4 btyes
            let bytes = &[i as u16]; // 2 bytes
            interned_int.push(interner.intern(int).raw());
            interned_str.push(interner.intern(str_).raw());
            interned_bytes.push(interner.intern(bytes).raw());
        }

        // Verifies every pointer is unique.
        interned_int.sort_unstable();
        interned_int.dedup();
        interned_str.sort_unstable();
        interned_str.dedup();
        interned_bytes.sort_unstable();
        interned_bytes.dedup();
        assert_eq!(interned_int.len(), N);
        assert_eq!(interned_str.len(), N);
        assert_eq!(interned_bytes.len(), N);
        let whole = interned_int
            .iter()
            .chain(&interned_str)
            .chain(&interned_bytes)
            .cloned()
            .collect::<fxhash::FxHashSet<_>>();
        assert_eq!(whole.len(), N * 3);

        // Verifies `get` method for every item.
        for i in 0..N {
            let int = &(i as u32); // 4 bytes
            let str_ = &*strs[i]; // greater than 4 btyes
            let bytes = &[i as u16]; // 2 bytes
            assert_eq!(*interner.get(int).unwrap(), int);
            assert_eq!(*interner.get(str_).unwrap(), str_);
            assert_eq!(*interner.get(bytes).unwrap(), bytes);
        }
    }

    #[rustfmt::skip]
    fn test_dropless_interner_alignment_handling() {
        #[derive(PartialEq, Eq, Hash, Debug)] #[repr(C, align(4))]  struct T4  (u16, [u8; 2]);
        #[derive(PartialEq, Eq, Hash, Debug)] #[repr(C, align(8))]  struct T8  (u16, [u8; 6]);
        #[derive(PartialEq, Eq, Hash, Debug)] #[repr(C, align(16))] struct T16 (u16, [u8; 14]);

        macro_rules! impl_for_first_2bytes {
            () => {
                unsafe fn hash<S: BuildHasher>(hash_builder: &S, bytes: &[u8]) -> u64 {
                    hash_builder.hash_one(&bytes[0..2])
                }
                unsafe fn eq(a: &[u8], b: &[u8]) -> bool {
                    a[0..2] == b[0..2]
                }
            };
        }

        impl Dropless for T4 { simple_impl_dropless_fn!(from_bytes); impl_for_first_2bytes!(); }
        impl Dropless for T8 { simple_impl_dropless_fn!(from_bytes); impl_for_first_2bytes!(); }
        impl Dropless for T16 { simple_impl_dropless_fn!(from_bytes); impl_for_first_2bytes!(); }

        let interner = DroplessInterner::new();

        let mut interned_4 = Vec::new();
        let mut interned_8 = Vec::new();
        let mut interned_16 = Vec::new();

        const N: usize = 1000;

        // Items being interned have the same first 2 bytes, so they will look like the smae from
        // perspective of interner. But, they must be distinguished from each other due to their
        // different layouts. Following bytes after the 2 bytes will be used to detect data
        // validity.
        for i in 0..N {
            let t4 = T4(i as u16, [4; _]);
            let t8 = T8(i as u16, [8; _]);
            let t16 = T16(i as u16, [16; _]);
            interned_4.push(interner.intern(&t4).raw());
            interned_8.push(interner.intern(&t8).raw());
            interned_16.push(interner.intern(&t16).raw());
        }

        for i in 0..N {
            let t4 = T4(i as u16, [4; _]);
            let t8 = T8(i as u16, [8; _]);
            let t16 = T16(i as u16, [16; _]);
            unsafe {
                assert_eq!(*<Interned<'_, T4>>::from_raw(interned_4[i]), &t4);
                assert_eq!(*<Interned<'_, T8>>::from_raw(interned_8[i]), &t8);
                assert_eq!(*<Interned<'_, T16>>::from_raw(interned_16[i]), &t16);
            }
        }
    }
}
