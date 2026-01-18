use parking_lot::{RawMutex, lock_api::RawMutex as _};
use std::{
    cell::UnsafeCell,
    fmt,
    hash::{Hash, Hasher},
    ops,
    ptr::{self, NonNull},
    slice,
};

/// Due to [`Private`], clients must create this type through [`Interned::unique`], but still
/// allowed to use pattern match.
pub struct Interned<'a, T: ?Sized>(pub &'a T, Private);

impl<'a, T: ?Sized> Interned<'a, T> {
    /// Caller should guarantee that the value is unique in an interner.
    pub fn unique(value: &'a T) -> Self {
        Self(value, Private)
    }

    pub fn raw(&self) -> RawInterned {
        let ptr = NonNull::from_ref(self.0).cast::<u8>();
        RawInterned(ptr)
    }
}

impl<'a, T> Interned<'a, T> {
    pub(crate) unsafe fn from_raw(raw: RawInterned) -> Self {
        let ref_ = unsafe { raw.0.cast::<T>().as_ref() };
        Self(ref_, Private)
    }
}

impl<T: ?Sized> PartialEq for Interned<'_, T> {
    /// Compares data addresses only, which is sufficient for interned values.
    fn eq(&self, other: &Self) -> bool {
        ptr::addr_eq(self.0, other.0)
    }
}

impl<T: ?Sized> Eq for Interned<'_, T> {}

impl<T: Hash + ?Sized> Hash for Interned<'_, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<T: ?Sized> Clone for Interned<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for Interned<'_, T> {}

impl<'a, T: ?Sized> ops::Deref for Interned<'a, T> {
    type Target = &'a T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: fmt::Debug + ?Sized> fmt::Debug for Interned<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<T: fmt::Display + ?Sized> fmt::Display for Interned<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

// This is exposed as public, but clients cannot make this type directly.
#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct RawInterned(pub(crate) NonNull<u8>);

impl ops::Deref for RawInterned {
    type Target = NonNull<u8>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Debug for RawInterned {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy)]
pub struct Private;

/// Clients must unlock this mutex manually after successful locking.
pub struct ManualMutex<T> {
    mutex: RawMutex,
    data: UnsafeCell<T>,
}

impl<T> ManualMutex<T> {
    pub(crate) const fn new(value: T) -> Self {
        Self {
            mutex: RawMutex::INIT,
            data: UnsafeCell::new(value),
        }
    }

    /// Caller should call to [`unlock`](Self::unlock).
    pub fn lock(&self) -> *mut T {
        self.mutex.lock();
        self.data.get()
    }

    /// # Safety
    ///
    /// Must be paired with successful locking.
    pub unsafe fn unlock(&self) {
        unsafe { self.mutex.unlock() };
    }
}

impl<T: fmt::Debug> fmt::Debug for ManualMutex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Safety: Lock & unlock are paired with each other.
        unsafe {
            let t = self.lock().as_ref().unwrap_unchecked();
            let ret = fmt::Debug::fmt(t, f);
            self.unlock();
            ret
        }
    }
}

pub(crate) unsafe fn cast_then_drop_slice<T>(ptr: *mut u8, num_elems: usize) {
    unsafe {
        let slice = slice::from_raw_parts_mut(ptr.cast::<T>(), num_elems);
        ptr::drop_in_place(slice);
    }
}

#[cfg(test)]
pub(crate) fn assert_group_addr_eq(groups: &[&[RawInterned]]) {
    for i in 0..groups.len() {
        // Inside of a group shares the same address.
        for w in groups[i].windows(2) {
            assert_eq!(w[0], w[1]);
        }

        // Groups have different addresses.
        let a = groups[i][0];
        for j in i + 1..groups.len() {
            let b = groups[j][0];
            assert_ne!(a, b);
        }
    }
}
