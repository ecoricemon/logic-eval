use parking_lot::{RawMutex, lock_api::RawMutex as _};
use std::{
    cell::UnsafeCell,
    fmt,
    hash::{Hash, Hasher},
    ops,
    ptr::{self, NonNull},
    slice,
    sync::Arc,
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

#[derive(Clone)]
pub struct UnsafeLock<T: ?Sized> {
    inner: Arc<ManualMutex<T>>,
}

/// Unlike [`Mutex`], this lock is `Send` and `Sync` regardless of whether `T` is `Send` or not.
/// That's because `T` is always under the protection of this lock whenever clients uphold the
/// safety of the lock.
///
/// # Safety
///
/// There must be no copies of the value inside this lock. Clients must not have copies before
/// and after creation of this lock. Because this lock assumes that the value inside the lock
/// has no copies, so the lock is `Send` and `Sync` even if `T` isn't.  
/// For example, imagine that you have multiple `Rc<T>`, which is not `Send`, and make
/// `UnsafeLock<Rc<T>>` from one copy of them, then you send the lock to another thread. It can
/// cause data race because of `Rc<T>` outside this lock.  
/// But if you have only one `T` and wrap it within `UnsafeLock`, then `T` is guaranteed to be
/// protected by this lock. Making copies of `UnsafeLock<T>`, sending it to another thread, and
/// accessing it from another thread does not break the guarantee. But you still can make copies
/// of `T` from its pointer, but you shouldn't.
///
/// [`Mutex`]: std::sync::Mutex
unsafe impl<T: ?Sized> Send for UnsafeLock<T> {}
unsafe impl<T: ?Sized> Sync for UnsafeLock<T> {}

impl<T> UnsafeLock<T> {
    /// # Safety
    ///
    /// There must be no copies of the value. See [`Send implementation`].
    ///
    /// [`Send implementation`]: UnsafeLock<T>#impl-Send-for-UnsafeLock<T>
    pub unsafe fn new(value: T) -> Self {
        Self {
            inner: Arc::new(ManualMutex {
                mutex: RawMutex::INIT,
                data: UnsafeCell::new(value),
            }),
        }
    }
}

impl<T: ?Sized> UnsafeLock<T> {
    /// # Safety
    ///
    /// * Do not dereference to the returned pointer after [`unlock`](Self::unlock).
    /// * Do not make copies of `T` from the returned pointer. See [`Send implementation`].
    ///
    /// [`Send implementation`]: UnsafeLock<T>#impl-Send-for-UnsafeLock<T>
    pub unsafe fn lock(&self) -> NonNull<T> {
        self.inner.mutex.lock();
        unsafe { NonNull::new_unchecked(self.inner.data.get()) }
    }

    /// # Safety
    ///
    /// Must follow [`lock`](Self::lock).
    pub unsafe fn unlock(&self) {
        unsafe { self.inner.mutex.unlock() };
    }
}

impl<T: fmt::Debug> fmt::Debug for UnsafeLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Safety: Lock & unlock are paired with each other.
        unsafe {
            let t = self.lock().as_ref();
            let ret = fmt::Debug::fmt(t, f);
            self.unlock();
            ret
        }
    }
}

struct ManualMutex<T: ?Sized> {
    mutex: RawMutex,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send + ?Sized> Send for ManualMutex<T> {}
unsafe impl<T: Send + ?Sized> Sync for ManualMutex<T> {}

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
