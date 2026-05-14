use core::{marker::PhantomData, ptr::NonNull};

/// A lightweight wrapper around a non-null pointer for a shared reference.
///
/// The wrappers are useful when you need a pointer instead of a reference while preserving
/// ergonomics and provenance. For example, if you make a [`Ref`] from a mutable reference, the
/// fact that the type came from a mutable reference is retained. Tools such as [miri] can use that
/// information to find pointer-safety violations.
///
/// [miri]: https://github.com/rust-lang/miri
#[derive(Debug)]
pub struct Ref<'a, T: 'a + ?Sized> {
    ptr: NonNull<T>,
    _marker: PhantomData<&'a T>,
}

impl<'a, T: 'a + ?Sized> Ref<'a, T> {
    /// Creates a pointer wrapper from a shared reference.
    pub const fn from_ref(r: &'a T) -> Self {
        // Safety: A reference is non-null.
        let ptr = unsafe { NonNull::new_unchecked(r as *const T as *mut T) };
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Creates a pointer wrapper from a mutable reference.
    pub fn from_mut(r: &'a mut T) -> Self {
        // Safety: A reference is non-null.
        let ptr = unsafe { NonNull::new_unchecked(r as *mut T) };
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    // Output lifetime is 'a, so we need this method rather than AsRef::as_ref or something like
    // that.
    /// Returns the original shared reference.
    #[allow(clippy::should_implement_trait)]
    pub fn as_ref(&self) -> &'a T {
        // Safety: The type actually has the `&'a T`.
        unsafe { self.ptr.as_ref() }
    }

    /// Returns this wrapper as a non-null pointer.
    pub const fn as_nonnull(self) -> NonNull<T> {
        self.ptr
    }

    /// Returns this wrapper as a raw mutable pointer.
    pub const fn as_ptr(self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<'a, T: 'a + ?Sized> Clone for Ref<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: 'a + ?Sized> Copy for Ref<'a, T> {}
