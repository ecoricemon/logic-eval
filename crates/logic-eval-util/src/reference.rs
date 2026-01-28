use std::{marker::PhantomData, ptr::NonNull};

/// A lightweight wrapper around a non-null pointer for a shared reference.
///
/// The wrappers are useful when you need a pointer instead of a reference while not degrading usage
/// and keeping its provenance. For instance, if you make the [`Ref`] from a mutable reference, the
/// fact the type is made from "mutable" won't be lost, which is an information which some tools,
/// such as [miri], used for finding pointer safety violations.
///
/// [miri]: https://github.com/rust-lang/miri
#[derive(Debug)]
pub struct Ref<'a, T: 'a + ?Sized> {
    ptr: NonNull<T>,
    _marker: PhantomData<&'a T>,
}

impl<'a, T: 'a + ?Sized> Ref<'a, T> {
    pub const fn from_ref(r: &'a T) -> Self {
        Self {
            ptr: NonNull::from_ref(r),
            _marker: PhantomData,
        }
    }

    pub const fn from_mut(r: &'a mut T) -> Self {
        Self {
            ptr: NonNull::from_mut(r),
            _marker: PhantomData,
        }
    }

    pub const fn as_ref(&self) -> &'a T {
        // Safety: The type actually has the `&'a T`.
        unsafe { self.ptr.as_ref() }
    }

    pub const fn as_nonnull(self) -> NonNull<T> {
        self.ptr
    }

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
