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
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::reference::Ref;
    ///
    /// let value = 42;
    /// let ptr = Ref::from_ref(&value);
    ///
    /// assert_eq!(*ptr.as_ref(), 42);
    /// ```
    pub const fn from_ref(r: &'a T) -> Self {
        // Safety: A reference is non-null.
        let ptr = unsafe { NonNull::new_unchecked(r as *const T as *mut T) };
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Creates a pointer wrapper from a mutable reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::reference::Ref;
    ///
    /// let mut value = 42;
    /// let ptr = Ref::from_mut(&mut value);
    ///
    /// assert_eq!(*ptr.as_ref(), 42);
    /// ```
    pub fn from_mut(r: &'a mut T) -> Self {
        // Safety: A reference is non-null.
        let ptr = unsafe { NonNull::new_unchecked(r as *mut T) };
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Returns the original shared reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::reference::Ref;
    ///
    /// let value = String::from("hello");
    /// let ptr = Ref::from_ref(&value);
    ///
    /// assert_eq!(ptr.as_ref(), "hello");
    /// ```
    //
    // Output lifetime is 'a, which is the difference from AsRef::as_ref.
    #[allow(clippy::should_implement_trait)]
    pub fn as_ref(&self) -> &'a T {
        // Safety: The type actually has the `&'a T`.
        unsafe { self.ptr.as_ref() }
    }

    /// Returns this wrapper as a non-null pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::reference::Ref;
    ///
    /// let value = 42;
    /// let ptr = Ref::from_ref(&value).as_nonnull();
    ///
    /// assert_eq!(ptr.as_ptr(), (&value as *const i32).cast_mut());
    /// ```
    pub const fn as_nonnull(self) -> NonNull<T> {
        self.ptr
    }

    /// Returns this wrapper as a raw mutable pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::reference::Ref;
    ///
    /// let value = 42;
    /// let ptr = Ref::from_ref(&value).as_ptr();
    ///
    /// assert_eq!(ptr, (&value as *const i32).cast_mut());
    /// ```
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
