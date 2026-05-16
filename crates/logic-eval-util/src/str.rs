use std::ops;

/// A string path with an absolute/relative marker.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct StrPath<'a> {
    /// Whether the path is absolute.
    pub is_absolute: bool,
    /// The path text.
    pub path: &'a str,
}

impl<'a> StrPath<'a> {
    /// Creates an absolute string path.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::str::StrPath;
    ///
    /// let path = StrPath::absolute("/root/module");
    ///
    /// assert!(path.is_absolute);
    /// assert_eq!(&*path, "/root/module");
    /// ```
    pub const fn absolute(path: &'a str) -> Self {
        Self {
            is_absolute: true,
            path,
        }
    }

    /// Creates a relative string path.
    ///
    /// # Examples
    ///
    /// ```
    /// use logic_eval_util::str::StrPath;
    ///
    /// let path = StrPath::relative("local");
    ///
    /// assert!(!path.is_absolute);
    /// assert_eq!(path.path, "local");
    /// ```
    pub const fn relative(path: &'a str) -> Self {
        Self {
            is_absolute: false,
            path,
        }
    }
}

impl ops::Deref for StrPath<'_> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.path
    }
}
