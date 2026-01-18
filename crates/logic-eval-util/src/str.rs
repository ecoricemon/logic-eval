use std::ops;

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct StrPath<'a> {
    pub is_absolute: bool,
    pub path: &'a str,
}

impl<'a> StrPath<'a> {
    pub const fn absolute(path: &'a str) -> Self {
        Self {
            is_absolute: true,
            path,
        }
    }

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
