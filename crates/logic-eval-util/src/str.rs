use crate::Set;
use std::{
    borrow, fmt, ops,
    rc::Rc,
    sync::{Arc, LazyLock, RwLock},
};

static STR_FACTORY: LazyLock<RwLock<Set<Arc<str>>>> = LazyLock::new(|| RwLock::new(Set::default()));

/// Use [`From::from`] to create this type.
#[derive(Clone, Eq, Hash)]
pub struct Str(Arc<str>);

impl Str {
    fn create_or_get(input: &str) -> Self {
        let mut factory = STR_FACTORY.write().unwrap();
        if let Some(s) = factory.get(input) {
            Self(s.clone())
        } else {
            let arc: Arc<str> = Arc::from(input);
            factory.insert(Arc::clone(&arc));
            Self(arc)
        }
    }

    pub fn as_str(&self) -> &str {
        self.as_ref()
    }
}

impl Default for Str {
    fn default() -> Self {
        Self::create_or_get("")
    }
}

impl ops::Deref for Str {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for Str {
    fn drop(&mut self) {
        let is_last_one = Arc::strong_count(&self.0) == 2; // This + key in the factory
        if is_last_one {
            if let Ok(mut factory) = STR_FACTORY.write() {
                factory.remove(self.as_str());
            }
        }
    }
}

impl From<&str> for Str {
    fn from(value: &str) -> Self {
        Self::create_or_get(value)
    }
}

impl From<String> for Str {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

impl From<Box<str>> for Str {
    fn from(value: Box<str>) -> Self {
        Self::from(value.as_ref())
    }
}

impl From<Rc<str>> for Str {
    fn from(value: Rc<str>) -> Self {
        Self::from(value.as_ref())
    }
}

impl From<Arc<str>> for Str {
    fn from(value: Arc<str>) -> Self {
        Self(value)
    }
}

impl AsRef<str> for Str {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

impl borrow::Borrow<str> for Str {
    fn borrow(&self) -> &str {
        self.0.as_ref()
    }
}

impl<T: AsRef<str> + ?Sized> PartialEq<T> for Str {
    fn eq(&self, other: &T) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl fmt::Display for Str {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.as_ref().fmt(f)
    }
}

impl fmt::Debug for Str {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.as_ref().fmt(f)
    }
}

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
