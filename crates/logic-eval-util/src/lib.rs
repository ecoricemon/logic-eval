//! Utility data structures used by `logic-eval`.

/// Reference wrapper utilities.
pub mod reference;
/// String-related helpers.
pub mod str;
/// Scoped symbol table utilities.
pub mod symbol;
/// Containers that preserve unique values.
pub mod unique;

// === Hash map and set used within this crate ===

pub(crate) type Map<K, V> = std::collections::HashMap<K, V, fxhash::FxBuildHasher>;
