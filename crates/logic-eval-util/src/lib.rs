pub mod reference;
pub mod str;
pub mod symbol;
pub mod unique;

// === Hash map and set used within this crate ===

pub(crate) type Map<K, V> = std::collections::HashMap<K, V, fxhash::FxBuildHasher>;
