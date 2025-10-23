pub mod str;
pub mod symbol;
pub mod unique;

// === Hash map and set used within this crate ===

#[cfg(not(test))]
pub(crate) type Map<K, V> = std::collections::HashMap<K, V>;

#[cfg(test)]
pub(crate) type Map<K, V> = std::collections::HashMap<K, V, FixedState>;

#[cfg(not(test))]
pub(crate) type Set<T> = std::collections::HashSet<T>;

#[cfg(test)]
pub(crate) type Set<T> = std::collections::HashSet<T, FixedState>;

#[cfg(test)]
#[derive(Default, Clone, Copy)]
struct FixedState;

#[cfg(test)]
impl std::hash::BuildHasher for FixedState {
    type Hasher = std::hash::DefaultHasher;
    fn build_hasher(&self) -> Self::Hasher {
        std::hash::DefaultHasher::new()
    }
}
