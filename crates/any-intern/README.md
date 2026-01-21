# any-intern

`any-intern` is a Rust crate for efficient interning of values. It provides flexible interners for
various use cases, ensuring that each unique value is stored only once, saving memory and improving
lookup efficiency.

## Examples

### Interner

A generic interner for storing and deduplicating values of various types.

```rust
use any_intern::Interner;

#[derive(PartialEq, Eq, Hash, Debug)]
struct A(u32);

#[derive(PartialEq, Eq, Hash, Debug)]
struct B(u32);

let interner = Interner::new();
let a1 = interner.intern_static(A(42));
let a2 = interner.intern_static(A(42));
let b1 = interner.intern_static(B(42));
assert_eq!(a1, a2); // Same value, same reference
assert_ne!(a1.erased_raw(), b1.erased_raw()); // Different value, different reference

let s1 = interner.intern_dropless("hello");
let s2 = interner.intern_dropless(&*String::from("hello"));
assert_eq!(s1, s2); // Same string, same reference
```

### AnyInterner

A type-erased interner for storing and deduplicating values of a single type.

```rust
use any_intern::AnyInterner;

#[derive(PartialEq, Eq, Hash, Debug)]
struct A(u32);

let interner = AnyInterner::of::<A>();
unsafe {
    let a1 = interner.intern(A(42));
    let a2 = interner.intern(A(42));
    assert_eq!(a1, a2); // Same value, same reference
}
```

### DroplessInterner

An interner for storing and deduplicating values without requiring ownership.

```rust
use any_intern::DroplessInterner;

let interner = DroplessInterner::new();
let hello = interner.intern("hello");
let another_hello = interner.intern(&*String::from("hello"));
assert_eq!(hello, another_hello); // Same value, same reference
```
