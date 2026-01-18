use any_intern::{AnyInterner, Interner};

fn main() {
    test_interner_should_call_drop();
    test_any_interner_should_call_drop();
}

fn test_interner_should_call_drop() {
    #[derive(PartialEq, Eq, Hash)]
    struct A(Box<u32>);

    #[derive(PartialEq, Eq, Hash)]
    struct B(Box<u32>);

    let interner = Interner::new();

    const N: u32 = 1000;
    for i in 0..N {
        interner.intern_static(A(Box::new(i)));
        interner.intern_static(B(Box::new(i)));
    }
}

fn test_any_interner_should_call_drop() {
    #[derive(PartialEq, Eq, Hash)]
    struct A(Box<u32>);

    let interner = AnyInterner::new::<A>();

    const N: u32 = 1000;
    for i in 0..N {
        unsafe {
            interner.intern(A(Box::new(i)));
        }
    }
}
