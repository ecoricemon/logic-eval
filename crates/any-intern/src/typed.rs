use bumpalo::Bump;
use std::{alloc::Layout, cell::Cell, marker::PhantomData, mem, ptr, slice};

pub struct TypedArena<T> {
    bump: Bump,
    len: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T> TypedArena<T> {
    /// Returns number of elements in this arena.
    pub const fn len(&self) -> usize {
        self.len.get()
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn alloc(&self, value: T) -> &mut T {
        self.len.set(self.len() + 1);
        self.bump.alloc(value)
    }

    pub fn clear(&mut self) {
        self.drop_all();
        self.bump.reset();
        self.len.set(0);
    }

    fn drop_all(&mut self) {
        if mem::needs_drop::<T>() {
            if size_of::<T>() > 0 {
                let stride = Layout::new::<T>().pad_to_align().size();
                unsafe {
                    for (ptr, len) in self.bump.iter_allocated_chunks_raw() {
                        // Chunk would not be divisible by the `stride` especially when the stride
                        // is greater than 16. In that case, we should ignore the remainder.
                        let num_elems = len / stride;
                        let ptr = ptr.cast::<T>();
                        let slice = slice::from_raw_parts_mut(ptr, num_elems);
                        ptr::drop_in_place(slice);
                    }
                }
            } else {
                let ptr = ptr::dangling_mut::<T>();
                unsafe {
                    let slice = slice::from_raw_parts_mut(ptr, self.len());
                    ptr::drop_in_place(slice);
                }
            }
        }
    }
}

impl<T> Default for TypedArena<T> {
    fn default() -> Self {
        Self {
            bump: Bump::new(),
            len: Cell::new(0),
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for TypedArena<T> {
    fn drop(&mut self) {
        self.drop_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena() {
        test_arena_alloc();
        test_arena_drop();
    }

    fn test_arena_alloc() {
        const START: u32 = 0;
        const END: u32 = 100;
        const EXPECTED: u32 = (END + START) * (END - START + 1) / 2;

        let arena = TypedArena::default();
        let mut refs = Vec::new();
        for i in START..=END {
            let ref_ = arena.alloc(i);
            refs.push(ref_);
        }
        let acc = refs.into_iter().map(|ref_| *ref_).sum::<u32>();
        assert_eq!(acc, EXPECTED);
    }

    fn test_arena_drop() {
        macro_rules! test {
            ($arr_len:literal, $align:literal) => {{
                thread_local! {
                    static SUM: Cell<u32> = Cell::new(0);
                    static CNT: Cell<u32> = Cell::new(0);
                }

                #[repr(align($align))]
                struct A([u8; $arr_len]);

                // Restricted by `u8` and `A::new()`.
                const _: () = const { assert!($arr_len < 256) };

                impl A {
                    fn new() -> Self {
                        Self(std::array::from_fn(|i| i as u8))
                    }

                    fn sum() -> u32 {
                        ($arr_len - 1) * $arr_len / 2
                    }
                }

                impl Drop for A {
                    fn drop(&mut self) {
                        let sum = self.0.iter().map(|n| *n as u32).sum::<u32>();
                        SUM.set(SUM.get() + sum);
                        CNT.set(CNT.get() + 1);
                    }
                }

                struct Zst;

                impl Drop for Zst {
                    fn drop(&mut self) {
                        CNT.set(CNT.get() + 1);
                    }
                }

                const REPEAT: u32 = 10;

                // === Non-ZST type ===

                let arena = TypedArena::default();
                for _ in 0..REPEAT {
                    arena.alloc(A::new());
                }
                drop(arena);

                assert_eq!(SUM.get(), A::sum() * REPEAT);
                assert_eq!(CNT.get(), REPEAT);
                SUM.set(0);
                CNT.set(0);

                // === ZST type ===

                let arena = TypedArena::default();
                for _ in 0..REPEAT {
                    arena.alloc(Zst);
                }
                drop(arena);

                assert_eq!(CNT.get(), REPEAT);
            }};
        }

        // Array len, align
        test!(1, 1);
        test!(1, 2);
        test!(1, 4);
        test!(1, 8);
        test!(1, 16);
        test!(1, 32);
        test!(1, 64);
        test!(1, 128);
        test!(1, 256);

        test!(100, 1);
        test!(100, 2);
        test!(100, 4);
        test!(100, 8);
        test!(100, 16);
        test!(100, 32);
        test!(100, 64);
        test!(100, 128);
        test!(100, 256);
    }
}
