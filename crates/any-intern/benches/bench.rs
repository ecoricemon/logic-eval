use any_intern::DroplessInterner;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn benchmark_intern_formatted(c: &mut Criterion) {
    let mut group = c.benchmark_group("DroplessInterner");

    const START: u32 = 0;
    const N: u32 = 10_000;
    const END: u32 = START + N;

    // Benchmark for interning strings from integers
    group.bench_function("intern_after_to_string", |b| {
        let interner = DroplessInterner::new();
        let integers: Vec<u32> = (START..END).collect();

        b.iter(|| {
            for i in &integers {
                let s = i.to_string(); // Convert integer to string
                black_box(interner.intern(s.as_str()));
            }
        });
    });

    // Benchmark for interning formatted strings (no alloc for `String`)
    group.bench_function("intern_formatted_str", |b| {
        let interner = DroplessInterner::new();
        let integers: Vec<u32> = (START..END).collect();

        b.iter(|| {
            for i in &integers {
                black_box(interner.intern_formatted_str(i, 10).unwrap());
            }
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_intern_formatted);
criterion_main!(benches);
