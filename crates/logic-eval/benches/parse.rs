mod common;

use common::{
    build_fact_source, delimiter_heavy_expr, fact_count, repeated_names,
    representative_fact_clause, representative_nested_term, representative_rule_body_expr,
    representative_rule_clause, unique_names, REQUESTS,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use logic_eval::{parse_str, Clause, ClauseDataset, Intern, StrInterner, Term};

fn benchmark_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("logic_eval_parse");
    group.sample_size(10);

    let fact_source = build_fact_source(REQUESTS);
    group.bench_function("parse_fact_dataset", |b| {
        b.iter(|| {
            let interner = StrInterner::new();
            let dataset: ClauseDataset<_> = parse_str(black_box(&fact_source), &interner).unwrap();
            black_box(dataset.len())
        });
    });

    let fact_clause = representative_fact_clause(REQUESTS);
    group.bench_function("parse_fact_clause", |b| {
        b.iter(|| {
            let interner = StrInterner::new();
            let clause: Clause<_> = parse_str(black_box(&fact_clause), &interner).unwrap();
            black_box(&clause);
        });
    });

    let rule_clause = representative_rule_clause();
    group.bench_function("parse_rule_clause", |b| {
        b.iter(|| {
            let interner = StrInterner::new();
            let clause: Clause<_> = parse_str(black_box(rule_clause), &interner).unwrap();
            black_box(&clause);
        });
    });

    let rule_body_expr = representative_rule_body_expr();
    group.bench_function("parse_rule_body_expr", |b| {
        b.iter(|| {
            let interner = StrInterner::new();
            let expr: logic_eval::Expr<_> =
                parse_str(black_box(rule_body_expr), &interner).unwrap();
            black_box(&expr);
        });
    });

    let nested_term = representative_nested_term(REQUESTS);
    group.bench_function("parse_nested_term", |b| {
        b.iter(|| {
            let interner = StrInterner::new();
            let term: Term<_> = parse_str(black_box(&nested_term), &interner).unwrap();
            black_box(&term);
        });
    });

    let delimiter_expr = delimiter_heavy_expr(REQUESTS);
    group.bench_function("parse_delimiter_heavy_expr", |b| {
        b.iter(|| {
            let interner = StrInterner::new();
            let expr: logic_eval::Expr<_> =
                parse_str(black_box(&delimiter_expr), &interner).unwrap();
            black_box(&expr);
        });
    });

    let name_count = fact_count(REQUESTS) * 4;
    let unique_names = unique_names(name_count);
    group.bench_function("parse_unique_atom_terms", |b| {
        b.iter(|| {
            let interner = StrInterner::new();
            let mut parsed = 0;
            for name in &unique_names {
                let term: Term<_> = parse_str(black_box(name.as_str()), &interner).unwrap();
                black_box(&term);
                parsed += 1;
            }
            black_box(parsed)
        });
    });

    let repeated_names = repeated_names(name_count);
    group.bench_function("parse_repeated_atom_terms", |b| {
        b.iter(|| {
            let interner = StrInterner::new();
            let mut parsed = 0;
            for name in &repeated_names {
                let term: Term<_> = parse_str(black_box(*name), &interner).unwrap();
                black_box(&term);
                parsed += 1;
            }
            black_box(parsed)
        });
    });

    group.bench_function("intern_unique_names", |b| {
        b.iter(|| {
            let interner = StrInterner::new();
            let mut interned = 0;
            for name in &unique_names {
                let name = interner.intern_str(black_box(name.as_str()));
                black_box(&name);
                interned += 1;
            }
            black_box(interned)
        });
    });

    group.bench_function("intern_repeated_names", |b| {
        b.iter(|| {
            let interner = StrInterner::new();
            let mut interned = 0;
            for name in &repeated_names {
                let name = interner.intern_str(black_box(*name));
                black_box(&name);
                interned += 1;
            }
            black_box(interned)
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_parse);
criterion_main!(benches);
