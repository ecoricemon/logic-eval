mod common;

use common::{build_source, request_name, REQUESTS};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use logic_eval::{parse_str, ClauseDataset, Database, Expr, InternedStr, Name, StrInterner};

type BenchName<'a> = Name<InternedStr<'a>>;

const QUERY_BATCH_LEN: usize = 10;

fn build_database<'a>(interner: &'a StrInterner, requests: usize) -> Database<BenchName<'a>> {
    let source = build_source(requests);
    let dataset: ClauseDataset<_> = parse_str(&source, interner).unwrap();
    let mut db = Database::default();
    db.insert_dataset(dataset);
    db
}

fn build_query<'a>(interner: &'a StrInterner, request: usize) -> Expr<BenchName<'a>> {
    parse_str(
        &format!("candidate_match({}, $Candidate)", request_name(request)),
        interner,
    )
    .unwrap()
}

fn build_queries<'a>(interner: &'a StrInterner, start_request: usize) -> Vec<Expr<BenchName<'a>>> {
    (start_request..start_request + QUERY_BATCH_LEN)
        .map(|request| build_query(interner, request))
        .collect()
}

fn count_first_answer_for_query(db: &Database<BenchName<'_>>, query: Expr<BenchName<'_>>) -> usize {
    db.query(query)
        .prove_next()
        .map_or(0, |answer| answer.count())
}

fn count_all_answers_for_query(db: &Database<BenchName<'_>>, query: Expr<BenchName<'_>>) -> usize {
    let mut cx = db.query(query);
    let mut count = 0;
    while let Some(answer) = cx.prove_next() {
        count += answer.count();
    }
    count
}

fn count_first_answers(db: &Database<BenchName<'_>>, queries: &[Expr<BenchName<'_>>]) -> usize {
    queries
        .iter()
        .cloned()
        .map(|query| count_first_answer_for_query(db, query))
        .sum()
}

fn count_all_answers(db: &Database<BenchName<'_>>, queries: &[Expr<BenchName<'_>>]) -> usize {
    queries
        .iter()
        .cloned()
        .map(|query| count_all_answers_for_query(db, query))
        .sum()
}

fn benchmark_candidate_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("logic_eval_candidate_filter");
    group.sample_size(10);

    let interner = StrInterner::new();
    let db = build_database(&interner, REQUESTS);
    let query_start = REQUESTS / 2;
    let queries = build_queries(&interner, query_start);
    let expected_first = count_first_answers(&db, &queries);
    let expected_all = count_all_answers(&db, &queries);

    group.bench_function("first_answer", |b| {
        b.iter(|| {
            let count = count_first_answers(&db, &queries);
            assert_eq!(count, expected_first);
            black_box(count)
        });
    });

    group.bench_function("all_answers", |b| {
        b.iter(|| {
            let count = count_all_answers(&db, &queries);
            assert_eq!(count, expected_all);
            black_box(count)
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_candidate_filter);
criterion_main!(benches);
