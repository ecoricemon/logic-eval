use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use logic_eval::{parse_str, ClauseDataset, Database, Expr, InternedStr, Name, StrInterner};

const NODES: usize = 240;
const QUERY_COUNT: usize = 64;

type BenchName<'a> = Name<InternedStr<'a>>;

fn node(index: usize) -> String {
    format!("n{index}")
}

fn build_database<'a>(interner: &'a StrInterner) -> Database<BenchName<'a>> {
    let mut source = String::new();

    // A chain makes each ancestry query produce many answers. A few skip edges add branching so the
    // engine does enough independent work for parallel throughput to show up in the benchmark.
    for i in 0..NODES - 1 {
        source.push_str(&format!("parent({}, {}).\n", node(i), node(i + 1)));
    }
    for i in 0..NODES - 3 {
        if i % 7 == 0 {
            source.push_str(&format!("parent({}, {}).\n", node(i), node(i + 3)));
        }
    }

    source.push_str(
        "
        ancestor($X, $Y) :- parent($X, $Y).
        ancestor($X, $Z) :- parent($X, $Y), ancestor($Y, $Z).
        ",
    );

    let dataset: ClauseDataset<_> = parse_str(&source, interner).unwrap();
    let mut db = Database::default();
    db.insert_dataset(dataset);
    db
}

fn build_queries<'a>(interner: &'a StrInterner) -> Vec<Expr<BenchName<'a>>> {
    (0..QUERY_COUNT)
        .map(|i| {
            let root = i % (NODES / 3);
            parse_str(&format!("ancestor({}, $Who)", node(root)), interner).unwrap()
        })
        .collect()
}

fn count_answers(db: &Database<BenchName<'_>>, query: Expr<BenchName<'_>>) -> usize {
    let mut cx = db.query(query);
    let mut count = 0;
    while let Some(answer) = cx.prove_next() {
        count += answer.count();
    }
    count
}

fn run_serial(db: &Database<BenchName<'_>>, queries: &[Expr<BenchName<'_>>]) -> usize {
    queries
        .iter()
        .cloned()
        .map(|query| count_answers(db, query))
        .sum()
}

fn run_threaded(
    db: &Database<BenchName<'_>>,
    queries: &[Expr<BenchName<'_>>],
    threads: usize,
) -> usize {
    let chunk_len = (queries.len() + threads - 1) / threads;

    std::thread::scope(|scope| {
        let handles = queries
            .chunks(chunk_len)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .cloned()
                        .map(|query| count_answers(db, query))
                        .sum::<usize>()
                })
            })
            .collect::<Vec<_>>();

        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .sum()
    })
}

fn benchmark_query_threads(c: &mut Criterion) {
    let interner = StrInterner::new();
    let db = build_database(&interner);
    let queries = build_queries(&interner);
    let expected = run_serial(&db, &queries);

    let mut group = c.benchmark_group("logic_eval_query_threads");
    group.sample_size(10);

    group.bench_with_input(BenchmarkId::new("threads", 1), &1, |b, _| {
        b.iter(|| {
            let count = run_serial(&db, &queries);
            assert_eq!(count, expected);
            black_box(count)
        });
    });

    for threads in [2, 4] {
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            &threads,
            |b, &threads| {
                b.iter(|| {
                    let count = run_threaded(&db, &queries, threads);
                    assert_eq!(count, expected);
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_query_threads);
criterion_main!(benches);
