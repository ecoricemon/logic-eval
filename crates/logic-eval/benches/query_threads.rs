use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use logic_eval::{Atom, Clause, ClauseDataset, Database, Expr, Term};

const NODES: usize = 240;
const QUERY_COUNT: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BenchAtom(String);

impl Atom for BenchAtom {
    fn is_variable(&self) -> bool {
        self.0.starts_with('$')
    }
}

fn sym(name: impl Into<String>) -> BenchAtom {
    BenchAtom(name.into())
}

fn atom(name: impl Into<String>) -> Term<BenchAtom> {
    Term::atom(sym(name))
}

fn compound(
    name: impl Into<String>,
    args: impl IntoIterator<Item = Term<BenchAtom>>,
) -> Term<BenchAtom> {
    Term::compound(sym(name), args)
}

fn parent(from: usize, to: usize) -> Clause<BenchAtom> {
    Clause::fact(compound("parent", [atom(node(from)), atom(node(to))]))
}

fn node(index: usize) -> String {
    format!("n{index}")
}

fn build_database() -> Database<BenchAtom> {
    let mut clauses = Vec::new();

    // A chain makes each ancestry query produce many answers. A few skip edges add branching so the
    // engine does enough independent work for parallel throughput to show up in the benchmark.
    for i in 0..NODES - 1 {
        clauses.push(parent(i, i + 1));
    }
    for i in 0..NODES - 3 {
        if i % 7 == 0 {
            clauses.push(parent(i, i + 3));
        }
    }

    clauses.push(Clause::rule(
        compound("ancestor", [atom("$X"), atom("$Y")]),
        Expr::term(compound("parent", [atom("$X"), atom("$Y")])),
    ));
    clauses.push(Clause::rule(
        compound("ancestor", [atom("$X"), atom("$Z")]),
        Expr::expr_and([
            Expr::term(compound("parent", [atom("$X"), atom("$Y")])),
            Expr::term(compound("ancestor", [atom("$Y"), atom("$Z")])),
        ]),
    ));

    let mut db = Database::default();
    db.insert_dataset(ClauseDataset(clauses));
    db
}

fn build_queries() -> Vec<Expr<BenchAtom>> {
    (0..QUERY_COUNT)
        .map(|i| {
            let root = i % (NODES / 3);
            Expr::term(compound("ancestor", [atom(node(root)), atom("$Who")]))
        })
        .collect()
}

fn count_answers(db: &Database<BenchAtom>, query: Expr<BenchAtom>) -> usize {
    let mut cx = db.query(query);
    let mut count = 0;
    while let Some(answer) = cx.prove_next() {
        count += answer.count();
    }
    count
}

fn run_serial(db: &Database<BenchAtom>, queries: &[Expr<BenchAtom>]) -> usize {
    queries
        .iter()
        .cloned()
        .map(|query| count_answers(db, query))
        .sum()
}

fn run_threaded(db: &Database<BenchAtom>, queries: &[Expr<BenchAtom>], threads: usize) -> usize {
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
    let db = build_database();
    let queries = build_queries();
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
