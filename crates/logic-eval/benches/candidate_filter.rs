use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use logic_eval::{parse_str, ClauseDataset, Database, Expr, InternedStr, Name, StrInterner};

type BenchName<'a> = Name<InternedStr<'a>>;

const NOISE_CANDIDATES_PER_REQUEST: usize = 24;
const QUERY_BATCH_LEN: usize = 10;

#[derive(Clone, Copy)]
struct CaseSize {
    name: &'static str,
    requests: usize,
}

const CASES: &[CaseSize] = &[
    CaseSize {
        name: "small",
        requests: 32,
    },
    CaseSize {
        name: "medium",
        requests: 128,
    },
];

fn type_name(index: usize) -> String {
    format!("ty{index}")
}

fn request_name(index: usize) -> String {
    format!("req{index}")
}

fn candidate_name(index: usize) -> String {
    format!("candidate{index}")
}

fn noise_candidate_name(request: usize, candidate: usize) -> String {
    format!("candidate_noise_{request}_{candidate}")
}

fn nested_type(name: &str) -> String {
    format!("vec(vec({name}))")
}

fn build_source(requests: usize) -> String {
    let mut source = String::new();

    for i in 0..requests {
        let ty = type_name(i);
        let expected = nested_type(&ty);
        source.push_str(&format!(
            "expected_value({}, {}).\n",
            request_name(i),
            expected
        ));
        source.push_str(&format!("enabled_candidate({}).\n", candidate_name(i)));
        source.push_str(&format!(
            "candidate_value({}, {}, {}).\n",
            request_name(i),
            candidate_name(i),
            expected
        ));

        for j in 0..NOISE_CANDIDATES_PER_REQUEST {
            let noise_ty = type_name(requests + i * NOISE_CANDIDATES_PER_REQUEST + j);
            source.push_str(&format!(
                "candidate_value({}, {}, {}).\n",
                request_name(i),
                noise_candidate_name(i, j),
                nested_type(&noise_ty)
            ));
        }
    }

    source.push_str(
        "
        same_shape($Value, $Value).
        same_shape(vec($A), vec($B)) :- same_shape($A, $B).
        candidate_match($Request, $Candidate) :-
            candidate_value($Request, $Candidate, $Value),
            expected_value($Request, $Expected),
            same_shape($Value, $Expected),
            enabled_candidate($Candidate).
        ",
    );

    source
}

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

    for case in CASES {
        group.bench_with_input(
            BenchmarkId::new("build_database", case.name),
            case,
            |b, case| {
                b.iter(|| {
                    let interner = StrInterner::new();
                    let db = build_database(&interner, case.requests);
                    black_box(db.clauses().count())
                });
            },
        );

        let interner = StrInterner::new();
        let db = build_database(&interner, case.requests);
        let query_start = case.requests / 2;
        let queries = build_queries(&interner, query_start);
        let expected_first = count_first_answers(&db, &queries);
        let expected_all = count_all_answers(&db, &queries);

        group.bench_with_input(BenchmarkId::new("first_answer", case.name), case, |b, _| {
            b.iter(|| {
                let count = count_first_answers(&db, &queries);
                assert_eq!(count, expected_first);
                black_box(count)
            });
        });

        group.bench_with_input(BenchmarkId::new("all_answers", case.name), case, |b, _| {
            b.iter(|| {
                let count = count_all_answers(&db, &queries);
                assert_eq!(count, expected_all);
                black_box(count)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_candidate_filter);
criterion_main!(benches);
