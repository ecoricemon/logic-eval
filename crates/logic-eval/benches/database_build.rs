mod common;

use common::{build_source, NOISE_CANDIDATES_PER_REQUEST, REQUESTS};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use logic_eval::{
    parse_str, Atom, Clause, ClauseDataset, Database, InternedStr, Name, StrInterner, Term,
};

type BenchName<'a> = Name<InternedStr<'a>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BenchAtom {
    ExpectedValue,
    EnabledCandidate,
    CandidateValue,
    SameShape,
    CandidateMatch,
    Vec,
    Req(usize),
    Candidate(usize),
    NoiseCandidate(usize, usize),
    Ty(usize),
    RequestVar,
    CandidateVar,
    ValueVar,
    ExpectedVar,
    AVar,
    BVar,
}

impl Atom for BenchAtom {
    fn is_variable(&self) -> bool {
        matches!(
            self,
            Self::RequestVar
                | Self::CandidateVar
                | Self::ValueVar
                | Self::ExpectedVar
                | Self::AVar
                | Self::BVar
        )
    }
}

fn atom(atom: BenchAtom) -> Term<BenchAtom> {
    Term::atom(atom)
}

fn term(functor: BenchAtom, args: impl IntoIterator<Item = Term<BenchAtom>>) -> Term<BenchAtom> {
    Term::compound(functor, args)
}

fn expr(
    functor: BenchAtom,
    args: impl IntoIterator<Item = Term<BenchAtom>>,
) -> logic_eval::Expr<BenchAtom> {
    logic_eval::Expr::term_compound(functor, args)
}

fn nested_atom_type(ty: usize) -> Term<BenchAtom> {
    term(
        BenchAtom::Vec,
        [term(BenchAtom::Vec, [atom(BenchAtom::Ty(ty))])],
    )
}

fn build_custom_atom_dataset(requests: usize) -> ClauseDataset<BenchAtom> {
    let mut clauses = Vec::new();

    for i in 0..requests {
        let expected = nested_atom_type(i);
        clauses.push(Clause::fact(term(
            BenchAtom::ExpectedValue,
            [atom(BenchAtom::Req(i)), expected.clone()],
        )));
        clauses.push(Clause::fact(term(
            BenchAtom::EnabledCandidate,
            [atom(BenchAtom::Candidate(i))],
        )));
        clauses.push(Clause::fact(term(
            BenchAtom::CandidateValue,
            [
                atom(BenchAtom::Req(i)),
                atom(BenchAtom::Candidate(i)),
                expected,
            ],
        )));

        for j in 0..NOISE_CANDIDATES_PER_REQUEST {
            let noise_ty = requests + i * NOISE_CANDIDATES_PER_REQUEST + j;
            clauses.push(Clause::fact(term(
                BenchAtom::CandidateValue,
                [
                    atom(BenchAtom::Req(i)),
                    atom(BenchAtom::NoiseCandidate(i, j)),
                    nested_atom_type(noise_ty),
                ],
            )));
        }
    }

    clauses.push(Clause::fact(term(
        BenchAtom::SameShape,
        [atom(BenchAtom::ValueVar), atom(BenchAtom::ValueVar)],
    )));
    clauses.push(Clause::rule(
        term(
            BenchAtom::SameShape,
            [
                term(BenchAtom::Vec, [atom(BenchAtom::AVar)]),
                term(BenchAtom::Vec, [atom(BenchAtom::BVar)]),
            ],
        ),
        expr(
            BenchAtom::SameShape,
            [atom(BenchAtom::AVar), atom(BenchAtom::BVar)],
        ),
    ));
    clauses.push(Clause::rule(
        term(
            BenchAtom::CandidateMatch,
            [atom(BenchAtom::RequestVar), atom(BenchAtom::CandidateVar)],
        ),
        logic_eval::Expr::expr_and([
            expr(
                BenchAtom::CandidateValue,
                [
                    atom(BenchAtom::RequestVar),
                    atom(BenchAtom::CandidateVar),
                    atom(BenchAtom::ValueVar),
                ],
            ),
            expr(
                BenchAtom::ExpectedValue,
                [atom(BenchAtom::RequestVar), atom(BenchAtom::ExpectedVar)],
            ),
            expr(
                BenchAtom::SameShape,
                [atom(BenchAtom::ValueVar), atom(BenchAtom::ExpectedVar)],
            ),
            expr(BenchAtom::EnabledCandidate, [atom(BenchAtom::CandidateVar)]),
        ]),
    ));

    ClauseDataset(clauses)
}

fn build_database<'a>(interner: &'a StrInterner, requests: usize) -> Database<BenchName<'a>> {
    let source = build_source(requests);
    let dataset: ClauseDataset<_> = parse_str(&source, interner).unwrap();
    let mut db = Database::default();
    db.insert_dataset(dataset);
    db
}

fn build_custom_atom_database(requests: usize) -> Database<BenchAtom> {
    let dataset = build_custom_atom_dataset(requests);
    let mut db = Database::default();
    db.insert_dataset(dataset);
    db
}

fn benchmark_database_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("logic_eval_database_build");
    group.sample_size(10);

    group.bench_function("build_source", |b| {
        b.iter(|| {
            let source = build_source(REQUESTS);
            black_box(source.len())
        });
    });

    let source = build_source(REQUESTS);
    group.bench_function("parse_dataset", |b| {
        b.iter(|| {
            let interner = StrInterner::new();
            let dataset: ClauseDataset<_> = parse_str(black_box(&source), &interner).unwrap();
            black_box(dataset.len())
        });
    });

    let interner = StrInterner::new();
    let dataset: ClauseDataset<_> = parse_str(&source, &interner).unwrap();
    group.bench_function("insert_dataset", |b| {
        b.iter_batched(
            || dataset.clone(),
            |dataset| {
                let mut db = Database::default();
                db.insert_dataset(dataset);
                black_box(db.clauses().count())
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("build_custom_atom_dataset", |b| {
        b.iter(|| {
            let dataset = build_custom_atom_dataset(REQUESTS);
            black_box(dataset.len())
        });
    });

    group.bench_function("build_custom_atom_database", |b| {
        b.iter(|| {
            let db = build_custom_atom_database(REQUESTS);
            black_box(db.clauses().count())
        });
    });

    group.bench_function("build_database", |b| {
        b.iter(|| {
            let interner = StrInterner::new();
            let db = build_database(&interner, REQUESTS);
            black_box(db.clauses().count())
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_database_build);
criterion_main!(benches);
