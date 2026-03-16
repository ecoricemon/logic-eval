# logic-eval

[![Crates.io][crates-badge]][crates-url]
[![CI Status][ci-badge]][ci-url]
[![Codecov][codecov-badge]][codecov-url]

[crates-badge]: https://img.shields.io/crates/v/logic-eval.svg
[crates-url]: https://crates.io/crates/logic-eval
[ci-badge]: https://github.com/ecoricemon/logic-eval/actions/workflows/test.yml/badge.svg
[ci-url]: https://github.com/ecoricemon/logic-eval/actions/workflows/test.yml
[codecov-badge]: https://codecov.io/gh/ecoricemon/logic-eval/graph/badge.svg?flag=logic-eval
[codecov-url]: https://app.codecov.io/gh/ecoricemon/logic-eval?flags%5B0%5D=logic-eval

A prolog-like logic evaluator.

## Example

```rust
use logic_eval::{Database, StrInterner, parse_str};

// Creates a DB.
let mut db = Database::new();
let interner = StrInterner::new();

// Initializes the DB with a little bit of logic.
let dataset = "
    child(a, b).
    child(b, c).
    descend($X, $Y) :- child($X, $Y).
    descend($X, $Z) :- child($X, $Y), descend($Y, $Z).
";
db.insert_dataset(parse_str(dataset, &interner).unwrap());
db.commit();

// Queries the DB.
let query = "descend($X, $Y).";
let mut cx = db.query(parse_str(query, &interner).unwrap());

let mut answer = Vec::new();
while let Some(eval) = cx.prove_next() {
    let s = eval.into_iter().map(|assign| assign.to_string()).collect::<Vec<_>>().join(", ");
    answer.push(s);
}

assert_eq!(answer, [
    "$X = a, $Y = b",
    "$X = b, $Y = c",
    "$X = a, $Y = c",
]);
```
