# logic-eval

A simple logic evaluator.

[![Crates.io][crates-badge]][crates-url]
[![CI Status][ci-badge]][ci-url]
[![Codecov][codecov-badge]][codecov-url]

[crates-badge]: https://img.shields.io/crates/v/logic-eval.svg
[crates-url]: https://crates.io/crates/logic-eval
[ci-badge]: https://github.com/ecoricemon/logic-eval/actions/workflows/test.yml/badge.svg
[ci-url]: https://github.com/ecoricemon/logic-eval/actions/workflows/test.yml
[codecov-badge]: https://codecov.io/gh/ecoricemon/logic-eval/graph/badge.svg?flag=logic-eval
[codecov-url]: https://app.codecov.io/gh/ecoricemon/logic-eval?flags%5B0%5D=logic-eval

## Example

```rust
use logic_eval::{Database, parse_str};

// Creates a DB with default interner.
let mut db = Database::new();

// Initializes the DB with a little bit of logic.
let dataset = "
    child(a, b).
    child(b, c).
    descend($X, $Y) :- child($X, $Y).
    descend($X, $Z) :- child($X, $Y), descend($Y, $Z).
";
db.insert_dataset(parse_str(dataset, db.interner()).unwrap());
db.commit();

// Queries the DB.
let query = "descend($X, $Y).";
let mut cx = db.query(parse_str(query, db.interner()).unwrap());

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

// If the database was created with default interner, it should be deallocated.
drop(cx);
db.dealloc();
```
