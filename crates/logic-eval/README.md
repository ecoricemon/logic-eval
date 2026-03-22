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

```text
+------------------+
|  Feel(fresh) :-  |
|   Sleep(well),   |
|   Sun(shine),    |
|   Air(cool).     |
+------------------+
```

`logic-eval` is a Prolog-like logic evaluation library for Rust.

## Features

- `SLG resolution`: handles recursive queries with tabling.
- `Custom type support`: use `&str`, interned strings, or your own `Atom` type.
- `Parsing`: parse facts, rules, and queries from a Prolog-like text syntax with `parse_str`.
- `Basic logical operators`: supports NOT, AND, and OR in rule bodies.

## Examples

### Parse text and query a database

```rust
use logic_eval::{Database, StrInterner, parse_str};

let mut db = Database::new();
let interner = StrInterner::new();

let dataset = "
    child(a, b).
    child(b, c).
    child(c, d).
    descend($X, $Y) :- child($X, $Y).
    descend($X, $Z) :- child($X, $Y), descend($Y, $Z).
";
db.insert_dataset(parse_str(dataset, &interner).unwrap());
db.commit();

let mut cx = db.query(parse_str("descend($X, $Y).", &interner).unwrap());

let mut answer = Vec::new();
while let Some(eval) = cx.prove_next() {
    let s = eval.into_iter().map(|assign| assign.to_string()).collect::<Vec<_>>().join(", ");
    answer.push(s);
}

assert_eq!(answer, [
    "$X = a, $Y = b",
    "$X = b, $Y = c",
    "$X = c, $Y = d",
    "$X = a, $Y = c",
    "$X = b, $Y = d",
    "$X = a, $Y = d",
]);
```
