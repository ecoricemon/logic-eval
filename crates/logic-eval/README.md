# logic-eval

A simple logic evaluator.

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
