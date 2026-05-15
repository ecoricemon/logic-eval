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

### Quick start: use it in your own project

Install Rust from <https://rustup.rs/>, then create a small project:

```sh
cargo new logic-eval-quickstart
cd logic-eval-quickstart
cargo add logic-eval
```

Replace `src/main.rs` with:

```rust
use logic_eval::{parse_str, Database, StrInterner};

fn main() {
    let mut db = Database::new();
    let interner = StrInterner::new();

    let pantry = r#"
        has(pasta).
        has(tomato).
        has(cheese).

        can_make(spaghetti) :- has(pasta), has(tomato).
        can_make(grilled_cheese) :- has(bread), has(cheese).
        can_make(mac_and_cheese) :- has(pasta), has(cheese).
    "#;

    db.insert_dataset(parse_str(pantry, &interner).unwrap());
    db.commit();

    let query = parse_str("can_make($Meal).", &interner).unwrap();
    let mut results = db.query(query);

    println!("You can make:");
    while let Some(answer) = results.prove_next() {
        for assignment in answer {
            println!("- {}", assignment.rhs());
        }
    }
}
```

Run it:

```sh
cargo run
```

You should see:

```text
You can make:
- spaghetti
- mac_and_cheese
```

### More examples

The `crates/logic-eval/examples` directory has more complete examples:

- `meal_recommendation.rs`: facts and simple rules.
- `access_control.rs`: policy checks with multiple queries.
- `dependency_graph.rs`: recursive rules for direct and indirect dependencies.

Run any example with:

```sh
cargo run -p logic-eval --example dependency_graph
```

### Logic syntax

logic-eval programs are made from facts, rules, and queries.

Facts say what is true:

```prolog
has(pasta).
has(tomato).
public(handbook).
```

Rules say what can be inferred. The part before `:-` is true when the part after `:-` is true:

```prolog
can_make(spaghetti) :- has(pasta), has(tomato).
can_read($User, handbook) :- role($User, admin).
```

Queries ask what follows from the facts and rules:

```prolog
can_make($Meal).
can_read($User, handbook).
```

Names that start with `$`, like `$Meal` and `$User`, are variables. logic-eval tries to find every
value that makes the query true.

Use `,` for AND. This rule means "you can make spaghetti if you have pasta and tomato":

```prolog
can_make(spaghetti) :- has(pasta), has(tomato).
```

Use `;` for OR. This rule means "a user can open a document if they can read it or edit it":

```prolog
can_open($User, $Document) :- can_read($User, $Document); can_edit($User, $Document).
```

Use `\+` for NOT. This rule means "a document needs review if it is a draft and is not approved":

```prolog
needs_review($Document) :- draft($Document), \+ approved($Document).
```

Use parentheses when you want to make grouping explicit:

```prolog
can_open($User, $Document) :-
    role($User, admin);
    (can_read($User, $Document), \+ archived($Document)).
```
