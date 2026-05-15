//! A first logic-eval example for finding meals from pantry ingredients.
//!
//! This example uses simple facts like `has(pasta).` and rules like `can_make(spaghetti) :-
//! has(pasta), has(tomato).`.

use logic_eval::{parse_str, Database, StrInterner};

fn main() {
    let mut db = Database::new();
    let interner = StrInterner::new();

    // Facts describe what is true. Rules describe what can be inferred.
    let pantry = r#"
        has(pasta).
        has(tomato).
        has(cheese).

        can_make(spaghetti) :- has(pasta), has(tomato).
        can_make(grilled_cheese) :- has(bread), has(cheese).
        can_make(mac_and_cheese) :- has(pasta), has(cheese).
    "#;

    // Load the facts and rules into the database before asking questions.
    db.insert_dataset(parse_str(pantry, &interner).unwrap());
    db.commit();

    // $Meal is a variable. logic-eval will find every value that works.
    let query = parse_str("can_make($Meal).", &interner).unwrap();
    let mut results = db.query(query);

    println!("You can make:");
    while let Some(answer) = results.prove_next() {
        for assignment in answer {
            println!("- {}", assignment.rhs());
        }
    }
}
