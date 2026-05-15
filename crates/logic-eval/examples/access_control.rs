//! A small access-control policy example.
//!
//! This example shows how facts and rules can answer permission questions such as "who can read the
//! handbook?" and "who can edit the report?".

use logic_eval::{parse_str, Database, StrInterner};

fn main() {
    let mut db = Database::new();
    let interner = StrInterner::new();

    // Facts describe users, roles, and documents.
    let policy = r#"
        role(alice, admin).
        role(bob, editor).
        role(carol, viewer).

        public(handbook).
        owns(carol, report).

        can_read($User, $Document) :- role($User, admin).
        can_read($User, $Document) :- role($User, editor), public($Document).
        can_read($User, $Document) :- owns($User, $Document).

        can_edit($User, $Document) :- role($User, admin).
        can_edit($User, $Document) :- owns($User, $Document).
    "#;

    db.insert_dataset(parse_str(policy, &interner).unwrap());
    db.commit();

    println!("Who can read the handbook?");
    {
        let query = parse_str("can_read($User, handbook).", &interner).unwrap();
        let mut results = db.query(query);
        while let Some(answer) = results.prove_next() {
            for assignment in answer {
                println!("- {}", assignment.rhs());
            }
        }
    }

    println!();
    println!("Who can edit the report?");
    {
        let query = parse_str("can_edit($User, report).", &interner).unwrap();
        let mut results = db.query(query);
        while let Some(answer) = results.prove_next() {
            for assignment in answer {
                println!("- {}", assignment.rhs());
            }
        }
    }
}
