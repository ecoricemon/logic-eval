//! A recursive dependency graph example.
//!
//! This example shows how a recursive rule can find both direct and indirect package dependencies.

use logic_eval::{parse_str, Database, StrInterner};

fn main() {
    let mut db = Database::new();
    let interner = StrInterner::new();

    // Direct dependencies are facts. The recursive rule finds indirect dependencies.
    let graph = r#"
        depends(app, api).
        depends(app, web).
        depends(api, db).
        depends(api, auth).
        depends(web, ui).
        depends(auth, crypto).

        requires($Package, $Dependency) :- depends($Package, $Dependency).
        requires($Package, $Dependency) :-
            depends($Package, $Direct),
            requires($Direct, $Dependency).
    "#;

    db.insert_dataset(parse_str(graph, &interner).unwrap());
    db.commit();

    println!("app requires:");
    {
        let query = parse_str("requires(app, $Package).", &interner).unwrap();
        let mut results = db.query(query);
        while let Some(answer) = results.prove_next() {
            for assignment in answer {
                println!("- {}", assignment.rhs());
            }
        }
    }

    println!();
    println!("Packages that eventually require crypto:");
    {
        let query = parse_str("requires($Package, crypto).", &interner).unwrap();
        let mut results = db.query(query);
        while let Some(answer) = results.prove_next() {
            for assignment in answer {
                println!("- {}", assignment.rhs());
            }
        }
    }
}
