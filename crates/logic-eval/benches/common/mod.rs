#![allow(dead_code)]

pub(crate) const NOISE_CANDIDATES_PER_REQUEST: usize = 24;
pub(crate) const REQUESTS: usize = 128;

pub(crate) fn type_name(index: usize) -> String {
    format!("ty{index}")
}

pub(crate) fn request_name(index: usize) -> String {
    format!("req{index}")
}

pub(crate) fn candidate_name(index: usize) -> String {
    format!("candidate{index}")
}

pub(crate) fn noise_candidate_name(request: usize, candidate: usize) -> String {
    format!("candidate_noise_{request}_{candidate}")
}

pub(crate) fn nested_type(name: &str) -> String {
    format!("vec(vec({name}))")
}

pub(crate) fn fact_count(requests: usize) -> usize {
    requests * (3 + NOISE_CANDIDATES_PER_REQUEST)
}

pub(crate) fn build_fact_source(requests: usize) -> String {
    let mut source = String::new();
    append_fact_clauses(&mut source, requests);
    source
}

pub(crate) fn build_source(requests: usize) -> String {
    let mut source = build_fact_source(requests);

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

pub(crate) fn representative_fact_clause(requests: usize) -> String {
    let index = requests.saturating_sub(1);
    format!(
        "candidate_value({}, {}, {}).",
        request_name(index),
        candidate_name(index),
        nested_type(&type_name(index))
    )
}

pub(crate) fn representative_rule_clause() -> &'static str {
    "
        candidate_match($Request, $Candidate) :-
            candidate_value($Request, $Candidate, $Value),
            expected_value($Request, $Expected),
            same_shape($Value, $Expected),
            enabled_candidate($Candidate).
        "
}

pub(crate) fn representative_rule_body_expr() -> &'static str {
    "
        candidate_value($Request, $Candidate, $Value),
        expected_value($Request, $Expected),
        same_shape($Value, $Expected),
        enabled_candidate($Candidate)
        "
}

pub(crate) fn representative_nested_term(requests: usize) -> String {
    let index = requests.saturating_sub(1);
    format!(
        "candidate_value({}, {}, {})",
        request_name(index),
        candidate_name(index),
        nested_type(&type_name(index))
    )
}

pub(crate) fn delimiter_heavy_expr(terms: usize) -> String {
    let mut expr = String::new();

    for i in 0..terms {
        if i > 0 {
            if i % 4 == 0 {
                expr.push_str("; ");
            } else {
                expr.push_str(", ");
            }
        }

        if i % 4 == 0 {
            expr.push('(');
        }

        expr.push_str(&format!(
            "candidate_value({}, {}, {})",
            request_name(i),
            candidate_name(i),
            nested_type(&type_name(i))
        ));

        if i % 4 == 3 || i + 1 == terms {
            expr.push(')');
        }
    }

    expr
}

pub(crate) fn unique_names(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("name{i}")).collect()
}

pub(crate) fn repeated_names(count: usize) -> Vec<&'static str> {
    const NAMES: &[&str] = &[
        "candidate_value",
        "expected_value",
        "enabled_candidate",
        "same_shape",
        "vec",
        "req",
        "candidate",
        "$Request",
        "$Candidate",
        "$Value",
        "$Expected",
    ];

    (0..count).map(|i| NAMES[i % NAMES.len()]).collect()
}

fn append_fact_clauses(source: &mut String, requests: usize) {
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
}
