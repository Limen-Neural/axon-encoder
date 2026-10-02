#![cfg(feature = "serde")]

use std::fmt::Debug;
use std::path::Path;

use serde::{Serialize, de::DeserializeOwned};

mod golden;
mod history;

fn fixture_text(file: &str, pair: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/serde")
        .join(file);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} ({pair}): {error}", path.display()))
}

fn read_fixture<T: DeserializeOwned>(file: &str, pair: &str) -> T {
    serde_json::from_str(&fixture_text(file, pair))
        .unwrap_or_else(|error| panic!("{file} ({pair}): expected load, got {error}"))
}

fn assert_json<T: Serialize>(value: &T, file: &str, pair: &str) {
    let expected: serde_json::Value = read_fixture(file, pair);
    // Compare the JSON text writer's shortest f32 representation, not
    // to_value's widened f64 (e.g. 0.1 becomes 0.10000000149011612 there).
    let text = serde_json::to_string(value)
        .unwrap_or_else(|error| panic!("{file} ({pair}): serialize failed: {error}"));
    let actual: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("{file} ({pair}): invalid serialized JSON: {error}"));
    assert_eq!(actual, expected, "{file} ({pair}): JSON schema/value drift");
}

fn assert_golden<T: DeserializeOwned + Serialize + PartialEq + Debug>(file: &str, expected: T) {
    let restored: T = read_fixture(file, "v0.5 writer -> HEAD reader");
    assert_eq!(
        restored, expected,
        "{file} (v0.5 writer -> HEAD reader): semantic drift"
    );
    assert_json(&expected, file, "HEAD writer -> v0.5 golden");
}

fn assert_unknown_fields<T: DeserializeOwned + Serialize>(file: &str, golden: &str) {
    let pair = "HEAD extended writer -> HEAD reader";
    let restored: T = read_fixture(file, pair);
    assert_json(
        &restored,
        golden,
        &format!("{pair}/writer via {file} (unknown field discarded)"),
    );
}

fn assert_rejected<T: DeserializeOwned + Debug>(file: &str, pair: &str, reason: &str) {
    let result = serde_json::from_str::<T>(&fixture_text(file, pair));
    let error = result.expect_err(&format!("{file} ({pair}): expected rejection"));
    assert!(
        error.to_string().contains(reason),
        "{file} ({pair}): expected {reason:?}, got {error}"
    );
}
