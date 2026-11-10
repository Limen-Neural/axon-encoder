//! Public-surface tests for the opt-in `nir` feature.
//!
//! The mapping itself is unit-tested next to the implementation in
//! `src/nir.rs`; these tests drive the same API through the public prelude, the
//! way a downstream consumer (or the manual gate in the module docs) does.
//!
//! The `.nir` write path is exercised only up to the documented boundary: this
//! repository never enables `nir-rs`'s `hdf5` feature, so no test here links
//! HDF5 or requires libhdf5 to be installed. Write tests use a directory Cargo
//! owns for this test target (`CARGO_TARGET_TMPDIR`), never the shared system
//! temp directory.
#![cfg(feature = "nir")]

use axon_encoder::prelude::*;

fn encoder(channels: usize, v_th: f32) -> EmbeddingRateEncoder {
    EmbeddingRateEncoder::try_new(channels, EmbeddingEncoderConfig { v_th })
        .expect("valid EmbeddingRateEncoder")
}

#[test]
fn prelude_exposes_the_nir_export_surface() {
    // Naming each item proves the prelude re-exports resolve.
    use axon_encoder::prelude::{ENCODER_NODE_NAME, NIR_METADATA_PREFIX, NirExportError, ToNir};

    assert_eq!(NIR_METADATA_PREFIX, "axon_encoder");
    assert_eq!(ENCODER_NODE_NAME, "encoder");

    let graph = encoder(2, 0.3).to_nir(2).expect("export succeeds");
    assert_eq!(graph.len(), 3);

    let error: NirExportError = encoder(2, 0.3).to_nir(7).expect_err("mismatch is rejected");
    assert!(error.to_string().contains("tracks 2"));
}

#[test]
fn exported_graph_is_loadable_by_the_nir_rs_graph_model() {
    // A consumer receives `nir_rs::NirGraph` values, not a private wrapper, so
    // `nir-rs`'s own accessors and validators apply directly.
    let graph = encoder(6, 0.15).to_nir(6).expect("export succeeds");

    graph.validate_structure().expect("valid structure");
    graph.validate_parameters().expect("valid parameters");

    let node = graph.get("encoder").expect("encoder node");
    assert_eq!(node.type_name(), "IF");
    assert!(graph.get("input").is_some());
    assert!(graph.get("output").is_some());
}

#[test]
fn exported_nodes_use_wire_accurate_type_strings() {
    // Downstream consumers dispatch on the wire `type` string, so an exported
    // graph must use NIR's own names and never a marketing alias.
    let graph = encoder(2, 0.35).to_nir(2).expect("export succeeds");

    let types: Vec<&str> = graph.nodes.values().map(|node| node.type_name()).collect();
    assert_eq!(types, ["Input", "IF", "Output"]);
}

#[test]
fn repeated_exports_are_independent() {
    // Each call builds a fresh graph: no exporter state leaks between exports.
    let encoder = encoder(3, 0.2);
    let first = encoder.to_nir(3).expect("first export succeeds");
    let second = encoder.to_nir(3).expect("second export succeeds");

    assert_eq!(first, second);
}

#[test]
fn the_encoder_keeps_encoding_after_an_export() {
    // Exporting is a read-only projection of the encoder's parameters; it must
    // not consume, reset, or otherwise disturb the live membrane state. The
    // second call fires only if the first call's accumulation survived.
    let mut encoder = encoder(1, 1.0);

    let first = encoder.encode(&[0.6]);
    assert!(first.spikes.is_empty(), "0.6 is below the 1.0 threshold");

    let _ = encoder.to_nir(1).expect("export succeeds");

    let second = encoder.encode(&[0.6]);
    assert_eq!(second.spikes.len(), 1, "membrane state survived the export");
}

/// Fresh, test-owned output path under Cargo's per-target scratch directory.
fn scratch_path(test: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("nir_export_{}_{test}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch directory");
    dir.join("encoder.nir")
}

/// Without `nir-rs/hdf5` a write reports `Unimplemented`; if another workspace
/// member unifies that feature on, it succeeds. Anything else is a failure.
fn assert_hdf5_boundary(result: Result<(), NirExportError>, path: &std::path::Path) {
    let _ = std::fs::remove_dir_all(path.parent().expect("scratch parent"));
    assert!(matches!(
        result,
        Ok(()) | Err(NirExportError::Write(NirError::Unimplemented(_)))
    ));
}

#[test]
fn write_nir_surfaces_the_nir_rs_hdf5_boundary() {
    let graph = encoder(1, 0.4).to_nir(1).expect("export succeeds");
    let path = scratch_path("free_fn");

    assert_hdf5_boundary(write_nir(&path, &graph), &path);
}

#[test]
fn write_nir_trait_method_delegates_to_to_nir() {
    let path = scratch_path("trait_method");

    assert_hdf5_boundary(encoder(1, 0.4).write_nir(&path, 1), &path);
}
