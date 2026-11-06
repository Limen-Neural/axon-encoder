//! Optional NIR (Neuromorphic Intermediate Representation) parameter export.
//!
//! This module is gated behind the opt-in **`nir`** feature and built on the
//! maintained [`nir_rs`] graph model. It exports an encoder's **static
//! parameters** as a [`NirGraph`]; it never dumps a live [`SpikeEvent`] stream
//! and it never hand-rolls the NIR wire format. `nir-rs` owns `.nir` file
//! serialization, so this crate has no HDF5 dependency — see
//! [`write_nir`] for the one place where that boundary is visible.
//!
//! [`SpikeEvent`]: crate::types::SpikeEvent
//!
//! # Export path
//!
//! ```text
//! encoder.to_nir(channels)? -> nir_rs::NirGraph -> nir_rs::io::write
//! ```
//!
//! # What "defensible representation" means here
//!
//! NIR models *spike-to-spike* network graphs. `axon-encoder` models
//! *analog-to-spike* encoding, and most of its encoders have no NIR node that
//! means the same thing. A mapping is implemented only when a NIR node's
//! parameter set is the encoder's own parameter set, and it is documented as a
//! **parameter carrier** — not as a claim that a NIR simulator reproduces the
//! encoder's exact discrete-time dynamics.
//!
//! | Encoder | Mapped? | Reason |
//! | --- | --- | --- |
//! | [`EmbeddingRateEncoder`] | **Yes** — [`NirNode::If`] | Integrate-and-fire neuron: `config.v_th` is the node's `v_threshold`. |
//! | `RateEncoder` | No | Rate coding has no NIR node; `base_rate`/`max_rate` are not NIR parameters. |
//! | `DeltaEncoder` | No | Threshold applies to a *change* signal; NIR has no difference node, so only a fragment would be representable. |
//! | `BaselineHoldTernaryEncoder` | No | Ternary signed events have no NIR node. |
//! | `LatencyEncoder` | No | Input-dependent spike *time* is not NIR's fixed `Delay`. |
//! | `PhaseEncoder` | No | `cycle_steps` has no NIR counterpart. |
//! | `PopulationEncoder` | No | Analytic Gaussian tuning curves carry no weight matrix to export. |
//! | `TemporalEncoder` | No | Pattern detection over history has no NIR node. |
//! | `PredictiveEncoder` | No | Prediction error has no NIR node. |
//! | `DerivativeEncoder` | No | Rate-of-change detection has no NIR node. |
//! | `PoissonEncoder` | No | Sampling, not a graph; it implements no [`Encoder`](crate::Encoder). |
//!
//! Adding an encoder here is a deliberate act: implement [`ToNir`], record the
//! deviation from the encoder's real dynamics in the rustdoc, and extend the
//! table above.
//!
//! # Provenance metadata
//!
//! Every graph this module builds carries provenance under the
//! [`NIR_METADATA_PREFIX`] key prefix, so a consumer can tell an exported
//! parameter graph apart from a hand-authored network and recover which encoder
//! produced it.
//!
//! # Example
//!
//! ```
//! use axon_encoder::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let encoder = EmbeddingRateEncoder::try_new(3, EmbeddingEncoderConfig { v_th: 0.4 })?;
//! let graph = encoder.to_nir(3)?;
//!
//! // `input` -> `encoder` -> `output`, one node per graph port.
//! assert_eq!(graph.len(), 3);
//! graph.validate_structure().expect("well-formed graph");
//! # Ok(())
//! # }
//! ```
//!
//! # Writing a `.nir` file
//!
//! [`write_nir`] delegates to [`nir_rs::io::write`], which exists in every
//! `nir-rs` build but only serializes when `nir-rs`'s `hdf5` feature is
//! enabled. Enable it in your own dependency and Cargo unifies the features.
//! The `nir` feature is not in the published 0.6.0, so until the next release
//! depend on the repository directly:
//!
//! ```toml
//! [dependencies]
//! axon-encoder = { git = "https://github.com/Limen-Neural/axon-encoder", features = ["nir"] }
//! nir-rs = { version = "0.4", features = ["hdf5"] }
//! ```
//!
//! Without that feature, `write_nir` returns
//! [`NirExportError::Write`] wrapping [`NirError::Unimplemented`] instead of
//! silently doing nothing.
//!
//! # Manual consumer gate
//!
//! CI runs no Python step, so acceptance by an independent consumer is a
//! documented manual check rather than an automated one. It has been performed
//! for the mapping in this module, against the upstream reference
//! implementation:
//!
//! | | |
//! | --- | --- |
//! | Producer | `axon-encoder` `nir` feature + `nir-rs` 0.4.5 `hdf5` |
//! | Consumer | Python `nir` 1.0.8 (`nir.read`) |
//! | Result | Loads; `input`/`encoder`/`output` nodes and both edges present; `IF` reports `v_threshold` and `r`; graph and node metadata preserved; `validate_structure()` passes |
//!
//! Reproduce it when a mapping changes:
//!
//! ```text
//! # 1. Write a graph (requires nir-rs/hdf5, see above).
//! cargo run --example nir_export --features nir -- /tmp/encoder.nir
//!
//! # 2. Read it back through the reference implementation.
//! pip install "nir==1.0.8"
//! python -c "import nir; print(nir.read('/tmp/encoder.nir').nodes)"
//! ```
//!
//! A successful load that lists the `input`, `encoder`, and `output` node names
//! is the acceptance check. `nir-rs` reads `/version` permissively (it stores
//! whatever the producer wrote), so the version string is not part of it.
//!
//! # Fidelity through the wire
//!
//! Serializing and reloading preserves graph **semantics**: node names and
//! types, ordered edges, parameter values, and metadata. Node *insertion order*
//! is not preserved — `nir-rs` documents this for its HDF5 round-trip — so
//! consumers must look nodes up by name rather than by position.

use crate::encoder::EmbeddingRateEncoder;
use nir_rs::nodes::{If, Input, Output};
use nir_rs::types::{MetadataMap, MetadataValue, Tensor};
use std::fmt;
use std::path::Path;

/// Re-exported `nir-rs` types needed to consume an exported graph.
///
/// The rest of `nir-rs` (nodes, tensors, metadata, read options) is available
/// by depending on `nir-rs` directly.
pub use nir_rs::{NirError, NirGraph, NirNode};

/// Metadata key prefix for provenance entries written by this crate.
///
/// Keys are `"{NIR_METADATA_PREFIX}.{field}"`, for example
/// `"axon_encoder.encoder"`.
pub const NIR_METADATA_PREFIX: &str = "axon_encoder";

/// Name of the NIR node carrying the encoder parameters.
pub const ENCODER_NODE_NAME: &str = "encoder";

/// Error returned by [`ToNir::to_nir`] and [`write_nir`].
///
/// Kept separate from [`EncoderError`](crate::EncoderError): that type reports
/// invalid *encoder construction* parameters, while this one reports a failed
/// *export* of an otherwise valid encoder.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum NirExportError {
    /// The requested channel count is zero, which has no NIR `Input`/`Output` shape.
    EmptyChannelCount,
    /// The requested channel count disagrees with the encoder's own width.
    ///
    /// Exporting a graph wider or narrower than the encoder would describe a
    /// network the encoder does not implement, so the mismatch is rejected
    /// rather than silently resized.
    ChannelCountMismatch {
        /// Channels the encoder actually tracks.
        encoder_channels: usize,
        /// Channels the caller asked the graph to describe.
        requested: usize,
    },
    /// The requested channel count exceeds the addressable spike-channel range.
    ///
    /// Spikes carry `u16` channel IDs, so an export above that range could not
    /// describe anything the encoder can emit.
    ChannelCountTooLarge {
        /// Channels the caller asked the graph to describe.
        requested: usize,
        /// Largest supported channel count.
        maximum: usize,
    },
    /// An encoder parameter is not exportable (for example a non-finite or
    /// non-positive `v_th` set through the public `config` field).
    InvalidParameter {
        /// Name of the offending parameter.
        parameter: &'static str,
        /// Its current value.
        value: f32,
    },
    /// `nir-rs` rejected a graph or tensor this crate constructed.
    ///
    /// A construction failure is a bug in the exporter, not in the caller's
    /// configuration.
    Graph(NirError),
    /// `nir-rs` could not serialize the graph.
    ///
    /// Most often [`NirError::Unimplemented`], returned when the linked
    /// `nir-rs` build has no `hdf5` feature, or an underlying HDF5 I/O error.
    Write(NirError),
}

impl fmt::Display for NirExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyChannelCount => {
                write!(f, "channel count must be greater than 0 for a NIR export")
            }
            Self::ChannelCountMismatch {
                encoder_channels,
                requested,
            } => write!(
                f,
                "requested {requested} channel(s) but the encoder tracks {encoder_channels}"
            ),
            Self::ChannelCountTooLarge { requested, maximum } => write!(
                f,
                "requested {requested} channel(s) exceeds the maximum of {maximum}"
            ),
            Self::InvalidParameter { parameter, value } => write!(
                f,
                "encoder parameter `{parameter}` = {value} cannot be exported"
            ),
            Self::Graph(error) => write!(f, "failed to build NIR graph: {error}"),
            Self::Write(error) => write!(f, "failed to write NIR graph: {error}"),
        }
    }
}

impl std::error::Error for NirExportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(error) | Self::Write(error) => Some(error),
            Self::EmptyChannelCount
            | Self::ChannelCountMismatch { .. }
            | Self::ChannelCountTooLarge { .. }
            | Self::InvalidParameter { .. } => None,
        }
    }
}

/// Exports an encoder's static parameters as a [`NirGraph`].
///
/// Implemented only for encoders with a defensible NIR representation; see the
/// [module docs](self) for the audited list and the reasoning per encoder.
///
/// The `channels` argument is the encoding width the consumer wants described.
/// An encoder with a fixed width rejects a mismatch with
/// [`NirExportError::ChannelCountMismatch`] rather than exporting a graph it
/// does not implement.
///
/// # Errors
///
/// - [`NirExportError::EmptyChannelCount`] when `channels == 0`
/// - [`NirExportError::ChannelCountTooLarge`] when `channels` exceeds the
///   `u16` channel-ID range
/// - [`NirExportError::ChannelCountMismatch`] for a fixed-width encoder
/// - [`NirExportError::InvalidParameter`] for a non-exportable parameter
/// - [`NirExportError::Graph`] if `nir-rs` rejects a constructed node
pub trait ToNir {
    /// Builds the parameter graph for `channels` channels.
    fn to_nir(&self, channels: usize) -> Result<NirGraph, NirExportError>;

    /// Builds the parameter graph and writes it to `path`.
    ///
    /// Convenience for `write_nir(path, &self.to_nir(channels)?)`. Serialization
    /// requires `nir-rs`'s `hdf5` feature; see the [module docs](self).
    fn write_nir(&self, path: impl AsRef<Path>, channels: usize) -> Result<(), NirExportError> {
        let graph = self.to_nir(channels)?;
        write_nir(path, &graph)
    }
}

/// Writes `graph` as a `.nir` file through `nir-rs`.
///
/// This crate does not serialize NIR itself: the wire format is `nir-rs`'s
/// responsibility, and delegating keeps a single implementation of it. The call
/// fails with [`NirExportError::Write`] wrapping [`NirError::Unimplemented`]
/// when the linked `nir-rs` build has no `hdf5` feature.
///
/// # Errors
///
/// Propagates any [`NirError`] from [`nir_rs::io::write`].
pub fn write_nir(path: impl AsRef<Path>, graph: &NirGraph) -> Result<(), NirExportError> {
    nir_rs::io::write(path, graph).map_err(NirExportError::Write)
}

/// Rejects channel counts no encoder could describe.
fn validate_channels(channels: usize) -> Result<(), NirExportError> {
    if channels == 0 {
        return Err(NirExportError::EmptyChannelCount);
    }
    let maximum = crate::error::MAX_SPIKE_CHANNELS;
    if channels > maximum {
        return Err(NirExportError::ChannelCountTooLarge {
            requested: channels,
            maximum,
        });
    }
    Ok(())
}

/// Provenance entries identifying the encoder that produced a graph.
fn provenance(encoder: &'static str, channels: usize) -> MetadataMap {
    let mut metadata = MetadataMap::new();
    metadata.insert(
        format!("{NIR_METADATA_PREFIX}.encoder"),
        MetadataValue::String(encoder.to_owned()),
    );
    metadata.insert(
        format!("{NIR_METADATA_PREFIX}.channels"),
        MetadataValue::I64(channels as i64),
    );
    metadata
}

/// Wraps `node` between an `Input` and an `Output` port of `channels` width.
///
/// Every mapping in this module uses this one shape, so all exported graphs
/// have the same ports and the encoder-specific node sits in the middle under
/// [`ENCODER_NODE_NAME`].
fn build_graph(
    encoder: &'static str,
    channels: usize,
    node: NirNode,
) -> Result<NirGraph, NirExportError> {
    let mut graph = NirGraph::new();
    graph.metadata = provenance(encoder, channels);

    let port_shape = vec![channels];
    graph
        .insert_node(
            "input",
            NirNode::Input(Input {
                shape: port_shape.clone(),
                metadata: MetadataMap::new(),
            }),
        )
        .map_err(NirExportError::Graph)?;
    graph
        .insert_node(ENCODER_NODE_NAME, node)
        .map_err(NirExportError::Graph)?;
    graph
        .insert_node(
            "output",
            NirNode::Output(Output {
                shape: port_shape,
                metadata: MetadataMap::new(),
            }),
        )
        .map_err(NirExportError::Graph)?;

    graph.add_edge("input", ENCODER_NODE_NAME);
    graph.add_edge(ENCODER_NODE_NAME, "output");

    // The exporter owns this graph, so a structural failure here is an
    // exporter bug; validating now means consumers never receive a graph that
    // fails `nir-rs`'s own structural check.
    graph.validate_structure().map_err(NirExportError::Graph)?;
    Ok(graph)
}

/// `EmbeddingRateEncoder` exports as a NIR integrate-and-fire neuron.
///
/// The encoder accumulates each call's input into a persistent membrane
/// potential and fires when a channel crosses `config.v_th`; NIR's
/// [`NirNode::If`] integrates its input current into a membrane potential and
/// fires at `v_threshold`. The mapping is therefore direct:
///
/// | `EmbeddingRateEncoder` | NIR `IF` field |
/// | --- | --- |
/// | `config.v_th` | `v_threshold` |
/// | per-call input addition | `r = 1` (unit resistance) |
/// | `num_channels()` | `Input`/`Output` shape and parameter width |
///
/// # Documented deviations
///
/// The exported graph carries the encoder's **threshold parameter**; it is not
/// a behavioural claim that a NIR simulator reproduces this encoder exactly.
/// Two differences are recorded in the node metadata under
/// [`NIR_METADATA_PREFIX`]:
///
/// - **Reset.** The encoder soft-resets (`v -= v_th`); NIR `IF` resets to
///   `v_reset`, so `v_reset` is left absent and the wire/`nir-rs` default of
///   zeros applies.
/// - **Time.** The encoder is discrete per call and owns no clock, while NIR
///   integrates continuously. The exported graph says nothing about a step
///   duration.
///
/// # Errors
///
/// - [`NirExportError::ChannelCountMismatch`] when `channels` is not
///   [`EmbeddingRateEncoder::num_channels`]
/// - [`NirExportError::EmptyChannelCount`] / [`NirExportError::ChannelCountTooLarge`]
/// - [`NirExportError::InvalidParameter`] when `config.v_th` is no longer finite
///   and positive (the field is public and can be mutated after construction)
/// - [`NirExportError::Graph`] if `nir-rs` rejects a constructed tensor
///
/// # Example
///
/// ```
/// use axon_encoder::prelude::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let encoder = EmbeddingRateEncoder::try_new(2, EmbeddingEncoderConfig { v_th: 0.25 })?;
/// let graph = encoder.to_nir(2)?;
///
/// let node = graph.get("encoder").expect("encoder node");
/// let nir_rs::NirNode::If(neuron) = node else {
///     panic!("expected an IF node");
/// };
/// assert_eq!(neuron.v_threshold.numel(), 2);
/// # Ok(())
/// # }
/// ```
impl ToNir for EmbeddingRateEncoder {
    fn to_nir(&self, channels: usize) -> Result<NirGraph, NirExportError> {
        validate_channels(channels)?;

        let encoder_channels = self.num_channels();
        if channels != encoder_channels {
            return Err(NirExportError::ChannelCountMismatch {
                encoder_channels,
                requested: channels,
            });
        }

        // `config` is public, so `v_th` may have been mutated since
        // construction; re-check the constructor's invariant before exporting.
        if !(self.config.v_th.is_finite() && self.config.v_th > 0.0) {
            return Err(NirExportError::InvalidParameter {
                parameter: "v_th",
                value: self.config.v_th,
            });
        }

        let shape = vec![channels];
        let v_threshold = Tensor::from_f32(shape.clone(), vec![self.config.v_th; channels])
            .map_err(NirExportError::Graph)?;
        let r = Tensor::from_f32(shape, vec![1.0_f32; channels]).map_err(NirExportError::Graph)?;

        let mut metadata = provenance("EmbeddingRateEncoder", channels);
        metadata.insert(
            format!("{NIR_METADATA_PREFIX}.reset"),
            MetadataValue::String("soft_reset_subtracts_v_threshold".to_owned()),
        );
        metadata.insert(
            format!("{NIR_METADATA_PREFIX}.time_model"),
            MetadataValue::String("discrete_per_call".to_owned()),
        );

        build_graph(
            "EmbeddingRateEncoder",
            channels,
            NirNode::If(If {
                r,
                v_threshold,
                v_reset: None,
                metadata,
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::EmbeddingEncoderConfig;

    fn encoder(channels: usize, v_th: f32) -> EmbeddingRateEncoder {
        EmbeddingRateEncoder::try_new(channels, EmbeddingEncoderConfig { v_th })
            .expect("valid EmbeddingRateEncoder")
    }

    #[test]
    fn exports_input_encoder_output_with_ordered_edges() {
        let graph = encoder(3, 0.4).to_nir(3).expect("export succeeds");

        let names: Vec<&str> = graph.nodes.keys().map(String::as_str).collect();
        assert_eq!(names, ["input", ENCODER_NODE_NAME, "output"]);
        assert_eq!(
            graph.edges,
            vec![
                ("input".to_owned(), ENCODER_NODE_NAME.to_owned()),
                (ENCODER_NODE_NAME.to_owned(), "output".to_owned()),
            ]
        );
    }

    #[test]
    fn exports_ports_at_the_requested_width() {
        let graph = encoder(4, 0.4).to_nir(4).expect("export succeeds");

        let NirNode::Input(input) = graph.get("input").expect("input node") else {
            panic!("input node has the wrong type");
        };
        let NirNode::Output(output) = graph.get("output").expect("output node") else {
            panic!("output node has the wrong type");
        };
        assert_eq!(input.shape, vec![4]);
        assert_eq!(output.shape, vec![4]);
    }

    #[test]
    fn maps_v_th_onto_the_if_threshold() {
        let graph = encoder(2, 0.25).to_nir(2).expect("export succeeds");

        let NirNode::If(neuron) = graph.get(ENCODER_NODE_NAME).expect("encoder node") else {
            panic!("encoder node has the wrong type");
        };
        assert_eq!(neuron.v_threshold.shape(), &[2]);
        let nir_rs::types::TensorData::F32(values) = neuron.v_threshold.data() else {
            panic!("v_threshold is not f32");
        };
        assert_eq!(values, &[0.25, 0.25]);
        assert_eq!(neuron.r.numel(), 2);
        assert!(neuron.v_reset.is_none(), "NIR default reset applies");
    }

    #[test]
    fn records_provenance_on_graph_and_node() {
        let graph = encoder(1, 0.5).to_nir(1).expect("export succeeds");

        assert_eq!(
            graph
                .metadata
                .get(&format!("{NIR_METADATA_PREFIX}.encoder")),
            Some(&MetadataValue::String("EmbeddingRateEncoder".to_owned()))
        );
        assert_eq!(
            graph
                .metadata
                .get(&format!("{NIR_METADATA_PREFIX}.channels")),
            Some(&MetadataValue::I64(1))
        );

        let NirNode::If(neuron) = graph.get(ENCODER_NODE_NAME).expect("encoder node") else {
            panic!("encoder node has the wrong type");
        };
        assert_eq!(
            neuron
                .metadata
                .get(&format!("{NIR_METADATA_PREFIX}.reset"))
                .cloned(),
            Some(MetadataValue::String(
                "soft_reset_subtracts_v_threshold".to_owned()
            ))
        );
    }

    #[test]
    fn exported_graph_passes_nir_rs_validation() {
        let graph = encoder(5, 0.1).to_nir(5).expect("export succeeds");

        graph.validate_structure().expect("structure is valid");
        graph.validate_parameters().expect("parameters are valid");
    }

    #[test]
    fn rejects_channel_count_mismatch() {
        let error = encoder(3, 0.4).to_nir(2).expect_err("mismatch is rejected");

        assert_eq!(
            error,
            NirExportError::ChannelCountMismatch {
                encoder_channels: 3,
                requested: 2,
            }
        );
    }

    #[test]
    fn rejects_v_th_mutated_to_an_invalid_value() {
        for v_th in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut enc = encoder(1, 0.4);
            enc.config.v_th = v_th;
            assert!(matches!(
                enc.to_nir(1),
                Err(NirExportError::InvalidParameter {
                    parameter: "v_th",
                    ..
                })
            ));
        }
    }

    #[test]
    fn rejects_zero_channels() {
        let error = encoder(1, 0.4).to_nir(0).expect_err("zero is rejected");

        assert_eq!(error, NirExportError::EmptyChannelCount);
    }

    #[test]
    fn rejects_channels_above_the_u16_id_range() {
        let error = encoder(1, 0.4)
            .to_nir(crate::error::MAX_SPIKE_CHANNELS + 1)
            .expect_err("oversized width is rejected");

        assert_eq!(
            error,
            NirExportError::ChannelCountTooLarge {
                requested: crate::error::MAX_SPIKE_CHANNELS + 1,
                maximum: crate::error::MAX_SPIKE_CHANNELS,
            }
        );
    }

    #[test]
    fn write_nir_surfaces_the_nir_rs_hdf5_boundary() {
        // This repository never enables `nir-rs`'s `hdf5` feature, so the
        // delegation must report `Unimplemented` rather than panic or claim
        // success. The same call serializes once a consumer enables it.
        let graph = encoder(1, 0.4).to_nir(1).expect("export succeeds");
        let path = std::env::temp_dir().join("axon_encoder_nir_boundary.nir");

        // Another workspace member may unify `nir-rs/hdf5` on; then the write
        // succeeds, which is also correct delegation.
        let result = write_nir(&path, &graph);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            result,
            Ok(()) | Err(NirExportError::Write(NirError::Unimplemented(_)))
        ));
    }

    #[test]
    fn write_nir_trait_method_delegates_to_to_nir() {
        let path = std::env::temp_dir().join("axon_encoder_nir_trait.nir");
        let result = encoder(1, 0.4).write_nir(&path, 1);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            result,
            Ok(()) | Err(NirExportError::Write(NirError::Unimplemented(_)))
        ));
    }

    #[test]
    fn write_nir_trait_method_reports_validation_before_writing() {
        let path = std::env::temp_dir().join("axon_encoder_nir_invalid.nir");
        let error = encoder(2, 0.4)
            .write_nir(&path, 9)
            .expect_err("mismatch is rejected");

        assert_eq!(
            error,
            NirExportError::ChannelCountMismatch {
                encoder_channels: 2,
                requested: 9,
            }
        );
    }

    #[test]
    fn error_display_and_source_are_wired() {
        use std::error::Error as _;

        let empty = NirExportError::EmptyChannelCount;
        assert!(empty.to_string().contains("greater than 0"));
        assert!(empty.source().is_none());

        let mismatch = NirExportError::ChannelCountMismatch {
            encoder_channels: 1,
            requested: 2,
        };
        assert!(mismatch.to_string().contains("tracks 1"));
        assert!(mismatch.source().is_none());

        let too_large = NirExportError::ChannelCountTooLarge {
            requested: 3,
            maximum: 2,
        };
        assert!(too_large.to_string().contains("exceeds the maximum of 2"));
        assert!(too_large.source().is_none());

        let invalid = NirExportError::InvalidParameter {
            parameter: "v_th",
            value: -1.0,
        };
        assert!(invalid.to_string().contains("`v_th`"));
        assert!(invalid.source().is_none());

        let graph = NirExportError::Graph(NirError::Unimplemented("io::read"));
        assert!(graph.to_string().contains("failed to build NIR graph"));
        assert!(graph.source().is_some());

        let write = NirExportError::Write(NirError::Unimplemented("io::write"));
        assert!(write.to_string().contains("failed to write NIR graph"));
        assert!(write.source().is_some());
    }
}
