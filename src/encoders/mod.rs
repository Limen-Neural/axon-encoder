pub mod delta;
pub mod phase;
pub mod population;
pub mod predictive;
pub mod rate;
pub mod temporal;

pub use delta::{encode_deltas_to_spikes, DeltaEncoder};
pub use phase::PhaseEncoder;
pub use population::PopulationEncoder;
pub use predictive::PredictiveEncoder;
pub use rate::RateEncoder;
pub use temporal::TemporalEncoder;
