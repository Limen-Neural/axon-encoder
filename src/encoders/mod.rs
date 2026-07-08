pub mod delta;
pub mod derivative;
pub mod latency;
pub mod population;
pub mod predictive;
pub mod rate;
pub mod temporal;

pub use delta::{encode_deltas_to_spikes, DeltaEncoder};
pub use derivative::DerivativeEncoder;
pub use latency::LatencyEncoder;
pub use population::PopulationEncoder;
pub use predictive::PredictiveEncoder;
pub use rate::RateEncoder;
pub use temporal::TemporalEncoder;
