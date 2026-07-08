use crate::prelude::*;

/// Encodes analog values as phase-locked spikes within a repeating oscillation cycle.
///
/// Each input channel produces exactly one positive spike per call, with the spike
/// timestamp positioned within the current cycle according to the normalized input value.
/// Higher values map to later phase bins.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PhaseEncoder {
    cycle_steps: u64,
    range: (f32, f32),
    current_phase: u64,
}

impl PhaseEncoder {
    /// Creates a new `PhaseEncoder`.
    ///
    /// # Panics
    ///
    /// Panics if `cycle_steps == 0` or `range.0 >= range.1`.
    pub fn new(cycle_steps: u64, range: (f32, f32)) -> Self {
        assert!(cycle_steps > 0, "cycle_steps must be greater than 0");
        assert!(range.0 < range.1, "range min must be less than range max");

        Self {
            cycle_steps,
            range,
            current_phase: 0,
        }
    }

    fn normalize(&self, value: f32) -> f32 {
        ((value - self.range.0) / (self.range.1 - self.range.0)).clamp(0.0, 1.0)
    }

    fn phase_offset(&self, normalized: f32) -> u64 {
        ((normalized * self.cycle_steps as f32).floor() as u64).min(self.cycle_steps - 1)
    }

    fn encode_current_cycle(&self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();

        for (channel, &value) in input.iter().enumerate() {
            let phase_offset = self.phase_offset(self.normalize(value));
            output.spikes.push(SpikeEvent {
                channel: channel as u16,
                timestamp: (self.current_phase + phase_offset) % self.cycle_steps,
                polarity: true,
            });
        }

        output
    }

    fn advance_phase(&mut self) {
        self.current_phase = (self.current_phase + 1) % self.cycle_steps;
    }
}

impl Encoder for PhaseEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let output = self.encode_current_cycle(input);
        self.advance_phase();
        output
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        let output = self.encode_current_cycle(input);
        self.advance_phase();
        output
    }

    fn reset(&mut self) {
        self.current_phase = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_mapping_clamps_and_quantizes() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 10.0));

        let output = encoder.encode(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
        let timestamps: Vec<u64> = output.spikes.iter().map(|spike| spike.timestamp).collect();
        let polarities: Vec<bool> = output.spikes.iter().map(|spike| spike.polarity).collect();

        assert_eq!(timestamps, vec![0, 0, 4, 7, 7]);
        assert_eq!(polarities, vec![true; 5]);
    }

    #[test]
    fn test_phase_advances_after_each_call() {
        let mut encoder = PhaseEncoder::new(4, (0.0, 1.0));

        assert_eq!(encoder.encode(&[0.0]).spikes[0].timestamp, 0);
        assert_eq!(encoder.encode(&[0.0]).spikes[0].timestamp, 1);
        assert_eq!(encoder.encode_step(&[0.0]).spikes[0].timestamp, 2);
        assert_eq!(encoder.encode_step(&[0.0]).spikes[0].timestamp, 3);
        assert_eq!(encoder.encode(&[0.0]).spikes[0].timestamp, 0);
    }

    #[test]
    fn test_reset_restores_initial_phase() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));

        encoder.encode(&[0.0]);
        encoder.encode(&[0.0]);
        encoder.reset();

        let output = encoder.encode(&[1.0]);
        assert_eq!(output.spikes[0].timestamp, 7);
    }

    #[test]
    fn test_empty_input_returns_no_spikes() {
        let mut encoder = PhaseEncoder::new(4, (0.0, 1.0));

        let output = encoder.encode(&[]);
        assert!(output.spikes.is_empty());

        let next_output = encoder.encode(&[0.0]);
        assert_eq!(next_output.spikes[0].timestamp, 1);
    }

    #[test]
    #[should_panic(expected = "cycle_steps must be greater than 0")]
    fn test_zero_cycle_steps_rejected() {
        let _ = PhaseEncoder::new(0, (0.0, 1.0));
    }

    #[test]
    #[should_panic(expected = "range min must be less than range max")]
    fn test_invalid_range_rejected() {
        let _ = PhaseEncoder::new(8, (1.0, 1.0));
    }

    #[test]
    fn test_encode_step_matches_encode() {
        let input = [2.5, 7.5];
        let mut encode_encoder = PhaseEncoder::new(8, (0.0, 10.0));
        let mut step_encoder = PhaseEncoder::new(8, (0.0, 10.0));

        assert_eq!(
            encode_encoder.encode(&input),
            step_encoder.encode_step(&input)
        );
        assert_eq!(
            encode_encoder.encode(&input),
            step_encoder.encode_step(&input)
        );
    }
}
