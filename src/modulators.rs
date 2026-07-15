const EVENT_DOPAMINE_DECAY: f32 = 0.95;
const CORTISOL_DECAY: f32 = 0.90;
const ACETYLCHOLINE_DECAY: f32 = 0.99;
const TEMPO_DECAY: f32 = 0.98;
/// Allow true zero gain (full silence / zero threshold). Non-finite values map
/// to identity; values above this cap are clamped for numerical stability.
const MIN_GAIN_SCALE: f32 = 0.0;
const MAX_GAIN_SCALE: f32 = 1e4;

fn sanitize_gain_scale(scale: f32) -> f32 {
    if !scale.is_finite() {
        return 1.0;
    }

    scale.clamp(MIN_GAIN_SCALE, MAX_GAIN_SCALE)
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NeuroModulators {
    pub dopamine: f32,
    pub cortisol: f32,
    pub acetylcholine: f32,
    pub tempo: f32,
}

impl NeuroModulators {
    pub fn decay(&mut self) {
        self.dopamine = (self.dopamine * EVENT_DOPAMINE_DECAY).max(0.0);
        self.cortisol = (self.cortisol * CORTISOL_DECAY).max(0.0);
        self.acetylcholine = (self.acetylcholine * ACETYLCHOLINE_DECAY).max(0.0);
        self.tempo = (self.tempo * TEMPO_DECAY).max(0.0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct GainCurve {
    pub input_range: (f32, f32),
    pub output_range: (f32, f32),
}

impl GainCurve {
    pub fn new(input_range: (f32, f32), output_range: (f32, f32)) -> Self {
        assert!(
            input_range.0.is_finite() && input_range.1.is_finite() && input_range.0 < input_range.1,
            "input_range min must be less than max and finite"
        );
        assert!(
            output_range.0.is_finite() && output_range.1.is_finite(),
            "output_range values must be finite"
        );

        Self {
            input_range,
            output_range,
        }
    }

    pub fn identity() -> Self {
        Self {
            input_range: (0.0, 1.0),
            output_range: (1.0, 1.0),
        }
    }

    /// Returns whether this curve has a valid, finite, ordered input range.
    fn has_valid_input_range(&self) -> bool {
        self.input_range.0.is_finite()
            && self.input_range.1.is_finite()
            && self.input_range.0 < self.input_range.1
    }

    /// Evaluate the gain curve at the given modulator level.
    ///
    /// Negative levels are clamped to `input_range.0`. NaN or non-finite
    /// levels return the identity gain (1.0).
    pub fn evaluate(&self, level: f32) -> f32 {
        // Guard against NaN levels and invalid ranges that can arise from
        // public fields or bypassed constructors (e.g. deserialization).
        if !level.is_finite()
            || !self.has_valid_input_range()
            || !self.output_range.0.is_finite()
            || !self.output_range.1.is_finite()
        {
            return 1.0;
        }

        let clamped_level = level.clamp(self.input_range.0, self.input_range.1);
        // Use f64 for span to avoid overflow for valid f32 ranges (e.g., f32::MIN..f32::MAX).
        let span = (self.input_range.1 as f64) - (self.input_range.0 as f64);
        // span is guaranteed > 0 by has_valid_input_range
        let position = ((clamped_level as f64 - self.input_range.0 as f64) / span) as f32;

        // Use lerp form to avoid overflow when output_range spans nearly f32::MAX.
        let raw_scale = self.output_range.0 * (1.0 - position) + self.output_range.1 * position;

        sanitize_gain_scale(raw_scale)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for GainCurve {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            input_range: (f32, f32),
            output_range: (f32, f32),
        }

        let helper = Helper::deserialize(deserializer)?;

        if !helper.input_range.0.is_finite()
            || !helper.input_range.1.is_finite()
            || helper.input_range.0 >= helper.input_range.1
        {
            return Err(serde::de::Error::custom(
                "input_range min must be less than max and finite",
            ));
        }
        if !helper.output_range.0.is_finite() || !helper.output_range.1.is_finite() {
            return Err(serde::de::Error::custom(
                "output_range values must be finite",
            ));
        }

        Ok(Self {
            input_range: helper.input_range,
            output_range: helper.output_range,
        })
    }
}

impl Default for GainCurve {
    fn default() -> Self {
        Self::identity()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModulatorGainCurves {
    pub threshold: Option<GainCurve>,
    pub sensitivity: Option<GainCurve>,
    pub firing_rate: Option<GainCurve>,
    pub latency: Option<GainCurve>,
}

/// Scales produced by neuromodulator gain curves for each encoder component.
///
/// # Zero-gain semantics
///
/// The meaning of a 0.0 gain depends on the component:
/// - `threshold_scale = 0.0` → effective threshold is 0 → every input spikes (maximum sensitivity)
/// - `sensitivity_scale = 0.0` → output is suppressed (no spikes for PopulationEncoder)
/// - `firing_rate_scale = 0.0` → firing rate is 0 → no spikes (silence)
/// - `latency_scale = 0.0` → max_latency is 0 → all spikes at timestamp 0 (instant response)
///
/// This asymmetry is intentional and reflects the physical semantics of each component.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct EncodingGains {
    pub threshold_scale: f32,
    pub sensitivity_scale: f32,
    pub firing_rate_scale: f32,
    pub latency_scale: f32,
}

impl EncodingGains {
    pub fn identity() -> Self {
        Self {
            threshold_scale: 1.0,
            sensitivity_scale: 1.0,
            firing_rate_scale: 1.0,
            latency_scale: 1.0,
        }
    }

    fn sanitize(self) -> Self {
        Self {
            threshold_scale: sanitize_gain_scale(self.threshold_scale),
            sensitivity_scale: sanitize_gain_scale(self.sensitivity_scale),
            firing_rate_scale: sanitize_gain_scale(self.firing_rate_scale),
            latency_scale: sanitize_gain_scale(self.latency_scale),
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for EncodingGains {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            #[serde(default = "default_gain_scale")]
            threshold_scale: f32,
            #[serde(default = "default_gain_scale")]
            sensitivity_scale: f32,
            #[serde(default = "default_gain_scale")]
            firing_rate_scale: f32,
            #[serde(default = "default_gain_scale")]
            latency_scale: f32,
        }

        fn default_gain_scale() -> f32 {
            1.0
        }

        let helper = Helper::deserialize(deserializer)?;
        let gains = Self {
            threshold_scale: helper.threshold_scale,
            sensitivity_scale: helper.sensitivity_scale,
            firing_rate_scale: helper.firing_rate_scale,
            latency_scale: helper.latency_scale,
        };
        Ok(gains.sanitize())
    }
}

impl Default for EncodingGains {
    fn default() -> Self {
        Self::identity()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NeuromodulatorGainCurves {
    #[cfg_attr(feature = "serde", serde(default))]
    pub dopamine: ModulatorGainCurves,
    #[cfg_attr(feature = "serde", serde(default))]
    pub cortisol: ModulatorGainCurves,
    #[cfg_attr(feature = "serde", serde(default))]
    pub acetylcholine: ModulatorGainCurves,
    #[cfg_attr(feature = "serde", serde(default))]
    pub tempo: ModulatorGainCurves,
}

impl NeuromodulatorGainCurves {
    pub fn evaluate(&self, modulators: &NeuroModulators) -> EncodingGains {
        let mut gains = EncodingGains::identity();

        Self::apply_curves(&mut gains, self.dopamine, modulators.dopamine);
        Self::apply_curves(&mut gains, self.cortisol, modulators.cortisol);
        Self::apply_curves(&mut gains, self.acetylcholine, modulators.acetylcholine);
        Self::apply_curves(&mut gains, self.tempo, modulators.tempo);

        gains.sanitize()
    }

    fn apply_curves(gains: &mut EncodingGains, curves: ModulatorGainCurves, level: f32) {
        if let Some(curve) = curves.threshold {
            gains.threshold_scale *= curve.evaluate(level);
        }
        if let Some(curve) = curves.sensitivity {
            gains.sensitivity_scale *= curve.evaluate(level);
        }
        if let Some(curve) = curves.firing_rate {
            gains.firing_rate_scale *= curve.evaluate(level);
        }
        if let Some(curve) = curves.latency {
            gains.latency_scale *= curve.evaluate(level);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gain_curve_clamps_input_range() {
        let curve = GainCurve::new((0.0, 1.0), (0.5, 2.0));

        assert_eq!(curve.evaluate(-5.0), 0.5);
        assert_eq!(curve.evaluate(5.0), 2.0);
    }

    #[test]
    fn gain_curve_interpolates_wide_f32_range() {
        let curve = GainCurve::new((f32::MIN, f32::MAX), (0.0, 2.0));

        assert_eq!(curve.evaluate(f32::MIN), 0.0);
        assert_eq!(curve.evaluate(f32::MAX), 2.0);
        assert!((curve.evaluate(0.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gain_curve_sanitizes_invalid_outputs() {
        let curve = GainCurve::new((0.0, 1.0), (-2.0, 2.0));

        assert_eq!(curve.evaluate(0.0), MIN_GAIN_SCALE);
        assert_eq!(curve.evaluate(f32::NAN), 1.0);
    }

    #[test]
    fn gain_curve_allows_true_zero_output() {
        let curve = GainCurve::new((0.0, 1.0), (0.0, 1.0));
        assert_eq!(curve.evaluate(0.0), 0.0);
    }

    #[test]
    fn gain_curve_invalid_range_returns_identity() {
        // Bypass constructor the same way a bad public-field mutation would.
        let curve = GainCurve {
            input_range: (1.0, 1.0),
            output_range: (0.0, 2.0),
        };
        assert_eq!(curve.evaluate(0.5), 1.0);
    }

    #[test]
    fn neuromodulator_curves_compose_multiplicatively() {
        let curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                firing_rate: Some(GainCurve::new((0.0, 1.0), (1.0, 2.0))),
                ..Default::default()
            },
            cortisol: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            acetylcholine: ModulatorGainCurves {
                firing_rate: Some(GainCurve::new((0.0, 1.0), (1.0, 1.5))),
                ..Default::default()
            },
            tempo: ModulatorGainCurves {
                sensitivity: Some(GainCurve::new((0.0, 1.0), (1.0, 1.25))),
                ..Default::default()
            },
        };
        let modulators = NeuroModulators {
            dopamine: 1.0,
            cortisol: 1.0,
            acetylcholine: 1.0,
            tempo: 1.0,
        };

        let gains = curves.evaluate(&modulators);

        assert_eq!(gains.threshold_scale, 0.5);
        assert_eq!(gains.sensitivity_scale, 1.25);
        assert_eq!(gains.firing_rate_scale, 3.0);
        assert_eq!(gains.latency_scale, 1.0); // no latency curve set
    }

    #[cfg(feature = "serde")]
    #[test]
    fn gain_curve_rejects_invalid_deserialize() {
        let json = r#"{"input_range":[1.0,0.0],"output_range":[0.0,1.0]}"#;
        let err = serde_json::from_str::<GainCurve>(json).unwrap_err();
        assert!(err.to_string().contains("input_range"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn encoding_gains_deserialize_sanitizes_values() {
        // Use out-of-range values that serde_json can parse (NaN is not valid JSON)
        let json =
            r#"{"threshold_scale":-999.0,"sensitivity_scale":999999.0,"firing_rate_scale":0.5}"#;
        let gains: EncodingGains = serde_json::from_str(json).unwrap();
        assert_eq!(gains.threshold_scale, 0.0); // -999 clamped to MIN_GAIN_SCALE (0.0)
        assert_eq!(gains.sensitivity_scale, MAX_GAIN_SCALE); // 999999 clamped to MAX_GAIN_SCALE
        assert_eq!(gains.firing_rate_scale, 0.5); // in range, unchanged
        assert_eq!(gains.latency_scale, 1.0); // defaults to 1.0 when omitted
    }

    #[test]
    fn sanitize_gain_scale_handles_nan_and_infinity() {
        assert_eq!(sanitize_gain_scale(f32::NAN), 1.0);
        assert_eq!(sanitize_gain_scale(f32::INFINITY), 1.0);
        assert_eq!(sanitize_gain_scale(f32::NEG_INFINITY), 1.0);
        assert_eq!(sanitize_gain_scale(0.0), 0.0);
        assert_eq!(sanitize_gain_scale(5.0), 5.0);
        assert_eq!(sanitize_gain_scale(1e10), MAX_GAIN_SCALE);
    }

    #[test]
    fn neuro_modulators_decay() {
        let mut mods = NeuroModulators {
            dopamine: 1.0,
            cortisol: 1.0,
            acetylcholine: 1.0,
            tempo: 1.0,
        };
        mods.decay();
        assert!((mods.dopamine - 0.95).abs() < 1e-6);
        assert!((mods.cortisol - 0.90).abs() < 1e-6);
        assert!((mods.acetylcholine - 0.99).abs() < 1e-6);
        assert!((mods.tempo - 0.98).abs() < 1e-6);

        // Decay floors at zero
        mods.dopamine = -0.5;
        mods.decay();
        assert_eq!(mods.dopamine, 0.0);
    }

    #[test]
    fn gain_curve_identity_returns_constant_one() {
        let curve = GainCurve::identity();
        assert_eq!(curve.evaluate(0.0), 1.0);
        assert_eq!(curve.evaluate(0.5), 1.0);
        assert_eq!(curve.evaluate(1.0), 1.0);
    }

    #[test]
    fn gain_curve_evaluate_non_finite_output_range_returns_identity() {
        let curve = GainCurve {
            input_range: (0.0, 1.0),
            output_range: (f32::NAN, 2.0),
        };
        assert_eq!(curve.evaluate(0.5), 1.0);

        let curve2 = GainCurve {
            input_range: (0.0, 1.0),
            output_range: (1.0, f32::INFINITY),
        };
        assert_eq!(curve2.evaluate(0.5), 1.0);
    }

    #[test]
    fn encoding_gains_sanitize_clamps_extremes() {
        let gains = EncodingGains {
            threshold_scale: f32::NAN,
            sensitivity_scale: f32::INFINITY,
            firing_rate_scale: -1.0,
            latency_scale: 0.5,
        };
        let sanitized = gains.sanitize();
        assert_eq!(sanitized.threshold_scale, 1.0);
        assert_eq!(sanitized.sensitivity_scale, 1.0);
        assert_eq!(sanitized.firing_rate_scale, 0.0);
        assert_eq!(sanitized.latency_scale, 0.5);
    }

    #[test]
    fn neuromodulator_curves_all_none_returns_identity() {
        let curves = NeuromodulatorGainCurves::default();
        let mods = NeuroModulators::default();
        let gains = curves.evaluate(&mods);
        assert_eq!(gains.threshold_scale, 1.0);
        assert_eq!(gains.sensitivity_scale, 1.0);
        assert_eq!(gains.firing_rate_scale, 1.0);
        assert_eq!(gains.latency_scale, 1.0);
    }

    #[test]
    fn neuromodulator_curves_partial_none() {
        let curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 2.0))),
                ..Default::default()
            },
            ..Default::default()
        };
        let mods = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };
        let gains = curves.evaluate(&mods);
        assert_eq!(gains.threshold_scale, 2.0);
        assert_eq!(gains.sensitivity_scale, 1.0);
        assert_eq!(gains.firing_rate_scale, 1.0);
        assert_eq!(gains.latency_scale, 1.0);
    }

    #[test]
    fn modulator_gain_curves_default_is_none() {
        let curves = ModulatorGainCurves::default();
        assert!(curves.threshold.is_none());
        assert!(curves.sensitivity.is_none());
        assert!(curves.firing_rate.is_none());
    }

    #[test]
    fn gain_curve_default_is_identity() {
        assert_eq!(GainCurve::default(), GainCurve::identity());
    }

    #[test]
    fn encoding_gains_default_is_identity() {
        assert_eq!(EncodingGains::default(), EncodingGains::identity());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn neuromodulator_gain_curves_partial_json_deserializes() {
        // Only set dopamine; cortisol/acetylcholine/tempo should default
        let json = r#"{
            "dopamine": {
                "firing_rate": {"input_range": [0.0, 1.0], "output_range": [1.0, 2.0]}
            }
        }"#;
        let curves: NeuromodulatorGainCurves = serde_json::from_str(json).unwrap();
        assert!(curves.dopamine.firing_rate.is_some());
        assert!(curves.cortisol.threshold.is_none());
        assert!(curves.acetylcholine.sensitivity.is_none());
        assert!(curves.tempo.firing_rate.is_none());
    }
}
