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

#[derive(Debug, Clone, Copy, Default)]
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
            input_range.0.is_finite()
                && input_range.1.is_finite()
                && input_range.0 < input_range.1,
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
        let span = self.input_range.1 - self.input_range.0;
        // span is guaranteed > 0 by has_valid_input_range
        let position = (clamped_level - self.input_range.0) / span;
        let raw_scale =
            self.output_range.0 + position * (self.output_range.1 - self.output_range.0);

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
            return Err(serde::de::Error::custom("output_range values must be finite"));
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EncodingGains {
    pub threshold_scale: f32,
    pub sensitivity_scale: f32,
    pub firing_rate_scale: f32,
}

impl EncodingGains {
    pub fn identity() -> Self {
        Self {
            threshold_scale: 1.0,
            sensitivity_scale: 1.0,
            firing_rate_scale: 1.0,
        }
    }

    fn sanitize(self) -> Self {
        Self {
            threshold_scale: sanitize_gain_scale(self.threshold_scale),
            sensitivity_scale: sanitize_gain_scale(self.sensitivity_scale),
            firing_rate_scale: sanitize_gain_scale(self.firing_rate_scale),
        }
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
    pub dopamine: ModulatorGainCurves,
    pub cortisol: ModulatorGainCurves,
    pub acetylcholine: ModulatorGainCurves,
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
    }

    #[cfg(feature = "serde")]
    #[test]
    fn gain_curve_rejects_invalid_deserialize() {
        let json = r#"{"input_range":[1.0,0.0],"output_range":[0.0,1.0]}"#;
        let err = serde_json::from_str::<GainCurve>(json).unwrap_err();
        assert!(err.to_string().contains("input_range"));
    }
}
