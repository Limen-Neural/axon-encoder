//! Population: preferred-value grid and center-of-mass endpoint bias.

use axon_encoder::prelude::*;

/// Reference-oracle configuration for a single population, bundling the three
/// primitives (`n`, `range`, `width`) the helpers used to pass around
/// individually. Grouping them removes the 5-argument `response` helper and the
/// primitive-argument churn CodeScene flagged, without changing any behavior.
struct PopulationOracle {
    n: usize,
    range: (f32, f32),
    width: f64,
}

impl PopulationOracle {
    /// The preferred value of neuron `i` in this `n`-neuron population.
    ///
    /// This mirrors the CURRENT source (`src/encoders/population.rs`): neuron 0
    /// is pinned to `min`, neuron `n - 1` to `max`, and the interior neurons sit
    /// on the endpoint-covering grid `min + (i / (n - 1)) * span`. This replaces
    /// the stale `i / n` grid in the LIM-1340 issue table (which put neuron 7 of
    /// 8 at 0.875); on the current grid neuron 7 of 8 sits at `1.0`.
    ///
    /// The interior term is computed in **f32 with the same operation order as
    /// `PopulationEncoder::get_rate_with_tuning_width`** (`range.0 + (i as f32 /
    /// (n - 1) as f32) * span`), then widened to f64. Computing it directly in
    /// f64 would diverge from the encoder for large-magnitude ranges, where the
    /// f32 rounding shifts a preferred value by several units.
    fn preferred(&self, i: usize) -> f64 {
        let (min, max) = self.range;
        if self.n == 1 || i == 0 {
            min as f64
        } else if i == self.n - 1 {
            max as f64
        } else {
            let span = max - min;
            let preferred = min + (i as f32 / (self.n - 1) as f32) * span;
            preferred as f64
        }
    }

    /// The effective tuning width the encoder's unmodulated (identity,
    /// `sensitivity_scale == 1.0`) path actually uses.
    ///
    /// `PopulationEncoder::effective_tuning_width(1.0)` takes the `>= 1.0`
    /// branch and returns `(self.tuning_width / 1.0).max(f32::EPSILON)` ==
    /// `self.tuning_width.max(f32::EPSILON)`: it FLOORS the configured width at
    /// `f32::EPSILON` (see `src/encoders/population.rs`). Any configured width in
    /// `(0, f32::EPSILON)` is therefore clamped up to `f32::EPSILON` before the
    /// Gaussian is evaluated. The oracle mirrors that exact floor so it does not
    /// diverge from the production encoder for sub-epsilon widths; for widths
    /// already `>= f32::EPSILON` (e.g. 0.1, 0.2) this is a no-op.
    fn effective_width(&self) -> f64 {
        self.width.max(f32::EPSILON as f64)
    }

    /// Gaussian tuning response of neuron `i` to `input`, matching the encoder's
    /// `get_rate_with_tuning_width` evaluated at the encoder's effective width.
    ///
    /// The width used here is [`Self::effective_width`], which floors the
    /// configured width at `f32::EPSILON` exactly as
    /// `effective_tuning_width(1.0)` does on the encoder's unmodulated path.
    fn response(&self, input: f64, i: usize) -> f64 {
        let preferred = self.preferred(i);
        let distance = (input - preferred).abs();
        let width = self.effective_width();
        (-(distance * distance) / (2.0 * width * width)).exp()
    }

    /// Pooled center-of-mass decode: the tuning-weighted mean of preferred
    /// values.
    fn center_of_mass(&self, input: f64) -> f64 {
        let mut weighted = 0.0f64;
        let mut total = 0.0f64;
        for i in 0..self.n {
            let response = self.response(input, i);
            weighted += self.preferred(i) * response;
            total += response;
        }
        weighted / total
    }
}

#[test]
fn population_preferred_grid_covers_both_endpoints() {
    // Document and pin the ACTUAL preferred-value formula: i / (n - 1), not the
    // stale i / n from the issue table. For N = 8 over (0, 1), neuron 7 sits at
    // 1.0 (not 0.875), and neuron 0 sits at 0.0.
    let n = 8usize;
    let range = (0.0f32, 1.0f32);
    let oracle = PopulationOracle {
        n,
        range,
        width: 0.2,
    };

    assert_eq!(oracle.preferred(0), 0.0);
    assert_eq!(oracle.preferred(n - 1), 1.0);
    // Interior neuron on the i/(n-1) grid. The oracle computes the interior term
    // in f32 (like the encoder) then widens, so compare against the f32 value.
    let expected_4 = (4.0f32 / 7.0f32) as f64;
    assert!((oracle.preferred(4) - expected_4).abs() < 1e-12);
    // Neuron 7 of 8 is the pinned endpoint, exactly 1.0 (not the stale 0.875).
    assert_eq!(oracle.preferred(7), 1.0);

    // The stale i/N grid would have put neuron 7 at 0.875; the current grid does
    // not. This guards against regressing to the old formula.
    assert!((oracle.preferred(7) - 0.875).abs() > 0.1);
}

#[test]
fn population_peak_firer_pins_the_encoder_to_the_endpoint_grid() {
    // Regression guard that OBSERVES THE PRODUCTION ENCODER, not just the copied
    // helper: drive `PopulationEncoder::encode` thousands of times and assert
    // that the empirically most-active neuron matches the neuron the current
    // `i / (n - 1)` grid predicts. This is what ties the oracle to real output
    // so a grid change breaks the test.
    //
    // Setup: N = 10 over (0, 100), width 10, probing at x = 800 / 9, which is
    // exactly neuron 8's preferred value on the CURRENT grid. Two things follow:
    //
    //   * On the current `i / (n - 1)` grid, preferred[8] == x, so neuron 8
    //     fires with probability exp(0) = 1.0 on EVERY call (a deterministic
    //     anchor), while its nearest competitors (neurons 7 and 9) fire at only
    //     ~0.54. Neuron 8 is therefore the unambiguous highest-frequency firer.
    //   * On the OLD `i / n` grid the same neurons would sit at 80 and 90, so
    //     the closest neuron to x = 88.9 would be neuron 9 (pref 90), not neuron
    //     8. The predicted peak differs between the two grids, so reverting the
    //     encoder to `i / n` moves the observed peak and fails this test.
    let n = 10usize;
    let range = (0.0f32, 100.0f32);
    let width = 10.0f32;
    let probe = 800.0f32 / 9.0; // == oracle.preferred(8) on the current grid.
    let mut encoder = PopulationEncoder::try_new(n, range, width).expect("valid population");
    let oracle = PopulationOracle {
        n,
        range,
        width: width as f64,
    };

    // The neuron whose preferred value is closest to the probe under the
    // current grid. Computed from the oracle helper, not hardcoded.
    let predicted_peak = (0..n)
        .min_by(|&a, &b| {
            let da = (oracle.preferred(a) - probe as f64).abs();
            let db = (oracle.preferred(b) - probe as f64).abs();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();
    assert_eq!(
        predicted_peak, 8,
        "current grid predicts neuron 8 as the peak"
    );

    // Empirical per-neuron firing counts over many independent encode calls.
    let iterations = 8_000u64;
    let mut counts = vec![0u64; n];
    for _ in 0..iterations {
        for spike in encoder.encode(&[probe]).spikes {
            counts[spike.channel as usize] += 1;
        }
    }

    // Neuron 8 fires with probability 1.0 on the current grid, so it must be
    // present on every call.
    assert_eq!(
        counts[predicted_peak], iterations,
        "neuron {predicted_peak} sits exactly on the probe and must fire every call"
    );

    // The empirically most-active neuron must be the one the current grid
    // predicts. On the old i/n grid the peak would land elsewhere (neuron 9),
    // so this assertion actually distinguishes the two grids.
    let observed_peak = (0..n).max_by_key(|&i| counts[i]).unwrap();
    assert_eq!(
        observed_peak, predicted_peak,
        "observed peak neuron {observed_peak} must match the i/(n-1) grid prediction {predicted_peak}; \
         counts = {counts:?}"
    );

    // The peak must be a strict, comfortable margin above its nearest rival so
    // sampling noise cannot flip the winner: neuron 8 fires ~2x as often as the
    // ~0.54-probability neurons 7 and 9.
    let runner_up = (0..n)
        .filter(|&i| i != observed_peak)
        .map(|i| counts[i])
        .max()
        .unwrap();
    assert!(
        counts[observed_peak] > runner_up + iterations / 4,
        "peak neuron {observed_peak} ({}) must dominate its runner-up ({runner_up}) by a wide margin",
        counts[observed_peak]
    );
}

#[test]
fn population_center_of_mass_is_biased_inward_at_the_maximum() {
    // Durable invariant (replaces the stale 0.951/0.875 numbers): for N > 1 the
    // center-of-mass decode at x = range max is STRICTLY LESS THAN max, because
    // every neuron below the top pulls the pooled mean down. This is the
    // endpoint-bias property a real decoder must account for.
    let range = (0.0f32, 1.0f32);
    // The strict inequality holds only where the neighbor Gaussian responses are
    // REPRESENTABLE (non-underflowing). width = 0.2 keeps every neighbor
    // response well above f64's smallest positive value, so the inward pull is
    // real. See `population_com_strict_bias_fails_under_narrow_width_underflow`
    // for the degenerate narrow-width case where this inequality does NOT hold.
    let width = 0.2f64;

    for n in [2usize, 4, 8, 16] {
        let oracle = PopulationOracle { n, range, width };
        let com_at_max = oracle.center_of_mass(range.1 as f64);
        assert!(
            com_at_max < range.1 as f64,
            "N={n}: CoM at max {com_at_max} must be strictly below {}",
            range.1
        );
        // Symmetrically, the CoM at the minimum is biased upward (strictly
        // above min), confirming the bias is an endpoint effect, not a global
        // offset.
        let com_at_min = oracle.center_of_mass(range.0 as f64);
        assert!(
            com_at_min > range.0 as f64,
            "N={n}: CoM at min {com_at_min} must be strictly above {}",
            range.0
        );
    }

    // Interior points are reconstructed with far smaller error than the
    // endpoints: the bias is specific to the range edges.
    let interior = 0.5f64;
    let interior_oracle = PopulationOracle { n: 8, range, width };
    let com_interior = interior_oracle.center_of_mass(interior);
    assert!(
        (com_interior - interior).abs() < 0.05,
        "interior CoM {com_interior} should closely track {interior}"
    );
}

#[test]
fn population_com_strict_bias_fails_under_narrow_width_underflow() {
    // Documents the LIMIT of the inward-bias invariant: for very narrow tuning
    // widths the lower neurons' Gaussian response underflows to exactly 0.0, so
    // their inward pull vanishes and the CoM at max equals max EXACTLY. A
    // downstream decoder must not rely on the strict inequality outside the
    // representable-neighbor regime.
    //
    // n = 2 over (0, 1), width = 0.01: the lower neuron sits at 0.0, so at
    // input = 1.0 its response is exp(-(1.0)^2 / (2 * 0.01^2)) = exp(-5000),
    // which underflows to 0.0 in f64. Only the top neuron (preferred 1.0,
    // response 1.0) contributes, so CoM == 1.0 == max, NOT strictly below it.
    let range = (0.0f32, 1.0f32);
    let oracle = PopulationOracle {
        n: 2,
        range,
        width: 0.01,
    };

    // The lower neuron's response has underflowed to exactly zero.
    let lower_response = oracle.response(range.1 as f64, 0);
    assert_eq!(
        lower_response, 0.0,
        "narrow width must underflow the lower neuron's response to exactly 0.0"
    );

    // Consequently the CoM at max equals max exactly: the strict inequality that
    // holds at width 0.2 does NOT hold here.
    let com_at_max = oracle.center_of_mass(range.1 as f64);
    assert_eq!(
        com_at_max, range.1 as f64,
        "under neighbor underflow the CoM at max collapses onto max exactly"
    );
    // The strict inward-bias inequality (`CoM < max`) that holds at width 0.2
    // does NOT hold here: the CoM sits exactly at max, not below it.
    assert!(
        com_at_max >= range.1 as f64,
        "the strict inward-bias inequality must NOT hold under underflow"
    );
}

#[test]
fn population_com_strict_bias_fails_when_neighbor_is_nonzero_but_lost_to_rounding() {
    // A SHARPER limit than plain underflow: the strict inward-bias inequality
    // can fail even when the lower neuron's response is REPRESENTABLE (strictly
    // nonzero, not underflowed). "Nonzero" is not the right condition; the
    // neighbor contribution must SURVIVE THE FLOATING-POINT ACCUMULATION — be
    // large enough that adding it to the endpoint's 1.0 actually changes the
    // sum.
    //
    // n = 2 over (0, 1), width = 0.1: the lower neuron sits at 0.0, so at
    // input = 1.0 its response is exp(-(1.0)^2 / (2 * 0.1^2)) = exp(-50)
    // ≈ 1.93e-22. That value is strictly greater than 0.0 (NOT underflowed),
    // yet `center_of_mass(1.0)` computes (0.0 * 1.93e-22 + 1.0 * 1.0) /
    // (1.93e-22 + 1.0), and `1.0 + 1.93e-22 == 1.0` in f64, so the ratio rounds
    // back to exactly 1.0 == max. The inward pull is lost to rounding even
    // though the response is representable.
    let range = (0.0f32, 1.0f32);
    let oracle = PopulationOracle {
        n: 2,
        range,
        width: 0.1,
    };

    // The lower neuron's response is strictly positive: it did NOT underflow to
    // zero. This is the crux — the condition is not "nonzero" but "survives
    // accumulation".
    let lower_response = oracle.response(range.1 as f64, 0);
    assert!(
        lower_response > 0.0,
        "width 0.1 keeps the lower neuron's response representable (nonzero), \
         got {lower_response}"
    );

    // Yet the neighbor contribution is lost when summed against the endpoint's
    // 1.0, so the CoM at max still equals max exactly: the strict inequality
    // does NOT hold despite the nonzero response.
    let com_at_max = oracle.center_of_mass(range.1 as f64);
    assert_eq!(
        com_at_max, range.1 as f64,
        "a representable-but-rounding-lost neighbor still collapses the CoM onto max"
    );

    // Contrast with width 0.2, where the neighbor contribution DOES survive
    // accumulation and the strict inward bias genuinely holds (≈ 0.99999627).
    let representable = PopulationOracle {
        n: 2,
        range,
        width: 0.2,
    };
    let com_representable = representable.center_of_mass(range.1 as f64);
    assert!(
        com_representable < range.1 as f64,
        "width 0.2 keeps the neighbor contribution alive, so CoM {com_representable} \
         is strictly below max"
    );
}

#[test]
fn population_oracle_floors_sub_epsilon_width_like_the_encoder() {
    // The oracle must mirror the encoder's unmodulated-path width floor. The
    // encoder's `effective_tuning_width(1.0)` returns
    // `self.tuning_width.max(f32::EPSILON)`, so a configured width in
    // (0, f32::EPSILON) is clamped UP to f32::EPSILON before the Gaussian is
    // evaluated (see `src/encoders/population.rs`). Without the floor the oracle
    // would underflow the neighbor to 0.0 and wrongly report CoM == max, while
    // the encoder still fires the lower neuron.
    //
    // Setup mirrors the encoder's own
    // `minimum_positive_width_still_fires_at_both_endpoint_preferences` unit
    // test: n = 2 over (0, 2e-7), configured width f32::MIN_POSITIVE (far below
    // f32::EPSILON). At the effective (floored) width f32::EPSILON the lower
    // neuron's response at max is exp(-(2e-7)^2 / (2 * f32::EPSILON^2)), a
    // healthy fraction (~0.24), NOT an underflow.
    let range = (0.0f32, 2e-7f32);
    let configured_width = f32::MIN_POSITIVE as f64;
    let oracle = PopulationOracle {
        n: 2,
        range,
        width: configured_width,
    };

    // The oracle floors the width at f32::EPSILON, matching effective_tuning_width(1.0).
    assert!(
        configured_width < f32::EPSILON as f64,
        "the configured width must be sub-epsilon for this test to be meaningful"
    );
    assert_eq!(
        oracle.effective_width(),
        f32::EPSILON as f64,
        "the oracle must clamp a sub-epsilon width up to f32::EPSILON"
    );

    // With the floored width the lower neuron's response at max is a healthy
    // fraction, not an underflow to 0.0.
    let lower_response = oracle.response(range.1 as f64, 0);
    let span = range.1 as f64;
    let floored = f32::EPSILON as f64;
    let expected = (-(span * span) / (2.0 * floored * floored)).exp();
    assert!(
        lower_response > 0.1,
        "floored width must keep the lower neuron firing (~0.24), got {lower_response}"
    );
    assert!(
        (lower_response - expected).abs() < 1e-12,
        "oracle response {lower_response} must match the analytically floored value {expected}"
    );

    // Because the neighbor now contributes meaningfully, the CoM at max is
    // strictly below max — the oracle no longer diverges from the encoder by
    // silently underflowing.
    let com_at_max = oracle.center_of_mass(range.1 as f64);
    assert!(
        com_at_max < range.1 as f64,
        "with the width floor the CoM at max {com_at_max} is biased inward, below {}",
        range.1
    );

    // Cross-check against the PRODUCTION encoder: configured with the same
    // sub-epsilon width it still fires the lower neuron (neuron 0) at the max
    // endpoint, exactly as its own unit test asserts. This ties the oracle's
    // floored behavior to real encoder output.
    let mut encoder =
        PopulationEncoder::try_new(2, range, f32::MIN_POSITIVE).expect("valid population");
    let mut lower_fired = false;
    for _ in 0..2_000 {
        if encoder
            .encode(&[range.1])
            .spikes
            .iter()
            .any(|s| s.channel == 0)
        {
            lower_fired = true;
            break;
        }
    }
    assert!(
        lower_fired,
        "the encoder fires the lower neuron under a sub-epsilon width, so the oracle must too"
    );
}

#[test]
fn population_fires_at_tick_zero_and_channels_map_to_neurons() {
    // Population spikes all land at TickOffset::ZERO, and for input channel j
    // neuron i emits on spike channel j * num_neurons + i. An empirical CoM over
    // firing neurons therefore maps channel -> neuron index -> preferred value.
    let n = 8usize;
    let range = (0.0f32, 1.0f32);
    let width = 0.2f32;
    let mut encoder = PopulationEncoder::try_new(n, range, width).expect("valid population");

    // Single input channel: all spikes are at tick zero and channels are in
    // 0..n mapping directly to neuron indices. Probe exactly at neuron 4's
    // preferred value (4 / 7 on the current grid) so that neuron fires with
    // probability exp(0) = 1.0 and the output is guaranteed non-empty. A bare
    // `encode(&[0.5])` could emit zero spikes (~1 in 2400 runs) because every
    // neuron independently misses on one unseeded-RNG call; anchoring on a
    // deterministic neuron removes that flake without weakening the tick-zero
    // or channel-mapping checks below.
    let deterministic_neuron = 4usize;
    let probe = (deterministic_neuron as f32) / ((n - 1) as f32); // == preferred[4].
    let out = encoder.encode(&[probe]);
    assert!(
        out.spikes
            .iter()
            .any(|s| s.channel as usize == deterministic_neuron),
        "the neuron sitting on the probe (prob 1.0) must always fire"
    );
    assert!(!out.spikes.is_empty());
    assert!(
        out.spikes.iter().all(|s| s.timestamp == TickOffset::ZERO),
        "all population spikes land at tick zero"
    );
    assert!(out.spikes.iter().all(|s| (s.channel as usize) < n));

    // Empirical pooled CoM at x = max, averaged over many calls, is strictly
    // below max (the same inward bias as the analytic oracle).
    let width_f64 = width as f64;
    let oracle = PopulationOracle {
        n,
        range,
        width: width_f64,
    };
    let mut weighted = 0.0f64;
    let mut spikes = 0u64;
    for _ in 0..5_000 {
        for spike in encoder.encode(&[range.1]).spikes {
            let neuron = spike.channel as usize;
            weighted += oracle.preferred(neuron);
            spikes += 1;
        }
    }
    assert!(spikes > 0);
    let empirical_com = weighted / spikes as f64;
    assert!(
        empirical_com < range.1 as f64,
        "empirical CoM at max {empirical_com} must be below {}",
        range.1
    );

    // Two input channels: the second population lives on channels [n, 2n).
    let out = encoder.encode(&[0.0, 1.0]);
    assert!(
        out.spikes
            .iter()
            .any(|s| (s.channel as usize) >= n && (s.channel as usize) < 2 * n),
        "input channel 1 must emit on the second population's channel block"
    );

    // The analytic oracle stays consistent with the width the encoder uses.
    let analytic_com = oracle.center_of_mass(range.1 as f64);
    assert!(analytic_com < range.1 as f64);
}
