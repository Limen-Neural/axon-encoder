//! Behavioral tests for the public time types (#62 / RM-368).
//!
//! `TickOffset`, `Timebase`, `TimeModel`, and `TimeCursor` are entirely public,
//! so they are exercised from outside the crate — the same vantage point a
//! downstream simulator has. `tests/time_semantics.rs` covers the companion
//! question: whether the encoders honour the contract these types describe.

use axon_encoder::prelude::*;

// --- TickOffset --------------------------------------------------------------

#[test]
fn tick_offset_round_trips_u64() {
    let offset = TickOffset::from(9u64);
    assert_eq!(offset.ticks(), 9);
    assert_eq!(u64::from(offset), 9);
}

#[test]
fn tick_offset_compares_against_u64() {
    // The migration lever: pre-0.5 reads keep compiling.
    let offset = TickOffset::new(9);
    assert_eq!(offset, 9u64);
    assert_eq!(9u64, offset);
}

#[test]
fn tick_offset_orders_against_u64_in_both_directions() {
    let offset = TickOffset::new(9);
    assert!(offset > 8u64, "PartialOrd<u64> for TickOffset");
    // Bound separately: this exercises the mirrored impl, not the same one.
    let u64_is_less = 8u64 < offset;
    assert!(u64_is_less, "PartialOrd<TickOffset> for u64");
}

#[test]
fn tick_offset_zero_is_the_default_and_displays_its_unit() {
    assert_eq!(TickOffset::default(), TickOffset::ZERO);
    assert!(TickOffset::ZERO.is_zero() && !TickOffset::new(1).is_zero());
    assert_eq!(TickOffset::new(9).to_string(), "9t");
}

#[test]
fn tick_offset_saturating_add_clamps_at_max() {
    assert_eq!(TickOffset::new(3).saturating_add(4), 7u64);
    assert_eq!(TickOffset::new(u64::MAX).saturating_add(1), u64::MAX);
}

#[test]
fn tick_offset_checked_add_reports_overflow() {
    assert_eq!(TickOffset::new(3).checked_add(4), Some(TickOffset::new(7)));
    assert_eq!(TickOffset::new(u64::MAX).checked_add(1), None);
}

// --- Timebase ----------------------------------------------------------------

#[test]
fn timebase_constructors_agree() {
    let from_seconds = Timebase::try_from_seconds(0.001).expect("1 ms");
    assert_eq!(from_seconds, Timebase::MILLISECOND);
    assert_eq!(Timebase::try_from_hz(1000.0).expect("1 kHz"), from_seconds);
}

#[test]
fn timebase_constants_match_their_names() {
    for (timebase, nanos) in [
        (Timebase::NANOSECOND, 1),
        (Timebase::MICROSECOND, 1_000),
        (Timebase::MILLISECOND, 1_000_000),
        (Timebase::SECOND, 1_000_000_000),
    ] {
        assert_eq!(timebase.tick_nanos(), nanos);
    }
}

#[test]
fn timebase_round_trips_through_nanos() {
    let timebase = Timebase::try_from(1_000u64).expect("1 us");
    assert_eq!(timebase, Timebase::MICROSECOND);
    assert_eq!(u64::from(timebase), 1_000);
}

#[test]
fn timebase_rejects_unrepresentable_seconds() {
    let expected = EncoderError::NonPositiveOrNonFinite {
        parameter: "tick_seconds",
    };
    // Non-positive, non-finite, rounding below a nanosecond, and overflowing the
    // u64 nanosecond range all fail the same way. The last case sits exactly on
    // the boundary: it rounds to 2^64 ns, which an `as u64` cast would saturate
    // back into range instead of rejecting.
    let boundary = (u64::MAX as f64) / 1e9;
    for seconds in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-12, 1e30, boundary] {
        assert_eq!(
            Timebase::try_from_seconds(seconds),
            Err(expected.clone()),
            "tick_seconds = {seconds}"
        );
    }
}

#[test]
fn timebase_rejects_unrepresentable_rates() {
    // Named for the parameter the caller actually supplied.
    let expected = EncoderError::NonPositiveOrNonFinite { parameter: "hz" };
    for hz in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            Timebase::try_from_hz(hz),
            Err(expected.clone()),
            "hz = {hz}"
        );
    }
}

#[test]
fn timebase_rejects_a_zero_tick() {
    assert_eq!(
        Timebase::try_from_nanos(0),
        Err(EncoderError::WindowMustBePositive {
            parameter: "tick_nanos"
        })
    );
}

#[test]
fn timebase_converts_offsets_to_nanos() {
    assert_eq!(
        Timebase::MILLISECOND.offset_nanos(TickOffset::new(5)),
        5_000_000
    );
    // Saturates rather than wrapping.
    assert_eq!(
        Timebase::SECOND.offset_nanos(TickOffset::new(u64::MAX)),
        u64::MAX
    );
}

#[test]
fn timebase_converts_offsets_to_seconds() {
    let seconds = Timebase::MILLISECOND.offset_seconds(TickOffset::new(2500));
    assert!((seconds - 2.5).abs() < 1e-9, "got {seconds}");
}

#[test]
fn timebase_reports_its_tick_in_seconds_and_hertz() {
    let timebase = Timebase::MILLISECOND;
    assert!((timebase.tick_seconds() - 0.001).abs() < 1e-12);
    assert!((timebase.hz() - 1000.0).abs() < 1e-9);
}

#[test]
fn timebase_counts_whole_ticks_in_a_duration() {
    assert_eq!(Timebase::MILLISECOND.ticks_from_nanos(2_500_000), 2);
    assert_eq!(Timebase::MILLISECOND.to_string(), "1000000ns/tick");
}

// --- TimeModel ---------------------------------------------------------------

#[test]
fn instant_model_is_one_dimensionless_tick_per_call() {
    assert_eq!(TimeModel::INSTANT, TimeModel::default());
    assert_eq!(TimeModel::INSTANT.step_ticks(), 1);
    assert_eq!(TimeModel::INSTANT.span_ticks(), 1);
}

#[test]
fn instant_model_neither_overlaps_nor_claims_a_duration() {
    assert!(!TimeModel::INSTANT.is_overlapping());
    assert!(TimeModel::INSTANT.timebase().is_none());
}

#[test]
fn window_model_advances_by_its_whole_span() {
    let window = TimeModel::window(11);
    assert_eq!(window.step_ticks(), 11);
    assert_eq!(window.span_ticks(), 11);
    assert!(!window.is_overlapping());
}

#[test]
fn window_model_bounds_offsets_exclusively() {
    let window = TimeModel::window(11);
    assert!(window.contains(TickOffset::new(10)));
    assert!(!window.contains(TickOffset::new(11)));
}

#[test]
fn overlapping_model_reaches_past_its_step() {
    let phase_like = TimeModel::overlapping(1, 16);
    assert_eq!(phase_like.step_ticks(), 1);
    assert_eq!(phase_like.span_ticks(), 16);
    assert!(phase_like.is_overlapping());
}

#[test]
fn models_clamp_zero_to_one_tick() {
    for model in [TimeModel::window(0), TimeModel::overlapping(0, 0)] {
        assert_eq!((model.step_ticks(), model.span_ticks()), (1, 1));
    }
}

#[test]
fn time_model_carries_an_optional_timebase() {
    let model = TimeModel::INSTANT.with_timebase(Timebase::MICROSECOND);
    assert_eq!(model.timebase(), Some(Timebase::MICROSECOND));
    assert_eq!(TimeModel::INSTANT.with_timebase_opt(None).timebase(), None);
}

// --- TimeCursor --------------------------------------------------------------

#[test]
fn cursor_advances_by_step_ticks() {
    let mut cursor = TimeCursor::new(TimeModel::window(11));
    assert_eq!(
        (cursor.origin(), cursor.absolute(TickOffset::new(3))),
        (0, 3)
    );
    assert_eq!(cursor.advance(), 11);
    assert_eq!(cursor.absolute(TickOffset::new(3)), 14);
}

#[test]
fn cursor_advances_past_several_calls_at_once() {
    let mut cursor = TimeCursor::new(TimeModel::window(11));
    assert_eq!(cursor.advance_by(3), 33);
}

#[test]
fn cursor_reset_keeps_the_model() {
    let mut cursor = TimeCursor::new(TimeModel::window(11));
    cursor.advance();
    cursor.reset();
    assert_eq!(cursor.origin(), 0);
    assert_eq!(cursor.model(), TimeModel::window(11));
}

#[test]
fn cursor_saturates_instead_of_wrapping() {
    let mut cursor = TimeCursor::starting_at(TimeModel::INSTANT, u64::MAX - 1);
    assert_eq!(cursor.absolute(TickOffset::new(5)), u64::MAX);
    assert_eq!(cursor.advance(), u64::MAX);
    assert_eq!(cursor.advance_by(u64::MAX), u64::MAX);
}

#[test]
fn cursor_nanos_require_a_timebase() {
    let dimensionless = TimeCursor::new(TimeModel::INSTANT);
    assert_eq!(dimensionless.absolute_nanos(TickOffset::ZERO), None);

    let physical =
        TimeCursor::starting_at(TimeModel::INSTANT.with_timebase(Timebase::MILLISECOND), 3);
    assert_eq!(physical.absolute_nanos(TickOffset::new(1)), Some(4_000_000));
}

#[test]
fn cursor_maps_a_whole_call() {
    let cursor = TimeCursor::starting_at(TimeModel::window(11), 22);
    let spikes = [
        SpikeEvent::new(0, TickOffset::new(0), true),
        SpikeEvent::new(1, TickOffset::new(5), true),
    ];
    let absolute: Vec<u64> = cursor.absolute_times(&spikes).collect();
    assert_eq!(absolute, vec![22, 27]);
}
