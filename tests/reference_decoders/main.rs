//! In-tree inverse oracles for the latency, phase, rate, and population
//! encoders (LIM-1340, parent LIM-1265).
//!
//! The crate ships **no production decoder**. Downstream count-rate, TTFS
//! (time-to-first-spike), and center-of-mass consumers therefore have no shared
//! reference for "what value produced this spike train". This file provides the
//! inverses used *only as test oracles*: they let LIM-1260's decoder-consistency
//! audit cite these numbers instead of re-deriving them, and they pin the
//! properties a real decoder must respect (collision points, quantization
//! bounds, the batch-versus-stream rate cap, and the population endpoint bias).
//!
//! Every oracle here is derived from the encoder source at `main` `fca28fa`,
//! not from the issue's `f3e375f` measurement table. Two things in that table
//! are stale and are corrected below:
//!
//!   * Population preferred values now cover both range endpoints via
//!     `preferred[i] = min + (i / (num_neurons - 1)) * span` (PR #101), not the
//!     old `i / num_neurons` grid. So for `N = 8` over `(0, 1)`, neuron 7 sits
//!     at `1.0`, not `0.875`.
//!   * The durable population invariant is therefore *not* a hardcoded
//!     `0.951 / 0.875` figure but the endpoint bias: the center-of-mass decode
//!     at `x = max` is strictly less than `max` for `N > 1` **when the neighbor
//!     tuning responses survive the floating-point accumulation** — i.e. when a
//!     lower neuron's contribution is large enough that adding it to the
//!     endpoint term actually changes the sum (`1.0 + contribution != 1.0`),
//!     rather than being lost to rounding against it. When that holds, the
//!     Gaussian neighbors all sit below the maximum and pull the CoM inward.
//!     A merely *nonzero* (representable, non-underflowing) neighbor response is
//!     **not** sufficient: for a narrow-but-not-underflowing width the lower
//!     neuron's response can be a tiny positive value (e.g. `exp(-50) ≈
//!     1.93e-22` at `n = 2`, range `(0, 1)`, width `0.1`) that is nonzero yet
//!     lost when summed against the endpoint's `1.0` (`1.0 + 1.93e-22 == 1.0`
//!     in f64), so the CoM still lands on `max` exactly. For even narrower
//!     widths the response underflows all the way to exactly `0.0` and the same
//!     collapse happens for a stronger reason. The strict inequality is only
//!     claimed and tested where the neighbor contribution survives accumulation.
//!
//! There is intentionally **no public decoder API**: these inverses live in the
//! test tree until a later 0.5 design explicitly asks for one.
//!
//! Each encoder family lives in its own submodule so that every module carries a
//! single responsibility (one encoder family's oracle plus its tests).

mod latency;
mod phase;
mod population;
mod rate;
