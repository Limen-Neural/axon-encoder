use crate::{Encoder, SpikeSink, types::EncodedOutput};
use ndarray::{ArrayView1, ArrayView2};

/// Feature-gated helpers for encoding `ndarray` views without changing the core trait.
///
/// The returning methods (`encode_array1`, `encode_step_array2`, …) allocate an
/// [`EncodedOutput`] per call or per row. Their `*_into` counterparts write the
/// same spikes into caller-owned [`SpikeSink`]s, so a runtime can reuse buffers
/// across views. 1-D helpers take `&mut dyn SpikeSink` and stay callable on
/// `&mut dyn Encoder`, matching [`Encoder::encode_into`]. 2-D helpers take a
/// slice of sinks — one destination per row — instead of constructing
/// `Vec<EncodedOutput>` or embedding a queue type.
pub trait NdarrayEncoderExt: Encoder {
    /// Encodes a 1-D view in batch mode, returning an owned [`EncodedOutput`].
    ///
    /// Contiguous views are passed through as a slice. A strided view is copied
    /// once into a temporary `Vec<f32>` only when `as_slice()` cannot provide
    /// one.
    fn encode_array1(&mut self, input: ArrayView1<'_, f32>) -> EncodedOutput {
        with_array1_input(input, |input| self.encode(input))
    }

    /// Encodes a 1-D view as one streaming step, returning an owned
    /// [`EncodedOutput`].
    ///
    /// Same contiguous/strided rule as [`encode_array1`](Self::encode_array1).
    fn encode_step_array1(&mut self, input: ArrayView1<'_, f32>) -> EncodedOutput {
        with_array1_input(input, |input| self.encode_step(input))
    }

    /// Encodes each row of `input` as an independent sample.
    ///
    /// Before any row is processed, the encoder is snapshotted via [`Clone`] and
    /// every row is encoded starting from that identical initial state. `self`
    /// is left unmodified by this call. State is **not** carried over between
    /// rows, so this is the right choice when each row represents a separate,
    /// unrelated sample.
    ///
    /// This differs from [`encode_step_array2`](Self::encode_step_array2), which
    /// streams a single mutable state across rows (i.e. later rows see the
    /// accumulated state from earlier rows). For encoders whose `encode_step`
    /// simply delegates to `encode` (e.g. [`DeltaEncoder`](crate::encoders::DeltaEncoder)),
    /// the two methods will produce different results for the same input.
    fn encode_array2(&self, input: ArrayView2<'_, f32>) -> Vec<EncodedOutput>
    where
        Self: Clone,
    {
        let standard = input.as_standard_layout();
        // each row gets a fresh clone so state never crosses row boundaries
        standard
            .rows()
            .into_iter()
            .map(|row| {
                let mut encoder = self.clone();
                encoder.encode_array1(row)
            })
            .collect()
    }

    /// Encodes each row of `input` as a step in a continuous stream.
    ///
    /// Unlike [`encode_array2`](Self::encode_array2), this threads a single
    /// mutable `self` across rows: each row is encoded via `encode_step` using
    /// the state left behind by the previous row, and `self` ends up holding
    /// the state accumulated after the last row. Use this when the rows of
    /// `input` represent a single continuous stream rather than independent
    /// samples.
    fn encode_step_array2(&mut self, input: ArrayView2<'_, f32>) -> Vec<EncodedOutput> {
        let standard = input.as_standard_layout();
        standard
            .rows()
            .into_iter()
            .map(|row| self.encode_step_array1(row))
            .collect()
    }

    /// Encodes a 1-D view in batch mode into caller-owned storage.
    ///
    /// Same spikes, in the same order, as [`encode_array1`](Self::encode_array1)
    /// and as [`Encoder::encode_into`] on the equivalent slice. `sink` is
    /// appended to, never cleared. A strided view is copied once only when a
    /// slice is unavailable.
    ///
    /// Takes `&mut dyn SpikeSink` so this stays callable on `&mut dyn Encoder`,
    /// matching the core sink API.
    ///
    /// # Examples
    ///
    /// Reuse one buffer across views; `clear` keeps the capacity the first
    /// call bought:
    ///
    /// ```rust
    /// use axon_encoder::prelude::*;
    /// use ndarray::arr1;
    /// # fn main() -> Result<(), EncoderError> {
    /// let mut encoder = DeltaEncoder::try_new(0.1, 3)?;
    /// let mut buffer: Vec<SpikeEvent> = Vec::new();
    ///
    /// for sample in [arr1(&[0.5, 0.0, 0.0]), arr1(&[0.5, 0.9, 0.0])] {
    ///     buffer.clear();
    ///     encoder.encode_array1_into(sample.view(), &mut buffer);
    /// }
    /// # let _ = buffer;
    /// # Ok(())
    /// # }
    /// ```
    fn encode_array1_into(&mut self, input: ArrayView1<'_, f32>, sink: &mut dyn SpikeSink) {
        with_array1_input(input, |input| self.encode_into(input, sink))
    }

    /// Encodes a 1-D view as one streaming step into caller-owned storage.
    ///
    /// Stands to [`encode_step_array1`](Self::encode_step_array1) as
    /// [`encode_array1_into`](Self::encode_array1_into) stands to
    /// [`encode_array1`](Self::encode_array1), including the append-don't-clear
    /// contract and the copy-once-if-strided rule.
    fn encode_step_array1_into(&mut self, input: ArrayView1<'_, f32>, sink: &mut dyn SpikeSink) {
        with_array1_input(input, |input| self.encode_step_into(input, sink))
    }

    /// Encodes each row independently into a caller-provided destination.
    ///
    /// Same row semantics as [`encode_array2`](Self::encode_array2): every row
    /// starts from a clone of this encoder's current state, and `self` is left
    /// unmodified. Unlike that method, this writes into `sinks` instead of
    /// allocating a `Vec<EncodedOutput>`.
    ///
    /// `sinks` must contain one [`SpikeSink`] per row. Each sink is appended
    /// to, never cleared. Non-contiguous rows are copied once, into a reused
    /// scratch buffer, only when a slice is unavailable — the whole 2-D view
    /// is not made standard-layout first.
    ///
    /// # Panics
    ///
    /// Panics if `sinks.len()` is not equal to `input.nrows()`.
    fn encode_array2_into<S: SpikeSink>(&self, input: ArrayView2<'_, f32>, sinks: &mut [S])
    where
        Self: Clone,
    {
        assert_one_sink_per_row(input.nrows(), sinks.len(), "encode_array2_into");
        let mut scratch = Vec::new();
        for (row, sink) in input.rows().into_iter().zip(sinks.iter_mut()) {
            let mut encoder = self.clone();
            with_array1_scratch(row, &mut scratch, |input| encoder.encode_into(input, sink));
        }
    }

    /// Encodes each row as a step in a continuous stream into a caller-provided
    /// destination.
    ///
    /// Same row semantics as [`encode_step_array2`](Self::encode_step_array2):
    /// a single mutable `self` is threaded across rows, each row is encoded
    /// via [`encode_step_into`](Encoder::encode_step_into), and `self` ends
    /// holding the state accumulated after the last row. Unlike that method,
    /// this writes into `sinks` instead of allocating a `Vec<EncodedOutput>`.
    ///
    /// `sinks` must contain one [`SpikeSink`] per row. Each sink is appended
    /// to, never cleared. To append every row into **one** sink, iterate
    /// `input.rows()` and call
    /// [`encode_step_array1_into`](Self::encode_step_array1_into) yourself —
    /// that keeps this helper free of any particular queue type.
    ///
    /// Non-contiguous rows are copied once, into a reused scratch buffer, only
    /// when a slice is unavailable. Row-major views whose rows are already
    /// slices skip that copy; column-major views pay it per row rather than
    /// copying the whole matrix.
    ///
    /// # Panics
    ///
    /// Panics if `sinks.len()` is not equal to `input.nrows()`.
    ///
    /// # Examples
    ///
    /// One reusable buffer per row; `clear` keeps the capacity across frames:
    ///
    /// ```rust
    /// use axon_encoder::prelude::*;
    /// use ndarray::arr2;
    /// # fn main() -> Result<(), EncoderError> {
    /// let frame = arr2(&[[0.5_f32, 0.0], [0.5, 0.9]]);
    /// let mut encoder = DeltaEncoder::try_new(0.1, 2)?;
    /// let mut buffers = vec![Vec::<SpikeEvent>::new(); frame.nrows()];
    ///
    /// encoder.encode_step_array2_into(frame.view(), &mut buffers);
    /// for buffer in &mut buffers {
    ///     buffer.clear();
    /// }
    /// encoder.encode_step_array2_into(frame.view(), &mut buffers);
    /// # let _ = buffers;
    /// # Ok(())
    /// # }
    /// ```
    fn encode_step_array2_into<S: SpikeSink>(
        &mut self,
        input: ArrayView2<'_, f32>,
        sinks: &mut [S],
    ) {
        assert_one_sink_per_row(input.nrows(), sinks.len(), "encode_step_array2_into");
        let mut scratch = Vec::new();
        for (row, sink) in input.rows().into_iter().zip(sinks.iter_mut()) {
            with_array1_scratch(row, &mut scratch, |input| {
                self.encode_step_into(input, sink)
            });
        }
    }
}

impl<T: Encoder + ?Sized> NdarrayEncoderExt for T {}

fn with_array1_input<R>(input: ArrayView1<'_, f32>, f: impl FnOnce(&[f32]) -> R) -> R {
    if let Some(slice) = input.as_slice() {
        f(slice)
    } else {
        let owned: Vec<f32> = input.iter().copied().collect();
        f(&owned)
    }
}

/// Like [`with_array1_input`], but reuses `scratch` for the strided copy so a
/// 2-D walk does not allocate a fresh `Vec<f32>` on every non-contiguous row.
fn with_array1_scratch<R>(
    input: ArrayView1<'_, f32>,
    scratch: &mut Vec<f32>,
    f: impl FnOnce(&[f32]) -> R,
) -> R {
    if let Some(slice) = input.as_slice() {
        f(slice)
    } else {
        scratch.clear();
        scratch.extend(input.iter().copied());
        f(scratch)
    }
}

fn assert_one_sink_per_row(nrows: usize, nsinks: usize, method: &str) {
    assert_eq!(
        nsinks, nrows,
        "{method} needs one sink per row (got {nsinks} sinks for {nrows} rows)"
    );
}

#[cfg(test)]
mod tests {
    use super::NdarrayEncoderExt;
    use crate::{
        Encoder, SpikeSink,
        encoders::{DeltaEncoder, LatencyEncoder, RateEncoder},
        types::{EncodedOutput, SpikeEvent},
    };
    use ndarray::{arr1, arr2};

    #[test]
    fn encode_array1_matches_slice_encoding() {
        let input = arr1(&[0.0_f32, 3.0, 1.0]);

        let mut slice_encoder = DeltaEncoder::new(2.0, input.len());
        let expected = slice_encoder.encode(input.as_slice().unwrap());

        let mut array_encoder = DeltaEncoder::new(2.0, input.len());
        let actual = array_encoder.encode_array1(input.view());

        assert_eq!(actual, expected);
    }

    #[test]
    fn encode_array2_encodes_each_row_independently() {
        let input = arr2(&[[0.0_f32, 0.0], [3.0, 0.0], [3.0, 4.0]]);

        let expected: Vec<_> = input
            .rows()
            .into_iter()
            .map(|row| {
                let mut enc = DeltaEncoder::new(2.0, input.ncols());
                enc.encode(row.as_slice().unwrap())
            })
            .collect();

        let array_encoder = DeltaEncoder::new(2.0, input.ncols());
        let actual = array_encoder.encode_array2(input.view());

        assert_eq!(actual, expected);
    }

    #[test]
    fn encode_step_array2_preserves_state_across_rows() {
        let input = arr2(&[[0.6_f32], [0.6], [0.6]]);

        let mut slice_encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        let expected: Vec<_> = input
            .rows()
            .into_iter()
            .map(|row| slice_encoder.encode_step(row.as_slice().unwrap()))
            .collect();

        let mut array_encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        let actual = array_encoder.encode_step_array2(input.view());

        assert_eq!(actual, expected);
    }

    #[test]
    fn encode_array1_falls_back_for_non_standard_layout_views() {
        let input = arr2(&[[0.0_f32, 3.0], [0.0, 0.0]]);
        let transposed = input.t();
        let non_standard = transposed.row(1);

        assert!(non_standard.as_slice().is_none());

        let expected_input = [3.0_f32, 0.0];

        let mut slice_encoder = DeltaEncoder::new(2.0, expected_input.len());
        let expected = slice_encoder.encode(&expected_input);

        let mut array_encoder = DeltaEncoder::new(2.0, expected_input.len());
        let actual = array_encoder.encode_array1(non_standard);

        assert_eq!(actual, expected);
    }

    #[test]
    fn encode_array2_handles_column_major_with_single_layout_copy() {
        let input = arr2(&[[0.0_f32, 0.0], [3.0, 0.0], [3.0, 4.0]]);
        let column_major = input.t().to_owned();
        let view = column_major.t();

        let expected: Vec<_> = input
            .rows()
            .into_iter()
            .map(|row| {
                let mut enc = DeltaEncoder::new(2.0, 2);
                enc.encode(row.as_slice().unwrap())
            })
            .collect();

        let array_encoder = DeltaEncoder::new(2.0, 2);
        let actual = array_encoder.encode_array2(view);

        assert_eq!(actual, expected);
    }

    #[test]
    fn encode_array2_leaves_self_unmodified() {
        let input = arr2(&[[0.0_f32, 0.0], [3.0, 0.0], [3.0, 4.0]]);
        let encoder = DeltaEncoder::new(2.0, input.ncols());
        let before = encoder.clone();

        let _ = encoder.encode_array2(input.view());

        assert_eq!(encoder, before, "independent-row mode must not mutate self");
    }

    fn assert_array1_into_matches_slice(
        view: ndarray::ArrayView1<'_, f32>,
        slice: &[f32],
        label: &str,
    ) {
        let mut returning = DeltaEncoder::new(2.0, slice.len());
        let mut expected = Vec::new();
        returning.encode_into(slice, &mut expected);

        let mut via_view = DeltaEncoder::new(2.0, slice.len());
        let mut actual = Vec::new();
        via_view.encode_array1_into(view, &mut actual);

        assert_eq!(actual, expected, "{label} ArrayView1 diverged from slice");
    }

    fn assert_step_array1_into_matches_slice(
        view: ndarray::ArrayView1<'_, f32>,
        slice: &[f32],
        label: &str,
    ) {
        let mut returning = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        let mut expected = Vec::new();
        returning.encode_step_into(slice, &mut expected);

        let mut via_view = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        let mut actual = Vec::new();
        via_view.encode_step_array1_into(view, &mut actual);

        assert_eq!(actual, expected, "{label} step ArrayView1 diverged");
    }

    fn step_encode_rows(
        input: ndarray::ArrayView2<'_, f32>,
    ) -> (RateEncoder, Vec<Vec<SpikeEvent>>) {
        let mut slice_encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        let expected: Vec<Vec<_>> = input
            .rows()
            .into_iter()
            .map(|row| {
                let mut buffer = Vec::new();
                slice_encoder.encode_step_into(row.as_slice().unwrap(), &mut buffer);
                buffer
            })
            .collect();
        (slice_encoder, expected)
    }

    #[test]
    fn encode_array1_into_matches_slice_encode_into_for_contiguous_and_strided_views() {
        let contiguous = arr1(&[0.0_f32, 3.0, 1.0]);
        assert_array1_into_matches_slice(
            contiguous.view(),
            contiguous.as_slice().unwrap(),
            "contiguous",
        );

        let matrix = arr2(&[[0.0_f32, 3.0], [0.0, 0.0]]);
        let transposed = matrix.t();
        let strided = transposed.row(1);
        assert!(strided.as_slice().is_none());
        assert_array1_into_matches_slice(strided, &[3.0_f32, 0.0], "strided");
    }

    #[test]
    fn encode_step_array1_into_matches_slice_encode_step_into_for_contiguous_and_strided_views() {
        let contiguous = arr1(&[0.6_f32, 0.6]);
        assert_step_array1_into_matches_slice(
            contiguous.view(),
            contiguous.as_slice().unwrap(),
            "contiguous",
        );

        let matrix = arr2(&[[0.6_f32, 0.0], [0.6, 0.0]]);
        let transposed = matrix.t();
        let strided = transposed.row(0);
        assert!(strided.as_slice().is_none());
        assert_step_array1_into_matches_slice(strided, &[0.6_f32, 0.6], "strided");
    }

    #[test]
    fn encode_step_array2_into_matches_repeated_encode_step_into_for_row_and_column_major() {
        let row_major = arr2(&[[0.6_f32, 0.2], [0.6, 0.2], [0.6, 0.2]]);
        let column_major = row_major.t().to_owned();
        let column_major_view = column_major.t();
        assert!(
            !column_major_view.is_standard_layout(),
            "fixture must exercise strided rows"
        );
        assert!(
            column_major_view.row(0).as_slice().is_none(),
            "column-major rows must not be slices"
        );

        let (slice_row_major, expected) = step_encode_rows(row_major.view());
        let mut array_encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        let mut actual = vec![Vec::new(); row_major.nrows()];
        array_encoder.encode_step_array2_into(row_major.view(), &mut actual);
        assert_eq!(actual, expected, "row-major streaming rows diverged");
        assert_eq!(
            array_encoder, slice_row_major,
            "row-major must advance one mutable state"
        );

        let (slice_column_major, expected) = step_encode_rows(row_major.view());
        let mut array_encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        let mut actual = vec![Vec::new(); row_major.nrows()];
        array_encoder.encode_step_array2_into(column_major_view, &mut actual);
        assert_eq!(actual, expected, "column-major streaming rows diverged");
        assert_eq!(
            array_encoder, slice_column_major,
            "column-major must advance one mutable state"
        );
    }

    #[test]
    fn encode_array2_into_leaves_self_unmodified_and_matches_independent_rows() {
        let input = arr2(&[[0.0_f32, 0.0], [3.0, 0.0], [3.0, 4.0]]);
        let encoder = DeltaEncoder::new(2.0, input.ncols());
        let before = encoder.clone();

        let expected: Vec<Vec<_>> = input
            .rows()
            .into_iter()
            .map(|row| {
                let mut enc = DeltaEncoder::new(2.0, input.ncols());
                let mut buffer = Vec::new();
                enc.encode_into(row.as_slice().unwrap(), &mut buffer);
                buffer
            })
            .collect();

        let mut actual = vec![Vec::new(); input.nrows()];
        encoder.encode_array2_into(input.view(), &mut actual);

        assert_eq!(actual, expected);
        assert_eq!(encoder, before, "independent-row mode must not mutate self");
    }

    #[test]
    fn ndarray_sink_helpers_append_and_never_clear() {
        let preexisting = SpikeEvent::new(99, 42u64, false);
        let view1 = arr1(&[1.0_f32]);
        let view2 = arr2(&[[1.0_f32], [0.0]]);

        let mut encoder = LatencyEncoder::new(3, (0.0, 1.0));
        let mut buffer = vec![preexisting];
        encoder.encode_array1_into(view1.view(), &mut buffer);
        assert_eq!(buffer[0], preexisting, "1-D batch must append");
        assert_eq!(buffer.len(), 2);

        let mut encoder = LatencyEncoder::new(3, (0.0, 1.0));
        let mut buffer = vec![preexisting];
        encoder.encode_step_array1_into(view1.view(), &mut buffer);
        assert_eq!(buffer[0], preexisting, "1-D step must append");
        assert_eq!(buffer.len(), 2);

        let mut encoder = LatencyEncoder::new(3, (0.0, 1.0));
        let mut sinks = vec![vec![preexisting], vec![preexisting]];
        encoder.encode_step_array2_into(view2.view(), &mut sinks);
        assert!(
            sinks.iter().all(|sink| sink[0] == preexisting),
            "2-D streaming must append per row"
        );

        let encoder = LatencyEncoder::new(3, (0.0, 1.0));
        let mut sinks = vec![vec![preexisting], vec![preexisting]];
        encoder.encode_array2_into(view2.view(), &mut sinks);
        assert!(
            sinks.iter().all(|sink| sink[0] == preexisting),
            "2-D independent-row must append per row"
        );
    }

    /// Encoder whose returning paths panic, so a helper that built an
    /// intermediate `EncodedOutput` would fail the test.
    #[derive(Clone, Debug, PartialEq)]
    struct SinkOnlyEncoder {
        spikes: usize,
    }

    impl Encoder for SinkOnlyEncoder {
        fn encode(&mut self, _input: &[f32]) -> EncodedOutput {
            panic!("encode allocates EncodedOutput; sink helpers must not call it");
        }

        fn encode_step(&mut self, _input: &[f32]) -> EncodedOutput {
            panic!("encode_step allocates EncodedOutput; sink helpers must not call it");
        }

        fn encode_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
            for (i, &value) in input.iter().enumerate() {
                if value > 0.0 {
                    sink.push(SpikeEvent::at_step_start(i as u16, true));
                    self.spikes += 1;
                }
            }
        }

        fn encode_step_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
            self.encode_into(input, sink);
        }

        fn reset(&mut self) {
            self.spikes = 0;
        }
    }

    #[test]
    fn ndarray_sink_helpers_do_not_allocate_an_intermediate_encoded_output() {
        let view1 = arr1(&[1.0_f32, 0.0, 2.0]);
        let view2 = arr2(&[[1.0_f32, 0.0], [0.0, 3.0]]);
        let mut encoder = SinkOnlyEncoder { spikes: 0 };

        let mut buffer = Vec::new();
        encoder.encode_array1_into(view1.view(), &mut buffer);
        assert_eq!(buffer.len(), 2);

        buffer.clear();
        encoder.encode_step_array1_into(view1.view(), &mut buffer);
        assert_eq!(buffer.len(), 2);

        let mut encoder = SinkOnlyEncoder { spikes: 0 };
        let mut sinks = vec![Vec::new(), Vec::new()];
        encoder.encode_step_array2_into(view2.view(), &mut sinks);
        assert_eq!(sinks[0].len(), 1);
        assert_eq!(sinks[1].len(), 1);

        let encoder = SinkOnlyEncoder { spikes: 0 };
        let before = encoder.clone();
        let mut sinks = vec![Vec::new(), Vec::new()];
        encoder.encode_array2_into(view2.view(), &mut sinks);
        assert_eq!(sinks[0].len(), 1);
        assert_eq!(sinks[1].len(), 1);
        assert_eq!(encoder, before);
    }

    #[test]
    fn encode_array1_into_is_callable_on_dyn_encoder() {
        let input = arr1(&[0.0_f32, 3.0, 1.0]);
        let mut encoder: Box<dyn Encoder> = Box::new(DeltaEncoder::new(2.0, input.len()));
        let mut buffer = Vec::new();
        encoder.encode_array1_into(input.view(), &mut buffer);

        let mut expected_encoder = DeltaEncoder::new(2.0, input.len());
        let mut expected = Vec::new();
        expected_encoder.encode_into(input.as_slice().unwrap(), &mut expected);
        assert_eq!(buffer, expected);
    }
}
