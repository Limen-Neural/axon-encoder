use crate::{types::EncodedOutput, Encoder};
use ndarray::{ArrayView1, ArrayView2};

/// Feature-gated helpers for encoding `ndarray` views without changing the core trait.
pub trait NdarrayEncoderExt: Encoder {
    fn encode_array1(&mut self, input: ArrayView1<'_, f32>) -> EncodedOutput {
        with_array1_input(input, |input| self.encode(input))
    }

    fn encode_step_array1(&mut self, input: ArrayView1<'_, f32>) -> EncodedOutput {
        with_array1_input(input, |input| self.encode_step(input))
    }

    fn encode_array2(&mut self, input: ArrayView2<'_, f32>) -> Vec<EncodedOutput>
    where
        Self: Clone,
    {
        let standard = input.as_standard_layout();
        let base = self.clone(); // each row gets a fresh clone so state never crosses row boundaries
        standard
            .rows()
            .into_iter()
            .map(|row| {
                let mut encoder = base.clone();
                encoder.encode_array1(row)
            })
            .collect()
    }

    fn encode_step_array2(&mut self, input: ArrayView2<'_, f32>) -> Vec<EncodedOutput> {
        let standard = input.as_standard_layout();
        standard
            .rows()
            .into_iter()
            .map(|row| self.encode_step_array1(row))
            .collect()
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

#[cfg(test)]
mod tests {
    use super::NdarrayEncoderExt;
    use crate::{
        encoders::{DeltaEncoder, RateEncoder},
        Encoder,
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

        let mut array_encoder = DeltaEncoder::new(2.0, input.ncols());
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

        let mut array_encoder = DeltaEncoder::new(2.0, 2);
        let actual = array_encoder.encode_array2(view);

        assert_eq!(actual, expected);
    }
}
