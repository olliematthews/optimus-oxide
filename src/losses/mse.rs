use crate::matrix::vector::FloatVector;

pub fn loss<const SIZE: usize>(output: FloatVector<SIZE>, label: FloatVector<SIZE>) -> f32 {
    output
        .iter()
        .zip(label.iter())
        .map(|(x, y)| {
            let e = y - x;
            e * e
        })
        .sum()
}

pub fn backwards<const SIZE: usize>(
    output: FloatVector<SIZE>,
    label: FloatVector<SIZE>,
) -> FloatVector<SIZE> {
    output
        .iter()
        .zip(label.iter())
        .map(|(x, y)| y - x)
        .collect()
}
