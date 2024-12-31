use crate::matrix::vector::FloatVector;

pub fn relu<const SIZE: usize>(mut state: FloatVector<SIZE>) -> FloatVector<SIZE> {
    state
        .iter_mut()
        .filter(|&&mut element| element < 0.)
        .for_each(|element| {
            *element = 0.;
        });
    state
}
