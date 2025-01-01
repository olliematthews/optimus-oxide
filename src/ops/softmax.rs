use crate::matrix::vector::FloatVector;

pub fn softmax<const SIZE: usize>(mut state: FloatVector<SIZE>) -> FloatVector<SIZE> {
    const EPS: f32 = 1e-5;
    state.iter_mut().for_each(|element: &mut f32| {
        *element = element.exp();
    });
    state /= state.iter().sum::<f32>() + EPS;
    state
}
