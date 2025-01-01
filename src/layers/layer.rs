use crate::matrix::vector::FloatVector;

pub trait Layer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, ParamType> {
    fn forward(&self, state: FloatVector<INPUT_SIZE>) -> FloatVector<OUTPUT_SIZE>;

    fn backward(
        &mut self,
        state: FloatVector<INPUT_SIZE>,
        dl_dout: FloatVector<OUTPUT_SIZE>,
    ) -> FloatVector<INPUT_SIZE>;

    fn get_grads(&self) -> ParamType;

    fn update_params(&mut self, param_updates: ParamType);
}
