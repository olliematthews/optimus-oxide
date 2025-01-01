use crate::matrix::matrix::Matrix2;
use crate::matrix::vector::FloatVector;
use crate::ops::relu::relu;

use super::layer::Layer;

pub struct LinearLayer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    weights: Matrix2<OUTPUT_SIZE, INPUT_SIZE>,
    bias: FloatVector<OUTPUT_SIZE>,
    weights_grad_tot: Matrix2<OUTPUT_SIZE, INPUT_SIZE>,
    bias_grad_tot: FloatVector<OUTPUT_SIZE>,
    backward_step_count: u16,
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> LinearLayer<INPUT_SIZE, OUTPUT_SIZE> {
    pub fn from_elements(
        weight_elements: [[f32; INPUT_SIZE]; OUTPUT_SIZE],
        bias_elements: [f32; OUTPUT_SIZE],
    ) -> Self {
        LinearLayer {
            weights: Matrix2::from_elements(weight_elements),
            bias: FloatVector::from_elements(bias_elements),
            weights_grad_tot: Matrix2::new_zeros(),
            bias_grad_tot: FloatVector::new_zeros(),
            backward_step_count: 0,
        }
    }

    pub fn from_wb(
        weights: Matrix2<OUTPUT_SIZE, INPUT_SIZE>,
        bias: FloatVector<OUTPUT_SIZE>,
    ) -> Self {
        LinearLayer {
            weights,
            bias,
            weights_grad_tot: Matrix2::new_zeros(),
            bias_grad_tot: FloatVector::new_zeros(),
            backward_step_count: 0,
        }
    }

    pub fn from(
        matrix_elements: [[f32; INPUT_SIZE]; OUTPUT_SIZE],
        bias_elements: [f32; OUTPUT_SIZE],
    ) -> Self {
        LinearLayer {
            weights: Matrix2::from_elements(matrix_elements),
            bias: FloatVector::from_elements(bias_elements),
            weights_grad_tot: Matrix2::new_zeros(),
            bias_grad_tot: FloatVector::new_zeros(),
            backward_step_count: 0,
        }
    }
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize>
    Layer<INPUT_SIZE, OUTPUT_SIZE, (Matrix2<OUTPUT_SIZE, INPUT_SIZE>, FloatVector<OUTPUT_SIZE>)>
    for LinearLayer<INPUT_SIZE, OUTPUT_SIZE>
{
    fn forward(&self, state: FloatVector<INPUT_SIZE>) -> FloatVector<OUTPUT_SIZE> {
        let mut ret_vector = self.weights.dot(&state);
        ret_vector += &self.bias;
        ret_vector
    }

    fn backward(
        &mut self,
        state: FloatVector<INPUT_SIZE>,
        dl_dout: FloatVector<OUTPUT_SIZE>,
    ) -> FloatVector<INPUT_SIZE> {
        let mut dl_din: FloatVector<INPUT_SIZE> = FloatVector::new_zeros();
        self.weights
            .rows
            .iter()
            .zip(dl_dout.iter())
            .for_each(|(row, grad)| {
                for (j, element) in row.iter().enumerate() {
                    dl_din[j] = element * grad;
                }
            });

        self.bias_grad_tot += &dl_dout;
        self.weights_grad_tot += &dl_dout.outer(&state);
        self.backward_step_count += 1;
        dl_din
    }

    fn get_grads(&self) -> (Matrix2<OUTPUT_SIZE, INPUT_SIZE>, FloatVector<OUTPUT_SIZE>) {
        (
            &self.weights_grad_tot / self.backward_step_count,
            &self.bias_grad_tot / self.backward_step_count,
        )
    }

    fn update_params(
        &mut self,
        param_updates: (Matrix2<OUTPUT_SIZE, INPUT_SIZE>, FloatVector<OUTPUT_SIZE>),
    ) {
        let (weight_update, bias_update) = param_updates;
        self.weights += &weight_update;
        self.bias += &bias_update;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ops() {
        const INPUT_SIZE: usize = 2;
        const OUTPUT_SIZE: usize = 3;

        let layer = LinearLayer::from_elements([[1., 0.], [0., -1.], [0., 0.]], [0., 0., 1.]);
        let input: FloatVector<INPUT_SIZE> = FloatVector::from_elements([5., 3.]);
        let label: FloatVector<OUTPUT_SIZE> = FloatVector::from_elements([5., 0., 1.]);

        assert_eq!(layer.forward(input), label);
    }
}
