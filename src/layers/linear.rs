use crate::matrix::matrix::Matrix2;
use crate::matrix::vector::FloatVector;
use crate::ops::relu::relu;

struct LinearLayer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    weights: Matrix2<OUTPUT_SIZE, INPUT_SIZE>,
    bias: FloatVector<OUTPUT_SIZE>,
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> LinearLayer<INPUT_SIZE, OUTPUT_SIZE> {
    pub fn from_elements(
        weight_elements: [[f32; INPUT_SIZE]; OUTPUT_SIZE],
        bias_elements: [f32; OUTPUT_SIZE],
    ) -> Self {
        LinearLayer {
            weights: Matrix2::from_elements(weight_elements),
            bias: FloatVector::from_elements(bias_elements),
        }
    }

    pub fn from_wb(
        weights: Matrix2<OUTPUT_SIZE, INPUT_SIZE>,
        bias: FloatVector<OUTPUT_SIZE>,
    ) -> Self {
        LinearLayer { weights, bias }
    }

    pub fn from(
        matrix_elements: [[f32; INPUT_SIZE]; OUTPUT_SIZE],
        bias_elements: [f32; OUTPUT_SIZE],
    ) -> Self {
        LinearLayer {
            weights: Matrix2::from_elements(matrix_elements),
            bias: FloatVector::from_elements(bias_elements),
        }
    }

    pub fn forward(&self, state: FloatVector<INPUT_SIZE>) -> FloatVector<OUTPUT_SIZE> {
        let mut ret_vector = self.weights.dot(&state);
        ret_vector += &self.bias;
        relu(ret_vector)
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
