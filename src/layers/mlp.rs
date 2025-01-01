use crate::{layers::linear::LinearLayer, matrix::vector::FloatVector, ops::relu::relu};

struct MLP<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, const HIDDEN_LAYER_SIZE: usize> {
    encode: LinearLayer<INPUT_SIZE, HIDDEN_LAYER_SIZE>,
    decode: LinearLayer<HIDDEN_LAYER_SIZE, OUTPUT_SIZE>,
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, const HIDDEN_LAYER_SIZE: usize>
    MLP<INPUT_SIZE, OUTPUT_SIZE, HIDDEN_LAYER_SIZE>
{
    pub fn forward(&self, state: FloatVector<INPUT_SIZE>) -> FloatVector<OUTPUT_SIZE> {
        let hidden_layer = relu(self.encode.forward(state));
        self.decode.forward(hidden_layer)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_identity_network() {
        let encode = LinearLayer::from_elements([[1.], [-1.]], [0., 0.]);
        let decode = LinearLayer::from_elements([[1., -1.]], [0.]);
        let mlp = MLP { encode, decode };

        for input in [1., -1., -9.].iter() {
            let input_vec = FloatVector::from_elements([*input]);
            assert_eq!(mlp.forward(input_vec.clone()), input_vec);
        }
    }
}
