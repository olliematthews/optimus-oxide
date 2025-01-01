use crate::{
    layers::{layer::Layer, linear::LinearLayer},
    losses::mse,
    matrix::vector::FloatVector,
};

struct LinearModel<const INPUT_SIZE: usize> {
    layer: LinearLayer<INPUT_SIZE, 1>,
    inputs: Vec<FloatVector<INPUT_SIZE>>,
    learning_rate: f32,
}

impl<const INPUT_SIZE: usize> LinearModel<INPUT_SIZE> {
    pub fn forward(&mut self, input: FloatVector<INPUT_SIZE>) -> f32 {
        self.inputs.push(input.clone());

        self.layer.forward(input)[0]
    }

    pub fn backward(
        &mut self,
        outputs: &mut Vec<FloatVector<1>>,
        labels: &mut Vec<FloatVector<1>>,
    ) {
        for ((output, label), input) in outputs
            .drain(..)
            .zip(labels.drain(..))
            .zip(self.inputs.drain(..))
        {
            let dl_dout = mse::backwards(output, label);

            self.layer.backward(input, dl_dout)[0];
        }
    }

    pub fn update_params(&mut self) {
        let (mut weight_grads, mut bias_grads) = self.layer.get_grads();
        weight_grads *= -self.learning_rate;
        bias_grads *= -self.learning_rate;

        self.layer.update_params((weight_grads, bias_grads));
    }
}
