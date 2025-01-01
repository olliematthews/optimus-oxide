use crate::layers::layer::Layer;

pub fn update<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize>(
    layers: Vec<Box<impl Layer<INPUT_SIZE, OUTPUT_SIZE>>>,
) {
}
