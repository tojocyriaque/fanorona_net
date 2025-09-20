use crate::{nn::NeuralNetwork, utils::inits::*, utils::loads::*};

#[allow(unused)]
impl NeuralNetwork {
    pub fn from_file(filename: String) -> Self {
        let params = load_parameters_binary(filename).expect("Failed to load parameters");
        NeuralNetwork {
            is: params.input_size,
            ls: params.layer_sizes,
            weights: params.weights,
            biases: params.biases,
            lr: params.learning_rate,
            ln: params.layer_num,
        }
    }

    #[allow(unused)]
    pub fn new(ls: &Vec<usize>, is: usize, lr: f64) -> Self {
        // let mut rng = SimpleRng::new(58);
        NeuralNetwork {
            is: is,
            lr: lr,
            ln: ls.len(),
            ls: ls.to_vec(),
            weights: init_matrixes(ls, is, true),
            biases: init_vectors(ls, false),
        }
    }
}
