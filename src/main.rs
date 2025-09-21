#[allow(unused)]
use crate::{
    nn::NeuralNetwork,
    testing::{predict::test_model, train::train_model},
};

mod data;
mod games;
mod maths;
mod nn;
mod testing;

const LEARNING_RATE: f64 = 0.1;
const EPOCHS: usize = 10;
const MODEL_PATH: &str = "models/fn_model_v2/fn_model_v2_E9.bin";
const TESTS_DATA_FILE: &str = "datasets/tests.txt";
const TRAIN_DATA_FILE: &str = "datasets/trainings.txt";

fn main() {
    // TESTING A MODEL
    test_model(MODEL_PATH, TESTS_DATA_FILE);

    // CREATING NEW MODELS
    // let layer_sizes: Vec<usize> = vec![120, 18];
    // let mut nn: NeuralNetwork = NeuralNetwork::new(&layer_sizes, 46, LEARNING_RATE);
    // train_model(&mut nn, TRAIN_DATA_FILE, EPOCHS);
}
