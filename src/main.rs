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

// ==================== TRAINING CONSTANTS ==============================
const LEARNING_RATE: f64 = 0.1;
const EPOCHS: usize = 20;
const TRAIN_DATA_FILE: &str = "datasets/trainings.txt";
const TRAIN_TEST_FILE: &str = "datasets/balanced_tests.txt";
// this is the directory where you model will be registered
const MODEL_PARAMS_DIR: &str = "models";
//This is the identification of your model type
const MODEL_TYPE: &str = "fn_model_v5";
const INPUT_SIZE: usize = 46;

fn main() {
    // TESTING MODELS
    // ==================== MODEL TESTING CONSTANTS ==========================
    const MODEL_V3: &str = "models/fn_model_v3/fn_model_v3_E9.bin";
    const MODEL_V5: &str = "models/fn_model_v5/fn_model_v5_E20.bin";
    const TESTS_FILE: &str = "datasets/tests.txt";
    test_model(MODEL_V3, TESTS_FILE);
    test_model(MODEL_V5, TESTS_FILE);

    // CREATING NEW MODELS
    // let layer_sizes: Vec<usize> = vec![64, 64, 18];
    // let mut nn: NeuralNetwork = NeuralNetwork::new(&layer_sizes, INPUT_SIZE, LEARNING_RATE);
    // train_model(&mut nn, TRAIN_DATA_FILE, EPOCHS);
}
