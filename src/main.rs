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
const EPOCHS: usize = 10;
const TRAIN_DATA_FILE: &str = "datasets/trainings.txt";
const TRAIN_TEST_FILE: &str = "datasets/balanced_tests.txt";
// this is the directory where you model will be registered
const MODEL_PARAMS_DIR: &str = "models";
//This is the identification of your model type
const MODEL_TYPE: &str = "fn_model_v3";

// ==================== MODEL TESTING CONSTANTS ==========================
const MODEL_V2_PATH: &str = "models/fn_model_v2/fn_model_v2_E9.bin";
const MODEL_V3_PATH: &str = "models/fn_model_v3/fn_model_v3_E9.bin";
const TESTS_DATA_FILE: &str = "datasets/tests.txt";

fn main() {
    // TESTING MODELS
    test_model(MODEL_V2_PATH, TESTS_DATA_FILE);
    test_model(MODEL_V3_PATH, TESTS_DATA_FILE);

    // CREATING NEW MODELS
    // let layer_sizes: Vec<usize> = vec![256, 18];
    // let mut nn: NeuralNetwork = NeuralNetwork::new(&layer_sizes, 46, LEARNING_RATE);
    // train_model(&mut nn, TRAIN_DATA_FILE, EPOCHS);
}
