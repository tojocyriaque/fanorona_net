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

fn main() {
    // TESTING MODELS
    // ==================== MODEL TESTING CONSTANTS ==========================
    // const TESTS_FILE: &str = "datasets/balanced_tests.txt";
    // test_model("models/fn_model_v5/fn_model_v5_E20.bin", TESTS_FILE);
    // test_model("models/fn_model_v6/fn_model_v6_E20.bin", TESTS_FILE);
    // =======================================================================

    // CREATING NEW MODELS
    // ==================== TRAINING CONSTANTS ===============================
    // const LEARNING_RATE: f64 = 0.1;
    // const EPOCHS: usize = 20;
    // const TRAIN_FILE: &str = "datasets/trainings.txt";
    // // this is the directory where you model will be registered

    // const MODELS_DIR: &str = "models"; // Directory where models will be saved
    // //This is the identification of your model type
    // const MODEL_NAME: &str = "fn_model_v6"; // Each model will be saved for every epochs of the training
    // const INPUT_SIZE: usize = 46;
    // // =======================================================================

    // let layer_sizes: Vec<usize> = vec![64, 64, 18];
    // let mut nn: NeuralNetwork = NeuralNetwork::new(&layer_sizes, INPUT_SIZE, LEARNING_RATE);
    // train_model(&mut nn, MODELS_DIR, TRAIN_FILE, MODEL_NAME, EPOCHS);
}
