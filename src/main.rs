#[allow(unused)]
use crate::{
    data::datasets::{balance_dataset_uniform, inspect_dataset},
    data::datasets::{generate_dataset, shuffle_dataset},
    games::fanorona::*,
    nn::NeuralNetwork,
    testing::train::continue_train_model,
    testing::{predict::test_model, train::train_model},
};

mod data;
mod games;
mod maths;
mod nn;
mod testing;

fn main() {
    // ==================== BALANCING TRAIN DATASET ==========================
    // let data_file: &str = "datasets/depth7/training.txt";
    // // shuffle_dataset(data_file);
    // let out_file: &str = "datasets/depth7/balanced_training.txt";
    // let target_per_class: usize = 300;
    // balance_dataset_uniform(data_file, out_file, target_per_class);
    // inspect_dataset(out_file);

    // TESTING MODELS
    // ==================== MODEL TESTING CONSTANTS ==========================
    const TESTS_FILE: &str = "datasets/depth7/training.txt";
    test_model("models/fn_model_d7_v8/fn_model_d7_v8_E69.bin", TESTS_FILE);
    // =======================================================================

    // CREATING NEW MODELS
    // ==================== TRAINING CONSTANTS ===============================
    // const LEARNING_RATE: f64 = 0.001;
    // const EPOCHS: usize = 100;
    // const TRAIN_FILE: &str = "datasets/depth7/balanced_training.txt";
    // // // this is the directory where you model will be registered

    // const MODELS_DIR: &str = "models"; // Directory where models will be saved
    // //This is the identification of your model type
    // const MODEL_NAME: &str = "fn_model_d7_ng_v2"; // Each model will be saved for every epochs of the training
    // const INPUT_SIZE: usize = 46;
    // // // // =======================================================================
    // let layer_sizes: Vec<usize> = vec![64, 18];
    // let mut nn: NeuralNetwork = NeuralNetwork::new(&layer_sizes, INPUT_SIZE, LEARNING_RATE);
    // train_model(&mut nn, MODELS_DIR, TRAIN_FILE, MODEL_NAME, EPOCHS);

    // // (if it is just an upgrade of a model you can continue it down here by loading the model)
    // let existent_model = "models/fn_model_d7_ng_v4/fn_model_d7_ng_v4_E100.bin";
    // let new_model_name = "fn_model_d7_ng_v5";

    // continue_train_model(
    //     existent_model,
    //     new_model_name,
    //     MODELS_DIR,
    //     TRAIN_FILE,
    //     EPOCHS,
    // );

    // =================================== DATASET GENERATION (with a depth as parameter)
    // redirect it into a file
    // generate_dataset(7);

    // =================================== PLAYING IN CONSOLE (for model testing maybe)
    let mut i_board = vec![0, 0, 1, 1, 1, -1, -1, 0, -1];
    let model = "models/fn_model_d7_v8/fn_model_d7_v8_E69.bin";
    play_fanorona(&mut i_board, model);
}
