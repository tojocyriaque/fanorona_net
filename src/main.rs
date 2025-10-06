use std::time::Instant;

use crate::testing::{predict::test_model, train::batch_train};
#[allow(unused)]
use crate::{
    data::datasets::{balance_dataset_uniform, inspect_dataset},
    data::datasets::{generate_dataset, shuffle_dataset},
    games::fanorona::*,
    maths::collectors::{mat::*, vec::*},
    nn::NeuralNetwork,
    testing::*,
};

mod data;
mod games;
mod maths;
mod nn;
mod testing;

fn main() {
    // ==================== BALANCING TRAIN DATASET ==========================
    // let data_file: &str = "datasets/depth7/balanced_training.txt";
    // // shuffle_dataset(data_file);
    // let out_file: &str = "datasets/depth7/bl_tr.txt";
    // let target_per_class: usize = 212;
    // balance_dataset_uniform(data_file, out_file, target_per_class);
    // inspect_dataset(data_file);

    // =================================== DATASET GENERATION (with a depth as parameter)
    // redirect it into a file
    // generate_dataset(7);

    // // ==================== MODEL CREATION ===============================
    let tr_file = "datasets/depth7/bl_tr.txt";
    let model_name = "fn_md_all_v5";
    let models_dir = "models";

    let layer_sizes: Vec<usize> = vec![64, 128, 256, 18];
    let learning_rate = 0.97;
    let input_size = 46;

    let initial_epoch = 1;
    let n_epoch = 100;
    let batch_size = 5000; // make it the tr_file length for full batch

    // ========== TESTING THE LAST MODEL before continue training
    let mut model: NeuralNetwork;

    if initial_epoch > 1 {
        // load the last model (before the initial_epoch)
        let model_bin = format!(
            "models/{model_name}/{model_name}_E_{}.bin",
            initial_epoch - 1
        );
        test_model(model_bin.as_str(), tr_file);
        model = NeuralNetwork::from_file(model_bin.to_owned()); // load a model
    } else {
        // create new model
        model = NeuralNetwork::new(&layer_sizes, input_size, learning_rate); // new model
    }

    let timer = Instant::now();
    // train the model
    batch_train(
        &mut model,
        tr_file,
        batch_size,
        initial_epoch,
        n_epoch,
        model_name,
        models_dir,
        learning_rate,
    );
    let elapsed = timer.elapsed();
    println!("(MINIMABTCH LEARNING) Time elapsed: {:?}", elapsed);

    // Testing the model
    // load the last model (before the initial_epoch)
    let model_bin = format!(
        "models/{model_name}/{model_name}_E_{}.bin",
        initial_epoch - 1 + n_epoch
    );
    test_model(model_bin.as_str(), "datasets/depth7/min_bl.txt");

    // =================================== PLAYING IN CONSOLE (for model testing maybe)
    // let mut i_board = vec![0, 0, 1, 1, 1, -1, -1, 0, -1];
    // play_fanorona(&mut i_board, model_bin.as_str());
}
