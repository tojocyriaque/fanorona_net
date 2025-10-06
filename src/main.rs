use std::{fmt::format, time::Instant};

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
    // let data_file: &str = "datasets/depth7/min_bl.txt";
    // shuffle_dataset(data_file);
    // let out_file: &str = "datasets/depth7/min_bl.txt";
    // let target_per_class: usize = 40;
    // balance_dataset_uniform(data_file, out_file, target_per_class);
    // inspect_dataset(out_file);

    // =================================== DATASET GENERATION (with a depth as parameter)
    // redirect it into a file
    // generate_dataset(7);

    // ==================== MODEL CREATION ===============================
    let layer_sizes: Vec<usize> = vec![128, 18];
    let input_size = 46;
    let batch_size = 1320; // make it the tr_file length for full batch
    let learning_rate = 0.01;
    let tr_file = "datasets/depth7/min_bl.txt";
    let initial_epoch = 8501;
    let n_epoch = 26200;
    let model_name = "fn_md_min_bl_v1";
    let models_dir = "models";

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
    );
    let elapsed = timer.elapsed();
    println!("(MINIMABTCH LEARNING) Time elapsed: {:?}", elapsed);

    // Testing the model
    // load the last model (before the initial_epoch)
    let model_bin = format!(
        "models/{model_name}/{model_name}_E_{}.bin",
        initial_epoch - 1 + n_epoch
    );
    test_model(model_bin.as_str(), tr_file);

    // =================================== PLAYING IN CONSOLE (for model testing maybe)
    // let mut i_board = vec![0, 0, 1, 1, 1, -1, -1, 0, -1];
    // let model = "models/fn_model_d7_v8/fn_model_d7_v8_E69.bin";
    // play_fanorona(&mut i_board, model);
}
