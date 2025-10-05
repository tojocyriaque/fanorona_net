use std::time::Instant;

use crate::testing::train::batch_train;
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
    // let data_file: &str = "datasets/depth7/training.txt";
    // // shuffle_dataset(data_file);
    // let out_file: &str = "datasets/depth7/balanced_training.txt";
    // let target_per_class: usize = 300;
    // balance_dataset_uniform(data_file, out_file, target_per_class);
    // inspect_dataset(out_file);

    // =================================== DATASET GENERATION (with a depth as parameter)
    // redirect it into a file
    // generate_dataset(7);

    // ==================== MODEL CREATION ===============================
    let layer_sizes: Vec<usize> = vec![32, 18];
    let input_size = 46;
    let batch_size = 10000;
    let learning_rate = 0.2;
    let tr_file = "datasets/depth7/balanced_training.txt";
    let initial_epoch = 1;
    let n_epoch = 1000;
    let model_name = "fn_md_v1";
    let models_dir = "models";

    let model_bin = "models/fn_md_v1/fn_md_v1_E_123.bin";
    let mut model = NeuralNetwork::from_file(model_bin.to_owned());
    let timer = Instant::now();
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

    // =================================== PLAYING IN CONSOLE (for model testing maybe)
    // let mut i_board = vec![0, 0, 1, 1, 1, -1, -1, 0, -1];
    // let model = "models/fn_model_d7_v8/fn_model_d7_v8_E69.bin";
    // play_fanorona(&mut i_board, model);
}
