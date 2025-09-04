use std::time::Instant;

#[allow(unused)]
use crate::{
    data::datasets::{balance_dataset_uniform, inspect_dataset},
    data::datasets::{generate_dataset, shuffle_dataset},
    games::fanorona::*,
    maths::collectors::{mat::*, vec::*},
    nn::NeuralNetwork,
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

    // ==================== MODEL CREATION
    let layer_sizes: Vec<usize> = vec![64, 128, 18];
    let input_size = 46;
    let batch_size = 100;
    let learning_rate = 0.02;
    let input = Matrix::init_xavier(batch_size, input_size);

    let nn = NeuralNetwork::new(&layer_sizes, input_size, learning_rate);
    let timer = Instant::now();
    nn.batch_forward(&input);
    let elapsed = timer.elapsed();

    println!("(BATCHED) Time elapsed: {:?}", elapsed);

    // =================================== PLAYING IN CONSOLE (for model testing maybe)
    // let mut i_board = vec![0, 0, 1, 1, 1, -1, -1, 0, -1];
    // let model = "models/fn_model_d7_v8/fn_model_d7_v8_E69.bin";
    // play_fanorona(&mut i_board, model);
}
