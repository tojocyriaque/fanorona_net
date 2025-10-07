#[allow(unused)]
use crate::{
    data::datasets::{balance_dataset_uniform, inspect_dataset},
    data::datasets::{generate_dataset, shuffle_dataset},
    games::{fanorona::*, minmax::*},
    maths::collectors::{mat::*, vec::*},
    nn::NeuralNetwork,
    testing::{predict::*, train::*, *},
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
    // let time = std::time::Instant::now();
    // generate_dataset(7);
    // let el = time.elapsed();
    // println!("Generated in {:?}", el);

    // // ==================== MODEL CREATION ===============================
    let tr_file = "datasets/depth6/all.txt";
    let model_name = "fn_d6";
    let models_dir = "new_models";

    let layer_sizes: Vec<usize> = vec![32, 81];
    let learning_rate = 0.4;
    let input_size = 46;

    let initial_epoch = 1201;
    let n_epoch = 500;
    let batch_size = 61288; // make it the tr_file length for full batch

    // ========== TESTING THE LAST MODEL before continue training
    let mut model: NeuralNetwork;

    if initial_epoch > 1 {
        // load the last model (before the initial_epoch)
        let model_bin = format!(
            "{models_dir}/{model_name}/{model_name}_E_{}.bin",
            initial_epoch - 1
        );
        test_model(model_bin.as_str(), tr_file);
        model = NeuralNetwork::from_file(model_bin.to_owned()); // load a model
    } else {
        // create new model
        model = NeuralNetwork::new(&layer_sizes, input_size, learning_rate); // new model
    }

    let timer = std::time::Instant::now();
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
    println!("(MINIBATCH LEARNING) Time elapsed: {:?}", elapsed);

    // Testing the model
    // load the last model (before the initial_epoch)
    let model_bin = format!(
        "{models_dir}/{model_name}/{model_name}_E_{}.bin",
        initial_epoch - 1 + n_epoch
    );
    test_model(model_bin.as_str(), tr_file);

    // =================================== PLAYING IN CONSOLE (for model testing maybe)
    // let i_board = vec![0, 0, 1, 1, 1, -1, -1, 0, -1];
    // let mut s_moves: Vec<GMove> = Vec::new();
    // minimax_multi(&i_board, 6, 1, true, &mut s_moves);
    // let moves_proba = moves_proba_comb(&s_moves, 9);

    // show_board(i_board);
    // println!("Moves: {:?}", s_moves);
    // println!("Moves proba:\n{:?}", moves_proba);

    // play_fanorona(&mut i_board, model_bin.as_str());
}
