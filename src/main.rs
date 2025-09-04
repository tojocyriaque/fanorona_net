#[allow(unused)]
use crate::{
    data::datasets::{balance_dataset_uniform, inspect_dataset, valid_pos},
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
    // generate_dataset(6);
    // let pos = vec![1, 0, 0, -1, 2, 0, -1, -1, 1];
    // let vl = valid_pos(&pos, -1);
    // println!("Valid pos: {:?} => {}", pos, vl);
    // let el = time.elapsed();
    // println!("Generated in {:?}", el);

    // // ==================== MODEL CREATION ===============================
    let tr_file = "datasets/depth6/all_1hot.txt";
    let model_name = "fn_d6_final";
    let models_dir = "new_models";

    let layer_sizes: Vec<usize> = vec![256, 81];
    let learning_rate = 0.1;
    let input_size = 46;

    let initial_epoch = 31;
    let n_epoch = 0;
    let batch_size = 32; // make it the tr_file length for full batch

    // // ========== TESTING THE LAST MODEL before continue training
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
    test_model(&model_bin, tr_file);

    // let old_model = "new_models/fn_d6/fn_d6_E_1700.bin";
    // let new_model = "new_models/fn_d6_v2/fn_d6_v2_E_1300.bin";
    // test_model(new_model, tr_file);

    let mut nn_board = vec![-1, 0, -2, -1, 0, 0, 1, 1, 1];

    // for predict_board in &[nn_board] {
    //     let mn_board = &predict_board[0..=8].to_vec();
    //     show_board(&mn_board);

    //     let pl = if predict_board[9] == 1 { 1 } else { -1 };
    //     let pl_str = if pl == 1 { "X" } else { "0" };

    //     println!("Board_9: {}, pl: {pl}, pl_str: {pl_str}", predict_board[9]);

    //     let mut moves = Vec::new();
    //     minimax_multi(&mn_board, 6, pl, true, &mut moves);
    //     println!("Best moves for player {pl_str} => {:?}", moves);
    //     println!("Models predictions:");
    //     println!("-------------------");
    //     for model_path in [old_model, new_model] {
    //         let mv = predict_from_pos(model_path, predict_board.to_vec());

    //         println!("({model_path}) => Player {pl_str}, Bot move: {:?}", mv);
    //     }
    // }

    show_board(&nn_board);
    play_fanorona(&mut nn_board, model_bin.as_str());
}
