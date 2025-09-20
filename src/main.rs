use rand::seq::SliceRandom;

use crate::{
    nn::NeuralNetwork,
    utils::{load_positions, one_hot, save_parameters_binary},
};

mod dataset_gen;
mod game;
mod nn;
mod utils;

const LEARNING_RATE: f64 = 0.1;
const EPOCHS: usize = 50;
const MODEL_PARAMS_DIR: &str = "models";
const TESTS_DATA_FILE: &str = "datasets/tests.txt";
const TRAIN_DATA_FILE: &str = "datasets/trainings.txt";

fn main() {
    // CHECK THE DATASET
    // println!("Dataset de l'entrainement: ");
    // inspect_dataset(TRAIN_DATA_FILE);
    // println!("Dataset de tests");
    // inspect_dataset(TESTS_DATA_FILE);

    // TESTING PREDICTIONS
    // let mut nn: NeuralNetwork =
    //     NeuralNetwork::from_file(format!("{MODEL_PARAMS_DIR}/FN_BOT_E49.bin"));
    // predict_moves(&mut nn, TESTS_DATA_FILE);

    // TRAINING
    // let layer_sizes: Vec<usize> = vec![34, 120, 18];
    // let mut nn: NeuralNetwork = NeuralNetwork::new(&layer_sizes, 46, LEARNING_RATE);

    // train(&mut nn, TRAIN_DATA_FILE, EPOCHS);
}

#[allow(unused)]
fn predict_moves(nn: &mut NeuralNetwork, filename: &str) {
    let mut shuffler = rand::thread_rng();
    let mut test_pos: Vec<Vec<i32>> = load_positions(filename);

    test_pos.shuffle(&mut shuffler);

    let mut correct = 0;
    let mut pos_len = 0;
    for (idx, pos) in test_pos.iter().enumerate() {
        let p = &pos[0..=8];
        let player = &pos[9];
        let d_star: usize = pos[10] as usize;
        let a_star: usize = pos[11] as usize;
        let cv_pos = one_hot(p.to_vec(), *player as usize);

        let ((d, pd), (a, pa)) = nn.predict(&cv_pos);
        if d == d_star && a == a_star {
            correct += 1;
        }
        pos_len += 1;
    }

    println!(
        "{}/{} => Précision:{:.2} %",
        correct,
        pos_len,
        100.0 * correct as f64 / pos_len as f64,
    );
}

#[allow(unused)]
fn train(nn: &mut NeuralNetwork, filename: &str, epochs: usize) {
    let mut training_pos: Vec<Vec<i32>> = load_positions(filename);

    let mut shuffler = rand::thread_rng();
    for epoch in 0..epochs {
        training_pos.shuffle(&mut shuffler);

        for (_, pos) in training_pos.iter().enumerate() {
            let p = &pos[0..=8];
            let player = &pos[9];
            let cv_pos = one_hot(p.to_vec(), *player as usize);

            let d_star: usize = pos[10] as usize;
            let a_star: usize = pos[11] as usize;

            // update the parameters
            nn.back_prop(&cv_pos, d_star, a_star + 9);
        }

        // Vérifier les poids de la dernière couche
        let w_sum: f64 = nn.weights[nn.ln - 1].iter().flatten().sum();
        let b_sum: f64 = nn.biases[nn.ln - 1].iter().sum();
        println!(
            "Poids dernière couche somme: {:.6}, Biais somme: {:.6}",
            w_sum, b_sum
        );

        predict_moves(nn, TESTS_DATA_FILE);
        // Save parameters after each epoch
        if let Err(e) =
            save_parameters_binary(nn, format!("{MODEL_PARAMS_DIR}/FN_BOT_E{epoch}.bin"))
        {
            eprintln!(
                "Erreur lors de la sauvegarde binaire à l'époque {}: {}",
                epoch + 1,
                e
            );
        }
    }
}
