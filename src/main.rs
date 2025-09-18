use std::fs::read_to_string;

use crate::{
    nn::NeuralNetwork,
    utils::{one_hot, save_parameters_binary},
};

mod dataset_gen;
mod game;
mod nn;
mod utils;

const LR: f64 = 0.1;
const EPOCHS: usize = 10;
const POS_LEN: usize = 28000;
const MODEL_PARAMS_DIR: &str = "models";
fn main() {
    // let params = load_parameters_binary(format!("{MODEL_PARAMS_DIR}/FN_BOT_E0.bin"))
    //     .expect("Échec du chargement");
    // println!("Taux d'apprentissage chargé : {}", params.learning_rate);
    // println!("Taille des couches: {:?}", params.layer_sizes);
    // println!(
    //     "Biais de la couche de sortie: {:?}",
    //     params.biases[params.layer_sizes.len() - 1]
    // );

    let layer_sizes: Vec<usize> = vec![128, 18];
    let mut nn: NeuralNetwork = NeuralNetwork::new(&layer_sizes, 46, LR);
    train(&mut nn, "datasets.txt", EPOCHS);
}

fn train(nn: &mut NeuralNetwork, data_filename: &str, epochs: usize) {
    for epoch in 0..epochs {
        let mut loss = 0.0;
        let mut count = 0;
        let mut correct = 0;

        for (idx, line) in read_to_string(data_filename).unwrap().lines().enumerate() {
            // Break after a certain number of positions (do not load all of the data)
            if idx + 1 == POS_LEN {
                break;
            }

            let pos: Vec<i32> = line.split(" ").map(|u| u.parse().unwrap()).collect();
            let p = &pos[0..=8];
            let player = &pos[9];
            let cv_pos = one_hot(p.to_vec(), *player as usize);

            let d_star: usize = pos[10] as usize;
            let a_star: usize = pos[11] as usize;

            nn.back_prop(&cv_pos, d_star, a_star + 9);
            let ((d, pd), (a, pa)) = nn.predict(&cv_pos);
            let sample_loss: f64 = -pa.ln() - pd.ln();

            // println!("Target: {:?} , Prediction: {:?}", (d_star, a_star), (d, a));
            if d == d_star && a == a_star {
                correct += 1;
            }

            // println!("Loss: {sample_loss}");
            loss += sample_loss;
            count += 1;
        }

        println!(
            "Époque {}/{} terminée, perte moyenne: {} Correctes: {}, Précision: {}",
            epoch + 1,
            epochs,
            loss / count as f64,
            correct,
            correct as f64 / POS_LEN as f64,
        );

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
