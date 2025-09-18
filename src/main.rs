use std::fs::read_to_string;

use crate::{nn::NeuralNetwork, utils::one_hot};

mod dataset_gen;
mod game;
mod nn;
mod utils;

const LR: f64 = 0.1;
const EPOCHS: usize = 10;
const POS_LEN: usize = 61238;

fn main() {
    let layer_sizes: Vec<usize> = vec![31, 18];
    let mut positions: Vec<Vec<i32>> = Vec::new();

    // let mut rnd = thread_rng();
    let mut _c = 0;
    for line in read_to_string("datasets.txt").unwrap().lines() {
        if _c == POS_LEN {
            break;
        }
        let pos: Vec<i32> = line.split(" ").map(|u| u.parse().unwrap()).collect();
        positions.push(pos.clone());
        _c += 1;
    }

    let mut nn: NeuralNetwork = NeuralNetwork::new(&layer_sizes, 46, LR);
    train(&mut nn, positions, EPOCHS);
}

fn train(nn: &mut NeuralNetwork, tr_pos: Vec<Vec<i32>>, epochs: usize) {
    for epoch in 0..epochs {
        let mut loss = 0.0;
        let mut count = 0;
        let mut correct = 0;

        tr_pos.iter().for_each(|pos| {
            let p = &pos[0..=8];
            let pl = &pos[9];
            let cv_pos = one_hot(p.to_vec(), *pl as usize);

            let d_star: usize = pos[10] as usize;
            let a_star: usize = pos[11] as usize;

            // Lock nn for back_prop
            nn.back_prop(&cv_pos, d_star, a_star + 9);
            let ((d, pd), (a, pa)) = nn.predict(&cv_pos);
            let sample_loss: f64 = -pa.ln() - pd.ln();

            // println!("Target: {:?} , Prediction: {:?}", (d_star, a_star), (d, a));
            if d == d_star && a == a_star {
                correct += 1;
            }

            loss += sample_loss;
            // println!("Loss: {sample_loss}");
            count += 1;
        });

        println!(
            "Époque {}/{} terminée, perte moyenne: {} Correctes: {}, Précision: {}",
            epoch + 1,
            epochs,
            loss / count as f64,
            correct,
            correct as f64 / POS_LEN as f64,
        );
    }
}
