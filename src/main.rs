use std::{fs::read_to_string, os::unix::thread};

use itertools::Itertools;
use rand::{Rng, random, seq::SliceRandom, thread_rng};

use crate::{nn::NeuralNetwork, utils::one_hot};

mod dataset_gen;
mod game;
mod nn;
mod utils;

const LR: f64 = 0.0001;
const EPOCHS: usize = 50;

fn main() {
    let mut positions: Vec<Vec<i32>> = Vec::new();
    let mut rnd = thread_rng();
    let mut _c = 0;
    for line in read_to_string("datasets.txt").unwrap().lines() {
        // if _c == 7000 {
        //     break;
        // }
        let pos: Vec<i32> = line.split(" ").map(|u| u.parse().unwrap()).collect();
        positions.push(pos.clone());
        _c += 1;
    }

    // positions.shuffle(&mut rnd);
    // for pos in &positions {
    //     println!("{}", pos.into_iter().join(" "));
    // }
    let mut nn: NeuralNetwork = NeuralNetwork::new(3, &vec![128, 64, 18], 46, LR);
    train(&mut nn, positions, EPOCHS);
}

fn train(nn: &mut NeuralNetwork, tr_pos: Vec<Vec<i32>>, epochs: usize) {
    for epoch in 0..epochs {
        let mut loss = 0.0;
        let mut count = 1;
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
            let sample_loss: f64 = -pa.max(1e-10).ln() - pd.max(1e-10).ln();
            println!("Target: {:?} , Prediction: {:?}", (d_star, a_star), (d, a));
            if d == d_star && a == a_star + 9 {
                correct += 1;
            }

            // Safely update loss
            loss += sample_loss;
            count += 1;
        });

        // println!(
        //     "Époque {}/{} terminée, perte moyenne: {} Correctes: {}",
        //     epoch + 1,
        //     epochs,
        //     loss / count as f64,
        //     correct
        // );
    }
}
