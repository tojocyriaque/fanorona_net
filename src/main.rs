use std::fs::read_to_string;

use rand::{seq::SliceRandom, thread_rng};

use crate::{nn::NeuralNetwork, utils::one_hot};

mod dataset_gen;
mod game;
mod nn;
mod utils;

const LR: f64 = 0.54;
const EPOCHS: usize = 10;

fn main() {
    let mut positions: Vec<Vec<i32>> = Vec::new();
    let mut c = 0;
    let mut random = thread_rng();
    for line in read_to_string("datasets.txt").unwrap().lines() {
        // if c == {
        //     break;
        // }
        let pos: Vec<i32> = line.split(" ").map(|u| u.parse().unwrap()).collect();
        positions.push(pos.clone());
        // println!("{:?}", pos);
        c += 1;
    }

    positions.shuffle(&mut random);
    let mut nn: NeuralNetwork = NeuralNetwork::new(3, &vec![64, 128, 18], 46, LR);
    train(&mut nn, positions, EPOCHS);
}

fn train(nn: &mut NeuralNetwork, tr_pos: Vec<Vec<i32>>, epochs: usize) {
    for epoch in 0..epochs {
        let mut loss = 0.0;
        let mut count = 0;
        for pos in &tr_pos {
            let p = &pos[0..=8];
            let pl = &pos[9];
            let cv_pos = one_hot(p.to_vec(), *pl as usize);

            let d_star: usize = pos[10] as usize;
            let a_star: usize = pos[11] as usize;

            nn.back_prop(&cv_pos, d_star, a_star + 8);
            let ((_d, pd), (_a, pa)) = nn.predict(&cv_pos);
            let sample_loss: f64 = -pa.ln() - pd.ln();
            // println!("{}", sample_loss);
            loss += sample_loss;
            count += 1;

            // println!("{}, {}", _d, _a);
        }
        // println!(
        //     "Époque {}/{} terminée , perte moyenne: {}",
        //     epoch + 1,
        //     epochs,
        //     loss / count as f64
        // );
    }
}

// fn train(nn: &mut NeuralNetwork, train_positions: Vec<GBoard>, epochs: i32, lr: f64) {
//     for epoch in 0..epochs {
//         let mut loss = 0.0;
//         let mut count = 0;

//         for pos in &train_positions {
// let player = 1 + pos.iter().filter(|&u| *u == 0).count() % 2;
//             // let x: Vector = one_hot(pos.clone(), player);

//             let mut bm = 0;
//             // minimax(&pos, 6, [-1, 1][player - 1], &mut bm, true);

//             // let y: Vector = encode_sol(bm);
//             nn.train(x.clone(), y.clone(), lr);
//             let (_, _, _, a2) = nn.forward(x);

//             let sample_loss: f64 = -y.iter().zip(a2).map(|(&t, p)| t * p.ln()).sum::<f64>();

//             loss += sample_loss;
//             count += 1;
//         }
//         println!(
//             "Époque {}/{} terminée , perte moyenne: {}",
//             epoch + 1,
//             epochs,
//             loss / count as f64
//         );
//     }
// }
