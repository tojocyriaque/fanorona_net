use std::fs::*;
use std::io::*;

use std::fs::*;
use std::io::*;
use crate::nn::*;
use crate::nn::init::one_hot;
use crate::data::restructured_datasets::all_possible_actions;
use crate::games::fanorona::{possible};

#[allow(unused)]
pub fn predict_moves(nn: &mut NeuralNetwork, filename: &str) -> (f64, f64, f64) {
    let file = std::fs::File::open(filename).unwrap();
    let reader = std::io::BufReader::new(file);

    let mut correct_exact = 0.0;
    let mut correct_soft = 0.0;
    let mut pos_len = 0.0;
    let mut loss = 0.0;

    let all_actions = all_possible_actions();

    for line_result in reader.lines() {
        let pos: Vec<i32> = line_result.unwrap()
            .split(" ")
            .map(|s| s.parse().unwrap())
            .collect();

        let p = &pos[0..=8];
        let player = pos[9];
        let best_idx: usize = *pos.last().unwrap() as usize; // dernier élément = meilleur coup

        // récupérer les indices des coups légaux
        let legal_moves = possible(&p.to_vec(), player as i32);
        let legal_moves_idx: Vec<usize> = legal_moves.iter()
                .filter_map(|m| all_actions.iter().position(|a| a == m))
                .collect();

        let cv_pos = one_hot(p.to_vec(), player as usize);

        // target
        let mut target: Vec<f64> = vec![0.0; all_actions.len()];
        target[best_idx] = 1.0;
        for &idx in legal_moves_idx.iter() {
            if idx != best_idx {
                target[idx] = 0.5; // petite pénalisation pour coups légaux non optimaux
            }
        }

        // Normalisation
        let sum: f64 = target.iter().sum();
        for i in 0..target.len() {
            target[i] /= sum;
        }

        // softmax du réseau
        let (_z, a) = nn.feed_forward(cv_pos.clone());
        let sf = a.last().unwrap();

        // Loss : cross-entropy avec soft target
        for i in 0..sf.len() {
            loss -= target[i] * sf[i].ln();
        }

        // Accuracy
        let (pred_idx, _) = nn.predict(cv_pos);
        if pred_idx == best_idx {
            correct_exact += 1.0;
            correct_soft += 1.0;
        } else if legal_moves_idx.contains(&pred_idx) {
            correct_soft += 0.5; // demi-point si coup légal
        }

        pos_len += 1.0;
    }

    loss /= pos_len;
    let acc_exact = 100.0 * correct_exact / pos_len;
    let acc_soft = 100.0 * correct_soft / pos_len;

    (acc_exact, acc_soft, loss)
}



#[allow(unused)]
pub fn predict_from_pos(model: &str, pos: Vec<i32>) -> (usize, usize) {
    let nn = NeuralNetwork::from_file(model.to_string());
    let p = &pos[0..=8];
    let player = pos[9];
    let cv_pos = one_hot(p.to_vec(), player as usize);

    let (action_idx, _prob) = nn.predict(cv_pos);
    all_possible_actions()[action_idx]
}


#[allow(unused)]
pub fn test_model(model_path: &str, filename: &str) {
    let mut nn = NeuralNetwork::from_file(model_path.to_string());
    let model_name = model_path.split("/").last().unwrap();

    let (acc_exact, acc_soft, loss) = predict_moves(&mut nn, filename);
    println!(
        "Model {model_name} (accuracy: {acc_exact:.4}%) (loss: {loss:.4})"
    );
}

