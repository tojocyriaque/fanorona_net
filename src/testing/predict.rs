use std::fs::*;
use std::io::*;

use crate::nn::init::one_hot;
use crate::nn::*;
use crate::data::restructured_datasets::{all_possible_actions};

#[allow(unused)]
pub fn predict_moves(nn: &mut NeuralNetwork, filename: &str) -> (f64, f64) {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let mut correct = 0;
    let mut pos_len = 0;
    let mut loss = 0.0;

    for line_result in reader.lines() {
        let pos: Vec<i32> = line_result
            .unwrap()
            .split(" ")
            .map(|s| s.parse().unwrap())
            .collect();

        let p = &pos[0..=8];
        let player = pos[9];
        let action_star: usize = *pos.last().unwrap() as usize; // dernier élément
        let cv_pos = one_hot(p.to_vec(), player as usize);

        let (action_pred, prob) = nn.predict(cv_pos);
        loss -= prob.ln();

        if action_pred == action_star {
            correct += 1;
        }
        pos_len += 1;
    }

    let acc = 100.0 * correct as f64 / pos_len as f64;
    loss /= pos_len as f64;

    (acc, loss)
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

    let (acc, loss) = predict_moves(&mut nn, filename);
    println!(
        "Model {model_name} (accuracy: {acc:.4}%) (loss: {loss:.4})"
    );
}

