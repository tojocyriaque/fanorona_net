use std::fs::*;
use std::io::*;

use crate::nn::init::one_hot;
use crate::nn::*;

#[allow(unused)]
pub fn predict_moves(nn: &mut NeuralNetwork, filename: &str) -> ((f64, f64, f64), f64) {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let mut correct = 0;
    let mut correct_d = 0;
    let mut correct_a = 0;

    let mut pos_len = 0;

    let mut loss = 0.0;

    let mut acc = 0.0;
    let mut acc_a = 0.0;
    let mut acc_d = 0.0;

    for (idx, line_result) in reader.lines().enumerate() {
        // read line one by one
        let pos: Vec<i32> = line_result
            .unwrap()
            .split(" ")
            .map(|s| s.parse().unwrap())
            .collect();

        let p = &pos[0..=8];
        let player = &pos[9];
        let d_star: usize = pos[10] as usize;
        let a_star: usize = pos[11] as usize;
        let cv_pos = one_hot(p.to_vec(), *player as usize);

        let ((d, pd), (a, pa)) = nn.predict(cv_pos);
        loss -= pd.ln() + pa.ln();

        if d == d_star {
            correct_d += 1;
        }
        if a == a_star {
            correct_a += 1;
        }
        if d == d_star && a == a_star {
            correct += 1;
        }
        pos_len += 1;
    }

    let acc_a = 100.0 * correct_a as f64 / pos_len as f64;
    let acc_d = 100.0 * correct_d as f64 / pos_len as f64;
    let acc = 100.0 * correct as f64 / pos_len as f64;
    loss = loss / pos_len as f64;

    // (loss, accuracy)
    ((acc_a, acc_d, acc), loss)
}

#[allow(unused)]
pub fn predict_from_pos(model: &str, pos: Vec<i32>) -> (usize, usize) {
    let mut nn = NeuralNetwork::from_file(model.to_string());
    let p = &pos[0..=8];
    let player = &pos[9];
    let cv_pos = one_hot(p.to_vec(), *player as usize);

    let ((d, pd), (a, pa)) = nn.predict(cv_pos);
    (d, a)
}

#[allow(unused)]
pub fn test_model(model_path: &str, filename: &str) {
    let mut nn = NeuralNetwork::from_file(model_path.to_string());
    let model_name = model_path.split("/").last().unwrap();

    let ((acc_a, acc_d, acc), loss) = predict_moves(&mut nn, filename);
    println!(
        "Model {model_name} (accuracy: {acc:.4}%, accuracy_d: {acc_d:.4}%, accuracy_a: {acc_a:.4}%) (loss: {loss:.4})"
    );
}
