use std::fs::*;
use std::io::*;

use crate::nn::init::one_hot;
use crate::nn::*;

#[allow(unused)]
pub fn predict_moves(nn: &mut NeuralNetwork, filename: &str) -> f64 {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let mut correct = 0;
    let mut pos_len = 0;

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
        if d == d_star && a == a_star {
            correct += 1;
        }
        pos_len += 1;
    }
    100.0 * correct as f64 / pos_len as f64
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

    let accuracy = predict_moves(&mut nn, filename);
    println!("Model {model_name} accuracy: {accuracy:.2}%");
}
