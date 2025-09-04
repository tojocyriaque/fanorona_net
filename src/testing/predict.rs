use std::fs::*;
use std::io::*;

use crate::nn::init::one_hot_fanorona;
use crate::nn::*;

#[allow(unused)]
pub fn predict_moves(nn: &mut NeuralNetwork, filename: &str) -> (f64, f64) {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let mut correct = 0;
    let mut pos_len = 0;
    let mut loss = 0.0;
    let mut acc = 0.0;
    let mut valid_count = 0;

    for (idx, line_result) in reader.lines().enumerate() {
        // read line one by one
        let pos: Vec<f64> = line_result
            .unwrap()
            .split(" ")
            .map(|s| s.parse().unwrap())
            .collect();

        let brd = &pos[0..=8].to_vec();
        let player = &pos[9];
        // let d_star: usize = pos[10] as usize;
        // let a_star: usize = pos[11] as usize;
        let cv_pos = one_hot_fanorona(brd.to_vec(), *player as usize);

        let ((d, a), proba, sf) = nn.predict(cv_pos);

        let board: Vec<i32> = brd.iter().map(|&f| f as i32).collect();
        let pl: i32 = if *player == 1.0 { 1 } else { -1 };

        let y = pos[10..].to_vec();
        let (idx, _) = y
            .iter()
            .enumerate()
            .max_by(|(_, v1), (_, v2)| v1.total_cmp(&v2))
            .unwrap();

        loss -= sf[idx].ln();

        let d_star = idx / 9;
        let a_star = idx % 9;

        if (d_star, a_star) == (d, a) {
            correct += 1
        }

        // valid move
        if board[d] * pl > 0 && board[a] == 0 {
            valid_count += 1;
        }

        pos_len += 1;
    }

    let acc = 100.0 * correct as f64 / pos_len as f64;
    loss = loss / pos_len as f64;

    println!(
        "Valid accuracy: {:.4}%",
        100.0 * valid_count as f64 / pos_len as f64
    );
    // (loss, accuracy)
    (acc, loss)
}

#[allow(unused)]
pub fn predict_from_pos(model: &str, pos: Vec<i32>) -> (usize, usize) {
    let mut nn = NeuralNetwork::from_file(model.to_string());
    let p: Vec<f64> = pos[0..=8].iter().map(|&f| f as f64).collect();
    let player = &pos[9];
    let cv_pos = one_hot_fanorona(p, *player as usize);

    let ((d, a), proba, sf) = nn.predict(cv_pos);
    (d, a)
}

#[allow(unused)]
pub fn test_model(model_path: &str, filename: &str) {
    let mut nn = NeuralNetwork::from_file(model_path.to_string());
    let model_name = model_path.split("/").last().unwrap();

    let (acc, loss) = predict_moves(&mut nn, filename);
    println!("Model {model_name} => (accuracy: {acc:.4}% , loss: {loss:.4})");
}
