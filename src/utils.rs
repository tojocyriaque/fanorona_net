pub mod datasets;
pub mod game;
pub mod inits;
pub mod loads;
pub mod matrixes;
pub mod vectors;

use vectors::Vector;

// ==================== ACTIVATION FUNCTIONS ====================
#[allow(dead_code)]
pub fn sigmoid(z: f64) -> f64 {
    let z = z.clamp(-100.0, 100.0);
    1.0 / (1.0 + (-z).exp())
}

#[allow(dead_code)]
pub fn re_lu(x: f64) -> f64 {
    if x >= 0. { x } else { 0. }
}

#[allow(dead_code)]
pub fn softmax(y: &Vector) -> Vector {
    let max_y = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vector = y
        .iter()
        .map(|&y_i| (y_i - max_y).clamp(-100.0, 100.0).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / y.len() as f64; y.len()] // Distribution uniforme si somme nulle
    } else {
        exps.iter().map(|&x| x / sum).collect()
    }
}
