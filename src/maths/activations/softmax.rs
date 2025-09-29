// ================================ IMPLEMENTATIONS FOR SOFTMAX ====================================

use crate::{maths::collectors::vec::Vector, vector};

pub trait Softmax {
    type Output;
    fn softmax(self) -> Self::Output;
}

impl Softmax for &Vec<f64> {
    type Output = Vector;
    fn softmax(self) -> Self::Output {
        let max_y = self.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f64> = self
            .iter()
            .map(|&y_i| (y_i - max_y).clamp(-100.0, 100.0).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        if sum == 0.0 {
            vector![1.0 / self.len() as f64; self.len()] // Distribution uniforme si somme nulle
        } else {
            Vector(exps.iter().map(|&x| x / sum).collect())
        }
    }
}

impl Softmax for &Vector {
    type Output = Vector;
    fn softmax(self) -> Self::Output {
        (&self.0).softmax()
    }
}

impl Softmax for &[f64] {
    type Output = Vector;
    fn softmax(self) -> Self::Output {
        self.to_vec().softmax()
    }
}
