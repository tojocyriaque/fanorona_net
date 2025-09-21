// ================================ IMPLEMENTATIONS FOR SOFTMAX ====================================

use crate::maths::collectors::vec::VecStruct;

pub trait Softmax {
    type Output;
    fn softmax(self) -> Self::Output;
}

impl Softmax for Vec<f64> {
    type Output = Self;
    fn softmax(self) -> Self::Output {
        let max_y = self.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f64> = self
            .iter()
            .map(|&y_i| (y_i - max_y).clamp(-100.0, 100.0).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        if sum == 0.0 {
            vec![1.0 / self.len() as f64; self.len()] // Distribution uniforme si somme nulle
        } else {
            exps.iter().map(|&x| x / sum).collect()
        }
    }
}

impl Softmax for &Vec<f64> {
    type Output = Vec<f64>;
    fn softmax(self) -> Self::Output {
        self.clone().softmax()
    }
}

impl Softmax for VecStruct {
    type Output = Self;
    fn softmax(self) -> Self::Output {
        VecStruct(self.0.softmax())
    }
}

impl Softmax for &VecStruct {
    type Output = VecStruct;
    fn softmax(self) -> Self::Output {
        VecStruct((&self.0).softmax())
    }
}
