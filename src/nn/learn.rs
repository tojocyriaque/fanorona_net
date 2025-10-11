#[allow(unused)]
use crate::{
    maths::{
        activations::{relu::ReLU, sigmoid::Sigmoid, softmax::Softmax},
        collectors::{mat::*, vec::*},
    },
    nn::{
        NeuralNetwork,
        init::{init_matrixes, init_vectors},
    },
};


impl NeuralNetwork {
    // forward propagation returning activations for all layers
    pub fn feed_forward(&self, x: Vector) -> (Vec<Vector>, Vec<Vector>) {
        let mut z: Vec<Vector> = Vec::new();
        let mut a: Vec<Vector> = Vec::new();

        // input layer

        // println!("x:{:?}; w:{:?}", x.len(), self.weights[0].dim());
        z.push(&(&x * &self.weights[0]) + &self.biases[0]);
        // a.push(z[0].relu()); // if using relu
        a.push(z[0].sigmoid()); // if using sigmoid

        for i in 1..self.ln {
            z.push(&(&a[i - 1] * &self.weights[i]) + &self.biases[i]);
            if i < self.ln - 1 {
                // a.push(z[i].relu());
                a.push(z[i].sigmoid());
            }
        }

        let sf = z[self.ln - 1].softmax(); // 32 éléments
        a.push(sf);
        (z, a)
    }

    pub fn compute_gradients(
        &self,
        x: &Vector,
        best_idx: usize,
        legal_moves_idx: &Vec<usize>,
        // d_star: usize,
        // a_star: usize,
    ) -> (Vec<Vector>, Vec<Matrix>) {
        let (z, a) = self.feed_forward(x.clone());

        let mut dz: Vec<Vector> = init_vectors(&self.ls, false);
        let mut dw: Vec<Matrix> = init_matrixes(&self.ls, self.is, false);

        // Compute dz for output layer (loss gradient)
        let mut target: Vec<f64> = vec![0.0; self.ls[self.ln - 1]];

        // best_idx = index du meilleur coup dans all_possible_actions
        target[best_idx] = 1.0;

        // legal_moves_idx = indices de tous les coups légaux dans all_possible_actions
        for &idx in legal_moves_idx.iter() {
            if idx != best_idx {
                target[idx] = 0.1; // petite pénalisation
            }
        }

        // Normalisation
        let sum: f64 = target.iter().sum();
        for i in 0..target.len() {
            target[i] /= sum;
        }

        for i in 0..self.ls[self.ln - 1] {
            dz[self.ln - 1][i] = a[self.ln - 1][i] - target[i];
            // let kron_di = if i == d_star { 1.0 } else { 0.0 };
            // let kron_ai = if i == a_star { 1.0 } else { 0.0 };
            // dz[self.ln - 1][i] = a[self.ln - 1][i] - kron_di - kron_ai;
        }

        // Backpropagate through hidden layers
        for k in (0..=self.ln - 1).rev() {
            let m = self.ls[k];
            let a_1 = if k == 0 { x } else { &a[k - 1] }; // previous activation
            let n = a_1.len();

            // Gradient for weights
            for i in 0..m {
                for j in 0..n {
                    dw[k][i][j] = dz[k][i] * a_1[j];
                }
            }

            // Gradient for previous layer
            if k > 0 {
                for i in 0..n {
                    let mut da_i = 0.0;
                    for j in 0..m {
                        da_i += dz[k][j] * self.weights[k][j][i];
                    }
                    dz[k - 1][i] = da_i * a[k - 1][i] * (1.0 - a[k - 1][i]); // sigmoid
                    // dz[k - 1][i] = da_i * if z[k - 1][i] > 0.0 { 1.0 } else { 0.0 }; // relu
                }
            }
        }

        (dz, dw)
    }

    /// Update weights and biases using precomputed gradients and learning rate
    pub fn apply_gradients(&mut self, dz: &[Vector], dw: &[Matrix]) {
        for k in 0..self.ln {
            self.biases[k] = &self.biases[k] - &(self.lr * &dz[k]);
            self.weights[k] = &self.weights[k] - &(self.lr * &dw[k]);
        }
    }

    // Classic backprop calling compute_gradients then apply_gradients
    // pub fn back_prop(&mut self, x: &Vector, d_star: usize, a_star: usize) {
    //     let (dz, dw) = self.compute_gradients(x, d_star, a_star);
    //     self.apply_gradients(&dz, &dw);
    // }
}
