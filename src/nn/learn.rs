use crate::{nn::NeuralNetwork, utils::inits::*, utils::matrixes::*, utils::vectors::*, utils::*};
use rayon::iter::*;

impl NeuralNetwork {
    // forward propagation returning activations for all layers
    pub fn feed_forward(&self, x: Vector) -> Vec2d {
        // println!("{:?}", self.biases[0]);
        let mut z: Vec2d = init_vectors(&self.ls, false);
        let mut a: Vec2d = init_vectors(&self.ls, false);

        // Hidden layers (sigmoid)
        for i in 0..self.ln {
            // if i is 0, the previous layer is the inputs
            let previous = if i == 0 { &x } else { &a[i - 1] };

            z[i] = vec_sum(&mat_vec_prod(&self.weights[i], previous), &self.biases[i]);
            // skip sigmoid for output layer
            if i < self.ln - 1 {
                a[i] = z[i].par_iter().map(|&u: &f64| sigmoid(u)).collect();
            }
        }

        // Output layer
        let sf1 = softmax(&z[self.ln - 1].clone()[0..=8].to_vec()); // first 9 elements
        let sf2 = softmax(&z[self.ln - 1].clone()[9..].to_vec()); // last 9 elements
        a[self.ln - 1] = [sf1.clone(), sf2.clone()].concat();

        // println!("{} {} => {}", sf1.len(), sf2.len(), a[self.ln - 1].len());
        return a;
    }

    pub fn back_prop(&mut self, x: &Vector, d_star: usize, a_star: usize) {
        let a: Vec2d = self.feed_forward(x.clone());
        // LOSS: - log(Pd(d*)) -log(Pa(a*));
        // dL / da

        let mut dz: Vec2d = init_vectors(&self.ls, false);
        let mut dw: Vec<Vec2d> = init_matrixes(&self.ls, self.is, false);
        for i in 0..self.ls[self.ln - 1] {
            let kron_di = if i == d_star { 1.0 } else { 0.0 }; // kroneker(d_star,i)
            let kron_ai = if i == a_star { 1.0 } else { 0.0 }; // kroneker(a_star,i)
            dz[self.ln - 1][i] = a[self.ln - 1][i] - kron_di - kron_ai;
        }

        // Descent for the hidden layers (ln-1 -> 0)
        for k in (0..=self.ln - 1).rev() {
            // previous activation (if it is the first layer then the prev_act is x)
            let prev_activation = if k == 0 { &x } else { &a[k - 1] };

            // dwk
            // db = dz
            for i in 0..self.ls[k] {
                for j in 0..prev_activation.len() {
                    dw[k][i][j] = dz[k][i] * prev_activation[j];
                }
            }

            // dz(k-1)
            if k > 0 {
                for i in 0..self.ls[k - 1] {
                    let mut da_i = 0.0; // d(a-1)i (not necessary to add this in RAM)
                    for j in 0..self.ls[k] {
                        da_i += dz[k][j] * self.weights[k][j][i];
                    }

                    // dzk (sigmoid except for the output layer)
                    dz[k - 1][i] = da_i * a[k - 1][i] * (1.0 - a[k - 1][i]);
                }
            }
        }

        // Updating parameters
        for k in 0..self.ln {
            // Size of the previous layer
            let prev_len = if k > 0 { self.ls[k - 1] } else { x.len() };

            // W = W - lr * DW
            for i in 0..self.ls[k] {
                for j in 0..prev_len {
                    self.weights[k][i][j] -= self.lr * dw[k][i][j];
                }
            }

            // B = B - lr * DB (But db = dz)
            for i in 0..self.ls[k] {
                self.biases[k][i] -= self.lr * dz[k][i];
            }
        }
    }
}
