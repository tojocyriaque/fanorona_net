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
        a.push(z[0].relu());

        for i in 1..self.ln {
            z.push(&(&a[i - 1] * &self.weights[i]) + &self.biases[i]);
            if i < self.ln - 1 {
                a.push(z[i].relu());
            }
        }

        let mut sf = z[self.ln - 1][0..=8].softmax();
        sf.extend(z[self.ln - 1][9..=17].softmax());

        a.push(sf);
        (z, a)
    }

    pub fn back_prop(&mut self, x: &Vector, d_star: usize, a_star: usize) {
        let (z, a) = self.feed_forward(x.clone());

        // Partial derivatives
        let mut dz: Vec<Vector> = init_vectors(&self.ls, false);
        let mut dw: Vec<Matrix> = init_matrixes(&self.ls, self.is, false);

        // LOSS: - log(Pd(d*)) -log(Pa(a*));
        // 1) dL / dz(n)i = a(n)i - kron(d,i) - krond(a,i)
        // 2) dL / da(k-1)i = SUM( dz(k)j * w(k)ji )j
        // 3) dL / dz(k-1)i = (dL / da(k-1)i) * a(k-1)i * (1 - a(k-1)i)

        for i in 0..self.ls[self.ln - 1] {
            let kron_di = if i == d_star { 1.0 } else { 0.0 };
            let kron_ai = if i == a_star { 1.0 } else { 0.0 };
            dz[self.ln - 1][i] = a[self.ln - 1][i] - kron_di - kron_ai;
        }

        for k in (0..=self.ln - 1).rev() {
            // m x n matrix
            let m = self.ls[k];
            let a_1 = if k == 0 { x } else { &a[k - 1] }; // previous activation
            let n = a_1.len();

            // dL / dW(k)ij = (dL/dz(k)i) * a(k-1)j
            for i in 0..m {
                for j in 0..n {
                    dw[k][i][j] = dz[k][i] * a_1[j]
                }
            }

            // dz(k-1)
            if k > 0 {
                for i in 0..n {
                    let mut da_i = 0.0;
                    for j in 0..m {
                        da_i += dz[k][j] * self.weights[k][j][i];
                    }

                    // dz[k - 1][i] = da_i * a[k - 1][i] * (1.0 - a[k - 1][i]) // (sigmoid)
                    dz[k - 1][i] = da_i * (if z[k - 1][i] > 0.0 { 1.0 } else { 0.0 }); // (relu)
                }
            }
        }

        // Updating parameters
        for k in 0..self.ln {
            self.biases[k] = &self.biases[k] - &(self.lr * &dz[k]);
            self.weights[k] = &self.weights[k] - &(self.lr * &dw[k]);
        }
    }
}
