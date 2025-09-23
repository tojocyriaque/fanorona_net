use crate::{
    maths::{
        activations::{sigmoid::Sigmoid, softmax::Softmax},
        collectors::mat::*,
        collectors::vec::*,
    },
    nn::{
        NeuralNetwork,
        init::{init_matrixes, init_vectors},
    },
};

impl NeuralNetwork {
    // forward propagation returning activations for all layers
    pub fn feed_forward(&self, x: Vector) -> Vec<Vector> {
        let mut z: Vec<Vector> = Vec::new();
        let mut a: Vec<Vector> = Vec::new();

        // input layer

        // println!("x:{:?}; w:{:?}", x.len(), self.weights[0].dim());
        z.push(&(&x * &self.weights[0]) + &self.biases[0]);
        a.push((&z[0]).sigmoid());

        for i in 1..self.ln {
            z.push(&(&a[i - 1] * &self.weights[i]) + &self.biases[i]);
            if i < self.ln - 1 {
                a.push((&z[i]).sigmoid());
            }
        }

        let sf1 = (&z[self.ln - 1][0..=8].to_vec()).softmax();
        let sf2 = (&z[self.ln - 1][9..=17].to_vec()).softmax();

        a.push(Vector([sf1, sf2].concat()));
        a
    }

    pub fn back_prop(&mut self, x: &Vector, d_star: usize, a_star: usize) {
        let a: Vec<Vector> = self.feed_forward(x.clone());

        // Partial derivatives
        let mut dz: Vec<Vector> = init_vectors(&self.ls, false);
        let mut dw: Vec<Matrix> = init_matrixes(&self.ls, self.is, false);

        // LOSS: - log(Pd(d*)) -log(Pa(a*));

        // dL / dz(n)i = a(n)i - kron(d,i) - krond(a,i)
        // dL / da(k-1)i = SUM( dz(k)j * w(k)ji )j
        // dL / dz(k-1)i = (dL / da(k-1)i) * a(k-1)i * (1 - a(k-1)i)

        for i in 0..self.ls[self.ln - 1] {
            let kron_di = if i == d_star { 1.0 } else { 0.0 };
            let kron_ai = if i == a_star { 1.0 } else { 0.0 };
            dz[self.ln - 1][i] = a[self.ln - 1][i] - kron_di - kron_ai;
        }

        for k in (0..=self.ln - 1).rev() {
            // m x n matrix
            let m = self.ls[k];
            let prev_act = if k == 0 { x } else { &a[k - 1] };
            let n = prev_act.len();

            // dL / dW(k)ij = (dL/dz(k)i) * a(k-1)j
            for i in 0..m {
                for j in 0..n {
                    dw[k][i][j] = dz[k][i] * prev_act[j]
                }
            }

            // dz(k-1)
            if k > 0 {
                for i in 0..n {
                    let mut da_i = 0.0; // da(k-1)i
                    for j in 0..m {
                        da_i += dz[k][j] * self.weights[k][j][i];
                    }
                    dz[k - 1][i] = da_i * a[k - 1][i] * (1.0 - a[k - 1][i])
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
