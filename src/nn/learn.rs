use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::nn::init::init_matrixes;
#[allow(unused)]
use crate::{
    mat,
    maths::{
        activations::{relu::ReLU, sigmoid::Sigmoid, softmax::Softmax},
        collectors::{mat::*, vec::*},
    },
    nn::NeuralNetwork,
    vector,
};

impl NeuralNetwork {
    #[allow(unused, non_snake_case)]
    pub fn batch_forward(&self, X: &Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut A_vec: Vec<Matrix> = Vec::new();
        let mut Z_vec: Vec<Matrix> = Vec::new();

        let mut Inp = X;

        for k in 0..self.ln - 1 {
            let Zk = (Inp * &self.weights[k]).add_each_line(&self.biases[k]);
            // set up this sequence to avoid cloning and borrowing (costs so much)
            A_vec.push(Zk.map_elms(|x| x.relu()));
            Z_vec.push(Zk);
            Inp = &A_vec.last().unwrap();
        }

        let l = self.ln - 1;
        // let n_l = self.ls[l];
        // let half = n_l / 2;

        let Zl = (Inp * &self.weights[l]).add_each_line(&self.biases[l]);
        let Al = Matrix(Zl.map_lines(|l| l.softmax()));

        Z_vec.push(Zl);
        A_vec.push(Al);

        (Z_vec, A_vec)
    }

    #[allow(unused, non_snake_case)]
    pub fn batch_grads(&mut self, X: &Matrix, Y: &Matrix) -> (Vec<Matrix>, Vec<Matrix>, f64) {
        let (Z, A) = self.batch_forward(X);

        let batch_size = X.len();
        let d = 1.0 / batch_size as f64;

        //  --------------- CALCULATION OF BATCH LOSS
        let A_out = &A[self.ln - 1]; // (batch_size, 18)
        //
        // // Pour éviter log(0), ajoute un petit epsilon
        let eps = 1e-12;
        let log_probs = A_out.map_elms(|x| (x.max(eps)).ln());
        //
        // // Perte = -sum(Y * log(A)) par ligne, puis moyenne
        let loss_per_sample: Vec<f64> = Y
            .map_zip_el(&log_probs, |a, b| a * b)
            .map_lines(|row| -row.sum());
        let batch_loss: f64 = Vector(loss_per_sample).mean();
        // ---------------------------------

        // initializations of gradients
        let mut GW: Vec<Matrix> = init_matrixes(&self.ls, self.is, false);
        let mut GZ: Vec<Matrix> = self
            .ls
            .par_iter()
            .map(|&n| mat![vector![0.0;n];batch_size])
            .collect();

        let mut A_prev = X;

        // OUTPUT LAYER (dZ)
        GZ[self.ln - 1] = d * &(&A[self.ln - 1] - Y);
        for k in (1..=self.ln - 1).rev() {
            GW[k] = &A[k - 1].tr() * &GZ[k];
            let mut GA_prev = &GZ[k] * &self.weights[k].tr();
            // let G_sig = A[k - 1].map_elms(|e| e * (e - 1.0));
            let G_relu = A[k - 1].map_elms(|e| if e > 0.0 { 1.0 } else { 0.0 });
            GZ[k - 1] = GA_prev.map_zip_el(&G_relu, |x, y| x * y); // element by element product
        }
        GW[0] = &X.tr() * &GZ[0];

        (GW, GZ, batch_loss)
    }

    #[allow(unused, non_snake_case)]
    pub fn batch_backward(&mut self, GW: &Vec<Matrix>, GZ: &Vec<Matrix>) {
        for k in 0..self.ln {
            self.weights[k] = &self.weights[k] - &(self.lr * &GW[k]);
            // Mean of each colums
            let gz_cm: Vector = Vector(GZ[k].map_cols(|v| v.mean()));
            self.biases[k] = &self.biases[k] - &(self.lr * &gz_cm);
        }
    }
}
