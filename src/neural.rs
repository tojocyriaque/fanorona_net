use std::{
    fs::File,
    io::{BufRead, BufReader, Read, Write},
};

use ndarray::prelude::*;
use ndarray_rand::{
    rand::{seq::SliceRandom, thread_rng},
    rand_distr::{Distribution, Uniform},
};
use serde::{Deserialize, Serialize};

use crate::{dataset::load_dataset, fanorona3::one_hot_fanorona};

pub type Matrix = Array2<f64>;
pub type Vector = Array1<f64>;

// ==================== SAVING AND LOADING PARAMETERS =========================
//  Struct to represent the network parameters

#[derive(Debug, Serialize, Deserialize)]
#[allow(unused)]
pub struct Neural {
    input_size: usize,
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Vector>,
}

#[allow(unused)]
impl Neural {
    pub fn from_params(
        input_size: usize,
        layers: Vec<usize>,
        weights: Vec<Matrix>,
        biases: Vec<Vector>,
    ) -> Self {
        Neural {
            input_size,
            layers,
            weights,
            biases,
        }
    }

    pub fn config(&self) -> Vec<&[usize]> {
        self.weights.iter().map(|w| w.shape()).collect()
    }

    pub fn zeros(layers: Vec<usize>, input_size: usize) -> Self {
        let mut weights: Vec<Matrix> = Vec::new();
        let mut biases: Vec<Vector> = Vec::new();

        let mut mat = Matrix::zeros((input_size, layers[0]));
        weights.push(mat);
        biases.push(Vector::zeros(layers[0]));

        for i in 1..layers.len() {
            let rows = layers[i - 1];
            let cols = layers[i];
            mat = Matrix::zeros((rows, cols));
            weights.push(mat);
            biases.push(Vector::zeros(layers[i]));
        }

        Neural {
            input_size,
            layers,
            biases,
            weights,
        }
    }

    pub fn xavier(layers: Vec<usize>, input_size: usize) -> Self {
        // xavier uniforme
        let limit = (6.0 / (input_size + layers[0]) as f64).sqrt();
        let mut rng = thread_rng();
        let dist = Uniform::new(-limit, limit);
        // ---------------------------

        let mut weights: Vec<Matrix> = Vec::new();
        let mut biases: Vec<Vector> = Vec::new();
        let mut mat = Matrix::from_shape_fn((input_size, layers[0]), |_| dist.sample(&mut rng));
        weights.push(mat);

        biases.push(Vector::zeros(layers[0]));
        for i in 1..layers.len() {
            let rows = layers[i - 1];
            let cols = layers[i];
            mat = Matrix::from_shape_fn((rows, cols), |_| dist.sample(&mut rng));
            weights.push(mat);
            biases.push(Vector::zeros(layers[i]));
        }

        Neural {
            input_size,
            layers,
            biases,
            weights,
        }
    }

    #[allow(non_snake_case)]
    pub fn batch_forward(&mut self, X: &Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut Z: Vec<Matrix> = Vec::new();
        let mut A: Vec<Matrix> = Vec::new();

        let mut Inp = X;
        let layer_num = self.layers.len();
        for k in 0..layer_num - 1 {
            let Zk = Inp.dot(&self.weights[k]) + &self.biases[k];
            // _____________ ACTIVATION _________________
            // Relu
            let Ak = Zk.map(|&elt| if elt > 0.0 { elt } else { 0.0 });
            // sigmoid
            // let Ak = Zk.map(|&elt| 1.0 / (1.0 + elt.exp()));
            // -------------------------------------------
            Z.push(Zk);
            A.push(Ak);
            Inp = &A[k];
        }

        let l = layer_num - 1;
        let Zl = Inp.dot(&self.weights[l]) + &self.biases[l];
        Z.push(Zl);

        // Softmax
        // for numeric stability
        let max_vals = Z[l]
            .map_axis(Axis(1), |row| {
                *row.iter()
                    .max_by(|&a, b| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
            })
            .insert_axis(Axis(1));

        let mut Al = &Z[l] - max_vals;

        Al = Al.mapv(f64::exp);
        let exp_sums = Al.sum_axis(Axis(1)).insert_axis(Axis(1));
        Al = Al / exp_sums;

        A.push(Al);
        (Z, A)
    }

    #[allow(non_snake_case)]
    pub fn batch_grads(&mut self, X: Matrix, Y: Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        let (Z, A) = self.batch_forward(&X);
        let l = self.layers.len() - 1;
        let batch_size = X.shape()[0];

        let mut GZ: Vec<Matrix> = Vec::new();
        let mut GW: Vec<Matrix> = Vec::new();

        // Initialisation correcte
        for &size in &self.layers {
            GZ.push(Matrix::zeros((batch_size, size)));
        }
        GW.push(Matrix::zeros((self.input_size, self.layers[0])));
        for i in 1..self.layers.len() {
            GW.push(Matrix::zeros((self.layers[i - 1], self.layers[i])));
        }

        GZ[l] = (&A[l] - &Y) / batch_size as f64;

        for k in (1..=l).rev() {
            GW[k] = A[k - 1].t().dot(&GZ[k]);
            let GA_prev = &GZ[k].dot(&self.weights[k].t());
            // GZ[k - 1] = GA_prev * A[k - 1].map(|a| a * (a - 1.0)); // Sigmoid
            GZ[k - 1] = GA_prev * Z[k - 1].mapv(|z| if z > 0.0 { 1.0 } else { 0.0 }); // Relu
        }

        GW[0] = X.t().dot(&GZ[0]);
        (GZ, GW)
    }

    #[allow(non_snake_case)]
    pub fn batch_backward(&mut self, GW: Vec<Matrix>, GZ: Vec<Matrix>, lr: f64) {
        let layers_num = self.layers.len();
        for k in 0..layers_num {
            self.weights[k] = &self.weights[k] - &(&GW[k] * lr);
            let GB_k = GZ[k].map_axis(Axis(0), |row| row.mean().unwrap()); // Means of cols
            self.biases[k] = &self.biases[k] - &(GB_k * lr);
        }
    }

    #[allow(non_snake_case)]
    pub fn batch_learn(&mut self, X: Matrix, Y: Matrix, lr: f64) -> f64 {
        let batch_size = X.shape()[0];

        let (GZ, GW) = self.batch_grads(X.clone(), Y.clone());
        self.batch_backward(GW, GZ, lr);

        // CALCULATING THE LOSS AFTER
        let (_, A) = self.batch_forward(&X);
        let Al = A.last().unwrap();

        let LogAl = Al.mapv(|e| e.max(1e-12).ln());
        let sum_lines = (Y * LogAl).sum_axis(Axis(1));

        -sum_lines.sum() / batch_size as f64
    }

    #[allow(non_snake_case)]
    pub fn train(
        &mut self,
        epochs: usize,
        init_lr: f64,
        input_size: usize,
        batch_size: usize,
        tr_file: &str,
        save_dir: &str,
    ) {
        let mut learning_rate = init_lr;
        let eps = 0.001;
        let output_size = *self.layers.last().unwrap();
        let mut datasets = load_dataset(tr_file, input_size, output_size);
        let mut shuffler = thread_rng();

        let lr_schedule = 100;

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let epoch_start = std::time::Instant::now();
            let mut batch_count = 0;

            datasets.shuffle(&mut shuffler);

            for (correct_best, batch) in datasets.chunks(batch_size).enumerate() {
                let bs = batch.len();
                let mut x_s: Vec<Vector> = Vec::new();
                let mut y_s: Vec<Vector> = Vec::new();

                for (x, y) in batch {
                    let mut x_1hot = one_hot_fanorona(x.as_slice().unwrap());
                    x_s.push(x_1hot.clone().into());
                    y_s.push(y.clone());
                }

                let X: Matrix = Matrix::from_shape_fn((bs, self.input_size), |(i, j)| x_s[i][j]);
                let Y: Matrix = Matrix::from_shape_fn((bs, output_size), |(i, j)| y_s[i][j]);

                let batch_start = std::time::Instant::now();
                let batch_loss = self.batch_learn(X, Y, learning_rate); // forward + backward
                let batch_elapsed = batch_start.elapsed();
                // println!(
                //     "Batch {} loss: {batch_loss} ({:?})",
                //     correct_best + 1,
                //     batch_elapsed
                // );

                epoch_loss += batch_loss;
                batch_count += 1;
            }

            epoch_loss = epoch_loss / batch_count as f64;
            let epoch_elapsed = epoch_start.elapsed();

            // saving the model
            let model_file = format!("{save_dir}/epoch_{}.bin", epoch + 1);
            self.save_to_bin(model_file.as_str());

            println!(
                "Epoch {} loss: {epoch_loss} ({:?}) lr: {learning_rate}",
                epoch + 1,
                epoch_elapsed
            );
            println!("-----------------------------------------");

            // if (epoch + 1) % lr_schedule == 0 {
            //     learning_rate *= 0.7;
            // }

            if epoch_loss < eps {
                break;
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn predict_best<F>(&mut self, board: &[f64], one_hot_conv: F) -> usize
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let board_1hot = one_hot_conv(board);
        let X: Matrix = Matrix::from_shape_fn((1, board_1hot.len()), |(i, j)| board_1hot[j] as f64);
        let (_, A) = self.batch_forward(&X);
        let sf = A.last().unwrap().row(0);

        let (best_idx, proba) = sf
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        best_idx
    }

    pub fn test<F, V>(
        &mut self,
        input_size: usize,
        test_fname: &str,
        one_hot_conv: F,
        mv_valid: V,
    ) -> f64
    where
        F: Fn(&[f64]) -> Vec<f64>,
        V: Fn(&[f64], usize) -> bool,
    {
        let test_file = File::open(test_fname).unwrap();
        let reader = BufReader::new(test_file);

        let mut correct = 0.0;
        let mut valid = 0.0;
        let mut data_size = 0.0;
        for line_data in reader.lines() {
            let data_line = line_data.unwrap();
            let data: Vec<f64> = data_line.split(" ").map(|f| f.parse().unwrap()).collect();

            let y: Vector = data[input_size..].to_vec().into();
            let (correct_best, _pb) = y
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            let mut input = &data[0..input_size];
            let best_pred = self.predict_best(input, &one_hot_conv);

            if mv_valid(input, best_pred) {
                valid += 1.0;
            }

            if best_pred == correct_best {
                correct += 1.0
            }

            data_size += 1.0;
        }

        println!("Mouvements valides: {:.4}%", 100.0 * valid / data_size);
        correct / data_size
    }

    pub fn save_to_bin(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        let mut file = File::create(filename)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    pub fn load_from_bin(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(filename)?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;
        let neural: Neural = bincode::deserialize(&contents)?;
        Ok(neural)
    }
}
