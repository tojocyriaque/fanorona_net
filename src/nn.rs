use crate::utils::{
    Vec2d, Vector, init_matrixes, init_vectors, load_parameters_binary, mat_vec_prod, sigmoid,
    softmax, vec_sum,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

pub struct NeuralNetwork {
    pub ln: usize,      // Layers number
    pub ls: Vec<usize>, // Layers sizes (the output layer is in count)
    pub weights: Vec<Vec2d>,
    pub biases: Vec2d,
    pub lr: f64,
    pub is: usize, // input size
}

impl NeuralNetwork {
    #[allow(unused)]
    pub fn from_file(filename: String) -> Self {
        let params = load_parameters_binary(filename).expect("Échec du chargement");
        NeuralNetwork {
            is: params.input_size,
            ls: params.layer_sizes,
            weights: params.weights,
            biases: params.biases,
            lr: params.learning_rate,
            ln: params.layer_num,
        }
    }

    // Let's add the weight here so it is easier to use new parameters from files
    pub fn new(ls: &Vec<usize>, is: usize, lr: f64) -> Self {
        // let mut rng = SimpleRng::new(58);
        NeuralNetwork {
            is: is,
            lr: lr,
            ln: ls.len(),
            ls: ls.to_vec(),
            weights: init_matrixes(ls, is, true),
            biases: init_vectors(ls, false),
        }
    }

    pub fn feed_forward(&self, x: Vector) -> Vec2d {
        // println!("{:?}", self.biases[0]);
        let mut z: Vec2d = init_vectors(&self.ls, false);
        let mut a: Vec2d = init_vectors(&self.ls, false);

        // Couche d'entrée
        z[0] = vec_sum(&mat_vec_prod(&self.weights[0], &x), &self.biases[0]);
        a[0] = z[0].par_iter().map(|&u: &f64| sigmoid(u)).collect();

        // Hidden layers (sigmoid)
        for i in 1..self.ln {
            z[i] = vec_sum(&mat_vec_prod(&self.weights[i], &a[i - 1]), &self.biases[i]);
            // skip output layer
            if i < self.ln - 1 {
                a[i] = z[i].par_iter().map(|&u: &f64| sigmoid(u)).collect();
            }
        }

        // Output layer
        let sf1 = softmax(&z[self.ln - 1].clone()[0..=8].to_vec());
        let sf2 = softmax(&z[self.ln - 1].clone()[9..].to_vec());
        a[self.ln - 1] = [sf1.clone(), sf2.clone()].concat();

        // println!("{} {} => {}", sf1.len(), sf2.len(), a[self.ln - 1].len());
        a
    }

    pub fn back_prop(&mut self, x: &Vector, d_star: usize, a_star: usize) {
        let a: Vec2d = self.feed_forward(x.clone());
        // LOSS: - log(Pd(d*)) -log(Pa(a*));
        // dL / da

        let mut dz: Vec2d = init_vectors(&self.ls, false);
        let mut dw: Vec<Vec2d> = init_matrixes(&self.ls, self.is, false);
        for i in 0..self.ls[self.ln - 1] {
            let kron_di = if i == d_star { 1.0 } else { 0.0 };
            let kron_ai = if i == a_star { 1.0 } else { 0.0 };
            dz[self.ln - 1][i] = a[self.ln - 1][i] - kron_di - kron_ai;
        }

        // Descent for the hidden layers
        for k in (0..=self.ln - 1).rev() {
            // previous activation (if it is the first layer then the prev_act is x)
            let prev_activation = if k == 0 { x.clone() } else { a[k - 1].clone() };

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
                    let da_i: f64 = (0..self.ls[k])
                        .map(|j| dz[k][j] * self.weights[k][j][i])
                        .sum();

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

    pub fn predict(&self, x: &Vector) -> ((usize, f64), (usize, f64)) {
        let sf: Vector = self.feed_forward(x.to_vec())[self.ln - 1].clone();
        let mut d_star = 0;
        let mut a_star = 0;

        let mut pd_star = 0.0;
        let mut pa_star = 0.0;

        // Finding the best probability in the first softmax
        for u in 0..18 {
            let p = sf[u];
            if pd_star < p && u < 9 {
                pd_star = p;
                d_star = u;
            }

            if pa_star < p && u > 8 {
                pa_star = p;
                a_star = u - 9;
            }
        }

        ((d_star, pd_star), (a_star, pa_star))
    }
}
