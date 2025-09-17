use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::utils::{mat_vec_prod, rand_f32, sigmoid, softmax, vec_sum};

pub type Vector = Vec<f64>;

// This is not always a matrix
pub type Vec2d = Vec<Vec<f64>>;

pub struct NeuralNetwork {
    pub ln: usize,      // Layers number
    pub ls: Vec<usize>, // Layers sizes
    pub weights: Vec<Vec2d>,
    pub biases: Vec2d,
    pub lr: f64,
    pub is: usize,
}

impl NeuralNetwork {
    // Let's add the weight here so it is easier to use new parameters from files
    pub fn new(ln: usize, ls: &Vec<usize>, is: usize, lr: f64) -> Self {
        // let mut rng = SimpleRng::new(58);
        NeuralNetwork {
            is: is,
            lr: lr,
            ln: ln,
            ls: ls.to_vec(),
            weights: ls
                .iter()
                .enumerate()
                .map(|(i, &e)| {
                    let col_num = if i == 0 { is } else { ls[i - 1] };
                    let mut v: Vec2d = Vec2d::new();
                    for _ in 0..e {
                        let mut vi = Vec::new();
                        for _ in 0..col_num {
                            vi.push((rand_f32() - 0.5) * 0.2);
                        }
                        v.push(vi);
                    }
                    v
                })
                .collect(),
            biases: ls
                .iter()
                .enumerate()
                .map(|(_, &e)| {
                    let mut v: Vector = Vector::new();
                    for _ in 0..e {
                        v.push((rand_f32() - 0.5) * 0.2);
                    }
                    v
                    // vec![(rand_f32() - 0.5) * 0.2; e]
                })
                .collect(),
        }
    }

    pub fn feed_forward(&self, x: Vector) -> (Vec2d, Vec2d) {
        // println!("{:?}", self.biases[0]);
        let mut z: Vec2d = self.ls.iter().map(|&l| vec![0.0; l]).collect();
        let mut a: Vec2d = self.ls.iter().map(|&l| vec![0.0; l]).collect();

        // Couche d'entrée
        z[0] = vec_sum(&mat_vec_prod(&self.weights[0], &x), &self.biases[0]);
        a[0] = z[0].par_iter().map(|&u: &f64| sigmoid(u)).collect();

        // Couches cachées (sigmoid)
        for i in 1..self.ln {
            z[i] = vec_sum(&mat_vec_prod(&self.weights[i], &a[i - 1]), &self.biases[i]);

            // println!("{:?}", z[i]);

            // skip the output layer
            if i < self.ln - 1 {
                a[i] = z[i].par_iter().map(|&u: &f64| sigmoid(u)).collect();
            }
        }

        // Couche de sortie
        let sf1 = softmax(z[self.ln - 1].clone()[0..=8].to_vec());
        let sf2 = softmax(z[self.ln - 1].clone()[9..].to_vec());
        a[self.ln - 1] = [sf1.clone(), sf2.clone()].concat();

        // println!("{:?}\n{:?}", z[2], a[2]);
        // println!("--------------");
        (z, a)
    }

    pub fn back_prop(&mut self, x: &Vector, d_star: usize, a_star: usize) {
        let (mut _z, a): (Vec2d, Vec2d) = self.feed_forward(x.clone());
        // LOSS: - log (Pd(d*) * Pa(a*));
        // dL / da
        let mut da: Vec2d = self.ls.iter().map(|&l| vec![0.0; l]).collect();
        let mut dz: Vec2d = self.ls.iter().map(|&l| vec![0.0; l]).collect();
        let mut db: Vec2d = self.ls.iter().map(|&l| vec![0.0; l]).collect();

        // Descent of the last layer
        da[self.ln - 1][d_star] = -1.0 / a[self.ln - 1][d_star];
        da[self.ln - 1][a_star] = -1.0 / a[self.ln - 1][a_star];

        for i in 0..self.ls[self.ln - 1] {
            let kron_di = if i == d_star { 1.0 } else { 0.0 };
            let kron_ai = if i == a_star { 1.0 } else { 0.0 };
            dz[self.ln - 1][i] = 2.0 * a[self.ln - 1][i] - kron_di - kron_ai;
        }
        let mut dw: Vec<Vec2d> = self
            .ls
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                if i == 0 {
                    vec![vec![0.0; self.is]; e]
                } else {
                    vec![vec![0.0; self.ls[i - 1]]; e]
                }
            })
            .collect();

        // Descent for the hidden layers
        for k in (0..(self.ln - 1)).rev() {
            let prev_act = if k == 0 { x.clone() } else { a[k - 1].clone() };
            // dzk (sigmoid except for the output layer)
            if k < self.ln - 1 {
                // activation is
                for i in 0..self.ls[self.ln - 1] {
                    dz[k][i] = da[k][i] * a[k][i] * (a[k][i] - 1.0);
                }
            }

            // dwk
            for i in 0..dw[k].len() {
                for j in 0..prev_act.len() {
                    dw[k][i][j] = dz[k][j] * prev_act[j];
                }
            }

            // dbk
            db[k] = dz[k].to_vec();

            // da(k-1)
            if k > 0 {
                for i in 0..da[k - 1].len() {
                    for j in 0..prev_act.len() {
                        da[k - 1][i] += dz[k][i] * dw[k][i][j];
                    }
                }
            }
        }

        for k in 0..self.ln {
            // W = W - lr * DW
            for i in 0..self.weights[k].len() {
                for j in 0..self.weights[k][i].len() {
                    self.weights[k][i][j] -= 0.0001; //self.lr * dw[k][i][j]; 
                }
            }

            println!("----------");
            for j in 0..self.weights[k].len() {
                println!("{:?}", &dw[k][j]);
            }
            // B = B - lr * DB
            for i in 0..self.ls[k] {
                self.biases[k][i] -= 0.0004;
                // self.lr * db[k][i];
            }
        }
    }

    pub fn predict(&self, x: &Vector) -> ((usize, f64), (usize, f64)) {
        let sf: Vector = self.feed_forward(x.to_vec()).1[self.ln - 1].clone();

        // println!("{:?} {:?}", x, sf);
        // println!("{:?}", sf);

        // Max of the first softmax
        let (d, pd): (usize, &f64) = sf[0..=8]
            .into_iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Max of the second softmax
        let (a, pa): (usize, &f64) = sf[9..=17]
            .into_iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        ((d, *pd), (a + 8, *pa))
    }
}
