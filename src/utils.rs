use std::{
    fs::File,
    io::{Read, Write},
};

use rand::random;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};

use crate::nn::NeuralNetwork;

// ==================== TYPES =========================
pub type Vector = Vec<f64>;
pub type Vec2d = Vec<Vector>;

// ==================== SAVING AND LOADING PARAMETERS =========================
//  Struct to represent the network parameters
#[derive(Serialize, Deserialize)]
pub struct NNParameters {
    pub layer_sizes: Vec<usize>,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
    pub learning_rate: f64,
}

// Saving the parameters in binary file
pub fn save_parameters_binary(nn: &NeuralNetwork, file_path: String) -> std::io::Result<()> {
    let params = NNParameters {
        layer_sizes: nn.ls.clone(),
        weights: nn.weights.clone(),
        biases: nn.biases.clone(),
        learning_rate: nn.lr,
    };

    let encoded: Vec<u8> = bincode::serialize(&params).expect("Échec de la sérialisation");
    let filename = format!("{file_path}");
    let mut file = File::create(&filename)?;
    file.write_all(&encoded)?;
    Ok(())
}

#[allow(dead_code)]
pub fn load_parameters_binary(filename: String) -> std::io::Result<NNParameters> {
    let mut file = File::open(filename)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let params: NNParameters = bincode::deserialize(&buffer).expect("Échec de la désérialisation");
    Ok(params)
}

// ==================== INITIALISATIONS OF COLLECTIONS =========================
// Initialisations of Vectors
pub fn init_vectors(ls: &Vec<usize>, rand_init: bool) -> Vec2d {
    ls.iter()
        .enumerate()
        .map(|(_, &l)| {
            (0..l)
                .map(|_| if rand_init { random::<f64>() } else { 0.0 })
                .collect()
        })
        .collect()
}

// If the random_init is set it will be Xavier / Glorot initialisation
pub fn init_matrixes(ls: &Vec<usize>, is: usize, rand_init: bool) -> Vec<Vec2d> {
    ls.iter()
        .enumerate()
        .map(|(i, &l)| {
            let col_num = if i == 0 { is } else { ls[i - 1] };
            (0..l)
                .map(|_| {
                    (0..col_num)
                        .map(|_| {
                            if rand_init {
                                let bound = (6.0 / (col_num + l) as f64).sqrt();
                                (random::<f64>() * 2.0 - 1.0) * bound
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

// Conversion
#[allow(dead_code)]
pub fn one_hot(pos: Vec<i32>, c_pl: usize) -> Vector {
    let mut v: Vector = pos
        .iter()
        .flat_map(|&idx| match idx {
            0 => vec![1., 0., 0., 0., 0.],
            1 => vec![0., 1., 0., 0., 0.],
            2 => vec![0., 0., 1., 0., 0.],
            -1 => vec![0., 0., 0., 0., 1.],
            -2 => vec![0., 0.0, 0., 1., 0.],
            _ => vec![0.; 5],
        })
        .collect::<Vector>();

    v.push([0., 1.][c_pl - 1]);
    v
}
// ================== VECTOR CALCULATIONS ======================
// Sum two vectors using multiple cores
#[allow(dead_code, unused_variables)]
pub fn vec_sum(v1: &Vector, v2: &Vector) -> Vector {
    v1.par_iter()
        .zip(v2.par_iter())
        .map(|(v1_i, v2_i)| v1_i + v2_i)
        .collect()
}

#[allow(dead_code, unused_variables)]
pub fn vec_mul(v: &Vector, q: f64) -> Vector {
    v.par_iter().map(|vi| vi * q).collect()
}

// Scalar product
#[allow(dead_code, unused_variables)]
pub fn scal_prod(v1: &Vector, v2: &Vector) -> f64 {
    v1.par_iter()
        .zip(v2.par_iter())
        .map(|(v1_i, v2_i)| v1_i * v2_i)
        .sum()
}

// ==================== MATRIX CALCULATIONS =====================
// Make matrix and Vector product using multiple cores
#[allow(dead_code, unused_variables)]
pub fn mat_vec_prod(m: &Vec2d, v: &Vector) -> Vector {
    m.par_iter().map(|row| scal_prod(row, v)).collect()
}

#[allow(dead_code, unused_variables)]
// transpose matrix
pub fn mat_tr(m: &Vec2d) -> Vec2d {
    (0..m[0].len())
        .into_par_iter()
        .map(|i: usize| {
            (0..m.len())
                .into_par_iter()
                .map(|j: usize| m[j][i])
                .collect()
        })
        .collect()
}

#[allow(dead_code, unused_variables)]
pub fn mat_prod(m1: &Vec2d, m2: &Vec2d) -> Vec2d {
    let m2_t: Vec2d = mat_tr(m2);

    (0..m1.len())
        .into_par_iter()
        .map(|i: usize| {
            m2_t.par_iter()
                .map(|m2_j| scal_prod(&m1[i], &m2_j))
                .collect()
        })
        .collect()
}

// ==================== ACTIVATION FUNCTIONS ====================
#[allow(dead_code)]
pub fn sigmoid(z: f64) -> f64 {
    let z = z.clamp(-100.0, 100.0);
    1.0 / (1.0 + (-z).exp())
}

#[allow(dead_code)]
pub fn re_lu(x: f64) -> f64 {
    if x >= 0. { x } else { 0. }
}

#[allow(dead_code)]
pub fn softmax(y: &Vector) -> Vector {
    let max_y = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vector = y
        .iter()
        .map(|&y_i| (y_i - max_y).clamp(-100.0, 100.0).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / y.len() as f64; y.len()] // Distribution uniforme si somme nulle
    } else {
        exps.iter().map(|&x| x / sum).collect()
    }
}
