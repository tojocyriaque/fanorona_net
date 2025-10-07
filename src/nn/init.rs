use crate::{
    data::loads::load_parameters_binary,
    maths::{collectors::mat::*, collectors::vec::*},
    nn::NeuralNetwork,
    vector,
};

// ==================== INITIALISATIONS OF COLLECTIONS =========================
// Initialisations of Vectors
pub fn init_vectors(ls: &Vec<usize>, rand_init: bool) -> Vec<Vector> {
    if !rand_init {
        ls.iter().map(|&n| vector![0.0;n]).collect()
    } else {
        ls.iter().map(|&n| Vector::init_rand(n)).collect()
    }
}

// If the random_init is set it will be Xavier / Glorot initialisation
#[allow(unused)]
pub fn init_matrixes(ls: &Vec<usize>, is: usize, rand_init: bool) -> Vec<Matrix> {
    let mut matrixes = Vec::new();
    matrixes.push(Matrix::init_xavier(is, ls[0])); // first layer
    for (idx, &m) in ls[1..].iter().enumerate() {
        let n = ls[idx];
        matrixes.push(Matrix::init_xavier(n, m));
    }
    matrixes
}

// ============================== CONVERSION =========================
#[allow(dead_code)]
pub fn one_hot_fanorona(pos: Vec<f64>, c_pl: usize) -> Vector {
    let mut v = pos
        .iter()
        .flat_map(|&idx| match idx {
            0.0 => vec![1., 0., 0., 0., 0.],
            1.0 => vec![0., 1., 0., 0., 0.],
            2.0 => vec![0., 0., 1., 0., 0.],
            -1.0 => vec![0., 0., 0., 0., 1.],
            -2.0 => vec![0., 0.0, 0., 1., 0.],
            _ => {
                panic!("Invalind value on the board: {}", idx); // ← ICI
            }
        })
        .collect::<Vec<f64>>();

    v.push([0., 1.][c_pl - 1]); // 1 ou 2
    Vector(v)
}

// ============================= INITIALIZATION OF NEURAL NETWORK =======================================
#[allow(unused)]
impl NeuralNetwork {
    pub fn from_file(filename: String) -> Self {
        let params = load_parameters_binary(filename).expect("Failed to load parameters");
        NeuralNetwork {
            is: params.input_size,
            ls: params.layer_sizes,
            weights: params.weights,
            biases: params.biases,
            lr: params.learning_rate,
            ln: params.layer_num,
        }
    }

    #[allow(unused)]
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
}
