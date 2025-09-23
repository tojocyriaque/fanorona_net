use crate::maths::collectors::{mat::*, vec::*};

pub mod init;
pub mod learn;
pub mod prediction;

pub struct NeuralNetwork {
    pub ln: usize,      // Layers number
    pub ls: Vec<usize>, // Layers sizes (the output layer is in count)
    pub weights: Vec<Matrix>,
    pub biases: Vec<Vector>,
    pub lr: f64,
    pub is: usize, // input size
}
