use crate::utils::matrixes::*;

pub mod init;
pub mod learn;
pub mod prediction;

pub struct NeuralNetwork {
    pub ln: usize,      // Layers number
    pub ls: Vec<usize>, // Layers sizes (the output layer is in count)
    pub weights: Vec<Vec2d>,
    pub biases: Vec2d,
    pub lr: f64,
    pub is: usize, // input size
}
