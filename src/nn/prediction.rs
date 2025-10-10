use crate::nn::*;
use crate::data::restructured_datasets::{all_possible_actions};

impl NeuralNetwork {
    pub fn predict(&self, x: Vector) -> (usize, f64) {
        let (_z, a) = &self.feed_forward(x);
        let sf = &a[self.ln - 1]; // 32 éléments
        let (action_idx, prob) = sf
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        (action_idx, *prob)
    }

    pub fn predict_move(&self, x: Vector) -> (usize, usize) {
        let (action_idx, _prob) = self.predict(x);
        let action = all_possible_actions()[action_idx]; // convert index -> (start, end)
        action
    }
}
