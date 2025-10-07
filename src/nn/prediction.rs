use crate::nn::*;

impl NeuralNetwork {
    #[allow(non_snake_case)]
    pub fn predict(&self, x: Vector) -> ((usize, usize), f64, Vector) {
        let X = Matrix(vec![x]);

        let (_, A) = &self.batch_forward(&X);

        let sf = &A[self.ln - 1][0];
        // println!("Softmax sum: {}", sf.sum());
        let mut mov_idx = 0;
        let mut max_proba = 0.0;

        // Finding the best probabilities
        let output_size = self.ls[self.ln - 1];
        for u in 0..output_size {
            let proba = sf[u];
            if max_proba < proba {
                max_proba = proba;
                mov_idx = u;
            }
        }

        let board_len = 9;
        let d_star = mov_idx / board_len;
        let a_star = mov_idx % board_len;

        return ((d_star, a_star), max_proba, sf.clone());
    }
}
