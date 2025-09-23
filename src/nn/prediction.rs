use crate::nn::*;

impl NeuralNetwork {
    pub fn predict(&self, x: Vector) -> ((usize, f64), (usize, f64)) {
        let sf: &Vector = &self.feed_forward(x)[self.ln - 1];
        let mut d_star = 0;
        let mut a_star = 0;

        let mut pd_star = 0.0;
        let mut pa_star = 0.0;

        // Finding the best probabilities
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

        return ((d_star, pd_star), (a_star, pa_star));
    }
}
