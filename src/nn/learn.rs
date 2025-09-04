use crate::{
    maths::{
        activations::{relu::ReLU, softmax::Softmax},
        collectors::mat::Matrix,
    },
    nn::NeuralNetwork,
};

impl NeuralNetwork {
    #[allow(unused, non_snake_case)]
    pub fn batch_forward(self, X: &Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut A_vec: Vec<Matrix> = Vec::new();
        let mut Z_vec: Vec<Matrix> = Vec::new();

        let mut Inp = X;

        for k in 0..self.ln - 1 {
            println!(
                "Dimensions => Inp: {:?}, W:{:?}",
                Inp.dim(),
                self.weights[k].dim()
            );

            let Zk = (Inp * &self.weights[k]).add_each_line(&self.biases[k]);
            // set up this sequence to avoid cloning and borrowing (costs so much)
            A_vec.push(Zk.map_elms(|x| x.relu()));
            Z_vec.push(Zk);
            Inp = &A_vec.last().unwrap();
        }

        let l = self.ln - 1;
        let n_l = self.ls[l];
        let half = n_l / 2;

        println!(
            "Dimensions => Inp: {:?}, W:{:?}",
            Inp.dim(),
            self.weights[l].dim()
        );

        Z_vec.push((Inp * &self.weights[l]).add_each_line(&self.biases[l]));
        let mut Al: Matrix = Matrix(Z_vec[l].map_lines(|v| {
            let mut sf = v[0..=half - 1].softmax();
            sf.extend(v[half..=n_l - 1].softmax());

            sf
        }));

        A_vec.push(Al);

        (Z_vec, A_vec)
    }
}
