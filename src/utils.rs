use std::u32;

use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

// ==================== TYPES =========================
pub type Vector = Vec<f64>;
pub type Matrix = Vec<Vector>;

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
pub fn mat_vec_prod(m: &Matrix, v: &Vector) -> Vector {
    m.par_iter().map(|row| scal_prod(row, v)).collect()
}

#[allow(dead_code, unused_variables)]
// transpose matrix
pub fn mat_tr(m: &Matrix) -> Matrix {
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
pub fn mat_prod(m1: &Matrix, m2: &Matrix) -> Matrix {
    let m2_t: Matrix = mat_tr(m2);

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
pub fn sigmoid(x: f64) -> f64 {
    1. / 1. + (-x).exp()
}

#[allow(dead_code)]
pub fn re_lu(x: f64) -> f64 {
    if x >= 0. { x } else { 0. }
}

#[allow(dead_code)]
pub fn softmax(y: Vector) -> Vector {
    let sum: f64 = y.par_iter().map(|y_i: &f64| y_i.exp()).sum();
    y.par_iter().map(|y_i: &f64| y_i.exp() / sum).collect()
}

// pub struct SimpleRng {
//     state: u32,
// }

// Fonction d'initialisation aléatoire simple
pub fn rand_f32() -> f64 {
    static mut STATE: u32 = 123456789;
    unsafe {
        STATE ^= STATE << 13;
        STATE ^= STATE >> 17;
        STATE ^= STATE << 5;
        (STATE as f64) / (u32::MAX as f64)
    }
}

// impl SimpleRng {
//     pub fn new(seed: u32) -> Self {
//         Self { state: seed }
//     }

//     /// retourne un u64 pseudo-aléatoire
//     fn next_u32(&mut self) -> u32 {
//         // constants for LCG (from Numerical Recipes style)
//         self.state = self.state.wrapping_mul(u32::MAX).wrapping_add(u32::MIN);
//         self.state
//     }

//     /// retourne un f64 dans (-0.5, 0.5)
//     pub fn next_f32_centered(&mut self) -> f64 {
//         let r = self.next_u32();
//         // convertir sur [0,1)
//         let u = (r as f64) / (u64::MAX as f64 + 1.0);
//         u - 0.5
//     }
// }

// /// Utilitaires linéaires et math
// #[allow(dead_code)]
// pub fn softmax_stable(x: &Vec<f64>) -> Vec<f64> {
//     // calcule softmax de x (stabilité numérique)
//     let max_x = x.iter().cloned().fold(std::f64::NEG_INFINITY, f64::max);
//     let mut exps: Vec<f64> = x.iter().map(|v| (v - max_x).exp()).collect();
//     let sum: f64 = exps.iter().sum();
//     if sum == 0.0 {
//         // éviter division par 0: renvoyer distribution uniforme
//         let n = exps.len();
//         vec![1.0 / n as f64; n]
//     } else {
//         for e in &mut exps {
//             *e /= sum;
//         }
//         exps
//     }
// }
