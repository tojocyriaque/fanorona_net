use rayon::iter::*;

// ================== VECTOR CALCULATIONS ======================
pub type Vector = Vec<f64>;

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
