use crate::utils::vectors::*;
use rayon::iter::*;

pub type Vec2d = Vec<Vector>;

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
        .into_iter()
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
