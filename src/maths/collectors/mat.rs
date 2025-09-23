use crate::vecstruct;

use super::vec::*;
use rand::random;
use rayon::iter::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Matrix(pub Vec<Vector>);
// ============== INITIALISATIONS =======================
impl Matrix {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn dim(&self) -> (usize, usize) {
        (self.len(), self[0].len())
    }
    // m x n matrix
    #[allow(dead_code)]
    pub fn init_0(m: usize, n: usize) -> Self {
        let result = vec![vecstruct![0.0;n]; m];
        Matrix(result)
    }

    #[allow(dead_code)]
    pub fn init_xavier(m: usize, n: usize) -> Self {
        let mut result = Vec::new();
        for _ in 0..m {
            let mut line = Vec::new();
            for _ in 0..n {
                let bound = (6.0 / (n + m) as f64).sqrt();
                line.push((random::<f64>() * 2.0 - 1.0) * bound);
            }
            result.push(Vector(line))
        }

        Matrix(result)
    }
}

// ============== INDEXING ==============================
impl std::ops::Index<usize> for Matrix {
    type Output = Vector;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self[i][j]
    }
}

impl std::ops::IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self[i][j]
    }
}

// ============== TRANSPOSE =============================
impl Matrix {
    fn tr(&self) -> Self {
        let result = (0..self[0].len())
            .into_iter()
            .map(|i: usize| {
                Vector(
                    (0..self.len())
                        .into_par_iter()
                        .map(|j: usize| self[j][i])
                        .collect(),
                )
            })
            .collect();
        Matrix(result)
    }
}

// ============== SUM AND DIFFERENCE ====================
// sum
impl std::ops::Add for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: Self) -> Self::Output {
        // if self.len() != rhs.len()
        //     || self
        //         .0
        //         .par_iter()
        //         .zip(rhs.0.par_iter())
        //         .any(|(u, v)| u.len() != v.len())
        // {
        //     panic!("Cannot add matrixes of different dimensions");
        // }

        let result = self
            .0
            .par_iter()
            .zip(rhs.0.par_iter())
            .map(|(u, v)| u + v)
            .collect();

        Matrix(result)
    }
}

// substraction
impl std::ops::Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Self) -> Self::Output {
        // if self.len() != rhs.len()
        //     || self
        //         .0
        //         .par_iter()
        //         .zip(rhs.0.par_iter())
        //         .any(|(u, v)| u.len() != v.len())
        // {
        //     panic!("Cannot substract matrixes of different dimensions");
        // }
        let result = self
            .0
            .par_iter()
            .zip(rhs.0.par_iter())
            .map(|(u, v)| u - v)
            .collect();

        Matrix(result)
    }
}

// ============== MULTIPLICATIONS =======================
// multiplication by matrix
impl std::ops::Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Self) -> Self::Output {
        // if self.0.iter().any(|u| u.0.len() != rhs.0.len()) {
        //     panic!("Cannot multiply matrixes with invalid dimensions match");
        // }

        // transpose the matrix rhs
        let rhs_t = rhs.tr();

        // M * T
        let result = self
            .0
            .par_iter()
            .map(|m| Vector(rhs_t.0.par_iter().map(|t| m * t).collect()))
            .collect();
        Matrix(result)
    }
}

// multiplication by a vector
impl std::ops::Mul<&Matrix> for &Vector {
    type Output = Vector;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        // if rhs.0.iter().any(|u| u.len() != self.len()) {
        //     panic!("Dimensions unmatched (Vector * Matrix)")
        // }

        let result = rhs.0.par_iter().map(|v| v * self).collect();
        Vector(result)
    }
}

// multiplication by a scalar
impl std::ops::Mul<&Matrix> for f64 {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        Matrix(rhs.0.par_iter().map(|u| self * u).collect())
    }
}

// =============== CODE GOLFING =========================
#[macro_export]
macro_rules! mat {
    ($([$($x:expr),*]),*) => {
        Matrix(vec![
                $(Vector(vec![$($x as f64),*])),*]
        )
    };
}
