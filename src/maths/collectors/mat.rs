use crate::vecstruct;

use super::vec::*;
use rand::random;
use rayon::iter::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mat(pub Vec<VecStruct>);
// ============== INITIALISATIONS =======================
impl Mat {
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
        Mat(result)
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
            result.push(VecStruct(line))
        }

        Mat(result)
    }
}

// ============== INDEXING ==============================
impl std::ops::Index<usize> for Mat {
    type Output = VecStruct;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::Index<(usize, usize)> for Mat {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self[i][j]
    }
}

impl std::ops::IndexMut<usize> for Mat {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self[i][j]
    }
}

// ============== TRANSPOSE =============================
impl Mat {
    fn tr(&self) -> Self {
        let result = (0..self[0].len())
            .into_iter()
            .map(|i: usize| {
                VecStruct(
                    (0..self.len())
                        .into_par_iter()
                        .map(|j: usize| self[j][i])
                        .collect(),
                )
            })
            .collect();
        Mat(result)
    }
}

// ============== SUM AND DIFFERENCE ====================
// sum
impl std::ops::Add for &Mat {
    type Output = Mat;
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

        Mat(result)
    }
}

// substraction
impl std::ops::Sub for &Mat {
    type Output = Mat;
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

        Mat(result)
    }
}

// ============== MULTIPLICATIONS =======================
// multiplication by matrix
impl std::ops::Mul for &Mat {
    type Output = Mat;
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
            .map(|m| VecStruct(rhs_t.0.par_iter().map(|t| m * t).collect()))
            .collect();
        Mat(result)
    }
}

// multiplication by a vector
impl std::ops::Mul<&Mat> for &VecStruct {
    type Output = VecStruct;
    fn mul(self, rhs: &Mat) -> Self::Output {
        // if rhs.0.iter().any(|u| u.len() != self.len()) {
        //     panic!("Dimensions unmatched (Vector * Matrix)")
        // }

        let result = rhs.0.par_iter().map(|v| v * self).collect();
        VecStruct(result)
    }
}

// multiplication by a scalar
impl std::ops::Mul<&Mat> for f64 {
    type Output = Mat;
    fn mul(self, rhs: &Mat) -> Self::Output {
        Mat(rhs.0.par_iter().map(|u| self * u).collect())
    }
}

// =============== CODE GOLFING =========================
#[macro_export]
macro_rules! mat {
    ($([$($x:expr),*]),*) => {
        Mat(vec![
                $(VecStruct(vec![$($x as f64),*])),*]
        )
    };
}
