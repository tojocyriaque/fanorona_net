use crate::vector;

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
        let result = vec![vector![0.0;n]; m];
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

    // vertical extension
    pub fn extend_ver(&mut self, rhs: Matrix) {
        self.0.extend(rhs.0);
    }

    // horizontal extension
    pub fn extend_hor(&mut self, rhs: &Matrix) {
        self.0
            .par_iter_mut()
            .enumerate()
            .for_each(|(u, v)| v.0.extend(&rhs[u].0));
    }
    pub fn elm_prod(&self, rhs: &Matrix) -> Matrix {
        self.map_zip_el(rhs, |a, b| a * b)
    }
    // Map with a function that takes each lement i,j of the two matrix (self, rhs)
    pub fn map_zip_el<F>(&self, rhs: &Matrix, f: F) -> Matrix
    where
        F: Fn(f64, f64) -> f64,
    {
        let (m, c1) = self.dim();
        let (l2, n) = rhs.dim();
        assert_eq!((m, c1), (l2, n), "Incompatible dimensions for map zip");
        let mut res = Matrix::init_0(m, n);

        for i in 0..m {
            for j in 0..n {
                res[i][j] = f(self[i][j], rhs[i][j]);
            }
        }

        res
    }

    pub fn map_elms<F>(&self, f: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        let (m, n) = self.dim();
        let mut mapped: Vec<Vector> = Vec::new();
        for i in 0..m {
            let mut ve: Vec<f64> = Vec::new();
            for j in 0..n {
                ve.push(f(self[i][j]));
            }
            mapped.push(Vector(ve));
        }
        Matrix(mapped)
    }

    pub fn map_lines<F, T>(&self, f: F) -> Vec<T>
    where
        F: Fn(&Vector) -> T,
    {
        let (m, _) = self.dim();
        let mut v: Vec<T> = Vec::new();
        for i in 0..m {
            v.push(f(&self[i]));
        }
        v
    }

    pub fn map_cols<F, T>(&self, f: F) -> Vec<T>
    where
        F: Fn(&Vector) -> T,
    {
        let mut v: Vec<T> = Vec::new();
        let tr = self.tr();
        for j in 0..tr.len() {
            v.push(f(&tr[j]))
        }
        v
    }

    pub fn add_each_line(self, rhs: &Vector) -> Matrix {
        let mut m: Vec<Vector> = Vec::new();
        for v in self.0 {
            m.push(&v + rhs);
        }
        Matrix(m)
    }

    pub fn add_each_col<F>(self, rhs: &Vector) -> Matrix {
        let (m, n) = self.dim();
        let mut matr: Vec<Vector> = Vec::new();

        for i in 0..m {
            let mut ve: Vec<f64> = Vec::new();
            for j in 0..n {
                ve.push(rhs[j] + self[i][j]);
            }
            matr.push(Vector(ve));
        }
        Matrix(matr)
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

impl std::ops::Index<std::ops::RangeInclusive<usize>> for Matrix {
    type Output = [Vector];
    fn index(&self, index: std::ops::RangeInclusive<usize>) -> &Self::Output {
        &self.0[index]
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
    pub fn tr(&self) -> Self {
        let result = (0..self[0].len())
            .into_par_iter()
            .map(|i: usize| {
                Vector(
                    (0..self.len())
                        .into_iter()
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

    ($x:expr;$y:expr)=>{
        Matrix(
            vec![$x;$y]
        )
    }
}
