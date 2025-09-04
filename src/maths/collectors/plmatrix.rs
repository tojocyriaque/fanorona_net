use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Index, IndexMut, Mul, Sub};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlMatrix {
    data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl PlMatrix {
    // === Constructeurs ===
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert!(data.len() == rows * cols, "Data size must be rows * cols");
        Self { data, rows, cols }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(vec![0.0; rows * cols], rows, cols)
    }

    pub fn xavier(rows: usize, cols: usize) -> Self {
        let bound = (6.0 / ((rows + cols) as f64)).sqrt();
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| (rand::random::<f64>() * 2.0 - 1.0) * bound)
            .collect();
        Self::new(data, rows, cols)
    }

    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    // === Accès élément ===
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }

    // === Opérations élémentaires ===
    pub fn map_elms<F>(&self, f: F) -> PlMatrix
    where
        F: Fn(f64) -> f64 + Sync + Send,
    {
        let data: Vec<f64> = self.data.par_iter().map(|&x| f(x)).collect();
        PlMatrix::new(data, self.rows, self.cols)
    }

    // Applique f à chaque ligne (retourne Vec<T>)
    pub fn map_lines<F, T>(&self, f: F) -> Vec<T>
    where
        F: Fn(&[f64]) -> T + Sync + Send,
        T: Send,
    {
        self.data.par_chunks(self.cols).map(f).collect()
    }

    // Ajoute un biais (vecteur de taille `cols`) à chaque ligne
    pub fn add_bias(&self, bias: &[f64]) -> PlMatrix {
        assert_eq!(bias.len(), self.cols, "Bias must match number of columns");
        let mut result = self.clone();
        result.data.par_chunks_mut(self.cols).for_each(|row| {
            for (r, &b) in row.iter_mut().zip(bias) {
                *r += b;
            }
        });
        result
    }

    // === Transposition (si vraiment nécessaire) ===
    // pub fn transpose(&self) -> PlMatrix {
    //     let mut tr = vec![0.0; self.rows * self.cols];
    //     // Parallélise sur les colonnes de l'originale (lignes de la transposée)
    //     (0..self.cols).into_par_iter().for_each(|j| {
    //         for i in 0..self.rows {
    //             tr[j * self.rows + i] = self.get(i, j);
    //         }
    //     });
    //     PlMatrix::new(tr, self.cols, self.rows)
    // }

    // === Extensions ===
    pub fn extend_ver(&mut self, other: &PlMatrix) {
        assert_eq!(
            self.cols, other.cols,
            "Cannot extend vertically: column mismatch"
        );
        self.data.extend_from_slice(&other.data);
        self.rows += other.rows;
    }

    pub fn extend_hor(&mut self, other: &PlMatrix) {
        assert_eq!(
            self.rows, other.rows,
            "Cannot extend horizontally: row mismatch"
        );
        let mut new_data = Vec::with_capacity(self.rows * (self.cols + other.cols));
        for i in 0..self.rows {
            new_data.extend_from_slice(&self.data[i * self.cols..(i + 1) * self.cols]);
            new_data.extend_from_slice(&other.data[i * other.cols..(i + 1) * other.cols]);
        }
        self.data = new_data;
        self.cols += other.cols;
    }
}

// === Indexing ===
impl Index<usize> for PlMatrix {
    type Output = [f64];
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

impl IndexMut<usize> for PlMatrix {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }
}

impl Index<(usize, usize)> for PlMatrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[i * self.cols + j]
    }
}

impl IndexMut<(usize, usize)> for PlMatrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data[i * self.cols + j]
    }
}

// === Opérations matricielles ===

// Addition
impl Add for &PlMatrix {
    type Output = PlMatrix;
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.dim(),
            rhs.dim(),
            "Matrix dimensions must match for addition"
        );
        let data: Vec<f64> = self
            .data
            .par_iter()
            .zip(&rhs.data)
            .map(|(a, b)| a + b)
            .collect();
        PlMatrix::new(data, self.rows, self.cols)
    }
}

// Soustraction
impl Sub for &PlMatrix {
    type Output = PlMatrix;
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.dim(),
            rhs.dim(),
            "Matrix dimensions must match for subtraction"
        );
        let data: Vec<f64> = self
            .data
            .par_iter()
            .zip(&rhs.data)
            .map(|(a, b)| a - b)
            .collect();
        PlMatrix::new(data, self.rows, self.cols)
    }
}

// Multiplication matricielle (OPTIMISÉE, sans transposition)
impl Mul for &PlMatrix {
    type Output = PlMatrix;
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.cols, rhs.rows,
            "Incompatible dimensions for matrix multiplication"
        );
        let m = self.rows;
        let k = self.cols; // = rhs.rows
        let n = rhs.cols;

        let mut data = vec![0.0; m * n];

        // Parallélisation sur les lignes de sortie (batch)
        data.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
            for kk in 0..k {
                let a_ik = self.get(i, kk);
                let rhs_row = &rhs.data[kk * n..(kk + 1) * n];
                for (j, &b_kj) in rhs_row.iter().enumerate() {
                    out_row[j] += a_ik * b_kj;
                }
            }
        });

        PlMatrix::new(data, m, n)
    }
}

// Multiplication scalaire
impl Mul<&PlMatrix> for f64 {
    type Output = PlMatrix;
    fn mul(self, rhs: &PlMatrix) -> Self::Output {
        let data: Vec<f64> = rhs.data.par_iter().map(|&x| self * x).collect();
        PlMatrix::new(data, rhs.rows, rhs.cols)
    }
}

// === Macros de commodité ===
#[macro_export]
macro_rules! plmat {
    // plmat![[1.0, 2.0], [3.0, 4.0]]
    ($([$($x:expr),*]),*) => {{
        let rows = vec![$(vec![$($x as f64),*]),*];
        let rows_count = rows.len();
        let cols_count = if rows_count > 0 { rows[0].len() } else { 0 };
        let data: Vec<f64> = rows.into_iter().flatten().collect();
        $crate::pl_matrix::PlMatrix::new(data, rows_count, cols_count)
    }};

    // plmat![0.0; rows, cols]
    ($val:expr; $rows:expr, $cols:expr) => {
        $crate::pl_matrix::PlMatrix::new(vec![$val as f64; $rows * $cols], $rows, $cols)
    };
}
