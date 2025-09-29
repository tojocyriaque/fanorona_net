use rand::*;
use rayon::iter::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector(pub Vec<f64>);

// ================ CONCATENATIONS ===============
impl Vector {
    pub fn extend(&mut self, v: Vector) {
        self.0.extend(v.0);
    }
}
// ================ INITIALISATION ===============
impl Vector {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[allow(unused)]
    pub fn init_rand(n: usize) -> Self {
        let mut result = Vec::new();
        for _ in 0..n {
            result.push(random())
        }
        Vector(result)
    }
}

// ================ ITERATION ====================
impl Vector {
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, f64> {
        self.0.iter_mut()
    }

    pub fn into_iter(self) -> std::vec::IntoIter<f64> {
        self.0.into_iter()
    }
}

// ================ PARALLEL ITERATION ===========
impl Vector {
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, f64> {
        self.0.par_iter()
    }

    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, f64> {
        self.0.par_iter_mut()
    }

    pub fn into_par_iter(self) -> rayon::vec::IntoIter<f64> {
        self.0.into_par_iter()
    }
}

// ================ INDEXING =====================
impl std::ops::Index<usize> for Vector {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::Index<std::ops::RangeInclusive<usize>> for Vector {
    type Output = [f64];
    fn index(&self, index: std::ops::RangeInclusive<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

// ================ SUM AND DIFFERENCE ===========
// ==> sum
impl std::ops::Add for &Vector {
    type Output = Vector;

    fn add(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            panic!("Cannot add two vectors of different dimensions")
        }

        let result: Vec<f64> = self
            .par_iter()
            .zip(rhs.par_iter())
            .map(|(u, v)| u + v)
            .collect();

        Vector(result)
    }
}

// ==> substraction
impl std::ops::Sub for &Vector {
    type Output = Vector;
    fn sub(self, rhs: Self) -> Self::Output {
        let result: Vec<f64> = self
            .par_iter()
            .zip(rhs.par_iter())
            .map(|(u, v)| u - v)
            .collect();

        Vector(result)
    }
}

// =================== MULTIPLICATIONS and SCALAR PRODUCT =============
// ==> multiplication by a vector
impl std::ops::Mul for &Vector {
    type Output = f64;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.0.len() != rhs.0.len() {
            panic!("Cannot add two vectors of different dimensions")
        }

        self.0
            .par_iter()
            .zip(rhs.0.par_iter())
            .map(|(v1_i, v2_i)| v1_i * v2_i)
            .sum()
    }
}

// ==> multiplication by a scalar
impl std::ops::Mul<&Vector> for f64 {
    type Output = Vector;
    fn mul(self, rhs: &Vector) -> Self::Output {
        Vector(rhs.0.par_iter().map(|u| u * self).collect())
    }
}

// =================== CODE GOLFING =============================
#[macro_export]
macro_rules! vector {
     ($($x:expr),*) => {
        Vector(vec![$($x as f64),*])
    };

    ($x:expr;$y:expr)=>{
        Vector(vec![$x as f64;$y])
    }
}
