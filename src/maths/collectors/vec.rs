use rand::*;
use rayon::iter::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecStruct(pub Vec<f64>);

// ================ INITIALISATION ===============
impl VecStruct {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[allow(unused)]
    pub fn init_rand(n: usize) -> Self {
        let mut result = Vec::new();
        for _ in 0..n {
            result.push(random())
        }
        VecStruct(result)
    }
}

// ================ ITERATION ====================
impl VecStruct {
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
impl VecStruct {
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
impl std::ops::Index<usize> for VecStruct {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for VecStruct {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl std::ops::Index<usize> for &VecStruct {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

// ================ SUM AND DIFFERENCE ===========
impl std::ops::Add for VecStruct {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            panic!("Cannot add two vectors of different dimensions")
        }

        let result: Vec<f64> = self
            .par_iter()
            .zip(rhs.par_iter())
            .map(|(u, v)| u + v)
            .collect();

        VecStruct(result)
    }
}

impl std::ops::Add for &VecStruct {
    type Output = VecStruct;

    fn add(self, rhs: Self) -> Self::Output {
        self.clone() + rhs
    }
}

impl std::ops::Add<&VecStruct> for VecStruct {
    type Output = VecStruct;

    fn add(self, rhs: &VecStruct) -> Self::Output {
        self + rhs.clone()
    }
}

impl std::ops::Add<VecStruct> for &VecStruct {
    type Output = VecStruct;

    fn add(self, rhs: VecStruct) -> Self::Output {
        self.clone() + rhs
    }
}

// substraction
impl std::ops::Sub for VecStruct {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let result: Vec<f64> = self
            .par_iter()
            .zip(rhs.par_iter())
            .map(|(u, v)| u - v)
            .collect();

        VecStruct(result)
    }
}

impl std::ops::Sub for &VecStruct {
    type Output = VecStruct;
    fn sub(self, rhs: Self) -> Self::Output {
        self.clone() - rhs.clone()
    }
}

impl std::ops::Sub<&VecStruct> for VecStruct {
    type Output = Self;
    fn sub(self, rhs: &VecStruct) -> Self::Output {
        self - rhs.clone()
    }
}

impl std::ops::Sub<VecStruct> for &VecStruct {
    type Output = VecStruct;
    fn sub(self, rhs: VecStruct) -> Self::Output {
        self.clone() - rhs
    }
}

// =================== MULTIPLICATIONS and SCALAR PRODUCT =============
// scalar product
impl std::ops::Mul for VecStruct {
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

impl std::ops::Mul for &VecStruct {
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

// multiplication by a scalar
impl std::ops::Mul<VecStruct> for f64 {
    type Output = VecStruct;
    fn mul(self, rhs: VecStruct) -> Self::Output {
        VecStruct(rhs.0.par_iter().map(|u| u * self).collect())
    }
}

impl std::ops::Mul<&VecStruct> for f64 {
    type Output = VecStruct;
    fn mul(self, rhs: &VecStruct) -> Self::Output {
        VecStruct(rhs.0.par_iter().map(|u| u * self).collect())
    }
}

// =================== CODE GOLFING =============================
#[macro_export]
macro_rules! vecstruct {
     ($($x:expr),*) => {
        VecStruct(vec![$($x as f64),*])
    };

    ($x:expr;$y:expr)=>{
        VecStruct(vec![$x as f64;$y])
    }
}
