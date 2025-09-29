use crate::maths::collectors::vec::Vector;

// ==================== IMPLEMENTATIONS FOR SIGMOID =================================
pub trait Sigmoid {
    type Output;
    fn sigmoid(self) -> Self::Output;
}

impl Sigmoid for f64 {
    type Output = Self;
    fn sigmoid(self) -> Self::Output {
        let z = self.clamp(-100.0, 100.0);
        1.0 / (1.0 + (-z).exp())
    }
}

impl Sigmoid for &Vec<f64> {
    type Output = Vector;
    fn sigmoid(self) -> Self::Output {
        Vector(self.iter().map(|u| u.sigmoid()).collect())
    }
}

impl Sigmoid for &Vector {
    type Output = Vector;
    fn sigmoid(self) -> Self::Output {
        self.0.sigmoid()
    }
}

impl Sigmoid for &[f64] {
    type Output = Vector;
    fn sigmoid(self) -> Self::Output {
        self.to_vec().sigmoid()
    }
}
