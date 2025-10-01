use crate::maths::collectors::vec::Vector;

#[allow(unused)]
// ==================== IMPLEMENTATIONS FOR SIGMOID =================================
pub trait ReLU {
    type Output;
    fn relu(self) -> Self::Output;
}

impl ReLU for f64 {
    type Output = Self;
    fn relu(self) -> Self::Output {
        self.max(0.0)
    }
}

impl ReLU for &Vec<f64> {
    type Output = Vec<f64>;
    fn relu(self) -> Self::Output {
        self.iter().map(|z| z.relu()).collect()
    }
}

impl ReLU for &Vector {
    type Output = Vector;
    fn relu(self) -> Self::Output {
        Vector(self.0.relu())
    }
}
