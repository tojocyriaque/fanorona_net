use crate::maths::collectors::vec::VecStruct;

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

impl Sigmoid for VecStruct {
    type Output = Self;
    fn sigmoid(self) -> Self::Output {
        let result = self.0.iter().map(|u| u.sigmoid()).collect();
        VecStruct(result)
    }
}

impl Sigmoid for &VecStruct {
    type Output = VecStruct;
    fn sigmoid(self) -> Self::Output {
        self.clone().sigmoid()
    }
}
