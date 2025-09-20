use super::matrixes::Vec2d;
use super::vectors::Vector;
use rand::random;

// ==================== INITIALISATIONS OF COLLECTIONS =========================
// Initialisations of Vectors
pub fn init_vectors(ls: &Vec<usize>, rand_init: bool) -> Vec2d {
    ls.iter()
        .enumerate()
        .map(|(_, &l)| {
            (0..l)
                .map(|_| if rand_init { random::<f64>() } else { 0.0 })
                .collect()
        })
        .collect()
}

// If the random_init is set it will be Xavier / Glorot initialisation
#[allow(unused)]
pub fn init_matrixes(ls: &Vec<usize>, is: usize, rand_init: bool) -> Vec<Vec2d> {
    ls.iter()
        .enumerate()
        .map(|(i, &l)| {
            let col_num = if i == 0 { is } else { ls[i - 1] };
            (0..l)
                .map(|_| {
                    (0..col_num)
                        .map(|_| {
                            if rand_init {
                                let bound = (6.0 / (col_num + l) as f64).sqrt();
                                (random::<f64>() * 2.0 - 1.0) * bound
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

// ============================== CONVERSION =========================
#[allow(dead_code)]
pub fn one_hot(pos: Vec<i32>, c_pl: usize) -> Vector {
    let mut v: Vector = pos
        .iter()
        .flat_map(|&idx| match idx {
            0 => vec![1., 0., 0., 0., 0.],
            1 => vec![0., 1., 0., 0., 0.],
            2 => vec![0., 0., 1., 0., 0.],
            -1 => vec![0., 0., 0., 0., 1.],
            -2 => vec![0., 0.0, 0., 1., 0.],
            _ => {
                panic!("Valeur invalide dans le plateau: {}", idx); // ← ICI
            }
        })
        .collect::<Vector>();

    v.push([0., 1.][c_pl - 1]);
    v
}
