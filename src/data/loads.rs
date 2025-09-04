use std::{
    fs::{read_to_string, File},
    io::{Read, Write},
};

use serde::{Deserialize, Serialize};

use crate::{
    maths::collectors::{mat::Matrix, vec::Vector},
    nn::NeuralNetwork,
};

// ==================== SAVING AND LOADING PARAMETERS =========================
//  Struct to represent the network parameters
#[derive(Serialize, Deserialize)]
pub struct NNParameters {
    pub input_size: usize,
    pub layer_num: usize,
    pub layer_sizes: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Vector>,
    pub learning_rate: f64,
}

// ============= LOAD POSISIONTS FROM FILE
pub fn load_positions(filename: &str) -> Vec<Vec<f64>> {
    match read_to_string(filename) {
        Ok(content) => {
            content
                .lines()
                .filter_map(|line| {
                    line.split_whitespace() // ← plus robuste que " "
                        .map(|s| s.parse::<f64>())
                        .collect::<Result<Vec<f64>, _>>()
                        .ok() // ← ignore les lignes mal formées
                })
                .collect()
        }
        Err(_) => {
            eprintln!("Erreur: impossible de lire le fichier {}", filename);
            vec![]
        }
    }
}

// Saving the parameters in binary file
#[allow(unused)]
pub fn save_parameters_binary(nn: &NeuralNetwork, file_path: String) -> std::io::Result<()> {
    let params = NNParameters {
        input_size: nn.is,
        layer_num: nn.ln,
        layer_sizes: nn.ls.clone(),
        weights: nn.weights.clone(),
        biases: nn.biases.clone(),
        learning_rate: nn.lr,
    };

    let encoded: Vec<u8> = bincode::serialize(&params).expect("Échec de la sérialisation");
    let filename = format!("{file_path}");
    let mut file = File::create(&filename)?;
    file.write_all(&encoded)?;
    Ok(())
}

#[allow(dead_code)]
pub fn load_parameters_binary(filename: String) -> std::io::Result<NNParameters> {
    let mut file = File::open(filename)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let params: NNParameters = bincode::deserialize(&buffer).expect("Échec de la désérialisation");
    Ok(params)
}
