use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use crate::nn::NeuralNetwork;

// === Structures ===
#[derive(Deserialize)]
pub struct PredictionRequest {
    pub x: Vec<i32>,
}

#[derive(Serialize)]
pub struct PredictionResponse {
    pub d: usize,
    pub a: usize,
}

// === Constante : chemin du modèle ===
// Include the model bytes directly into the wasm binary at compile time.
const MODEL_BYTES: &[u8] = include_bytes!("../models/fn_model_dReal_ng_v4/fn_model_dReal_ng_v4_E15.bin");

// === Fonction principale exportée vers JS ===
#[wasm_bindgen]
pub fn make_ai_move(input_json: &str) -> String {
    // 1️⃣ Désérialiser la requête JSON reçue depuis JS
    let parsed: PredictionRequest = serde_json::from_str(input_json)
        .expect("Invalid JSON input for AI prediction");

    // Load the neural network from the included bytes
    let nn = NeuralNetwork::from_binary(MODEL_BYTES).expect("Failed to load model from bytes");

    // 2️⃣ Appeler ton modèle existant
    let (d, a) = nn.predict_from_pos(parsed.x);

    // 3️⃣ Créer la réponse
    let response = PredictionResponse { d, a };

    // 4️⃣ Sérialiser en JSON (retour vers JS)
    serde_json::to_string(&response).unwrap()
}
