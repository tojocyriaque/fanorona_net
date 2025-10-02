use crate::testing::predict::predict_from_pos;
use axum::{
    extract::State,
    http::{Method, StatusCode},
    response::Json,
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tower_http::cors::{CorsLayer, Any, AllowOrigin};
use axum::http::HeaderValue;

mod data;
mod games;
mod maths;
mod nn;
mod testing;

// --- Structures pour l'API ---
#[derive(Deserialize)]
struct PredictionRequest {
    x: Vec<i32>,
}

#[derive(Serialize)]
struct PredictionResponse {
    d: usize,
    a: usize,
}

// --- Alias pour l’état ---
type AppState = String;

// --- Point d’entrée ---
#[tokio::main]
async fn main() {
    const MODEL_TO_USE: &str = "models/fn_model_vTest/fn_model_vTest_E20.bin";
    const SERVER_ADDRESS: &str = "0.0.0.0:3000";

    println!("Chargement du modèle : {}", MODEL_TO_USE);
    println!("Modèle chargé avec succès.");

    // --- CORS ---
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::exact(HeaderValue::from_static(
            "exp://192.168.137.153:8081",
        ))) // Autorise React Native en dev
        .allow_methods([Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    // --- Router ---
    let app = Router::new()
        .route("/tour", post(predict_handler))
        .with_state(MODEL_TO_USE.to_string())
        .layer(cors); // on ajoute le middleware CORS

    let addr: SocketAddr = SERVER_ADDRESS.parse().expect("Adresse du serveur invalide");
    println!("Serveur démarré, en écoute sur http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// --- Handler de la prédiction ---
async fn predict_handler(
    State(model_path): State<AppState>,
    Json(payload): Json<PredictionRequest>,
) -> (StatusCode, Json<PredictionResponse>) {
    println!("Requête de prédiction reçue pour le joueur {}", payload.x[9]);

    let (d, a) =
        predict_from_pos(&model_path, payload.x);

    let response = PredictionResponse {
        d,
        a,
    };

    (StatusCode::OK, Json(response))
}
