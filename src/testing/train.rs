use crate::maths::collectors::vec::Vector;
#[allow(unused)]
use crate::{
    data::loads::{load_positions, save_parameters_binary},
    maths::collectors::mat::Matrix,
    nn::{NeuralNetwork, init::one_hot_fanorona},
    testing::predict::predict_moves,
};

use rand::seq::SliceRandom;

#[allow(dead_code, non_snake_case)]
pub fn batch_train(
    model: &mut NeuralNetwork,
    tr_file: &str,
    batch_size: usize,
    i_epoch: usize,
    n_epoch: usize,
    model_name: &str,
    models_dir: &str,
    learning_rate: f64,
) {
    let output_size = model.ls[model.ln - 1];
    let mut positions = load_positions(tr_file);
    let mut shuffler = rand::thread_rng();

    let eps = 1e-1;
    let mut batch_loss = 0.0;
    model.lr = learning_rate;

    // Creating the model directory if it does not exist
    let model_dir = models_dir.to_owned() + "/" + model_name;
    match std::fs::create_dir_all(&model_dir) {
        Ok(()) => println!("Directory ensured to exist: {}", &model_dir),
        Err(e) => eprintln!("Failed to create directory: {}", e),
    }

    for epoch in i_epoch..i_epoch + n_epoch {
        positions.shuffle(&mut shuffler); // shuffle the positions each epoch

        let epoch_timer = std::time::Instant::now();
        for (b_idx, batch) in positions.chunks(batch_size).enumerate() {
            let norm_size = batch_size.min(batch.len());
            let mut X = Matrix::init_0(norm_size, model.is);
            let mut Y = Matrix::init_0(norm_size, output_size);

            for (k, pos) in batch.iter().enumerate() {
                let board = pos[0..=8].to_vec();
                let player: usize = pos[9] as usize;

                X[k] = one_hot_fanorona(board, player);
                Y[k] = Vector(pos[10..].to_vec());
            }

            let (GW, GZ, bl) = model.batch_grads(&X, &Y);
            batch_loss = bl;
            model.batch_backward(&GW, &GZ);
            println!("Batch {b_idx} loss: {batch_loss:.4}, pas: {:.4}", model.lr);
        }

        let elapsed = epoch_timer.elapsed();
        println!(
            "Epoque {}/{} terminée en {:.4?}",
            epoch,
            i_epoch + n_epoch - 1,
            elapsed
        );
        println!("---------------------------------------");

        // ================ MAKING THE PREDICTION IS SLOWER THAN THE TRAINING
        // let ((acc_a, acc_d, acc), loss) = predict_moves(model, tr_file);
        // println!("Prediction => a:{acc_a:.2}%, d:{acc_d:.2}%, (d,a):{acc:.2}%, loss:{loss:.2}");
        //
        // save_parameters_binary(nn: &NeuralNetwork, file_path: String)
        let model_bin = format!("{models_dir}/{model_name}/{model_name}_E_{epoch}.bin");
        if let Err(e) = save_parameters_binary(model, model_bin.clone()) {
            eprintln!(
                "Error on saving parameters at {model_bin} for epoch {}: {}",
                epoch + 1,
                e
            );
        }

        if batch_loss <= eps {
            break;
        }
    }
}
