use rand::seq::SliceRandom;

use crate::data::loads::*;
use crate::nn::init::one_hot;
use crate::nn::*;
use crate::testing::predict::predict_moves;

// math tools
use crate::maths::collectors::vec::Vector;
use crate::maths::collectors::mat::Matrix;
use crate::testing::train::init::init_vectors;
use crate::testing::train::init::init_matrixes;

#[allow(unused)]
pub fn continue_train_model(
    model: &str,      // the model to train
    model_name: &str, // name of the new model
    models_dir: &str,
    train_filename: &str,
    validation_filename: &str,
    epochs: usize,
) {
    let mut nn = NeuralNetwork::from_file(model.to_string());
    train_model(&mut nn, models_dir, train_filename,validation_filename, model_name, epochs);
}

#[allow(unused)]
pub fn train_model(
    nn: &mut NeuralNetwork,
    models_dir: &str,
    train_file: &str,
    val_file: &str,
    model_name: &str,
    epochs: usize,
) {
    let mut training_pos: Vec<Vec<i32>> = load_positions(train_file);

    // Create the model directory if it does not exist
    let model_dir = format!("{}/{}", models_dir, model_name);
    match std::fs::create_dir_all(&model_dir) {
        Ok(()) => println!("Directory ensured to exist: {}", &model_dir),
        Err(e) => eprintln!("Failed to create directory: {}", e),
    }

    let mut shuffler = rand::thread_rng();
    for epoch in 0..epochs {
        // Shuffle the training positions each epoch
        training_pos.shuffle(&mut shuffler);

        // Training loop
        for pos in training_pos.iter() {
            let p = &pos[0..=8];
            let player = pos[9];
            let cv_pos = one_hot(p.to_vec(), player as usize);

            let d_star: usize = pos[10] as usize;
            let a_star: usize = pos[11] as usize;

            nn.back_prop(&cv_pos, d_star, a_star + 9);
        }

        // Evaluate on training set
        let ((acc_a_train, acc_d_train, acc_train), loss_train) = predict_moves(nn, train_file);

        // Evaluate on validation set
        let ((acc_a_val, acc_d_val, acc_val), loss_val) = predict_moves(nn, val_file);

        println!(
            "Epoch {}/{} | Train Acc: {:.4}%, Train Loss: {:.4} | Val Acc: {:.4}%, Val Loss: {:.4}",
            epoch + 1,
            epochs,
            acc_train,
            loss_train,
            acc_val,
            loss_val
        );

        // Save parameters after each epoch
        if let Err(e) = save_parameters_binary(
            nn,
            format!("{}/{}_E{}.bin", model_dir, model_name, epoch + 1),
        ) {
            eprintln!("Error saving parameters for epoch {}: {}", epoch + 1, e);
        }
    }
}

#[allow(unused)]
pub fn continue_train_model_with_batch(
    model: &str,      // the model to train
    model_name: &str, // name of the new model
    models_dir: &str,
    train_filename: &str,
    validation_filename: &str,
    epochs: usize,
    batch_size: usize,
    step_size: usize,
) {
    let mut nn = NeuralNetwork::from_file(model.to_string());
    train_model_with_batch(&mut nn, models_dir, train_filename,validation_filename, model_name, epochs,batch_size,step_size);
}

#[allow(unused)]
pub fn train_model_with_batch(
    nn: &mut NeuralNetwork,
    models_dir: &str,
    train_file: &str,
    val_file: &str,
    model_name: &str,
    epochs: usize,
    batch_size: usize,
    step_size: usize,
) {
    let mut training_pos: Vec<Vec<i32>> = load_positions(train_file);

    // Create the model directory if it does not exist
    let model_dir = format!("{}/{}", models_dir, model_name);
    match std::fs::create_dir_all(&model_dir) {
        Ok(()) => println!("Directory ensured to exist: {}", &model_dir),
        Err(e) => eprintln!("Failed to create directory: {}", e),
    }

    let mut shuffler = rand::thread_rng();
    let in_lr = 0.0032352028870043046;
    for epoch in 0..epochs {
        training_pos.shuffle(&mut shuffler);
        nn.lr = in_lr * 0.5_f64.powf(epoch as f64 / step_size as f64);
        println!("Lr: {}",nn.lr);

        for batch in training_pos.chunks(batch_size) {
            // Initialize accumulators
            let mut batch_dz: Vec<Vector> = init_vectors(&nn.ls, false);
            let mut batch_dw: Vec<Matrix> = init_matrixes(&nn.ls, nn.is, false);

            for pos in batch.iter() {
                let p = &pos[0..=8];
                let player = pos[9];
                let cv_pos = one_hot(p.to_vec(), player as usize);

                let d_star: usize = pos[10] as usize;
                let a_star: usize = pos[11] as usize;

                // Compute gradients
                let (dz, dw) = nn.compute_gradients(&cv_pos, d_star, a_star + 9);

                // Accumulate gradients safely
                for k in 0..nn.ln {
                    for i in 0..dz[k].len() {
                        batch_dz[k][i] += dz[k][i];
                    }
                    for i in 0..dw[k].len() {
                        for j in 0..dw[k][i].len() {
                            batch_dw[k][i][j] += dw[k][i][j];
                        }
                    }
                }
            }

            // Average gradients
            let batch_len = batch.len() as f64;
            for k in 0..nn.ln {
                for i in 0..batch_dz[k].len() {
                    batch_dz[k][i] /= batch_len;
                }
                for i in 0..batch_dw[k].len() {
                    for j in 0..batch_dw[k][i].len() {
                        batch_dw[k][i][j] /= batch_len;
                    }
                }
            }

            // Apply averaged gradients
            nn.apply_gradients(&batch_dz, &batch_dw);
        }

        // Evaluate
        let ((acc_a_train, acc_d_train, acc_train), loss_train) = predict_moves(nn, train_file);
        let ((acc_a_val, acc_d_val, acc_val), loss_val) = predict_moves(nn, val_file);

        println!(
            "Epoch {}/{} | Train Acc: {:.4}%, Train Loss: {:.4} | Val Acc: {:.4}%, Val Loss: {:.4}",
            epoch + 1, epochs, acc_train, loss_train, acc_val, loss_val
        );

        // Save model parameters
        if let Err(e) = save_parameters_binary(
            nn,
            format!("{}/{}_E{}.bin", model_dir, model_name, epoch + 1),
        ) {
            eprintln!("Error saving parameters for epoch {}: {}", epoch + 1, e);
        }
    }
}
