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
pub fn continue_train_model_with_batch(
    model: &str,      // the model to train
    models_dir: &str,
    train_filename: &str,
    validation_filename: &str,
    model_name: &str, // name of the new model
    epochs: usize,
    batch_size: usize,
    step_size: usize,
) {
    let mut nn = NeuralNetwork::from_file(model.to_string());
    train_model_with_batch(&mut nn, models_dir, train_filename, validation_filename, model_name, epochs, batch_size, step_size);
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

    let model_dir = format!("{}/{}", models_dir, model_name);
    match std::fs::create_dir_all(&model_dir) {
        Ok(()) => println!("Directory ensured to exist: {}", &model_dir),
        Err(e) => eprintln!("Failed to create directory: {}", e),
    }

    let mut shuffler = rand::thread_rng();
    let in_lr = nn.lr;
    // let in_lr = 0.001;

    // ReduceLROnPlateau variables
    let mut best_val_acc = 0.0;
    let mut epochs_no_improve = 0;
    let plateau_factor = 0.5;
    let patience = 3;
    let min_lr = 1e-6;
    let mut decalage: f64 = 0.0;
    let mut step_decay: f64 = 0.0;

    for epoch in 0..epochs {
        training_pos.shuffle(&mut shuffler);

        // --- Step Decay ---
        step_decay = in_lr * 0.5_f64.powf((epoch) as f64 / step_size as f64);
        nn.lr = ((step_decay) - decalage).max(step_decay);

        for batch in training_pos.chunks(batch_size) {
            let mut batch_dz: Vec<Vector> = init_vectors(&nn.ls, false);
            let mut batch_dw: Vec<Matrix> = init_matrixes(&nn.ls, nn.is, false);

            for pos in batch.iter() {
                let p = &pos[0..=8];
                let player = pos[9];
                let cv_pos = one_hot(p.to_vec(), player as usize);

                // let d_star: usize = pos[10] as usize;
                // let a_star: usize = pos[11] as usize;

                let action_idx: usize = pos[10] as usize;

                let (dz, dw) = nn.compute_gradients(&cv_pos, action_idx);

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

            nn.apply_gradients(&batch_dz, &batch_dw);
        }

        // --- Evaluate ---
        let (acc_train, loss_train) = predict_moves(nn, train_file);
        let (acc_val, loss_val) = predict_moves(nn, val_file);
        // let (acc_val, loss_val) = (acc_train, loss_train);
        println!(
            "Epoch {}/{} | Train Acc: {:.4}%, Train Loss: {:.4} | Val Acc: {:.4}%, Val Loss: {:.4} | Lr: {:.6}",
            epoch + 1, epochs, acc_train, loss_train, acc_train, loss_train, nn.lr
        );

        // --- ReduceLROnPlateau ---
        if acc_val > best_val_acc {
            best_val_acc = acc_val;
            epochs_no_improve = 0;
        } else {
            epochs_no_improve += 1;
        }

        // --- ReduceLROnPlateau ---
        if epochs_no_improve >= patience && nn.lr > min_lr {
            decalage += nn.lr - (nn.lr * plateau_factor).max(min_lr);
            epochs_no_improve = 0;
            println!("LR reduced by plateau: {}", nn.lr);
        }

        // --- Save model parameters ---
        if let Err(e) = save_parameters_binary(
            nn,
            format!("{}/{}_E{}.bin", model_dir, model_name, epoch + 1),
        ) {
            eprintln!("Error saving parameters for epoch {}: {}", epoch + 1, e);
        }
    }
}
