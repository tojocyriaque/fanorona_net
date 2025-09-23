use rand::seq::SliceRandom;

use crate::data::loads::*;
use crate::nn::init::one_hot;
use crate::nn::*;
use crate::testing::predict::predict_moves;

#[allow(unused)]
pub fn train_model(
    nn: &mut NeuralNetwork,
    models_dir: &str,
    train_filename: &str,
    model_name: &str,
    epochs: usize,
) {
    let mut training_pos: Vec<Vec<i32>> = load_positions(train_filename);

    // Creating the model directory if it does not exist
    let model_dir = models_dir.to_owned() + "/" + model_name;
    match std::fs::create_dir_all(&model_dir) {
        Ok(()) => println!("Directory ensured to exist: {}", &model_dir),
        Err(e) => eprintln!("Failed to create directory: {}", e),
    }

    let mut shuffler = rand::thread_rng();
    for epoch in 0..epochs {
        // avoir the network to learn from the same positions
        // allow it to view various positions order
        training_pos.shuffle(&mut shuffler);
        // println!("-----------");
        // println!("{:.2?}", nn.weights[nn.ln - 1]);
        // println!("-----------");
        for (_, pos) in training_pos.iter().enumerate() {
            let p = &pos[0..=8];
            let player = &pos[9];
            let cv_pos = one_hot(p.to_vec(), *player as usize);

            let d_star: usize = pos[10] as usize;
            let a_star: usize = pos[11] as usize;

            // update the parameters
            nn.back_prop(&cv_pos, d_star, a_star + 9);
        }

        let accuracy = predict_moves(nn, train_filename);
        println!(
            "Epoch {}/{epochs} finished! => {accuracy:.2}% accuracy",
            epoch + 1
        );

        // Save parameters after each epoch
        if let Err(e) =
            // Saving the model parameters after each epoch
            save_parameters_binary(
                nn,
                format!("{model_dir}/{model_name}_E{}.bin", epoch + 1),
            )
        {
            eprintln!("Error on saving parameters for epoch {}: {}", epoch + 1, e);
        }
    }
}
