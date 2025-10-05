use rand::seq::SliceRandom;
use crate::{
    data::loads::{load_positions, save_parameters_binary},
    maths::collectors::mat::Matrix,
    nn::{NeuralNetwork, init::one_hot},
    testing::predict::predict_moves,
};

#[allow(dead_code, non_snake_case)]
pub fn batch_train(
    model: &mut NeuralNetwork,
    tr_file: &str,
    batch_size: usize,
    i_epoch: usize,
    n_epoch: usize,
    model_name:&str,
    models_dir:&str,
) {
    let output_size = model.ls[model.ln - 1];
    let mut positions = load_positions(tr_file);
    let mut shuffler = rand::thread_rng();

    // Creating the model directory if it does not exist
    let model_dir = models_dir.to_owned() + "/" + model_name;
    match std::fs::create_dir_all(&model_dir) {
        Ok(()) => println!("Directory ensured to exist: {}", &model_dir),
        Err(e) => eprintln!("Failed to create directory: {}", e),
    }


    for epoch in i_epoch..i_epoch + n_epoch {
        // positions.shuffle(&mut shuffler); // shuffle the positions each epoch
        
        let epoch_timer = std::time::Instant::now();
        for batch in positions.chunks(batch_size) {
            let mut X = Matrix::init_0(batch_size, model.is);
            let mut Y = Matrix::init_0(batch_size, output_size);

            for (k, pos) in batch.iter().enumerate() {
                let board = pos[0..=8].to_vec();
                let player = pos[9];
                let converted = one_hot(board, player as usize);
                let d_star: usize = pos[10] as usize;
                let a_star: usize = pos[11] as usize + 9;

                X[k] = converted;

                Y[k][d_star] = 1.0;
                Y[k][a_star] = 1.0;
            }

            let (GW, GZ) = model.batch_grads(&X, &Y);
            model.batch_backward(&GW, &GZ);
        }
        
        let elapsed = epoch_timer.elapsed();
        println!("Epoque {}/{} terminée en {:?}", epoch, i_epoch + n_epoch - 1, elapsed);

        // let ((acc_a, acc_d, acc), loss) = predict_moves(model, tr_file);
        // println!("Prediction => a:{acc_a:.2}%, d:{acc_d:.2}%, (d,a):{acc:.2}%, loss:{loss:.2}");
        //
        // save_parameters_binary(nn: &NeuralNetwork, file_path: String) 
        let model_bin = format!("{models_dir}/{model_name}/{model_name}_E_{epoch}.bin");
        if let Err(e) = save_parameters_binary(model, model_bin.clone()){
            eprintln!("Error on saving parameters at {model_bin} for epoch {}: {}", epoch + 1, e);
        }
    }
}

