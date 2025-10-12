#[allow(unused)]
use crate::{
    fanorona3::{Fanorontelo, one_hot_fanorona, valid_fn3_move},
    neural::Neural,
};

mod dataset;
mod fanorona3;
mod neural;
mod tictactoe;

#[allow(non_snake_case, unused)]
fn main() { 
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 || args[1] != "play" && args[1] != "cm" {
        eprintln!("Usage:
                    cargo run [options] play <model_path>
                    cargo run [options] cm <model_save_dir>");
        std::process::exit(1);
    }
    let todo = &args[1];
    
    if todo == "play" {
        play_to_test(&args[2]);
    } else if todo == "cm" {
        create_model(&args[2]);
    }
    else {
        println!("Invalid command !!");
    }
}

#[allow(unused)]
pub fn play_to_test(model_path:&str){
  let mut ne_load = Neural::load_from_bin(model_path);
  
  let mut ne :Neural;
    match ne_load {
        Ok(nn) => ne = nn,
        Err(e) => {
            panic!("Failed to load parameters from : {model_path}")
        }
    }

    // == for tictactoe game model
    // let mut tictactoe = Game::new(); 
    // play(&mut tictactoe, &mut ne);

    // == for fanorona 3x3 game
    let mut fn3_game = Fanorontelo {
        // the last element is the current player [ 1(=X) or 2(=0) ]
        board: [1, -1, 1, -1, 1, -1, 0, 0, 0, 1], // initial board here
    };
    fn3_game.play_with_bot(&mut ne);
}

// == Creating new models or continue training a model
#[allow(unused)]
pub fn create_model(save_dir:&str) {
// ============== MODEL AND TRAINING CONSTANTS ================
    let input_size = 46;
    let output_size = 81;
    let lr = 0.3;
    let epochs = 200;
    let layers = vec![512, output_size];
    let batch_size = 32;

    let board_size = 10; // the size of the game board (for fanorona3x3: 10 <the last element is the current player>)

    // ============= train dataset file ===================
    let tr_file = "dataset/fanorona/all.txt"; // actually it contains all possible positions

    // ==== load model from binaries
    let model = "models/fanorona3/epoch_200.bin";
    let mut ne = Neural::load_from_bin(model).unwrap();

    // ==== new model (creating new model from nothing)
    // let mut ne = Neural::xavier(layers, input_size);

    // ================= TRAINING ========================
    let train_start = std::time::Instant::now();
    ne.train(epochs, lr, board_size, batch_size, tr_file, save_dir); // training the model
    let train_elapsed = train_start.elapsed();

    let acc = ne.test(board_size, tr_file, one_hot_fanorona, valid_fn3_move); // testing with validation dataset
    println!(
        "Best moves: {:.4}%, Train time: {:?}",
        acc * 100.0,
        train_elapsed
    );
}