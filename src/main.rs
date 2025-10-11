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
    let input_size = 46;
    let output_size = 81;
    let lr = 0.5;
    let epochs = 0;
    let layers = vec![512, output_size];

    let batch_size = 32;
    let board_size = 10;
    let tr_file = "dataset/fanorona/all.txt";

    // ==== load model
    let model = "models/fn_d6_7/epoch_10.bin";
    let mut ne = Neural::load_from_bin(model).unwrap();

    // ==== new model
    // let mut ne = Neural::xavier(layers, input_size);
    let train_start = std::time::Instant::now();
    let save_dir = "models/fn_d6_8";
    ne.train(epochs, lr, board_size, batch_size, tr_file, save_dir);
    let train_elapsed = train_start.elapsed();

    let acc = ne.test(board_size, tr_file, one_hot_fanorona, valid_fn3_move);
    println!(
        "Meilleurs coups: {:.4}%, Train time: {:?}",
        acc * 100.0,
        train_elapsed
    );
    // println!("Meilleures coups: {:?}%", acc * 100.0);

    // let mut tictactoe = Game::new();
    // play(&mut tictactoe, &mut ne);

    // the last element if the current player (1 or 2)
    let mut fn3_game = Fanorontelo {
        board: [1, -1, 1, -1, 1, -1, 0, 0, 1, 2],
    };
    fn3_game.play_with_bot(&mut ne);
}
