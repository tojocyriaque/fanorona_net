use crate::{
    fanorona3::{Fanorontelo, one_hot_fanorona, valid_fn3_move},
    neural::Neural,
    tictactoe::one_hot_tictactoe,
};

mod dataset;
mod fanorona3;
mod neural;
mod tictactoe;

#[allow(non_snake_case)]
fn main() {
    let input_size = 46;
    let output_size = 81;
    let lr = 0.8;
    let epochs = 30;
    let layers = vec![512, output_size];

    let batch_size = 32;
    let board_size = 10;
    let tr_file = "dataset/fanorona/all.txt";

    // ==== load model
    let model = "models/fn_d6/epoch_30.bin";
    let mut ne = Neural::load_from_bin(model).unwrap();

    // ==== new model
    // let mut ne = Neural::xavier(layers, input_size);
    // let train_start = std::time::Instant::now();
    // let save_dir = "models/fn_d6";
    // ne.train(epochs, lr, board_size, batch_size, tr_file, save_dir);
    // let train_elapsed = train_start.elapsed();

    let acc = ne.test(board_size, tr_file, one_hot_fanorona, valid_fn3_move);
    // println!(
    //     "Accuracy: {:.4}%, Train time: {:?}",
    //     acc * 100.0,
    //     train_elapsed
    // );
    println!("Meilleures coups: {:?}%", acc * 100.0);

    // let mut tictactoe = Game::new();
    // play(&mut tictactoe, &mut ne);

    let mut fn3_game = Fanorontelo {
        board: [1, 0, 0, -1, -1, 0, -1, 1, 1, 1],
    };
    fn3_game.play_with_bot(&mut ne);
}

#[allow(unused)]
fn play(tictactoe: &mut tictactoe::Game, ne: &mut Neural) {
    loop {
        tictactoe.show();
        println!("------");

        let winner = tictactoe.game_over();
        if winner != 0 {
            let winner_str = if winner == 1 { "X" } else { "0" };
            println!("Player {} wins !", winner_str);
            break;
        }
        if tictactoe.draw() {
            println!("Draw !!");
            break;
        }

        let mut input = String::new();
        if let Err(e) = std::io::stdin().read_line(&mut input) {
            eprintln!("Error occured: {e} ");
        }

        input = input.strip_suffix("\n").unwrap().to_string();
        let mut square: usize = 0;
        match input.parse::<usize>() {
            Ok(sq) => {
                assert!(sq < 9, "Invalid square !!");
                square = sq
            }
            Err(e) => {
                eprintln!("Error occured: {e} ");
            }
        }

        tictactoe.play(square);
        //  === MINIMAX
        // let mut best = 0;
        // tictactoe.best_move(true, &mut best);
        // println!("Best from PC: ({best})");
        let board_f64 = tictactoe.board.map(|f| f as f64);
        let best = ne.predict_best(&board_f64, one_hot_tictactoe);

        // println!("Board: {:?} => Best from CPU: {best}", board_f64);

        if tictactoe.game_over() == 0 && !tictactoe.draw() {
            tictactoe.play(best);
        }
    }
}
