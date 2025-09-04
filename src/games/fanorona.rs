// =============== GAME LOGIC HERE =====================

use std::{collections::HashMap, usize};

use crate::{games::minmax::minimax_multi, nn::NeuralNetwork, testing::predict::predict_from_pos};

pub type FanoronaBoard = Vec<i32>;
pub type GMove = (usize, usize);

// Heuristic evaluation
pub fn evaluate_board(b: &FanoronaBoard) -> i32 {
    let winner: i32 = g_over(b);

    if winner != 0 {
        return 10 * winner;
    }

    b.iter().sum()
}

pub fn g_over(b: &FanoronaBoard) -> i32 {
    let win_lines = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8], // lignes
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8], // colonnes
        [0, 4, 8],
        [2, 4, 6], // diagonales
    ];

    for win_line in win_lines {
        let sum: i32 = win_line.map(|k| b[k]).iter().sum();
        if sum.abs() == 6 {
            return sum / 6;
        }
    }

    0
}

pub fn play(b: &mut FanoronaBoard, (s, e): GMove, pl: i32) -> bool {
    let winner: i32 = g_over(b);
    if winner != 0 {
        return false;
    }

    let moves: Vec<GMove> = possible(b, pl);
    let move_is_possible = moves.contains(&(s, e));
    if move_is_possible {
        let st_v = b[s];
        let end_v = b[e];

        if end_v == 0 && pl * st_v > 0 && winner == 0 {
            b[e] = if st_v.abs() < 2 { st_v * 2 } else { st_v };
            b[s] = 0;

            return true;
        }
    }

    // println!(
    //     "Mouvements possibles: {:?}, {:?} ?? {}",
    //     moves,
    //     (s, e),
    //     move_is_possible
    // );
    false
}

pub fn neighbours() -> HashMap<usize, Vec<usize>> {
    HashMap::from([
        (0, vec![1, 3, 4]),
        (1, vec![0, 2, 4]),
        (2, vec![1, 4, 5]),
        (3, vec![0, 4, 6]),
        (4, vec![0, 1, 2, 3, 5, 6, 7, 8]),
        (5, vec![2, 4, 8]),
        (6, vec![3, 4, 7]),
        (7, vec![4, 6, 8]),
        (8, vec![4, 5, 7]),
    ])
}

pub fn possible(b: &FanoronaBoard, pl: i32) -> Vec<GMove> {
    let g_neighbours = neighbours();
    // println!("Neigh: {:?}", g_neighbours);
    let mut moves: Vec<GMove> = vec![];
    for sq in 0..b.len() {
        let p = b[sq];
        if p * pl > 0 {
            // println!("SQ: {sq}");
            let nei = g_neighbours.get(&sq).unwrap();
            for &n in nei {
                if b[n] == 0 {
                    moves.push((sq, n));
                }
            }
        }
    }
    moves
}

// DISPLAY
#[allow(unused)]
pub fn show_board(board: &FanoronaBoard) {
    println!("-----------------------");
    for i in 0..3 {
        for j in 0..3 {
            let idx = 3 * i + j;
            let p = board[idx];
            if p > 0 {
                print!("X ");
            } else if p < 0 {
                print!("O ");
            } else {
                print!("_ ");
            }
        }
        println!()
    }
    println!("-----------------------")
}

#[allow(unused)]
pub fn play_fanorona(board: &mut FanoronaBoard, model: &str) {
    let mut nn: NeuralNetwork = NeuralNetwork::from_file(model.to_string());
    let mut human_player = -1;
    while g_over(board) == 0 {
        // Bot play
        println!("CPU...");

        let mut moves: Vec<GMove> = Vec::new();
        let pos: Vec<f64> = board[0..=8].iter().map(|&f| f as f64).collect();
        let bot_player = if -human_player == 1 { 1 } else { 2 };
        minimax_multi(&board, 7, -human_player, true, &mut moves);

        let mut pos_to_predict = board.to_vec();
        pos_to_predict.push(bot_player);
        let nn_move = predict_from_pos(model, pos_to_predict);
        println!("Minimax: {:?}, NN: {:?}", moves[0], nn_move);

        let best_move = moves[0];

        let played = play(board, best_move, -human_player);
        if !played {
            println!("CPU did not play !! it's move: {:?}", best_move);
        }

        let mut input = String::new();
        show_board(&board.to_vec());
        // println!("> Joueur {} : ", (3 - human_player) / 2);

        std::io::stdin().read_line(&mut input);
        let mv: Vec<usize> = input
            .strip_suffix("\n")
            .unwrap()
            .split(" ")
            .map(|u| u.parse().unwrap())
            .collect();

        let d: usize = mv[0];
        let a: usize = mv[1];

        let valid_play: bool = play(board, (d, a), human_player);

        if !valid_play {
            println!("Invalid !!")
        }
        // println!("Board => {:?}", board);
    }

    println!("Game is over!!")
}
