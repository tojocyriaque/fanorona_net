// =============== GAME LOGIC HERE =====================

use std::{collections::HashMap, usize};

use crate::{games::minmax::minimax, nn::NeuralNetwork};

pub type FanoronaBoard = Vec<i32>;
pub type GMove = (usize, usize);

fn swap_minmax(min: &mut i32, max: &mut i32, val: &mut i32) {
    *max = *max.max(val);
    *min = *min.min(val);
}

pub fn g_over(b: &FanoronaBoard) -> i32 {
    let mut max_sum: i32 = -100;
    let mut min_sum: i32 = 100;
    let mut index: usize;

    //     HORIZONTAL
    for i in 0..3 {
        let mut sum = 0;
        for j in 0..3 {
            index = 3 * i + j;
            sum += b[index];
        }
        swap_minmax(&mut min_sum, &mut max_sum, &mut sum);
    }
    // println!("HORIZONTAL: {max_sum} {min_sum}");

    //     VERTICAL
    for i in 0..3 {
        let mut sum = 0;
        for j in 0..3 {
            index = 3 * j + i;
            sum += b[index];
        }
        swap_minmax(&mut min_sum, &mut max_sum, &mut sum);
    }

    //     DIAGONAL
    let mut sd1 = b[0] + b[4] + b[8];
    let mut sd2 = b[2] + b[4] + b[6];

    swap_minmax(&mut min_sum, &mut max_sum, &mut sd1);
    swap_minmax(&mut min_sum, &mut max_sum, &mut sd2);

    if max_sum == 6 {
        1
    } else if min_sum == -6 {
        -1
    } else {
        0
    }
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

    println!(
        "Mouvements possibles: {:?}, {:?} ?? {}",
        moves,
        (s, e),
        move_is_possible
    );
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
    let mut moves: Vec<GMove> = vec![];
    for sq in 0..b.len() {
        let p = b[sq];
        if p * pl > 0 {
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
pub fn show_board(board: FanoronaBoard) {
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
    let mut curr_player = 1;
    while g_over(board) == 0 {
        let mut input = String::new();
        show_board(board.to_vec());
        // println!("> Joueur {} : ", (3 - curr_player) / 2);

        std::io::stdin().read_line(&mut input);
        let pos: Vec<usize> = input
            .strip_suffix("\n")
            .unwrap()
            .split(" ")
            .map(|u| u.parse().unwrap())
            .collect();

        let d: usize = pos[0];
        let a: usize = pos[1];

        let valid_play: bool = play(board, (d, a), curr_player);
        if valid_play {
            println!("human: G_over: {}", g_over(board));

            println!("CPU...");

            let mut best_move = (0, 0);
            minimax(board, 8, -curr_player, &mut best_move, true);
            play(board, best_move, -curr_player);
            println!("CPU: G_over: {}", g_over(board));
        } else {
            println!("Invalide !!")
        }
        println!("Board => {:?}", board);
    }

    println!("Game is over!!")
}
