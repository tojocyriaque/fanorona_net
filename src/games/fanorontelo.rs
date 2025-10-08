// =============== GAME LOGIC HERE =====================

use std::{collections::HashMap, usize};

use crate::{
    games::{fanorona::GMove, minmax::minimax_multi},
    nn::{NeuralNetwork, init::one_hot_fanorona},
};

type BoardType = Vec<i32>;
type MoveType = (usize, usize);

pub struct Fanorontelo {
    board: BoardType,
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

#[allow(unused)]
impl Fanorontelo {
    // Heuristic evaluation
    pub fn evaluate_board(&self) -> i32 {
        let winner: i32 = self.game_over();

        if winner != 0 {
            return 10 * winner;
        }

        self.board.iter().sum()
    }

    pub fn game_over(&self) -> i32 {
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
            let sum: i32 = win_line.map(|k| self.board[k]).iter().sum();
            if sum.abs() == 6 {
                return sum / 6;
            }
        }

        0
    }

    pub fn play_move(&mut self, (s, e): MoveType, pl: i32) -> bool {
        let winner: i32 = self.game_over();
        if winner != 0 {
            return false;
        }

        let moves: Vec<MoveType> = self.possible_moves(pl);
        let move_is_possible = moves.contains(&(s, e));
        if move_is_possible {
            let st_v = self.board[s];
            let end_v = self.board[e];

            if end_v == 0 && pl * st_v > 0 && winner == 0 {
                self.board[e] = if st_v.abs() < 2 { st_v * 2 } else { st_v };
                self.board[s] = 0;

                return true;
            }
        }
        false
    }

    pub fn possible_moves(&self, pl: i32) -> Vec<MoveType> {
        let g_neighbours = neighbours();
        let mut moves: Vec<MoveType> = vec![];
        for sq in 0..self.board.len() {
            let p = self.board[sq];
            if p * pl > 0 {
                let nei = g_neighbours.get(&sq).unwrap();
                for &n in nei {
                    if self.board[n] == 0 {
                        moves.push((sq, n));
                    }
                }
            }
        }
        moves
    }

    // DISPLAY
    #[allow(unused)]
    pub fn show_board(&self) {
        println!("-----------------------");
        for i in 0..3 {
            for j in 0..3 {
                let idx = 3 * i + j;
                let p = self.board[idx];
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
    pub fn play_with_bot(&mut self, model: &str) {
        let mut nn: NeuralNetwork = NeuralNetwork::from_file(model.to_string());
        let mut human_player = 1;
        while self.game_over() == 0 {
            let mut input = String::new();
            self.show_board();
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

            let valid_play: bool = self.play_move((d, a), human_player);
            if valid_play {
                // println!("human: G_over: {}", g_over(board));

                println!("CPU...");

                // minimax(board, 8, -human_player, &mut best_move, true); // using minimax bot

                let p: Vec<f64> = self.board[0..=8].iter().map(|&f| f as f64).collect();
                let player = if -human_player == 1 { 1 } else { 2 };
                let cv_pos = one_hot_fanorona(p, player as usize);

                // let ((d, pd), (a, pa)) = nn.predict(cv_pos.clone());
                // (d, a);
                //
                // let ((d, _), (a, _)) = nn.predict(cv_pos); // using the network
                let mut best_moves: Vec<GMove> = Vec::new();
                minimax_multi(&self.board, 7, -human_player, true, &mut best_moves);

                let mut best_move = best_moves[0];
                let played = self.play_move(best_move, -human_player);
                if !played {
                    println!("CPU did not play !! it's move: {:?}", best_move);
                }
                // println!("CPU: G_over: {}", g_over(board));
            } else {
                println!("Invalid !!")
            }
            // println!("Board => {:?}", board);
        }

        println!("Game is over!!")
    }
}
