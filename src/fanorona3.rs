// =============== GAME LOGIC HERE =====================

use std::{collections::HashMap, usize};

use crate::neural::Neural;

type BoardType = [i32; 10];
type MoveType = (usize, usize);

pub struct Fanorontelo {
    pub board: BoardType,
}

pub fn one_hot_fanorona(board: &[f64]) -> Vec<f64> {
    let mut board_1hot: Vec<f64> = board[0..=8]
        .iter()
        .flat_map(|&v| match v {
            0. => [1., 0., 0., 0., 0.],
            1. => [0., 1., 0., 0., 0.],
            2.0 => [0., 0., 1., 0., 0.],
            -1. => [0., 0., 0., 0., 1.],
            -2.0 => [0., 0., 0., 1., 0.],
            _ => [0., 0., 0., 0., 0.],
        })
        .collect();

    board_1hot.push([0.0, 1.0][board[9] as usize - 1]);
    board_1hot
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

pub fn valid_fn3_move(board: &[f64], mv: usize) -> bool {
    let mut gm_brd: [i32; 10] = [0; 10];
    for (idx, &v) in board.iter().enumerate() {
        gm_brd[idx] = v as i32;
    }

    let fn_game = Fanorontelo { board: gm_brd };

    let valids = fn_game.possible_moves();
    for (start, end) in valids {
        if start * 9 + end == mv {
            return true;
        }
    }

    false
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

    pub fn play_move(&mut self, (s, e): MoveType) -> bool {
        let pl = self.board[9];
        let player = if pl == 1 { 1 } else { -1 };
        let winner: i32 = self.game_over();
        if winner != 0 {
            return false;
        }

        let moves: Vec<MoveType> = self.possible_moves();
        let move_is_possible = moves.contains(&(s, e));
        if move_is_possible {
            let st_v = self.board[s];
            let end_v = self.board[e];

            if end_v == 0 && player * st_v > 0 && winner == 0 {
                self.board[e] = if st_v.abs() < 2 { st_v * 2 } else { st_v };
                self.board[s] = 0;
                let next_player = if pl == 1 { 2 } else { 1 };
                self.board[9] = next_player;
                return true;
            }
        }

        false
    }

    pub fn possible_moves(&self) -> Vec<MoveType> {
        let pl = self.board[9];
        let player = if pl == 1 { 1 } else { -1 };

        let g_neighbours = neighbours();
        let mut moves: Vec<MoveType> = vec![];
        for sq in 0..self.board.len() - 1 {
            let piece = self.board[sq];
            if piece * player > 0 {
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
    pub fn play_with_bot(&mut self, ne: &mut Neural) {
        loop {
            let mut input = String::new();
            self.show_board();
            // println!("> Joueur {} : ", (3 - human_player) / 2);

            let winner = self.game_over();
            if winner != 0 {
                let winner_str = if winner == 1 { "X" } else { "0" };
                println!("{winner_str} wins !!");
                break;
            }

            let human = if self.board[9] == 1 { "X" } else { "0" };
            println!("You... ({human})");
            std::io::stdin().read_line(&mut input);
            let mv: Vec<usize> = input
                .strip_suffix("\n")
                .unwrap()
                .split(" ")
                .map(|u| u.parse().unwrap())
                .collect();

            let d: usize = mv[0];
            let a: usize = mv[1];

            let valid_play: bool = self.play_move((d, a));
            if valid_play {
                // println!("human: G_over: {}", g_over(board));
                let cpu = if self.board[9] == 1 { "X" } else { "0" };
                println!("CPU... ({cpu})");
                let mut board: [f64; 10] = self.board.map(|f| f as f64);
                let cpu_best = ne.predict_best(&board.as_slice(), one_hot_fanorona);
                let start = cpu_best / 9;
                let end = cpu_best % 9;

                println!("CPU best: {start} -> {end}");
                self.play_move((start, end));
                // println!("CPU: G_over: {}", g_over(board));
            } else {
                println!("Invalid !!")
            }
            // println!("Board => {:?}", board);
        }

        println!("Game is over!!")
    }
}
