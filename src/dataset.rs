use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::{neural::Vector, tictactoe::Game};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Board {
    cells: [usize; 9], // 0 = vide, 1 = X, -1 = O
}

#[allow(dead_code, unused)]
impl Board {
    fn new() -> Self {
        Board { cells: [0; 9] }
    }

    fn count(&self, player: usize) -> usize {
        self.cells.iter().filter(|&&c| c == player).count()
    }

    fn is_winner(&self, player: usize) -> bool {
        let w = player;
        let c = &self.cells;

        // Lignes
        (c[0] == w && c[1] == w && c[2] == w) ||
        (c[3] == w && c[4] == w && c[5] == w) ||
        (c[6] == w && c[7] == w && c[8] == w) ||
        // Colonnes
        (c[0] == w && c[3] == w && c[6] == w) ||
        (c[1] == w && c[4] == w && c[7] == w) ||
        (c[2] == w && c[5] == w && c[8] == w) ||
        // Diagonales
        (c[0] == w && c[4] == w && c[8] == w) ||
        (c[2] == w && c[4] == w && c[6] == w)
    }

    fn is_valid(&self) -> bool {
        let count_x = self.count(1);
        let count_o = self.count(2);

        // Condition de base
        if count_x < count_o || count_x > count_o + 1 {
            return false;
        }

        let x_wins = self.is_winner(1);
        let o_wins = self.is_winner(2);

        // Les deux ne peuvent pas gagner
        if x_wins && o_wins {
            return false;
        }

        // Si X gagne, il doit avoir un pion de plus
        if x_wins && count_x != count_o + 1 {
            return false;
        }

        // Si O gagne, ils doivent avoir le même nombre
        if o_wins && count_x != count_o {
            return false;
        }

        true
    }
}

#[allow(unused)]
pub fn generate_all_boards() -> Vec<[usize; 9]> {
    let mut valid_boards = Vec::new();

    // Génère toutes les combinaisons possibles (3^9 = 19683)
    for i in 0..19683 {
        let mut board = Board::new();
        let mut temp = i;

        // Convertit en base 3
        for j in 0..9 {
            let digit = temp % 3;
            board.cells[j] = digit;
            temp /= 3;
        }

        if board.is_valid() {
            valid_boards.push(board.cells);
        }
    }

    valid_boards
}

#[allow(unused)]
pub fn generate_dataset() {
    let mut bmv = 0;
    let boards: Vec<[usize; 9]> = generate_all_boards();
    for (_, board) in boards.into_iter().enumerate() {
        let mut game_t = Game { board };
        game_t.best_move(true, &mut bmv);

        if !game_t.draw() && game_t.game_over() == 0 {
            let mut bvec: [usize; 9] = [0; 9];
            bvec[bmv] = 1;
            let data_vec = [board, bvec].concat();

            let data_str = data_vec
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<String>>()
                .join(" ");

            println!("{}", data_str);
        }
    }
}

pub fn load_dataset(fname: &str, input_size: usize, label_size: usize) -> Vec<(Vector, Vector)> {
    let mut datasets: Vec<(Vector, Vector)> = Vec::new();
    let file = File::open(fname).unwrap();
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let data_line = line.unwrap();
        let data: Vec<f64> = data_line.split(" ").map(|f| f.parse().unwrap()).collect();

        let x: Vector = data[0..input_size].to_vec().into();
        let y: Vector = data[input_size..input_size + label_size].to_vec().into();

        datasets.push((x, y));
    }

    // for (idx, (inp, lab)) in datasets.iter().enumerate() {
    //     // println!("{idx} X:{inp} => Y:{lab}")
    //     println!("X count: {} Y count: {}", inp.len(), lab.len());
    // }

    datasets
}
