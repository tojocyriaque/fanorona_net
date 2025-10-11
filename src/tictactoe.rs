use crate::neural::Neural;

#[allow(dead_code)]
pub fn one_hot_tictactoe(board: &[f64]) -> Vec<f64> {
    let mut board_1hot: Vec<f64> = Vec::new();
    for v in board {
        let mut _1hot = [0.0, 0.0, 0.0];
        match v {
            0.0 => _1hot = [1.0, 0.0, 0.0],
            1.0 => _1hot = [0.0, 1.0, 0.0],
            2.0 => _1hot = [0.0, 0.0, 1.0],
            _ => _1hot = [0.0, 0.0, 0.0],
        };
        for h in _1hot {
            board_1hot.push(h);
        }
    }

    board_1hot
}

#[derive(Clone, Copy)]
pub struct Game {
    pub board: [usize; 9],
}

#[allow(unused)]
pub fn valid_ttt_move(board: &[f64], mv: usize) -> bool {
    board[mv] == 0.0
}

#[allow(unused)]
impl Game {
    pub fn new() -> Self {
        Game { board: [0; 9] }
    }

    // counting empty cells
    pub fn empty(&self) -> Vec<usize> {
        (0..9).filter(|&idx| self.board[idx].eq(&0)).collect()
    }

    pub fn game_over(&self) -> usize {
        let win_lines = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];

        for wl in win_lines {
            let pl = self.board[wl[0]];
            if wl.iter().all(|&val| self.board[val] == pl) {
                return pl;
            }
        }

        0
    }

    // checking is the game is a tie
    pub fn draw(&self) -> bool {
        self.empty().len() == 0 && self.game_over() == 0
    }

    // playing a move (sq is from 0 to 8)
    pub fn play(&mut self, sq: usize) {
        assert!(self.board[sq] == 0, "Non empty square !!");
        assert!(self.game_over() == 0, "Game is over !!");
        assert!(self.empty().len() != 0, "Draw !!");

        let zeros = self.empty().len();
        let player = 2 - zeros % 2;
        self.board[sq] = player;
    }

    pub fn show(&self) {
        for i in 0..3 {
            let line: Vec<&str> = (0..3)
                .map(|j| {
                    let pl = self.board[i * 3 + j];
                    ["_", "X", "0"][pl]
                })
                .collect();
            println!("{}", line.join(" "));
        }
    }

    // heuristic evaluation of the game board
    pub fn eval_board(&self) -> i32 {
        let winner = self.game_over();
        let mut score = 0;

        if self.empty().len() == 0 {
            return 0;
        }

        for sq in self.board {
            if sq == 1 {
                score += 1
            } else if sq == 2 {
                score -= 1
            }
        }

        [score, 100, -100][winner]
    }

    // finding the best move from the current game state (using minimax)
    pub fn best_move(&mut self, is_root: bool, best: &mut usize) -> i32 {
        let empties = self.empty().len();
        let player = 2 - empties % 2;

        let mut minimax_score = if player == 1 { i32::MIN } else { i32::MAX };

        let draw = empties == 0;
        let winner = self.game_over();

        if winner != 0 {
            let scr = if winner == 1 { 100 } else { -100 };
            return scr;
        }
        if draw {
            return 0;
        }

        let zeros = self.empty();
        for emp in zeros {
            let mut game_t = Game::from(*self);
            game_t.play(emp);
            let ev = game_t.best_move(false, best);

            if player == 1 && minimax_score < ev || player == 2 && minimax_score > ev {
                minimax_score = ev;
                if is_root {
                    *best = emp;
                }
            }
        }

        minimax_score
    }

    // playing a game against a NN model
    #[allow(unused)]
    fn play_with_bot(&mut self, ne: &mut Neural) {
        loop {
            self.show();
            println!("------");

            let winner = self.game_over();
            if winner != 0 {
                let winner_str = if winner == 1 { "X" } else { "0" };
                println!("Player {} wins !", winner_str);
                break;
            }
            if self.draw() {
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

            self.play(square);
            //  === MINIMAX
            // let mut best = 0;
            // self.best_move(true, &mut best);
            // println!("Best from PC: ({best})");
            let board_f64 = self.board.map(|f| f as f64);
            let best = ne.predict_best(&board_f64, one_hot_tictactoe);

            // println!("Board: {:?} => Best from CPU: {best}", board_f64);

            if self.game_over() == 0 && !self.draw() {
                self.play(best);
            }
        }
    }
}
