use crate::games::fanorona::*;

// pub fn minimax_generic<Game, GMvType>(
//     game: &Game,
//     depth: usize,
//     is_max: i32,
//     is_root: bool, // to see if it's a root node of the move sequence tree
//     b_mv: &mut GMvType,
// ) -> f64
// where
//     Game:,
// {
//     0.0
// }

// Minimax algorithm to find best move
pub fn minimax(
    b: &FanoronaBoard,
    depth: usize,
    is_max: i32,
    b_mv: &mut GMove,
    is_root: bool,
) -> i32 {
    let winner = g_over(b);
    if depth == 0 || winner != 0 {
        return evaluate_board(b);
    }

    let mut minmax_score = if is_max == 1 { i32::MIN } else { i32::MAX };
    let p_mvs = possible(b, is_max);

    if p_mvs.is_empty() {
        // No valid moves
        return evaluate_board(b);
    }

    for mv in p_mvs {
        let mut b_t = b.clone();
        play(&mut b_t, mv, is_max);
        let ev: i32 = minimax(&b_t, depth - 1, -is_max, b_mv, false);

        if is_max == 1 && ev > minmax_score || is_max == -1 && ev < minmax_score {
            minmax_score = ev;
            // root node of the minimax tree
            if is_root {
                *b_mv = mv;
            }
        }
    }
    minmax_score
}

// Minimax algo to get multiple moves
pub fn minimax_multi(
    b: &FanoronaBoard,
    depth: usize,
    is_max: i32,
    is_root: bool,
    moves: &mut Vec<(GMove)>,
) -> i32 {
    let winner = g_over(b);
    if depth == 0 || winner != 0 {
        return evaluate_board(b);
    }

    let mut minmax_score = if is_max == 1 { i32::MIN } else { i32::MAX };
    let p_mvs = possible(b, is_max);

    if p_mvs.is_empty() {
        // No valid moves
        return evaluate_board(b);
    }

    for mv in p_mvs {
        let mut board_t = b.clone();
        play(&mut board_t, mv, is_max);
        let ev: i32 = minimax_multi(&board_t, depth - 1, -is_max, false, moves);

        if is_max == 1 && ev > minmax_score || is_max == -1 && ev < minmax_score {
            minmax_score = ev;
            if is_root {
                let idx = moves.len();
                moves.insert(idx, mv);
            }
        } else if is_root {
            moves.push(mv);
        }
    }
    minmax_score
}

// Transforming array of sorted moves into distrubution of probability
#[allow(unused)]
pub fn moves_proba(s_moves: &Vec<GMove>, sq_num: usize) -> (Vec<f64>, Vec<f64>) {
    // probabilty for start square
    let mut start_proba: Vec<f64> = vec![0.0; sq_num];
    // probabilty for end square
    let mut end_proba: Vec<f64> = vec![0.0; sq_num];

    let l = s_moves.len() as f64;
    for (idx, &(s, e)) in s_moves.iter().enumerate() {
        let k = idx as f64 + 1.0;
        // best move has best proba (got this sequence formula from the fact that the sum is always 1)
        let proba = (1.0 - k / (l + 1.0)) * (2.0 / l);

        start_proba[s] = proba;
        end_proba[e] = proba;

        println!("Move: {:?} Proba: {proba}", (s, e));
    }
    (start_proba, end_proba)
}
