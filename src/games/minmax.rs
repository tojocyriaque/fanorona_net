use crate::games::fanorona::*;

// Heuristic evaluation
pub fn evaluate_board(b: &GBoard) -> i32 {
    let winner: i32 = g_over(b);

    if winner != 0 {
        return 10 * winner;
    }

    b.iter().sum()
}

// Minimax algorithm to find best move
pub fn minimax(b: &GBoard, depth: usize, is_max: i32, b_mv: &mut GMove, max_depth: bool) -> i32 {
    let winner = g_over(b);
    if depth == 0 || winner != 0 {
        return evaluate_board(b);
    }

    let mut minmax_score = -100 * is_max;
    let p_mvs = possible(b, is_max);
    for mv in p_mvs {
        let mut b_t = b.clone();
        play(&mut b_t, mv, is_max);
        let ev: i32 = minimax(b, depth - 1, -is_max, b_mv, false);

        if is_max == 1 && ev > minmax_score || is_max == -1 && ev < minmax_score {
            minmax_score = ev;
            if max_depth {
                *b_mv = mv;
            }
        }
    }
    minmax_score
}
