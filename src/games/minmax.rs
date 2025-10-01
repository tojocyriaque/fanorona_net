use crate::games::fanorona::*;

// Heuristic evaluation
pub fn evaluate_board(b: &FanoronaBoard) -> i32 {
    let winner: i32 = g_over(b);

    if winner != 0 {
        return 10 * winner;
    }
    let material_score: i32 = b.iter().sum(); 
    let mobility_score = possible(b, 1).len() as i32 - possible(b, -1).len() as i32; 
    material_score * 10 + mobility_score
}

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
