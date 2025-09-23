// =============== GAME LOGIC HERE =====================

use std::{collections::HashMap, usize};

pub type GBoard = Vec<i32>;
pub type GMove = (usize, usize);

fn swap_minmax(min: &mut i32, max: &mut i32, val: &mut i32) {
    *max = *max.max(val);
    *min = *min.min(val);
}

pub fn g_over(b: &GBoard) -> i32 {
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
    // println!("VERTICAL: {max_sum} {min_sum}");

    //     DIAGONAL
    let mut sd1 = b[0] + b[4] + b[8];
    let mut sd2 = b[2] + b[4] + b[6];

    swap_minmax(&mut min_sum, &mut max_sum, &mut sd1);
    swap_minmax(&mut min_sum, &mut max_sum, &mut sd2);
    // println!("DIAGONAL: {max_sum} {min_sum}");

    if max_sum == 6 {
        1
    } else if min_sum == -6 {
        -1
    } else {
        0
    }
}

pub fn play(b: &mut GBoard, (s, e): GMove, pl: i32) -> bool {
    let winner: i32 = g_over(b);
    if winner != 0 {
        return false;
    }

    let moves: Vec<GMove> = possible(b, pl);
    if moves.contains(&(s, e)) {
        let st_v = b[s];
        let end_v = b[e];

        if end_v == 0 && pl * st_v > 0 && winner == 0 {
            b[e] = if st_v.abs() == 1 { st_v } else { st_v * 2 };
            b[s] = 0;
            return true;
        }
    }
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

pub fn possible(b: &GBoard, pl: i32) -> Vec<GMove> {
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
pub fn show_board(board: GBoard) {
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
