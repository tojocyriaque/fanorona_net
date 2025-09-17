#[allow(dead_code, unused)]
use itertools::Itertools;
use std::{collections::HashMap, usize};

use crate::game::{GMove, g_over, minimax};

#[allow(dead_code, unused)]
const DEPTH: usize = 5;

#[allow(dead_code, unused)]
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

#[allow(dead_code, unused)]
pub fn generate_combinations() {
    // Étape 1 : Générer les positions pour 3 zéros, 3 positifs, 3 négatifs
    let positions: Vec<usize> = (0..9).collect();
    for zero_pos in positions.into_iter().combinations(3) {
        // zero_pos est maintenant Vec<usize>
        let remaining: Vec<usize> = (0..9).filter(|&p| !zero_pos.contains(&p)).collect();

        for pos_pos in remaining.clone().into_iter().combinations(3) {
            // pos_pos est Vec<usize>
            let neg_pos: Vec<usize> = remaining
                .iter()
                .filter(|&&p| !pos_pos.contains(&p))
                .copied()
                .collect();

            // Étape 2 : Générer les valeurs pour les positifs (1 ou 2)
            let pos_iter = (0..3).map(|_| vec![1i32, 2]);
            for pos_vals in pos_iter.multi_cartesian_product() {
                // Étape 3 : Générer les valeurs pour les négatifs (-1 ou -2)
                let neg_iter = (0..3).map(|_| vec![-1i32, -2]);
                for neg_vals in neg_iter.multi_cartesian_product() {
                    // Créer la séquence de 9 chiffres
                    let mut combination = vec![0i32; 9];
                    // Placer les zéros
                    for &pos in &zero_pos {
                        combination[pos] = 0;
                    }
                    // Placer les positifs
                    for (&pos, &val) in pos_pos.iter().zip(pos_vals.iter()) {
                        combination[pos] = val;
                    }
                    // Placer les négatifs
                    for (&pos, &val) in neg_pos.iter().zip(neg_vals.iter()) {
                        combination[pos] = val;
                    }

                    // Skip the calculation ended games
                    if g_over(&combination) != 0 {
                        continue;
                    }

                    let v1 = valid_pos(combination.to_vec(), 1);
                    let v2 = valid_pos(combination.to_vec(), -1);
                    let mut b_mv: GMove = (0, 0);

                    if v1 {
                        minimax(&combination, DEPTH, 1, &mut b_mv, true);
                        println!(
                            "{} 1 {} {}",
                            combination.clone().into_iter().join(" "),
                            b_mv.0,
                            b_mv.1
                        );
                    }
                    if v2 {
                        minimax(&combination, DEPTH, -1, &mut b_mv, true);
                        println!(
                            "{} 2 {} {}",
                            combination.into_iter().join(" "),
                            b_mv.0,
                            b_mv.1
                        );
                    }
                }
            }
        }
    }
}

#[allow(dead_code, unused)]
fn valid_pos(position: Vec<i32>, pl: i32) -> bool {
    let nei: HashMap<usize, Vec<usize>> = neighbours();

    let opp: Vec<usize> = (0..9).filter(|&u| pl * position[u] == -2).collect();
    let ally: Vec<usize> = (0..9).filter(|&u| pl * position[u] > 0).collect();

    // See if the previous player had a valid position
    let mut opp_had_played: bool = false;
    for i in opp {
        let nb_has_0 = nei.get(&i).unwrap().iter().any(|&u| u == 0);
        if nb_has_0 {
            opp_had_played = true;
            break;
        }
    }

    let mut pl_has_valid: bool = false;
    for i in ally {
        let nb_has_0 = nei.get(&i).unwrap().iter().any(|&u| u == 0);
        if nb_has_0 {
            pl_has_valid = true;
            break;
        }
    }

    opp_had_played && pl_has_valid
}
