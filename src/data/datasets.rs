// ======================= GENERATING DATASETS ===========================
#[allow(dead_code, unused)]
use itertools::*;
use rand::{seq::SliceRandom, thread_rng};
use std::{
    collections::HashMap,
    path::Path,
    fs::{File, create_dir_all},
    io::{BufRead, BufReader, Write},
    usize,
};

// data loading
use crate::games::fanorona::*;
use crate::{data::loads::load_positions, games::minmax::*}; // game implementations

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
pub fn generate_dataset(depth: usize, output_filename: &str) {
    let mut output_file = File::create(output_filename).unwrap();
    // Étape 1 : Generate positon for 3 zeros, 3 positive, 3 negative
    let positions: Vec<usize> = (0..9).collect();
    for zero_pos in positions.into_iter().combinations(3) {
        // zero_pos is now  Vec<usize>
        let remaining: Vec<usize> = (0..9).filter(|&p| !zero_pos.contains(&p)).collect();

        for pos_pos in remaining.clone().into_iter().combinations(3) {
            // pos_pos is Vec<usize>
            let neg_pos: Vec<usize> = remaining
                .iter()
                .filter(|&&p| !pos_pos.contains(&p))
                .copied()
                .collect();

            // Step 2 : Generate values for the positive player (1 ou 2)
            let pos_iter = (0..3).map(|_| vec![1i32, 2]);
            for pos_vals in pos_iter.multi_cartesian_product() {
                // Step 3 : Generate values for the negative player (-1 ou -2)
                let neg_iter = (0..3).map(|_| vec![-1i32, -2]);
                for neg_vals in neg_iter.multi_cartesian_product() {
                    // Create the sequence of 9 numbers
                    let mut combination = vec![0i32; 9];
                    // Place the zeros
                    for &pos in &zero_pos {
                        combination[pos] = 0;
                    }
                    // Place the positive
                    for (&pos, &val) in pos_pos.iter().zip(pos_vals.iter()) {
                        combination[pos] = val;
                    }
                    // Place the negative
                    for (&pos, &val) in neg_pos.iter().zip(neg_vals.iter()) {
                        combination[pos] = val;
                    }

                    // Skip the calculation ended games
                    if g_over(&combination) != 0 {
                        continue;
                    }

                    // check if the position is valid for each player
                    let v1 = valid_pos(combination.to_vec(), 1);
                    let v2 = valid_pos(combination.to_vec(), -1);
                    let mut b_mv: GMove = (0, 0);

                    // valid if player one is the current player
                    if v1 {
                        minimax(&combination, depth, 1, &mut b_mv, true);
                        let line = format!(
                            "{} 1 {} {}",
                            combination.clone().into_iter().join(" "),
                            b_mv.0,
                            b_mv.1
                        );
                        writeln!(output_file, "{}", line).unwrap();
                        println!("{}", line);
                    }

                    // valid if the player 2 is the current player
                    if v2 {
                        minimax(&combination, depth, -1, &mut b_mv, true);
                        let line = format!(
                            "{} 2 {} {}",
                            combination.into_iter().join(" "),
                            b_mv.0,
                            b_mv.1
                        );
                        writeln!(output_file, "{}", line).unwrap();
                        println!("{}", line);                        
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

// ======================= BALANCIND DATASETS ============================

#[allow(unused)]
pub fn inspect_dataset(filename: &str) {
    let data: Vec<Vec<i32>> = load_positions(filename);
    let d_count = vec![0; 9];
    let mut a_count = vec![0; 9];

    let mut buckets: Vec<usize> = vec![0; 81]; // 9x9 = 81 combinations

    for pos in &data {
        if pos.len() < 12 {
            continue;
        }
        let d_star = pos[10] as usize;
        let a_star = pos[11] as usize;

        buckets[d_star * 9 + a_star] += 1;
    }

    println!("{:?}", buckets);
}

#[allow(unused)]
pub fn balance_dataset_uniform(
    input_filename: &str,
    output_filename: &str,
    target_per_class: usize,
) {
    let file = File::open(input_filename).unwrap();
    let reader = BufReader::new(file);

    // Grouping by (d_star, a_star)
    let mut buckets: Vec<Vec<String>> = vec![Vec::new(); 81]; // 9x9 = 81 combinations

    // Fill the buckets
    for line in reader.lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split(" ").collect();
        if parts.len() < 12 {
            continue;
        }

        let d_star: usize = parts[10].parse().unwrap_or(9);
        let a_star: usize = parts[11].parse().unwrap_or(9);

        if d_star < 9 && a_star < 9 {
            let idx = d_star * 9 + a_star; // Index 0..80
            buckets[idx].push(line);
        }
    }

    // shuffle bucjers
    let mut rng = thread_rng();
    for bucket in &mut buckets {
        bucket.shuffle(&mut rng);
    }

    // Écrire le dataset équilibré
    let mut output_file = File::create(output_filename).unwrap();
    for (_, bucket) in buckets.iter().enumerate() {
        let take = bucket.len().min(target_per_class); // ← Limite each pair
        for line in bucket.iter().take(take) {
            writeln!(output_file, "{}", line).unwrap();
        }
    }
}

#[allow(unused)]
pub fn split_dataset(
    input_filename: &str,
    output_dir: &str,
    train_ratio: f64,
    val_ratio: f64,
) {
    // Read the input file
    let file = File::open(input_filename).expect("Cannot open input file");
    let reader = BufReader::new(file);
    let mut lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

    // Shuffle the lines randomly
    let mut rng = thread_rng();
    lines.shuffle(&mut rng);

    // Compute the number of lines for each split
    let total = lines.len();
    let train_count = (total as f64 * train_ratio).round() as usize;
    let val_count = (total as f64 * val_ratio).round() as usize;
    let test_count = total - train_count - val_count;

    // Create the output directory if it does not exist
    create_dir_all(output_dir).expect("Cannot create output directory");

    // Build paths for the output files
    let train_path = Path::new(output_dir).join("training.txt");
    let val_path = Path::new(output_dir).join("validation.txt");
    let test_path = Path::new(output_dir).join("test.txt");

    // Write the datasets to their respective files
    // for (path, slice) in [
    //     (train_path, &lines[0..train_count]),
    //     (val_path, &lines[train_count..train_count + val_count]),
    //     (test_path, &lines[train_count + val_count..]),
    // ] {
    //     let mut f = File::create(path).expect("Cannot create output file");
    //     for line in slice {
    //         writeln!(f, "{}", line).unwrap();
    //     }
    // }

    // Copy all into those 3 files
    for path in [train_path, val_path, test_path] {
        let mut f = File::create(path).expect("Cannot create output file");
        for line in &lines {
            writeln!(f, "{}", line).unwrap();
        }
    }


    println!(
        "Dataset split: {} train, {} validation, {} test",
        train_count, val_count, test_count
    );
}

#[allow(unused)]
pub fn shuffle_dataset(filename: &str) {
    let mut positions = load_positions(filename);
    let mut shuffler = rand::thread_rng();
    positions.shuffle(&mut shuffler);

    for pos in positions {
        println!("{}", pos.into_iter().join(" "))
    }
}
