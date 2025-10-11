use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::collections::HashMap;
use rand::seq::SliceRandom;

type GMove = (usize, usize);


pub fn split_dataset(input: &str, train_out: &str, val_out: &str, ratio: f64) {
    let file = File::open(input).unwrap();
    let reader = BufReader::new(file);
    let mut lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();
    let mut rng = rand::thread_rng();
    lines.shuffle(&mut rng);

    let split_idx = (lines.len() as f64 * ratio) as usize;
    let (train, val) = lines.split_at(split_idx);

    let mut train_file = File::create(train_out).unwrap();
    let mut val_file = File::create(val_out).unwrap();

    for l in train {
        writeln!(train_file, "{}", l).unwrap();
    }
    for l in val {
        writeln!(val_file, "{}", l).unwrap();
    }
}

/// Transforme le dataset : remplace (start,end) par l'index d'action
pub fn transform_dataset(input_path: &str, output_path: &str) {
    let infile = File::open(input_path).expect("Impossible d'ouvrir le fichier d'entrée");
    let reader = BufReader::new(infile);
    let mut outfile = File::create(output_path).expect("Impossible de créer le fichier de sortie");

    let actions = all_possible_actions();
    let mut i = 0;
    let mut invalid_count = 0;

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                eprintln!("⚠️ Erreur lecture ligne: {}", e);
                continue;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let mut values: Vec<i32> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<i32>().ok())
            .collect();

        if values.len() < 12 {
            eprintln!("⚠️ Ligne ignorée (trop courte) : {}", line);
            continue;
        }

        let start = values[values.len() - 2] as usize;
        let end = values[values.len() - 1] as usize;

        i += 1;
        // Trouver l’index de l'action
        if let Some(idx) = actions.iter().position(|&(s, e)| s == start && e == end) {
            // values.truncate(values.len() - 2);
            // values.push(idx as i32);

            let line_out = values.iter()
                .map(|v| v.to_string())
                .collect::<Vec<String>>()
                .join(" ");
            writeln!(outfile, "{}", line_out).expect("Impossible d'écrire dans le fichier");

            println!("Ligne {} → index action: {}", i, idx);
        } else {
            eprintln!("⚠️ Coup invalide ({}, {}) ignoré", start, end);
            invalid_count += 1;
            continue;
        }

    }

    println!("Transformation terminée !");
    println!("Nombre de coups invalides ignorés : {}", invalid_count);
}

/// Graphe des voisins sur le plateau 3x3
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

/// Toutes les actions possibles du plateau (stable et triée)
pub fn all_possible_actions() -> Vec<GMove> {
    let neigh = neighbours();
    let mut actions = vec![];
    let mut starts: Vec<usize> = neigh.keys().copied().collect();
    starts.sort();

    for s in starts {
        let mut ends = neigh.get(&s).unwrap().clone();
        ends.sort();
        for e in ends {
            actions.push((s, e));
        }
    }
    // println!("{:?}", actions);
    actions
}
