use crate::parser;
use crate::repr;

use parser::Instance;
use rand::RngExt;
use rand::rngs::StdRng;
use repr::Genome;

pub fn edge_recombination(
    parent1: &Genome,
    parent2: &Genome,
    instance: &Instance,
    rng: &mut StdRng,
) -> Genome {
    let n = parent1.sequence.len();
    let mut sequence = Vec::with_capacity(n);

    if n > 0 {
        let mut adjacency = vec![Vec::<usize>::new(); n];

        for parent in [&parent1.sequence, &parent2.sequence] {
            for i in 0..n {
                let node = parent[i];
                if node >= n {
                    continue;
                }

                let left = parent[(i + n - 1) % n];
                let right = parent[(i + 1) % n];

                if left < n && !adjacency[node].contains(&left) {
                    adjacency[node].push(left);
                }
                if right < n && !adjacency[node].contains(&right) {
                    adjacency[node].push(right);
                }
            }
        }

        let mut remaining = vec![true; n];
        let mut current = if rng.random_bool(0.5) {
            parent1.sequence[0]
        } else {
            parent2.sequence[0]
        };
        if current >= n {
            current = 0;
        }

        while sequence.len() < n {
            if !remaining[current] {
                current = (0..n).find(|&idx| remaining[idx]).unwrap();
            }

            sequence.push(current);
            remaining[current] = false;

            for edges in adjacency.iter_mut() {
                edges.retain(|&neighbor| neighbor != current);
            }

            if sequence.len() == n {
                break;
            }

            let candidates: Vec<usize> = adjacency[current]
                .iter()
                .copied()
                .filter(|&neighbor| remaining[neighbor])
                .collect();

            if !candidates.is_empty() {
                let mut min_degree = usize::MAX;
                let mut best = Vec::new();

                for candidate in candidates {
                    let degree = adjacency[candidate]
                        .iter()
                        .filter(|&&neighbor| remaining[neighbor])
                        .count();

                    if degree < min_degree {
                        min_degree = degree;
                        best.clear();
                        best.push(candidate);
                    } else if degree == min_degree {
                        best.push(candidate);
                    }
                }

                current = best[rng.random_range(0..best.len())];
            } else {
                let unvisited: Vec<usize> = (0..n).filter(|&idx| remaining[idx]).collect();
                current = unvisited[rng.random_range(0..unvisited.len())];
            }
        }
    }

    let lengths = if rng.random_bool(0.5) {
        parent1.lengths.clone()
    } else {
        parent2.lengths.clone()
    };

    let mut child = Genome::new(sequence, lengths);
    child.calculate_fitness(&instance);
    child
}
