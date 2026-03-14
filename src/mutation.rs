use crate::Genome;
use rand::RngExt;
use rand::rngs::StdRng;

fn swap_mutation(genome: &mut Genome, rng: &mut StdRng) {
    let i = rng.random_range(0..genome.sequence.len());
    let j = rng.random_range(0..genome.sequence.len());
    genome.sequence.swap(i, j);
}

fn inversion_mutation(genome: &mut Genome, rng: &mut StdRng) {
    if genome.sequence.len() > 1 {
        let start = rng.random_range(0..genome.sequence.len());
        let end = rng.random_range(start..genome.sequence.len());
        genome.sequence[start..=end].reverse();
    }
}

#[allow(dead_code)]
fn route_inversion_mutation(genome: &mut Genome, rng: &mut StdRng) {
    // Find routes with at least than two elements
    let eligible_routes: Vec<usize> = genome
        .lengths
        .iter()
        .enumerate()
        .filter(|&(_, &len)| len > 1)
        .map(|(i, _)| i)
        .collect();

    if eligible_routes.is_empty() {
        return;
    }

    let route_idx = eligible_routes[rng.random_range(0..eligible_routes.len())];

    let route_start = genome.lengths[..route_idx].iter().sum::<usize>();
    let route_len = genome.lengths[route_idx];
    let route_end = route_start + route_len;

    let idx_a = rng.random_range(route_start..route_end);
    let idx_b = rng.random_range(idx_a..route_end);

    if idx_a != idx_b {
        genome.sequence[idx_a..=idx_b].reverse();
    }
}

fn relocation_mutation(genome: &mut Genome, rng: &mut StdRng) {
    let route_count = genome.lengths.len();
    if route_count < 2 {
        return;
    }

    let non_empty_routes = genome.get_non_empty_routes();
    if non_empty_routes.is_empty() {
        return;
    }

    let from_route = non_empty_routes[rng.random_range(0..non_empty_routes.len())];
    let mut to_route = rng.random_range(0..route_count);
    while to_route == from_route {
        to_route = rng.random_range(0..route_count);
    }

    let from_start = genome.lengths[..from_route].iter().sum::<usize>();
    let from_end = from_start + genome.lengths[from_route];
    if from_end == from_start {
        return;
    }
    let from_idx = rng.random_range(from_start..from_end);

    let to_start = genome.lengths[..to_route].iter().sum::<usize>();
    let to_end = to_start + genome.lengths[to_route];
    let insert_pos = rng.random_range(to_start..=to_end);

    let moved = genome.sequence.remove(from_idx);
    let adjusted_insert = if insert_pos > from_idx {
        insert_pos - 1
    } else {
        insert_pos
    };
    genome.sequence.insert(adjusted_insert, moved);

    genome.lengths[from_route] -= 1;
    genome.lengths[to_route] += 1;
}

fn route_length_mutation(genome: &mut Genome, rng: &mut StdRng) {
    let route_count = genome.lengths.len();
    if route_count < 2 {
        return;
    }

    let non_empty_routes = genome.get_non_empty_routes();
    if non_empty_routes.is_empty() {
        return;
    }

    let from_route = non_empty_routes[rng.random_range(0..non_empty_routes.len())];
    let mut to_route = rng.random_range(0..route_count);
    while to_route == from_route {
        to_route = rng.random_range(0..route_count);
    }

    genome.lengths[from_route] -= 1;
    genome.lengths[to_route] += 1;
}

pub fn random_mutate(genome: &mut Genome, moves: usize, mutation_chance: f64, rng: &mut StdRng) {
    if genome.sequence.is_empty() || genome.lengths.is_empty() {
        return;
    }

    if rng.random_bool(mutation_chance) {
        for _ in 0..moves {
            match rng.random_range(0..4) {
                0 => swap_mutation(genome, rng),
                1 => inversion_mutation(genome, rng),
                2 => relocation_mutation(genome, rng),
                _ => route_length_mutation(genome, rng),
            }
        }
    }
}
