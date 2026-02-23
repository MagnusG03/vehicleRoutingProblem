mod read_json;

use plotters::prelude::*;
use rand::RngExt;
use rand::rng;
use rand::seq::SliceRandom;
use read_json::{get_travel_time, read_json};
use std::error::Error;

//TODO: Create islands that rejoin after like 2000 generations into one island and also stop populate function from being deterministic.
//TODO: Also test simpler GA types but still with local search.
#[derive(Clone)]
struct Genome {
    sequence: Vec<usize>,
    lengths: Vec<usize>,
    fitness: f64,
    shared_fitness: f64,
    travel_time: f64,
    feasible: bool,
}

impl Genome {
    fn new(sequence: Vec<usize>, lengths: Vec<usize>) -> Self {
        Genome {
            sequence,
            lengths,
            fitness: 0.0,
            shared_fitness: 0.0,
            travel_time: 0.0,
            feasible: true,
        }
    }

    fn calculate_fitness(&mut self, instance: &read_json::Instance) {
        // Check if all patients are visited within their time windows, check nurse load, calculate total travel time, and check if all patients have been visited

        const STRUCTURE_PENALTY: f64 = 1000.0;
        const TW_BASE_PENALTY: f64 = 50.0;
        const TW_LINEAR_PENALTY: f64 = 5.0;
        const TW_QUADRATIC_PENALTY: f64 = 0.05;
        const CAPACITY_LINEAR_PENALTY: f64 = 100.0;
        const RETURN_LINEAR_PENALTY: f64 = 20.0;
        const RETURN_QUADRATIC_PENALTY: f64 = 0.1;
        const INFEASIBLE_GAP: f64 = 10000.0;

        let n_patients = instance.patients.len();
        let mut total_travel_time = 0.0;
        let mut nurse = 0;
        let mut seen = vec![false; n_patients];
        let mut penalty = 0.0;
        self.feasible = true;

        if self.sequence.len() != n_patients {
            penalty += STRUCTURE_PENALTY;
            self.feasible = false;
        }

        if self.lengths.len() != instance.nbr_nurses {
            penalty += STRUCTURE_PENALTY;
            self.feasible = false;
        }

        if self.lengths.iter().sum::<usize>() != n_patients {
            penalty += STRUCTURE_PENALTY;
            self.feasible = false;
        }

        while nurse < instance.nbr_nurses {
            let mut total_time = 0.0;
            let mut load = 0;
            let mut current_location = 0;
            let mut total_nurse_travel_time = 0.0;

            let mut gene_index: usize = self.lengths[..nurse].iter().sum::<usize>();
            let gene_end_index = gene_index + self.lengths[nurse];
            while gene_index < gene_end_index {
                let patient = self.sequence[gene_index];
                if patient >= n_patients {
                    penalty += STRUCTURE_PENALTY;
                    self.feasible = false;
                    gene_index += 1;
                    continue;
                }
                let patient_node = patient + 1; // Because 0 is the depot

                // Check if patient has already been seen
                if seen[patient] {
                    penalty += STRUCTURE_PENALTY;
                    self.feasible = false;
                }
                seen[patient] = true;

                let patient_info: &read_json::Patient = &instance.patients[patient];

                let travel_time =
                    get_travel_time(&instance.travel_times, current_location, patient_node);

                // Update time, and location
                total_nurse_travel_time += travel_time;
                total_time += travel_time;
                current_location = patient_node;

                // Wait until the start time if we arrive early.
                if total_time < patient_info.start_time as f64 {
                    total_time = patient_info.start_time as f64;
                }

                // If we finish care after the patient end time.
                total_time += patient_info.care_time as f64;
                let lateness = (total_time - patient_info.end_time as f64).max(0.0);
                if lateness > 0.0 {
                    penalty += TW_BASE_PENALTY
                        + TW_LINEAR_PENALTY * lateness
                        + TW_QUADRATIC_PENALTY * lateness * lateness;
                    self.feasible = false;
                }

                // If we exceed nurse capacity.
                load += patient_info.demand;
                let overload = (load as f64 - instance.capacity_nurse as f64).max(0.0);
                if overload > 0.0 {
                    penalty += CAPACITY_LINEAR_PENALTY * overload;
                    self.feasible = false;
                }

                gene_index += 1;
            }

            total_travel_time += total_nurse_travel_time;

            // If we cant make it back to the depot on time.
            let back_to_depot = get_travel_time(&instance.travel_times, current_location, 0);
            let return_lateness =
                (total_time + back_to_depot - instance.depot.return_time as f64).max(0.0);
            if return_lateness > 0.0 {
                penalty += RETURN_LINEAR_PENALTY * return_lateness
                    + RETURN_QUADRATIC_PENALTY * return_lateness * return_lateness;
                self.feasible = false;
            }

            total_travel_time += back_to_depot;

            nurse += 1;
        }

        // Check if all patients have been visited
        if !seen.iter().all(|&b| b) {
            penalty += STRUCTURE_PENALTY;
            self.feasible = false;
        }

        // Calculate and store fitness and travel time.
        let feasibility_gap = if self.feasible { 0.0 } else { INFEASIBLE_GAP };
        self.fitness = 1.0 / (1.0 + total_travel_time + feasibility_gap + penalty);
        self.travel_time = total_travel_time;
    }
}

fn calculate_shared_fitnesses(population: &mut Vec<Genome>) {
    const THRESHOLD: f64 = 0.2;
    const ALPHA: f64 = 1.0;

    for i in 0..population.len() {
        let mut denominator = 0.0;
        for l in 0..population.len() {
            let distance = genome_difference(&population[i], &population[l]);
            if distance <= THRESHOLD {
                denominator += 1.0 - (distance / THRESHOLD).powf(ALPHA);
            }
        }

        population[i].shared_fitness = population[i].fitness / denominator;
    }
}

fn shared_fitness_with_reference(genome: &Genome, reference: &[Genome]) -> f64 {
    const THRESHOLD: f64 = 0.2;
    const ALPHA: f64 = 1.0;

    let mut denominator = 1.0; // Include the genome itself.
    for other in reference {
        let distance = genome_difference(genome, other);
        if distance <= THRESHOLD {
            denominator += 1.0 - (distance / THRESHOLD).powf(ALPHA);
        }
    }
    genome.fitness / denominator
}

fn calculate_shared_fitness(genome: &mut Genome, population: &[Genome]) {
    genome.shared_fitness = shared_fitness_with_reference(genome, population);
}

fn calculate_entropy(population: &[Genome]) -> f64 {
    if population.is_empty() {
        return 0.0;
    }

    let sequence_len = population[0].sequence.len();
    if sequence_len == 0 {
        return 0.0;
    }

    let pop_size = population.len() as f64;
    let normalizer = (sequence_len as f64).ln();
    let mut total_entropy = 0.0;

    for pos in 0..sequence_len {
        let mut counts = vec![0usize; sequence_len];
        for genome in population {
            let patient = genome.sequence[pos];
            if patient < sequence_len {
                counts[patient] += 1;
            }
        }

        let mut entropy = 0.0;
        for count in counts {
            if count > 0 {
                let p = count as f64 / pop_size;
                entropy -= p * p.ln();
            }
        }

        if normalizer > 0.0 {
            entropy /= normalizer;
        }

        total_entropy += entropy;
    }

    total_entropy / sequence_len as f64
}

fn plot_metrics(
    fitness_history: &[f64],
    entropy_history: &[f64],
    feasible_travel_time_history: &[Option<f64>],
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    if fitness_history.is_empty()
        || entropy_history.is_empty()
        || feasible_travel_time_history.is_empty()
    {
        return Ok(());
    }

    let root = BitMapBackend::new(output_path, (1000, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((3, 1));

    let fitness_min = fitness_history
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let fitness_max = fitness_history
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let fitness_padding = if (fitness_max - fitness_min).abs() < f64::EPSILON {
        (fitness_max.abs() * 0.01).max(1e-12)
    } else {
        0.0
    };

    let mut fitness_chart = ChartBuilder::on(&areas[0])
        .caption("Best Fitness by Generation", ("sans-serif", 24))
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(55)
        .build_cartesian_2d(
            0usize..fitness_history.len(),
            (fitness_min - fitness_padding)..(fitness_max + fitness_padding),
        )?;

    fitness_chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Best Fitness")
        .draw()?;

    fitness_chart.draw_series(LineSeries::new(
        fitness_history.iter().enumerate().map(|(i, v)| (i, *v)),
        &BLUE,
    ))?;

    let mut entropy_chart = ChartBuilder::on(&areas[1])
        .caption("Population Entropy by Generation", ("sans-serif", 24))
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(55)
        .build_cartesian_2d(0usize..entropy_history.len(), 0f64..1f64)?;

    entropy_chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Entropy (normalized)")
        .draw()?;

    entropy_chart.draw_series(LineSeries::new(
        entropy_history.iter().enumerate().map(|(i, v)| (i, *v)),
        &RED,
    ))?;

    let feasible_travel_points: Vec<(usize, f64)> = feasible_travel_time_history
        .iter()
        .enumerate()
        .filter_map(|(i, travel_time)| travel_time.map(|travel_time| (i, travel_time)))
        .collect();

    let travel_min = feasible_travel_points
        .iter()
        .map(|(_, travel_time)| *travel_time)
        .fold(f64::INFINITY, f64::min);
    let travel_max = feasible_travel_points
        .iter()
        .map(|(_, travel_time)| *travel_time)
        .fold(f64::NEG_INFINITY, f64::max);
    let (travel_min, travel_max) = if feasible_travel_points.is_empty() {
        (0.0, 1.0)
    } else {
        (travel_min, travel_max)
    };
    let travel_padding = if (travel_max - travel_min).abs() < f64::EPSILON {
        (travel_max.abs() * 0.01).max(1e-12)
    } else {
        0.0
    };

    let mut travel_chart = ChartBuilder::on(&areas[2])
        .caption(
            "Lowest Feasible Travel Time by Generation",
            ("sans-serif", 24),
        )
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(55)
        .build_cartesian_2d(
            0usize..feasible_travel_time_history.len(),
            (travel_min - travel_padding)..(travel_max + travel_padding),
        )?;

    travel_chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Lowest Feasible Travel Time")
        .draw()?;

    travel_chart.draw_series(LineSeries::new(feasible_travel_points, &GREEN))?;

    root.present()?;
    Ok(())
}

fn greedy_seed_candidate(instance: &read_json::Instance) -> Genome {
    let n_patients = instance.patients.len();
    let n_nurses = instance.nbr_nurses;
    let mut rng = rng();

    let urgency_weight = rng.random_range(0.8..1.6);
    let width_weight = rng.random_range(0.3..1.2);
    let start_weight = rng.random_range(0.1..0.7);
    let demand_weight = rng.random_range(0.0..0.5);

    let lateness_weight = rng.random_range(9000.0..15000.0);
    let overload_weight = rng.random_range(9000.0..15000.0);
    let return_weight = rng.random_range(4000.0..9000.0);
    let travel_weight = rng.random_range(0.8..2.0);
    let balance_weight = rng.random_range(0.0..3.0);
    let score_noise = rng.random_range(0.2..2.0);

    let tie_noise: Vec<f64> = (0..n_patients)
        .map(|_| rng.random_range(0.0..40.0))
        .collect();

    let mut order: Vec<usize> = (0..n_patients).collect();
    order.sort_by(|&a, &b| {
        let pa = &instance.patients[a];
        let pb = &instance.patients[b];

        let width_a = pa.end_time.saturating_sub(pa.start_time) as f64;
        let width_b = pb.end_time.saturating_sub(pb.start_time) as f64;

        let key_a = urgency_weight * pa.end_time as f64
            + width_weight * width_a
            + start_weight * pa.start_time as f64
            + demand_weight * pa.demand as f64
            + tie_noise[a];
        let key_b = urgency_weight * pb.end_time as f64
            + width_weight * width_b
            + start_weight * pb.start_time as f64
            + demand_weight * pb.demand as f64
            + tie_noise[b];
        key_a.total_cmp(&key_b)
    });

    if n_patients > 1 {
        let max_band_size = n_patients.min(12);
        let band_size = if max_band_size > 2 {
            rng.random_range(2..(max_band_size + 1))
        } else {
            max_band_size
        };
        for chunk in order.chunks_mut(band_size.max(1)) {
            chunk.shuffle(&mut rng);
        }
    }

    let mut routes: Vec<Vec<usize>> = vec![Vec::new(); n_nurses];
    let mut route_time = vec![0.0f64; n_nurses];
    let mut route_load = vec![0usize; n_nurses];
    let mut route_loc = vec![0usize; n_nurses];
    let mut assigned = 0usize;

    for patient in order {
        let p = &instance.patients[patient];
        let patient_node = patient + 1;
        let average_route_size = (assigned as f64 + 1.0) / n_nurses as f64;
        let mut choices: Vec<(f64, usize, f64, usize, usize)> = Vec::with_capacity(n_nurses);

        for nurse in 0..n_nurses {
            let travel = get_travel_time(&instance.travel_times, route_loc[nurse], patient_node);
            let mut t = route_time[nurse] + travel;
            if t < p.start_time as f64 {
                t = p.start_time as f64;
            }
            t += p.care_time as f64;

            let lateness = (t - p.end_time as f64).max(0.0);
            let overload = (route_load[nurse] + p.demand).saturating_sub(instance.capacity_nurse);
            let return_lateness = (t + get_travel_time(&instance.travel_times, patient_node, 0)
                - instance.depot.return_time as f64)
                .max(0.0);
            let projected_route_size = routes[nurse].len() as f64 + 1.0;
            let imbalance = (projected_route_size - average_route_size).abs();

            let violation_cost = lateness_weight * lateness
                + overload_weight * overload as f64
                + return_weight * return_lateness;
            let score = violation_cost
                + travel_weight * travel
                + balance_weight * imbalance
                + rng.random_range(0.0..score_noise);

            choices.push((score, nurse, t, route_load[nurse] + p.demand, patient_node));
        }

        choices.sort_by(|a, b| a.0.total_cmp(&b.0));
        let max_rcl_size = choices.len().min(4);
        let rcl_size = if max_rcl_size > 1 {
            rng.random_range(1..(max_rcl_size + 1))
        } else {
            1
        };
        let (_, selected_nurse, selected_time, selected_load, selected_loc) =
            choices[rng.random_range(0..rcl_size)];

        routes[selected_nurse].push(patient);
        route_time[selected_nurse] = selected_time;
        route_load[selected_nurse] = selected_load;
        route_loc[selected_nurse] = selected_loc;
        assigned += 1;
    }

    let mut nurse_order: Vec<usize> = (0..n_nurses).collect();
    nurse_order.shuffle(&mut rng);

    let mut sequence = Vec::with_capacity(n_patients);
    let mut lengths = vec![0usize; n_nurses];
    for (slot, nurse) in nurse_order.into_iter().enumerate() {
        lengths[slot] = routes[nurse].len();
        sequence.extend(routes[nurse].iter().copied());
    }

    let mut genome = Genome::new(sequence, lengths);
    genome.calculate_fitness(instance);
    genome
}

fn greedy_seed_genome(instance: &read_json::Instance) -> Genome {
    const CONSTRUCTION_ATTEMPTS: usize = 6;
    const LOCAL_SEARCH_REFINEMENTS: usize = 1;
    const LOCAL_SEARCH_MIN_STEPS: usize = 8;
    const LOCAL_SEARCH_MAX_STEPS: usize = 24;
    const ELITE_POOL_SIZE: usize = 5;

    let mut rng = rng();
    let mut candidates = Vec::with_capacity(CONSTRUCTION_ATTEMPTS);

    for _ in 0..CONSTRUCTION_ATTEMPTS {
        candidates.push(greedy_seed_candidate(instance));
    }

    candidates.sort_by(solution_cmp);

    for idx in 0..LOCAL_SEARCH_REFINEMENTS.min(candidates.len()) {
        let steps = rng.random_range(LOCAL_SEARCH_MIN_STEPS..(LOCAL_SEARCH_MAX_STEPS + 1));
        let improved = local_search(candidates[idx].clone(), instance, steps, None);
        candidates[idx] = improved;
    }

    candidates.sort_by(solution_cmp);

    let elite_pool_size = ELITE_POOL_SIZE.min(candidates.len()).max(1);
    let mut total_weight = 0.0;
    for rank in 0..elite_pool_size {
        total_weight += 1.0 / (rank as f64 + 1.0);
    }

    let mut draw = rng.random_range(0.0..total_weight);
    let mut chosen_rank = elite_pool_size - 1;
    for rank in 0..elite_pool_size {
        draw -= 1.0 / (rank as f64 + 1.0);
        if draw <= 0.0 {
            chosen_rank = rank;
            break;
        }
    }

    candidates.swap_remove(chosen_rank)
}

fn random_genome(instance: &read_json::Instance) -> Genome {
    let mut rng = rng();
    let n_patients = instance.patients.len();
    let n_nurses = instance.nbr_nurses;

    let mut sequence: Vec<usize> = (0..n_patients).collect();
    sequence.shuffle(&mut rng);

    let mut lengths = vec![0usize; n_nurses];
    for _ in 0..n_patients {
        let k = rng.random_range(0..n_nurses);
        lengths[k] += 1;
    }

    let mut genome = Genome::new(sequence, lengths);
    genome.calculate_fitness(instance);
    genome
}

fn immigrate(population: &mut Vec<Genome>, instance: &read_json::Instance) {
    const SEED_RATE: f64 = 0.7;
    const IMMIGRATION_SIZE: usize = 100;
    let mut rng = rng();

    for _ in 0..IMMIGRATION_SIZE {
        if rng.random_bool(SEED_RATE) {
            population.push(greedy_seed_genome(instance));
        } else {
            population.push(random_genome(instance));
        }
    }
}

fn populate(population_size: usize, instance: &read_json::Instance) -> Vec<Genome> {
    let n_patients = instance.patients.len();
    let n_nurses = instance.nbr_nurses;

    let mut rng = rng();
    let mut population = Vec::with_capacity(population_size);
    let max_attempts_per_genome = 200;

    for _ in 0..population_size {
        let mut best_any: Option<Genome> = None;
        let mut best_feasible: Option<Genome> = None;

        // Seed from a constructive heuristic so population starts near feasible space.
        let seeded = greedy_seed_genome(instance);
        if best_any.as_ref().is_none_or(|b| seeded.fitness > b.fitness) {
            best_any = Some(seeded.clone());
        }
        if seeded.feasible {
            best_feasible = Some(seeded);
        }

        // Avoid costly random retries when the seeded solution is already feasible.
        if best_feasible.is_none() {
            for _ in 0..max_attempts_per_genome {
                let mut sequence: Vec<usize> = (0..n_patients).collect();
                sequence.shuffle(&mut rng);

                let mut lengths = vec![0usize; n_nurses];
                for _ in 0..n_patients {
                    let k = rng.random_range(0..n_nurses);
                    lengths[k] += 1;
                }

                let mut genome = Genome::new(sequence, lengths);
                genome.calculate_fitness(instance);

                if best_any.as_ref().is_none_or(|b| genome.fitness > b.fitness) {
                    best_any = Some(genome.clone());
                }

                if genome.feasible {
                    if best_feasible
                        .as_ref()
                        .is_none_or(|b| genome.travel_time < b.travel_time)
                    {
                        best_feasible = Some(genome);
                    }
                    break; // stop attempts once we found a feasible one
                }
            }
        }

        population.push(
            best_feasible
                .or(best_any)
                .expect("Failed to generate genome"),
        );
    }

    population
}

fn tournament_selection(
    population: &[Genome],
    population_size: usize,
    use_shared_fitness: bool,
) -> Vec<Genome> {
    let mut rng = rng();
    let mut selected: Vec<Genome> = Vec::with_capacity(population_size);
    while selected.len() < population_size {
        let a = rng.random_range(0..population.len());
        let b = rng.random_range(0..population.len());

        let a_fitness = if use_shared_fitness {
            population[a].shared_fitness
        } else {
            population[a].fitness
        };
        let b_fitness = if use_shared_fitness {
            population[b].shared_fitness
        } else {
            population[b].fitness
        };

        if a_fitness > b_fitness {
            selected.push(population[a].clone());
        } else {
            selected.push(population[b].clone());
        }
    }
    selected
}

fn order_crossover(parent1: &Genome, parent2: &Genome, instance: &read_json::Instance) -> Genome {
    let mut rng = rng();
    let n = parent1.sequence.len();

    let mut sequence = vec![usize::MAX; n];
    if n > 0 {
        let start = rng.random_range(0..n);
        let end = rng.random_range(start..n);

        for i in start..=end {
            sequence[i] = parent1.sequence[i];
        }

        let mut insert_idx = (end + 1) % n;
        for offset in 0..n {
            let candidate = parent2.sequence[(end + 1 + offset) % n];
            if !sequence.contains(&candidate) {
                sequence[insert_idx] = candidate;
                insert_idx = (insert_idx + 1) % n;
            }
        }
    }

    // Keep one full parent length vector to ensure validity.
    let lengths = if rng.random_bool(0.5) {
        parent1.lengths.clone()
    } else {
        parent2.lengths.clone()
    };

    let mut child: Genome = Genome::new(sequence, lengths);
    child.calculate_fitness(instance);
    child
}

fn edge_crossover(parent1: &Genome, parent2: &Genome, instance: &read_json::Instance) -> Genome {
    let mut rng = rng();
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
    child.calculate_fitness(instance);
    child
}

fn swap_mutate(genome: &mut Genome, path_mutation_rate: f64, length_mutation_rate: f64) {
    let mut rng = rng();
    // Swap two patients in the sequence.
    if rng.random_bool(path_mutation_rate) {
        let i = rng.random_range(0..genome.sequence.len());
        let j = rng.random_range(0..genome.sequence.len());
        genome.sequence.swap(i, j);
    }

    // Transfer one patient from one nurse to another to preserve total.
    if rng.random_bool(length_mutation_rate) {
        let from = rng.random_range(0..genome.lengths.len());
        let mut to = rng.random_range(0..genome.lengths.len());
        while to == from {
            to = rng.random_range(0..genome.lengths.len());
        }

        if genome.lengths[from] > 0 {
            genome.lengths[from] -= 1;
            genome.lengths[to] += 1;
        }
    }
}

fn inversion_mutate(genome: &mut Genome, path_mutation_rate: f64, length_mutation_rate: f64) {
    // Invert a subsequence of patients in the sequence.
    let mut rng = rng();
    let start = rng.random_range(0..genome.sequence.len());
    let end = rng.random_range(start..genome.sequence.len());
    if rng.random_bool(path_mutation_rate) {
        genome.sequence[start..=end].reverse();
    }

    // Transfer one patient from one nurse to another to preserve total.
    if rng.random_bool(length_mutation_rate) {
        let from = rng.random_range(0..genome.lengths.len());
        let mut to = rng.random_range(0..genome.lengths.len());
        while to == from {
            to = rng.random_range(0..genome.lengths.len());
        }

        if genome.lengths[from] > 0 {
            genome.lengths[from] -= 1;
            genome.lengths[to] += 1;
        }
    }
}

fn local_search_better(
    candidate: &Genome,
    incumbent: &Genome,
    shared_reference: Option<&[Genome]>,
) -> bool {
    let (candidate_shared, incumbent_shared) = if let Some(reference) = shared_reference {
        (
            shared_fitness_with_reference(candidate, reference),
            shared_fitness_with_reference(incumbent, reference),
        )
    } else {
        // Local-search contexts without a population reference fall back to raw fitness.
        (candidate.fitness, incumbent.fitness)
    };

    match (candidate.feasible, incumbent.feasible) {
        (true, false) => true,
        (false, true) => false,
        (true, true) => {
            if candidate.travel_time < incumbent.travel_time {
                true
            } else if (candidate.travel_time - incumbent.travel_time).abs() < 1e-9 {
                candidate_shared > incumbent_shared
            } else {
                false
            }
        }
        (false, false) => candidate_shared > incumbent_shared,
    }
}

fn solution_cmp(a: &Genome, b: &Genome) -> std::cmp::Ordering {
    match (a.feasible, b.feasible) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (true, true) => a
            .travel_time
            .total_cmp(&b.travel_time)
            .then_with(|| b.fitness.total_cmp(&a.fitness)),
        (false, false) => b.fitness.total_cmp(&a.fitness),
    }
}

fn route_offsets(lengths: &[usize], sequence_len: usize) -> Option<Vec<usize>> {
    let mut offsets = Vec::with_capacity(lengths.len() + 1);
    offsets.push(0);
    let mut acc = 0usize;

    for &len in lengths {
        acc += len;
        offsets.push(acc);
    }

    if acc == sequence_len {
        Some(offsets)
    } else {
        None
    }
}

fn first_improving_two_opt(
    base: &Genome,
    instance: &read_json::Instance,
    shared_reference: Option<&[Genome]>,
) -> Option<Genome> {
    let offsets = route_offsets(&base.lengths, base.sequence.len())?;

    for nurse in 0..base.lengths.len() {
        let start = offsets[nurse];
        let end = offsets[nurse + 1];
        if end - start < 2 {
            continue;
        }

        for i in start..(end - 1) {
            for j in (i + 1)..end {
                let mut candidate = base.clone();
                candidate.sequence[i..=j].reverse();
                candidate.calculate_fitness(instance);
                if local_search_better(&candidate, base, shared_reference) {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

fn first_improving_relocate(
    base: &Genome,
    instance: &read_json::Instance,
    shared_reference: Option<&[Genome]>,
) -> Option<Genome> {
    let offsets = route_offsets(&base.lengths, base.sequence.len())?;
    let n_routes = base.lengths.len();

    for from_route in 0..n_routes {
        let from_start = offsets[from_route];
        let from_end = offsets[from_route + 1];
        if from_end == from_start {
            continue;
        }

        for from_idx in from_start..from_end {
            for to_route in 0..n_routes {
                if to_route == from_route {
                    continue;
                }

                let to_start = offsets[to_route];
                let to_end = offsets[to_route + 1];

                for insert_pos in to_start..=to_end {
                    let mut candidate = base.clone();
                    let moved = candidate.sequence.remove(from_idx);
                    let adjusted_insert = if insert_pos > from_idx {
                        insert_pos - 1
                    } else {
                        insert_pos
                    };
                    candidate.sequence.insert(adjusted_insert, moved);

                    candidate.lengths[from_route] -= 1;
                    candidate.lengths[to_route] += 1;
                    candidate.calculate_fitness(instance);

                    if local_search_better(&candidate, base, shared_reference) {
                        return Some(candidate);
                    }
                }
            }
        }
    }
    None
}

fn first_improving_inter_route_swap(
    base: &Genome,
    instance: &read_json::Instance,
    shared_reference: Option<&[Genome]>,
) -> Option<Genome> {
    let offsets = route_offsets(&base.lengths, base.sequence.len())?;
    let n_routes = base.lengths.len();

    for route_a in 0..n_routes {
        let a_start = offsets[route_a];
        let a_end = offsets[route_a + 1];
        if a_end == a_start {
            continue;
        }

        for route_b in (route_a + 1)..n_routes {
            let b_start = offsets[route_b];
            let b_end = offsets[route_b + 1];
            if b_end == b_start {
                continue;
            }

            for idx_a in a_start..a_end {
                for idx_b in b_start..b_end {
                    let mut candidate = base.clone();
                    candidate.sequence.swap(idx_a, idx_b);
                    candidate.calculate_fitness(instance);

                    if local_search_better(&candidate, base, shared_reference) {
                        return Some(candidate);
                    }
                }
            }
        }
    }

    None
}

fn local_search(
    mut genome: Genome,
    instance: &read_json::Instance,
    max_steps: usize,
    shared_reference: Option<&[Genome]>,
) -> Genome {
    genome.calculate_fitness(instance);

    for _ in 0..max_steps {
        if let Some(next) = first_improving_two_opt(&genome, instance, shared_reference) {
            genome = next;
            continue;
        }

        if let Some(next) = first_improving_relocate(&genome, instance, shared_reference) {
            genome = next;
            continue;
        }

        if let Some(next) = first_improving_inter_route_swap(&genome, instance, shared_reference) {
            genome = next;
            continue;
        }

        break;
    }

    genome
}

fn repopulate(
    population: &mut Vec<Genome>,
    instance: &read_json::Instance,
    path_mutation_rate: f64,
    length_mutation_rate: f64,
    population_size: usize,
) {
    let mut rng = rng();
    let mut new_population = Vec::with_capacity(population_size);

    while new_population.len() < population_size {
        let parent1 = population[rng.random_range(0..population.len())].clone();
        let parent2 = population[rng.random_range(0..population.len())].clone();

        let mut child = order_crossover(&parent1, &parent2, instance);
        swap_mutate(&mut child, path_mutation_rate, length_mutation_rate);
        child.calculate_fitness(instance);
        new_population.push(child);
    }
    *population = new_population;
}

fn deterministic_crowding_repopulation(
    population: &mut Vec<Genome>,
    instance: &read_json::Instance,
    path_mutation_rate: f64,
    length_mutation_rate: f64,
    population_size: usize,
) {
    let mut rng = rng();
    let mut new_population = Vec::with_capacity(population_size);
    calculate_shared_fitnesses(population);

    while new_population.len() < population_size {
        let parent1 = population[rng.random_range(0..population.len())].clone();
        let parent2 = population[rng.random_range(0..population.len())].clone();

        let mut child = order_crossover(&parent1, &parent2, instance);
        swap_mutate(&mut child, path_mutation_rate, length_mutation_rate);
        child.calculate_fitness(instance);
        calculate_shared_fitness(&mut child, population);

        if genome_difference(&child, &parent1) < genome_difference(&child, &parent2) {
            if child.shared_fitness > parent1.shared_fitness {
                new_population.push(child);
            } else {
                new_population.push(parent1);
            }
        } else {
            if child.shared_fitness > parent2.shared_fitness {
                new_population.push(child);
            } else {
                new_population.push(parent2);
            }
        }
    }
    *population = new_population;
}

fn elitism_deterministic_crowding_repopulation(
    population: &mut Vec<Genome>,
    instance: &read_json::Instance,
    path_mutation_rate: f64,
    length_mutation_rate: f64,
    elite_size: usize,
    population_size: usize,
) {
    let mut rng = rng();
    let mut new_population = Vec::with_capacity(population_size);
    calculate_shared_fitnesses(population);

    new_population.extend(get_elites(population, elite_size));

    while new_population.len() < population_size {
        let parent1 = population[rng.random_range(0..population.len())].clone();
        let parent2 = population[rng.random_range(0..population.len())].clone();

        let mut child = order_crossover(&parent1, &parent2, instance);
        swap_mutate(&mut child, path_mutation_rate, length_mutation_rate);
        child.calculate_fitness(instance);
        calculate_shared_fitness(&mut child, population);

        if genome_difference(&child, &parent1) < genome_difference(&child, &parent2) {
            if child.shared_fitness > parent1.shared_fitness {
                new_population.push(child);
            } else {
                new_population.push(parent1);
            }
        } else {
            if child.shared_fitness > parent2.shared_fitness {
                new_population.push(child);
            } else {
                new_population.push(parent2);
            }
        }
    }
    *population = new_population;
}

fn elitism_scaling_probabilistic_crowding_repopulation(
    population: &mut Vec<Genome>,
    instance: &read_json::Instance,
    path_mutation_rate: f64,
    length_mutation_rate: f64,
    elite_size: usize,
    scaling_factor: f64,
    population_size: usize,
) {
    let mut rng = rng();
    let mut new_population = Vec::with_capacity(population_size);

    new_population.extend(get_elites(population, elite_size));

    while new_population.len() < population_size {
        let parent1 = population[rng.random_range(0..population.len())].clone();
        let parent2 = population[rng.random_range(0..population.len())].clone();

        let mut child1 = edge_crossover(&parent1, &parent2, instance);
        inversion_mutate(&mut child1, path_mutation_rate, length_mutation_rate);
        child1.calculate_fitness(instance);

        let mut child2 = edge_crossover(&parent2, &parent1, instance);
        inversion_mutate(&mut child2, path_mutation_rate, length_mutation_rate);
        child2.calculate_fitness(instance);

        if genome_difference(&child1, &parent1) + genome_difference(&child2, &parent2)
            < genome_difference(&child1, &parent2) + genome_difference(&child2, &parent1)
        {
            if child1.fitness > parent1.fitness {
                let child_prob = child1.fitness / (child1.fitness + parent1.fitness * scaling_factor);
                if rng.random_bool(child_prob) {
                    new_population.push(child1);
                } else {
                    new_population.push(parent1);
                }
            } else {
                let child_prob = scaling_factor * child1.fitness
                    / (scaling_factor * child1.fitness + parent1.fitness);
                if rng.random_bool(child_prob) {
                    new_population.push(child1);
                } else {
                    new_population.push(parent1);
                }
            }
            if child2.fitness > parent2.fitness {
                let child_prob = child2.fitness / (child2.fitness + parent2.fitness * scaling_factor);
                if rng.random_bool(child_prob) {
                    new_population.push(child2);
                } else {
                    new_population.push(parent2);
                }
            } else {
                let child_prob = scaling_factor * child2.fitness
                    / (scaling_factor * child2.fitness + parent2.fitness);
                if rng.random_bool(child_prob) {
                    new_population.push(child2);
                } else {
                    new_population.push(parent2);
                }
            }
        } else {
            if child1.fitness > parent2.fitness {
                let child_prob = child1.fitness / (child1.fitness + parent2.fitness * scaling_factor);
                if rng.random_bool(child_prob) {
                    new_population.push(child1);
                } else {
                    new_population.push(parent2);
                }
            } else {
                let child_prob = scaling_factor * child1.fitness
                    / (scaling_factor * child1.fitness + parent2.fitness);
                if rng.random_bool(child_prob) {
                    new_population.push(child1);
                } else {
                    new_population.push(parent2);
                }
            }
            if child2.fitness > parent1.fitness {
                let child_prob = child2.fitness / (child2.fitness + parent1.fitness * scaling_factor);
                if rng.random_bool(child_prob) {
                    new_population.push(child2);
                } else {
                    new_population.push(parent1);
                }
            } else {
                let child_prob = scaling_factor * child2.fitness
                    / (scaling_factor * child2.fitness + parent1.fitness);
                if rng.random_bool(child_prob) {
                    new_population.push(child2);
                } else {
                    new_population.push(parent1);
                }
            }
        }
    }
    *population = new_population;
}

fn best_fitness(population: &[Genome], use_shared_fitness: bool) -> f64 {
    population
        .iter()
        .map(|g| {
            if use_shared_fitness {
                g.shared_fitness
            } else {
                g.fitness
            }
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

fn lowest_feasible_travel_time(population: &[Genome]) -> Option<f64> {
    population
        .iter()
        .filter(|g| g.feasible)
        .map(|g| g.travel_time)
        .min_by(|a, b| a.total_cmp(b))
}

fn get_elites(population: &[Genome], elite_size: usize) -> Vec<Genome> {
    let mut sorted = population.to_vec();
    sorted.sort_by(|a, b| solution_cmp(a, b));
    sorted.into_iter().take(elite_size).collect()
}

fn genome_difference(a: &Genome, b: &Genome) -> f64 {
    let sequence_len = a.sequence.len().max(b.sequence.len());
    let lengths_len = a.lengths.len().max(b.lengths.len());

    if sequence_len == 0 && lengths_len == 0 {
        return 0.0;
    }

    // Order difference.
    let mut sequence_diff = 0usize;
    for i in 0..sequence_len {
        if a.sequence.get(i) != b.sequence.get(i) {
            sequence_diff += 1;
        }
    }
    let sequence_distance = if sequence_len > 0 {
        sequence_diff as f64 / sequence_len as f64
    } else {
        0.0
    };

    // Route split difference.
    let mut lengths_l1 = 0usize;
    let mut lengths_scale = 0usize;
    for i in 0..lengths_len {
        let ai = a.lengths.get(i).copied().unwrap_or(0);
        let bi = b.lengths.get(i).copied().unwrap_or(0);
        lengths_l1 += ai.abs_diff(bi);
        lengths_scale += ai.max(bi);
    }
    let lengths_distance = if lengths_scale > 0 {
        lengths_l1 as f64 / lengths_scale as f64
    } else {
        0.0
    };

    // Weighted average.
    0.7 * sequence_distance + 0.3 * lengths_distance
}

fn genetic_algorithm(
    instance: &read_json::Instance,
    population_size: usize,
    generations: usize,
) -> (Genome, Vec<f64>, Vec<f64>, Vec<Option<f64>>, Vec<Genome>) {
    const USE_SHARED_FITNESS_SELECTION: bool = false;
    const LIGHT_LS_INTERVAL: usize = 200;
    const HEAVY_LS_INTERVAL: usize = 1000;
    const LIGHT_LS_TOP_K: usize = 3;
    const HEAVY_LS_TOP_K: usize = 6;
    const LIGHT_LS_STEPS: usize = 6;
    const HEAVY_LS_STEPS: usize = 20;

    let mut population = populate(population_size, instance);
    let mut fitness_history = Vec::with_capacity(generations + 1);
    let mut entropy_history = Vec::with_capacity(generations + 1);
    let mut feasible_travel_time_history = Vec::with_capacity(generations + 1);
    let mut running_lowest_feasible_travel_time: Option<f64> = None;

    if USE_SHARED_FITNESS_SELECTION {
        calculate_shared_fitnesses(&mut population);
    }

    let mut global_best_feasible = population
        .iter()
        .filter(|g| g.feasible)
        .min_by(|a, b| a.travel_time.total_cmp(&b.travel_time))
        .cloned();

    fitness_history.push(best_fitness(&population, USE_SHARED_FITNESS_SELECTION));
    entropy_history.push(calculate_entropy(&population));
    if let Some(lowest_feasible) = lowest_feasible_travel_time(&population) {
        running_lowest_feasible_travel_time = Some(lowest_feasible);
    }
    feasible_travel_time_history.push(running_lowest_feasible_travel_time);

    // Adaptive scaling factor
    let initial_scaling_factor = 1.0;
    let mut scaling_factor;

    // Adaptive mutation factors
    let initial_path_mutation_rate = 0.95;
    let initial_length_mutation_rate = 0.95;
    let mut path_mutation_rate;
    let mut length_mutation_rate;

    let initial_entropy = calculate_entropy(&population);
    let mut current_entropy;

    for i in 0..generations {
        let mut ran_local_search = false;

        // Add light local search here
        if i % LIGHT_LS_INTERVAL == 0 && i % HEAVY_LS_INTERVAL != 0 && i != 0 {
            let mut ranked: Vec<usize> = (0..population.len()).collect();
            ranked.sort_by(|&a, &b| solution_cmp(&population[a], &population[b]));

            for idx in ranked.into_iter().take(LIGHT_LS_TOP_K) {
                let improved = local_search(population[idx].clone(), instance, LIGHT_LS_STEPS, None);
                population[idx] = improved;
            }
            ran_local_search = true;
        }

        // Add heavy local search here
        if i % HEAVY_LS_INTERVAL == 0 && i != 0 {
            let mut ranked: Vec<usize> = (0..population.len()).collect();
            ranked.sort_by(|&a, &b| solution_cmp(&population[a], &population[b]));

            for idx in ranked.into_iter().take(HEAVY_LS_TOP_K) {
                let improved = local_search(population[idx].clone(), instance, HEAVY_LS_STEPS, None);
                population[idx] = improved;
            }
            ran_local_search = true;
        }

        current_entropy = calculate_entropy(&population);

        scaling_factor = initial_scaling_factor * current_entropy / initial_entropy.max(1e-12);
        path_mutation_rate =
            initial_path_mutation_rate * (1.0 - current_entropy / initial_entropy.max(1e-12));
        length_mutation_rate =
            initial_length_mutation_rate * (1.0 - current_entropy / initial_entropy.max(1e-12));

        if ran_local_search && USE_SHARED_FITNESS_SELECTION {
            calculate_shared_fitnesses(&mut population);
        }

        population = tournament_selection(&population, population_size, USE_SHARED_FITNESS_SELECTION);

        elitism_scaling_probabilistic_crowding_repopulation(
            &mut population,
            instance,
            path_mutation_rate,
            length_mutation_rate,
            4,
            scaling_factor,
            population_size,
        );
        if USE_SHARED_FITNESS_SELECTION {
            calculate_shared_fitnesses(&mut population);
        }

        fitness_history.push(best_fitness(&population, USE_SHARED_FITNESS_SELECTION));
        entropy_history.push(calculate_entropy(&population));
        if let Some(lowest_feasible) = lowest_feasible_travel_time(&population) {
            running_lowest_feasible_travel_time = Some(
                running_lowest_feasible_travel_time.map_or(lowest_feasible, |best_so_far| {
                    best_so_far.min(lowest_feasible)
                }),
            );
        }
        feasible_travel_time_history.push(running_lowest_feasible_travel_time);
        if (i + 1) % 500 == 0 || i == 0 {
            match feasible_travel_time_history.last().unwrap() {
                Some(v) => println!(
                    "Generation {}: Best Fitness = {}, Entropy = {}, Lowest Feasible Travel Time = {}",
                    i + 1,
                    fitness_history.last().unwrap(),
                    entropy_history.last().unwrap(),
                    v
                ),
                None => println!(
                    "Generation {}: Best Fitness = {}, Entropy = {}, Lowest Feasible Travel Time = None",
                    i + 1,
                    fitness_history.last().unwrap(),
                    entropy_history.last().unwrap()
                ),
            }
        }

        if let Some(current_best_feasible) = population
            .iter()
            .filter(|g| g.feasible)
            .min_by(|a, b| a.travel_time.total_cmp(&b.travel_time))
        {
            if global_best_feasible
                .as_ref()
                .is_none_or(|best| current_best_feasible.travel_time < best.travel_time)
            {
                global_best_feasible = Some(current_best_feasible.clone());
            }
        }
    }

    let best = if let Some(best_feasible) = global_best_feasible {
        best_feasible
    } else {
        population
            .iter()
            .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
            .cloned()
            .unwrap()
    };

    (
        best,
        fitness_history,
        entropy_history,
        feasible_travel_time_history,
        population,
    )
}

fn main() {
    let instance = read_json("src/train/train_0.json");
    let (best, fitness_history, entropy_history, feasible_travel_time_history, _population) =
        genetic_algorithm(&instance, 100, 2000);

    // Add heavy local search on best individual
    let best = local_search(best, &instance, 120, None);

    println!("Best fitness: {}", best.fitness);
    println!("Total travel time: {}", best.travel_time);
    println!("Benchmark: {}", instance.benchmark);

    plot_metrics(
        &fitness_history,
        &entropy_history,
        &feasible_travel_time_history,
        "40000_100_elitism_scaling_probabilistic_crowding_repopulation_0.01_mutation13.png",
    )
    .expect("Failed to plot metrics");
}
