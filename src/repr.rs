use crate::parser;
use rand::{RngExt, rngs::StdRng, seq::SliceRandom};

const STRUCTURE_PENALTY: f64 = 1000.0;
const TW_BASE_PENALTY: f64 = 50.0;
const TW_LINEAR_PENALTY: f64 = 5.0;
const TW_QUADRATIC_PENALTY: f64 = 0.05;
const CAPACITY_LINEAR_PENALTY: f64 = 100.0;
const RETURN_LINEAR_PENALTY: f64 = 20.0;
const RETURN_QUADRATIC_PENALTY: f64 = 0.1;
const INFEASIBLE_GAP: f64 = 10000.0;

#[derive(Clone)]
pub struct Genome {
    pub sequence: Vec<usize>,
    pub lengths: Vec<usize>,
    pub fitness: f64,
    pub travel_time: f64,
    pub feasible: bool,
    pub nurse_travel_times: Vec<f64>,
    pub nurse_covered_demands: Vec<usize>,
}

impl Genome {
    pub fn new(sequence: Vec<usize>, lengths: Vec<usize>) -> Self {
        Genome {
            sequence,
            lengths,
            fitness: 0.0,
            travel_time: 0.0,
            feasible: true,
            nurse_travel_times: Vec::new(),
            nurse_covered_demands: Vec::new(),
        }
    }

    pub fn calculate_fitness(&mut self, instance: &parser::Instance) {
        // Check if all patients are visited within their time windows, check nurse load, calculate total travel time, and check if all patients have been visited

        let n_patients = instance.patients.len();
        let mut total_travel_time = 0.0;
        let mut nurse = 0;
        let mut seen = vec![false; n_patients];
        let mut penalty = 0.0;
        self.feasible = true;
        self.nurse_travel_times = vec![0.0; instance.nbr_nurses];
        self.nurse_covered_demands = vec![0; instance.nbr_nurses];

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

                let patient_info: &parser::Patient = &instance.patients[patient];

                let travel_time = instance.get_travel_time(current_location, patient_node);

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
            let back_to_depot = instance.get_travel_time(current_location, 0);
            let return_lateness =
                (total_time + back_to_depot - instance.depot.return_time as f64).max(0.0);
            if return_lateness > 0.0 {
                penalty += RETURN_LINEAR_PENALTY * return_lateness
                    + RETURN_QUADRATIC_PENALTY * return_lateness * return_lateness;
                self.feasible = false;
            }

            total_travel_time += back_to_depot;
            self.nurse_travel_times[nurse] = total_nurse_travel_time + back_to_depot;
            self.nurse_covered_demands[nurse] = load;

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

    pub fn get_non_empty_routes(&self) -> Vec<usize> {
        self.lengths
            .iter()
            .enumerate()
            .filter_map(|(idx, &len)| if len > 0 { Some(idx) } else { None })
            .collect()
    }
    pub fn genome_difference(&self, b: &Genome) -> f64 {
        let sequence_len = self.sequence.len().max(b.sequence.len());
        let lengths_len = self.lengths.len().max(b.lengths.len());

        if sequence_len == 0 && lengths_len == 0 {
            return 0.0;
        }

        // Order difference.
        let mut sequence_diff = 0usize;
        for i in 0..sequence_len {
            if self.sequence.get(i) != b.sequence.get(i) {
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
            let ai = self.lengths.get(i).copied().unwrap_or(0);
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
}

pub fn calculate_entropy(population: &[Genome]) -> f64 {
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

pub fn best_fitness(population: &[Genome]) -> f64 {
    population
        .iter()
        .map(|g| g.fitness)
        .fold(f64::NEG_INFINITY, f64::max)
}

pub fn solution_cmp(a: &Genome, b: &Genome) -> std::cmp::Ordering {
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

pub fn get_feasible_genomes(population: &[Genome]) -> Vec<Genome> {
    population.iter().filter(|g| g.feasible).cloned().collect()
}

pub fn best_feasible_solution(population: &[Genome]) -> Option<&Genome> {
    population
        .iter()
        .filter(|g| g.feasible)
        .min_by(|a, b| a.travel_time.total_cmp(&b.travel_time))
}

pub fn is_better_solution(candidate: &Genome, current: &Genome) -> bool {
    solution_cmp(candidate, current) == std::cmp::Ordering::Less
}

pub fn lowest_feasible_travel_time(population: &[Genome]) -> Option<f64> {
    population
        .iter()
        .filter(|g| g.feasible)
        .map(|g| g.travel_time)
        .min_by(|a, b| a.total_cmp(b))
}

pub fn get_elites(population: &[Genome], elite_size: usize) -> Vec<Genome> {
    let mut sorted = population.to_vec();
    sorted.sort_by(|a, b| solution_cmp(a, b));
    sorted.into_iter().take(elite_size).collect()
}

pub fn populate(
    population_size: usize,
    instance: &parser::Instance,
    rng: &mut StdRng,
) -> Vec<Genome> {
    let n_patients = instance.patients.len();
    let n_nurses = instance.nbr_nurses;
    let mut population = Vec::with_capacity(population_size);

    for _ in 0..population_size {
        let mut sequence: Vec<usize> = (0..n_patients).collect();
        sequence.shuffle(rng);

        let mut lengths = vec![0usize; n_nurses];
        for _ in 0..n_patients {
            let k = rng.random_range(0..n_nurses);
            lengths[k] += 1;
        }

        let mut genome = Genome::new(sequence, lengths);
        genome.calculate_fitness(instance);
        population.push(genome);
    }

    population
}
