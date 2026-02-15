mod read_json;

use rand::RngExt;
use rand::rng;
use rand::seq::SliceRandom;
use read_json::{get_travel_time, read_json};

#[derive(Clone)]
struct Genome {
    sequence: Vec<usize>,
    lengths: Vec<usize>,
    fitness: f64,
    travel_time: f64,
}

impl Genome {
    fn new(sequence: Vec<usize>, lengths: Vec<usize>) -> Self {
        Genome {
            sequence,
            lengths,
            fitness: 0.0,
            travel_time: 0.0,
        }
    }

    fn calculate_fitness(&mut self, instance: &read_json::Instance) {
        // Check if all patients are visited within their time windows, check nurse load, calculate total travel time, and check if all patients have been visited

        let n_patients = instance.patients.len();
        let mut total_travel_time = 0.0;
        let mut nurse = 0;
        let mut seen = vec![false; n_patients];
        let mut penalty = 0.0;

        if self.sequence.len() != n_patients {
            penalty += 1000.0;
        }

        if self.lengths.len() != instance.nbr_nurses {
            penalty += 1000.0;
        }

        if self.lengths.iter().sum::<usize>() != n_patients {
            penalty += 1000.0;
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
                let patient_node = patient + 1; // Because 0 is the depot

                // Check if patient has already been seen
                if patient >= n_patients || seen[patient] {
                    penalty += 1000.0;
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
                if total_time > patient_info.end_time as f64 {
                    penalty += 1000.0;
                }

                // If we exceed nurse capacity.
                load += patient_info.demand;
                if load > instance.capacity_nurse {
                    penalty += 1000.0;
                }

                gene_index += 1;
            }

            total_travel_time += total_nurse_travel_time;

            // If we cant make it back to the depot on time.
            if total_time + get_travel_time(&instance.travel_times, current_location, 0)
                > instance.depot.return_time as f64
            {
                penalty += 1000.0;
            }

            total_travel_time += get_travel_time(&instance.travel_times, current_location, 0);

            nurse += 1;
        }

        // Check if all patients have been visited
        if !seen.iter().all(|&b| b) {
            penalty += 1000.0;
        }

        // Calculate and store fitness and travel time.
        self.fitness = 1.0 / (1.0 + total_travel_time + penalty);
        self.travel_time = total_travel_time;
    }
}

fn populate(population_size: usize, instance: &read_json::Instance) -> Vec<Genome> {
    let n_patients = instance.patients.len();
    let n_nurses = instance.nbr_nurses;

    let mut rng = rng();
    let mut population = Vec::with_capacity(population_size);

    // Try a few times per genome to find a non-zero fitness individual.
    let max_attempts_per_genome = 300;

    for _ in 0..population_size {
        let mut best: Option<Genome> = None;

        for _ in 0..max_attempts_per_genome {
            // Create a sequence of all patients and shuffle it.
            let mut sequence: Vec<usize> = (0..n_patients).collect();
            sequence.shuffle(&mut rng);

            // Randomly distribute patients across nurses, nurses can have 0 patients.
            let mut lengths = vec![0usize; n_nurses];
            for _ in 0..n_patients {
                let k = rng.random_range(0..n_nurses);
                lengths[k] += 1;
            }

            let mut genome: Genome = Genome::new(sequence, lengths);
            genome.calculate_fitness(instance);

            if match &best {
                None => true,
                Some(b) => genome.fitness > b.fitness,
            } {
                best = Some(genome);
            }

            if match &best {
                Some(b) => b.fitness > 0.0,
                None => false,
            } {
                break;
            }
        }

        population.push(best.expect("Failed to generate genome"));
    }

    population
}

fn tournament_selection(population: &[Genome]) -> Vec<Genome> {
    let mut rng = rng();
    let mut selected: Vec<Genome> = Vec::with_capacity(population.len());
    while selected.len() < population.len() {
        let a = rng.random_range(0..population.len());
        let b = rng.random_range(0..population.len());

        if population[a].fitness > population[b].fitness {
            selected.push(population[a].clone());
        } else {
            selected.push(population[b].clone());
        }
    }
    selected
}

fn crossover(parent1: &Genome, parent2: &Genome, instance: &read_json::Instance) -> Genome {
    let mut rng = rng();
    let n = parent1.sequence.len();

    // Using order crossover and ensuring all patients are visited.
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

fn mutate(genome: &mut Genome, swap_rate: f64, length_mutation_rate: f64) {
    let mut rng = rng();
    // Swap two patients in the sequence.
    if rng.random_bool(swap_rate) {
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

fn repopulate(
    population: &mut Vec<Genome>,
    instance: &read_json::Instance,
    swap_rate: f64,
    length_mutation_rate: f64,
) {
    let mut rng = rng();
    let mut new_population = Vec::with_capacity(population.len());

    while new_population.len() < population.len() {
        let parent1 = population[rng.random_range(0..population.len())].clone();
        let parent2 = population[rng.random_range(0..population.len())].clone();

        let mut child = crossover(&parent1, &parent2, instance);
        mutate(&mut child, swap_rate, length_mutation_rate);
        child.calculate_fitness(instance);
        new_population.push(child);
    }
    *population = new_population;
}

fn genetic_algorithm(
    instance: &read_json::Instance,
    population_size: usize,
    generations: usize,
) -> Genome {
    let mut population = populate(population_size, instance);

    for _ in 0..generations {
        let mut selected = tournament_selection(&population);
        repopulate(&mut selected, instance, 0.1, 0.1);
        population = selected;
    }
    population
        .into_iter()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        .unwrap()
}

fn main() {
    let instance = read_json("src/train/train_0.json");
    let best = genetic_algorithm(&instance, 100, 100);
    println!("Best fitness: {}", best.fitness);
    println!("Total travel time: {}", best.travel_time);
    println!("Benchmark: {}", instance.benchmark);
}
