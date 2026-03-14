use rand::RngExt;
use rand::SeedableRng;

use rand::rngs::StdRng;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::cli;
use crate::crossover;
use crate::mutation;
use crate::parser;
use crate::plotting;
use crate::replacement::generalized_crowding_select;
use crate::repr;
use crate::repr::{
    Genome, best_feasible_solution, best_fitness, calculate_entropy, get_feasible_genomes,
    is_better_solution, lowest_feasible_travel_time, solution_cmp,
};

#[derive(Debug, Clone)]
pub struct SolveOptions {
    pub early_stop: bool,
    pub verbose: bool,
    pub save_refinement_plots: bool,
    pub progress_log_interval: usize,
    pub metrics_sample_interval: usize,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            early_stop: false,
            verbose: true,
            save_refinement_plots: true,
            progress_log_interval: 100,
            metrics_sample_interval: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunMetricSnapshot {
    pub run_idx: usize,
    pub generation: usize,
    pub best_fitness: f64,
    pub running_best_fitness: f64,
    pub entropy: f64,
    pub lowest_feasible_travel_time: Option<f64>,
}

pub struct SolveOutput {
    pub best: Genome,
    pub fitness_history: Vec<f64>,
    pub entropy_history: Vec<f64>,
    pub feasible_travel_time_history: Vec<Option<f64>>,
    pub best_feasible_per_run: Vec<f64>,
    pub run_metric_snapshots: Vec<RunMetricSnapshot>,
}

fn sample_run_metric_snapshots(
    run_idx: usize,
    fitness_history: &[f64],
    entropy_history: &[f64],
    feasible_history: &[Option<f64>],
    interval: usize,
) -> Vec<RunMetricSnapshot> {
    let safe_interval = interval.max(1);
    let len = fitness_history
        .len()
        .min(entropy_history.len())
        .min(feasible_history.len());
    if len == 0 {
        return Vec::new();
    }

    let mut snapshots = Vec::new();
    let mut generation = 0usize;
    while generation < len {
        let running_best = fitness_history[..=generation]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        snapshots.push(RunMetricSnapshot {
            run_idx,
            generation,
            best_fitness: fitness_history[generation],
            running_best_fitness: running_best,
            entropy: entropy_history[generation],
            lowest_feasible_travel_time: feasible_history[generation],
        });
        generation += safe_interval;
    }

    let last_idx = len - 1;
    if snapshots.last().is_none_or(|s| s.generation != last_idx) {
        let running_best = fitness_history
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        snapshots.push(RunMetricSnapshot {
            run_idx,
            generation: last_idx,
            best_fitness: fitness_history[last_idx],
            running_best_fitness: running_best,
            entropy: entropy_history[last_idx],
            lowest_feasible_travel_time: feasible_history[last_idx],
        });
    }

    snapshots
}

pub fn write_run_metrics_csv(
    snapshots: &[RunMetricSnapshot],
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = Path::new(output_path).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let mut file = fs::File::create(output_path)?;
    writeln!(
        file,
        "run,generation,best_fitness,running_best_fitness,entropy,lowest_feasible_travel_time"
    )?;

    for s in snapshots {
        let feasible = s
            .lowest_feasible_travel_time
            .map(|v| format!("{:.8}", v))
            .unwrap_or_else(|| "None".to_string());
        writeln!(
            file,
            "{},{},{:.8},{:.8},{:.8},{}",
            s.run_idx,
            s.generation,
            s.best_fitness,
            s.running_best_fitness,
            s.entropy,
            feasible
        )?;
    }

    Ok(())
}

fn local_search_reverse_route(genome: &Genome, instance: &parser::Instance) -> Option<Genome> {
    for nurse in 0..genome.lengths.len() {
        let start = genome.lengths[..nurse].iter().sum::<usize>();
        let end = start + genome.lengths[nurse];
        if end - start < 2 {
            continue;
        }

        for i in start..(end - 1) {
            for j in (i + 1)..end {
                let mut candidate = genome.clone();
                candidate.sequence[i..=j].reverse();
                candidate.calculate_fitness(instance);
                if is_better_solution(&candidate, genome) {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

fn local_search_relocate_patient(genome: &Genome, instance: &parser::Instance) -> Option<Genome> {
    let n_routes = genome.lengths.len();

    for from_route in 0..n_routes {
        let from_start = genome.lengths[..from_route].iter().sum::<usize>();
        let from_end = from_start + genome.lengths[from_route];
        if from_end == from_start {
            continue;
        }

        for from_idx in from_start..from_end {
            for to_route in 0..n_routes {
                if to_route == from_route {
                    continue;
                }

                let to_start = genome.lengths[..to_route].iter().sum::<usize>();
                let to_end = to_start + genome.lengths[to_route];

                for insert_pos in to_start..=to_end {
                    let mut candidate = genome.clone();
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

                    if is_better_solution(&candidate, genome) {
                        return Some(candidate);
                    }
                }
            }
        }
    }
    None
}

fn local_search_swap_patient(genome: &Genome, instance: &parser::Instance) -> Option<Genome> {
    let n_routes = genome.lengths.len();

    for route_a in 0..n_routes {
        let a_start = genome.lengths[..route_a].iter().sum::<usize>();
        let a_end = a_start + genome.lengths[route_a];
        if a_end == a_start {
            continue;
        }

        for route_b in (route_a + 1)..n_routes {
            let b_start = genome.lengths[..route_b].iter().sum::<usize>();
            let b_end = b_start + genome.lengths[route_b];
            if b_end == b_start {
                continue;
            }

            for idx_a in a_start..a_end {
                for idx_b in b_start..b_end {
                    let mut candidate = genome.clone();
                    candidate.sequence.swap(idx_a, idx_b);
                    candidate.calculate_fitness(instance);

                    if is_better_solution(&candidate, genome) {
                        return Some(candidate);
                    }
                }
            }
        }
    }

    None
}

fn local_search(mut genome: Genome, instance: &parser::Instance, max_steps: usize) -> Genome {
    // First-improving descent: keep applying one improving move until no move improves.
    genome.calculate_fitness(instance);

    for _ in 0..max_steps {
        if let Some(next) = local_search_reverse_route(&genome, instance) {
            genome = next;
            continue;
        }

        if let Some(next) = local_search_relocate_patient(&genome, instance) {
            genome = next;
            continue;
        }

        if let Some(next) = local_search_swap_patient(&genome, instance) {
            genome = next;
            continue;
        }

        break;
    }

    genome
}

fn iterated_local_search(
    seed: Genome,
    instance: &parser::Instance,
    iterations: usize,
    local_search_steps: usize,
    rng: &mut StdRng,
    verbose: bool,
) -> (Genome, Vec<f64>) {
    // Local search + mutate + local search, repeated until no improvement.
    let mut current = local_search(seed, instance, local_search_steps);
    let mut best = current.clone();
    let mut best_travel_history = vec![best.travel_time];
    let mut iterations_since_improvement = 0usize;
    let mut iteration = 0;

    while iterations_since_improvement < iterations {
        let mut candidate = current.clone();
        let shake_moves = 2 + iterations_since_improvement / 40;
        mutation::random_mutate(&mut candidate, shake_moves, 1.0, rng);
        candidate.calculate_fitness(instance);
        candidate = local_search(candidate, instance, local_search_steps);

        if is_better_solution(&candidate, &best) {
            best = candidate.clone();
            current = candidate;
            iterations_since_improvement = 0;
            best_travel_history.push(best.travel_time);
            continue;
        }

        let accept = match (candidate.feasible, current.feasible) {
            (true, false) => true,
            (false, true) => false,
            (true, true) => {
                if candidate.travel_time <= current.travel_time {
                    true
                } else {
                    let progress = iteration as f64 / iterations.max(1) as f64;
                    let temperature = (20.0 * (1.0 - progress)).max(0.5);
                    let delta = candidate.travel_time - current.travel_time;
                    let acceptance_probability = (-delta / temperature).exp().clamp(0.0, 1.0);
                    rng.random_bool(acceptance_probability)
                }
            }
            (false, false) => candidate.fitness > current.fitness || rng.random_bool(0.05),
        };

        if accept {
            current = candidate;
        }

        iterations_since_improvement += 1;
        iteration += 1;

        best_travel_history.push(best.travel_time);

        if verbose && (iteration + 1) % 50 == 0 {
            if best.feasible {
                println!(
                    "    Iteration {}, best travel time {:.4}",
                    iteration + 1,
                    best.travel_time
                );
            } else {
                println!("    Iteration {}, current best infeasible", iteration + 1,);
            }
        }
    }

    (best, best_travel_history)
}

fn make_child(
    p1: &Genome,
    p2: &Genome,
    instance: &parser::Instance,
    mutation_rate: f64,
    rng: &mut StdRng,
) -> Genome {
    let mut c = crossover::edge_recombination(p1, p2, instance, rng);
    mutation::random_mutate(&mut c, 1, mutation_rate, rng);
    c.calculate_fitness(instance);
    c
}

fn genetic_algorithm(
    instance: &parser::Instance,
    config: &cli::RunConfig,
    mut rng: &mut StdRng,
    verbose: bool,
    progress_log_interval: usize,
) -> (Genome, Vec<f64>, Vec<f64>, Vec<Option<f64>>, Vec<Genome>) {
    let population_size = config.population_size;
    let mut population = repr::populate(population_size, instance, rng);
    let mut fitness_history = Vec::with_capacity(config.generations + 1);
    let mut entropy_history = Vec::with_capacity(config.generations + 1);
    let mut feasible_travel_time_history = Vec::with_capacity(config.generations + 1);
    let mut running_lowest_feasible_travel_time: Option<f64> = None;

    let mut global_best_feasible = best_feasible_solution(&population).cloned();

    fitness_history.push(repr::best_fitness(&population));
    entropy_history.push(calculate_entropy(&population));
    if let Some(lowest_feasible) = lowest_feasible_travel_time(&population) {
        running_lowest_feasible_travel_time = Some(lowest_feasible);
    }
    feasible_travel_time_history.push(running_lowest_feasible_travel_time);

    let initial_scaling_factor = config.initial_scaling_factor;
    let mut scaling_factor;

    let initial_mutation_rate = config.initial_mutation_rate;
    let mut mutation_rate;

    let initial_entropy = calculate_entropy(&population);
    let mut current_entropy;

    let mut generation = 0;
    let mut generations_since_improvement = 0;

    let mut last_fitness = 0.0;
    let mut current_fitness;

    while generation < config.generations && generations_since_improvement < config.stagnation_limit
    {
        if generation > 0 && generation % config.ls_every == 0 {
            for genome in population.iter_mut() {
                *genome = local_search(genome.clone(), instance, config.ls_steps);
            }
        }

        current_entropy = calculate_entropy(&population);

        scaling_factor = initial_scaling_factor * current_entropy / initial_entropy.max(1e-12);
        mutation_rate =
            initial_mutation_rate * (1.0 - current_entropy / initial_entropy.max(1e-12));

        let mut new_population = Vec::with_capacity(population_size);

        let elite_size = config.elite_size.min(population_size);
        new_population.extend(repr::get_elites(&population, elite_size));

        while new_population.len() + 1 < population_size {
            let p1_idx = rng.random_range(0..population_size);
            let mut p2_idx = rng.random_range(0..population_size - 1);
            if p2_idx >= p1_idx {
                p2_idx += 1;
            }

            let p1 = population[p1_idx].to_owned();
            let p2 = population[p2_idx].to_owned();

            let c1 = make_child(&p1, &p2, instance, mutation_rate, rng);
            let c2 = make_child(&p2, &p1, instance, mutation_rate, rng);

            let (a1, a2) = if c1.genome_difference(&p1) + c2.genome_difference(&p2)
                < c1.genome_difference(&p2) + c2.genome_difference(&p1)
            {
                (p1, p2)
            } else {
                (p2, p1)
            };

            new_population.push(generalized_crowding_select(
                c1,
                a1,
                scaling_factor,
                &mut rng,
            ));
            new_population.push(generalized_crowding_select(
                c2,
                a2,
                scaling_factor,
                &mut rng,
            ));
        }

        if new_population.len() < population_size {
            let p1_idx = rng.random_range(0..population_size);
            let mut p2_idx = rng.random_range(0..population_size - 1);
            if p2_idx >= p1_idx {
                p2_idx += 1;
            }

            let p1 = population[p1_idx].to_owned();
            let p2 = population[p2_idx].to_owned();
            let c = make_child(&p1, &p2, instance, mutation_rate, rng);
            new_population.push(generalized_crowding_select(c, p1, scaling_factor, &mut rng));
        }

        population = new_population;

        current_fitness = best_fitness(&population);
        fitness_history.push(current_fitness);
        entropy_history.push(calculate_entropy(&population));

        if let Some(time) = lowest_feasible_travel_time(&population) {
            running_lowest_feasible_travel_time =
                Some(running_lowest_feasible_travel_time.map_or(time, |current| current.min(time)));
        }

        feasible_travel_time_history.push(running_lowest_feasible_travel_time);
        if verbose
            && ((generation + 1) % progress_log_interval.max(1) == 0 || generation == 0)
        {
            let running_best_fitness = fitness_history
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            match feasible_travel_time_history.last().unwrap() {
                Some(v) => println!(
                    "Generation {}: Best Fitness = {}, Running Best Fitness = {}, Entropy = {}, Lowest Feasible Travel Time = {}",
                    generation + 1,
                    fitness_history.last().unwrap(),
                    running_best_fitness,
                    entropy_history.last().unwrap(),
                    v
                ),
                None => println!(
                    "Generation {}: Best Fitness = {}, Running Best Fitness = {}, Entropy = {}, Lowest Feasible Travel Time = None",
                    generation + 1,
                    fitness_history.last().unwrap(),
                    running_best_fitness,
                    entropy_history.last().unwrap()
                ),
            }
        }

        if let Some(current_best_feasible) = best_feasible_solution(&population) {
            if global_best_feasible
                .as_ref()
                .is_none_or(|best| current_best_feasible.travel_time < best.travel_time)
            {
                global_best_feasible = Some(current_best_feasible.clone());
            }
        }

        generation += 1;
        if current_fitness > last_fitness {
            generations_since_improvement = 0;
        } else {
            generations_since_improvement += 1;
        }
        last_fitness = current_fitness;
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

fn multithreaded_solve(
    instance: &parser::Instance,
    options: &SolveOptions,
    config: &cli::RunConfig,
) -> (
    Genome,
    Vec<f64>,
    Vec<f64>,
    Vec<Option<f64>>,
    Vec<f64>,
    Vec<RunMetricSnapshot>,
) {
    let target_gap_percent = 5.0;
    let target_travel_time = instance.benchmark * (1.0 + target_gap_percent / 100.0);

    let mut best_overall: Option<Genome> = None;
    let mut best_fitness_history = Vec::new();
    let mut best_entropy_history = Vec::new();
    let mut best_feasible_history = Vec::new();
    let mut best_feasible_per_run = vec![-1f64; config.max_runs];
    let mut run_metric_snapshots = Vec::new();
    let mut best_refinement_histories_by_run: Vec<(usize, Vec<f64>)> =
        Vec::with_capacity(config.max_runs);

    for run_idx in 1..=config.max_runs {
        if options.verbose {
            println!("\n=== Run {}/{} ===", run_idx, config.max_runs);
        }
        let seed = run_idx as u64;
        let mut rng = StdRng::seed_from_u64(seed);

        let (run_best, fitness_history, entropy_history, feasible_history, population) =
            genetic_algorithm(
                instance,
                config,
                &mut rng,
                options.verbose,
                options.progress_log_interval,
            );
        run_metric_snapshots.extend(sample_run_metric_snapshots(
            run_idx,
            &fitness_history,
            &entropy_history,
            &feasible_history,
            options.metrics_sample_interval,
        ));

        let mut refinement_pool = get_feasible_genomes(&population);
        refinement_pool.sort_by(solution_cmp);
        refinement_pool.truncate(config.refinement_candidates.saturating_sub(1));
        refinement_pool.push(run_best.clone());

        let mut refined_best = run_best;
        let refinement_total = refinement_pool.len();
        let mut refined_candidates = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(refinement_total);

            for (candidate_idx, candidate) in refinement_pool.into_iter().enumerate() {
                let candidate_number = candidate_idx + 1;
                if options.verbose {
                    println!(
                        "  Refining candidate {}/{} (threaded)...",
                        candidate_number, refinement_total
                    );
                }
                handles.push(scope.spawn(move || {
                    let mut thread_rng = StdRng::seed_from_u64(candidate_number as u64);
                    let (refined, history) = iterated_local_search(
                        candidate,
                        instance,
                        config.ils_iterations,
                        config.ils_local_search_steps,
                        &mut thread_rng,
                        options.verbose,
                    );
                    (candidate_number, refined, history)
                }));
            }

            let mut results = Vec::with_capacity(refinement_total);
            for handle in handles {
                results.push(handle.join().expect("Refinement thread panicked"));
            }
            results
        });

        refined_candidates.sort_by_key(|(thread_number, _, _)| *thread_number);
        let mut best_refinement_history: Vec<f64> = Vec::new();
        for (_, refined, history) in &refined_candidates {
            if solution_cmp(&refined, &refined_best) == std::cmp::Ordering::Less {
                refined_best = refined.clone();
                best_refinement_history = history.clone();
            } else if best_refinement_history.is_empty()
                && solution_cmp(&refined, &refined_best) == std::cmp::Ordering::Equal
            {
                best_refinement_history = history.clone();
            }
        }
        if best_refinement_history.is_empty() && !refined_candidates.is_empty() {
            best_refinement_history = refined_candidates[0].2.clone();
        }
        if !best_refinement_history.is_empty() {
            best_refinement_histories_by_run.push((run_idx, best_refinement_history));
        }

        if refined_best.feasible {
            let gap_pct =
                100.0 * (refined_best.travel_time - instance.benchmark) / instance.benchmark;
            if options.verbose {
                println!(
                    "Run {} best travel time = {:.4} (gap {:.2}%)",
                    run_idx, refined_best.travel_time, gap_pct
                );
            }
            best_feasible_per_run[run_idx - 1] = refined_best.travel_time;
        } else if options.verbose {
            println!(
                "Run {} best solution is infeasible (fitness = {:.8})",
                run_idx, refined_best.fitness
            );
        }

        let is_better = best_overall.as_ref().is_none_or(|current_best| {
            solution_cmp(&refined_best, current_best) == std::cmp::Ordering::Less
        });
        if is_better {
            best_overall = Some(refined_best.clone());
            best_fitness_history = fitness_history;
            best_entropy_history = entropy_history;
            best_feasible_history = feasible_history;
        }

        if options.early_stop
            && refined_best.feasible
            && refined_best.travel_time <= target_travel_time
        {
            if options.verbose {
                println!(
                    "Target reached on run {} (<= {:.4})",
                    run_idx, target_travel_time
                );
            }
            break;
        }
    }

    if options.save_refinement_plots && !best_refinement_histories_by_run.is_empty() {
        let output_path = "refinement_best_candidate_by_run.png";
        if let Err(err) = plotting::plot_best_refinement_by_run(
            &best_refinement_histories_by_run,
            target_travel_time,
            &output_path,
            config.dataset.as_str(),
        ) {
            if options.verbose {
                println!(
                    "Failed to plot combined best-candidate refinement histories: {}",
                    err
                );
            }
        } else if options.verbose {
            println!("Saved combined refinement plot: {}", output_path);
        }
    }

    (
        best_overall.expect("No solution produced across restarts"),
        best_fitness_history,
        best_entropy_history,
        best_feasible_history,
        best_feasible_per_run,
        run_metric_snapshots,
    )
}

pub fn solve(
    instance: &parser::Instance,
    config: &cli::RunConfig,
    options: &SolveOptions,
) -> SolveOutput {
    let (
        best,
        fitness_history,
        entropy_history,
        feasible_travel_time_history,
        best_feasible_per_run,
        run_metric_snapshots,
    ) = multithreaded_solve(instance, options, &config);

    SolveOutput {
        best,
        fitness_history,
        entropy_history,
        feasible_travel_time_history,
        best_feasible_per_run,
        run_metric_snapshots,
    }
}
