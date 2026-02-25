mod read_json;

use plotters::prelude::*;
use rand::RngExt;
use rand::rng;
use rand::seq::SliceRandom;
use read_json::{get_travel_time, read_json};
use std::error::Error;

#[derive(Clone)]
struct Genome {
    sequence: Vec<usize>,
    lengths: Vec<usize>,
    fitness: f64,
    travel_time: f64,
    feasible: bool,
    nurse_travel_times: Vec<f64>,
    nurse_covered_demands: Vec<usize>,
}

impl Genome {
    fn new(sequence: Vec<usize>, lengths: Vec<usize>) -> Self {
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

fn plot_nurse_route_network(
    instance: &read_json::Instance,
    best: &Genome,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut min_x = instance.depot.x_coord as f64;
    let mut max_x = instance.depot.x_coord as f64;
    let mut min_y = instance.depot.y_coord as f64;
    let mut max_y = instance.depot.y_coord as f64;

    for patient in &instance.patients {
        let px = patient.x_coord as f64;
        let py = patient.y_coord as f64;
        min_x = min_x.min(px);
        max_x = max_x.max(px);
        min_y = min_y.min(py);
        max_y = max_y.max(py);
    }

    let pad_x = ((max_x - min_x) * 0.05).max(1.0);
    let pad_y = ((max_y - min_y) * 0.05).max(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Nurse Route Network", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d((min_x - pad_x)..(max_x + pad_x), (min_y - pad_y)..(max_y + pad_y))?;

    chart.configure_mesh().x_desc("X").y_desc("Y").draw()?;

    let depot_point = (instance.depot.x_coord as f64, instance.depot.y_coord as f64);
    chart.draw_series(std::iter::once(Circle::new(depot_point, 7, BLACK.filled())))?;

    for patient in &instance.patients {
        chart.draw_series(std::iter::once(Circle::new(
            (patient.x_coord as f64, patient.y_coord as f64),
            2,
            BLACK.mix(0.25).filled(),
        )))?;
    }

    for nurse in 0..best.lengths.len() {
        let start = best.lengths[..nurse].iter().sum::<usize>();
        let end = start + best.lengths[nurse];
        let route = &best.sequence[start..end];
        if route.is_empty() {
            continue;
        }

        let color = Palette99::pick(nurse);
        let mut polyline = Vec::with_capacity(route.len() + 2);
        polyline.push(depot_point);
        for &patient_idx in route {
            let patient = &instance.patients[patient_idx];
            polyline.push((patient.x_coord as f64, patient.y_coord as f64));
        }
        polyline.push(depot_point);

        chart
            .draw_series(LineSeries::new(polyline.iter().copied(), &color))?
            .label(format!("Nurse {}", nurse + 1))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], Palette99::pick(nurse)));

        chart.draw_series(
            polyline
                .iter()
                .skip(1)
                .take(route.len())
                .map(|&p| Circle::new(p, 4, color.filled())),
        )?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn plot_refinement_travel_time(
    histories: &[(usize, Vec<f64>)],
    target_travel_time: f64,
    output_path: &str,
    title: &str,
) -> Result<(), Box<dyn Error>> {
    if histories.is_empty() {
        return Ok(());
    }

    let mut max_len = 0usize;
    let mut travel_min = f64::INFINITY;
    let mut travel_max = f64::NEG_INFINITY;

    for (_, history) in histories {
        if history.is_empty() {
            continue;
        }
        max_len = max_len.max(history.len());
        for &value in history {
            travel_min = travel_min.min(value);
            travel_max = travel_max.max(value);
        }
    }

    if max_len == 0 || !travel_min.is_finite() || !travel_max.is_finite() {
        return Ok(());
    }

    travel_min = travel_min.min(target_travel_time);
    travel_max = travel_max.max(target_travel_time);
    let travel_padding = if (travel_max - travel_min).abs() < f64::EPSILON {
        (travel_max.abs() * 0.01).max(1e-12)
    } else {
        0.0
    };

    let root = BitMapBackend::new(output_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(65)
        .build_cartesian_2d(
            0usize..max_len,
            (travel_min - travel_padding)..(travel_max + travel_padding),
        )?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc("Best Travel Time")
        .draw()?;

    for (idx, (candidate_number, history)) in histories.iter().enumerate() {
        if history.is_empty() {
            continue;
        }
        chart
            .draw_series(LineSeries::new(
                history.iter().enumerate().map(|(i, v)| (i, *v)),
                &Palette99::pick(idx),
            ))?
            .label(format!("Candidate {}", candidate_number))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], Palette99::pick(idx))
            });
    }

    chart
        .draw_series(LineSeries::new(
            (0..max_len).map(|i| (i, target_travel_time)),
            &BLACK.mix(0.4),
        ))?
        .label("Target (benchmark +5%)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.mix(0.4)));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn create_seeded_genome(instance: &read_json::Instance) -> Genome {
    const LATENESS_PENALTY: f64 = 1000.0;
    const OVERLOAD_PENALTY: f64 = 3000.0;
    const RETURN_LATE_PENALTY: f64 = 800.0;
    const ROUTE_BALANCE_WEIGHT: f64 = 2.0;
    const SCORE_NOISE: f64 = 1e-4;

    let n_patients = instance.patients.len();
    let n_nurses = instance.nbr_nurses;
    let mut rng = rng();

    let mut patient_order: Vec<usize> = (0..n_patients).collect();
    patient_order.sort_by(|&a, &b| {
        let pa = &instance.patients[a];
        let pb = &instance.patients[b];
        pa.end_time
            .cmp(&pb.end_time)
            .then(pa.start_time.cmp(&pb.start_time))
            .then(pb.demand.cmp(&pa.demand))
    });

    if n_patients > 1 {
        for chunk in patient_order.chunks_mut(4) {
            chunk.shuffle(&mut rng);
        }
    }

    let mut routes: Vec<Vec<usize>> = vec![Vec::new(); n_nurses];
    let mut route_time = vec![0.0f64; n_nurses];
    let mut route_load = vec![0usize; n_nurses];
    let mut route_location = vec![0usize; n_nurses];

    for patient in patient_order {
        let patient_info = &instance.patients[patient];
        let patient_node = patient + 1;
        let average_route_size =
            (routes.iter().map(|r| r.len()).sum::<usize>() as f64 + 1.0) / n_nurses as f64;

        let mut best_nurse = 0usize;
        let mut best_score = f64::INFINITY;
        let mut best_time = 0.0f64;
        let mut best_load = 0usize;

        for nurse in 0..n_nurses {
            let travel =
                get_travel_time(&instance.travel_times, route_location[nurse], patient_node);

            let mut projected_time = route_time[nurse] + travel;
            if projected_time < patient_info.start_time as f64 {
                projected_time = patient_info.start_time as f64;
            }
            projected_time += patient_info.care_time as f64;

            let projected_load = route_load[nurse] + patient_info.demand;
            let lateness = (projected_time - patient_info.end_time as f64).max(0.0);
            let overload = projected_load.saturating_sub(instance.capacity_nurse);
            let return_lateness = (projected_time
                + get_travel_time(&instance.travel_times, patient_node, 0)
                - instance.depot.return_time as f64)
                .max(0.0);
            let route_size_after_insert = routes[nurse].len() as f64 + 1.0;
            let balance_penalty = (route_size_after_insert - average_route_size).abs();

            let score = travel
                + LATENESS_PENALTY * lateness
                + OVERLOAD_PENALTY * overload as f64
                + RETURN_LATE_PENALTY * return_lateness
                + ROUTE_BALANCE_WEIGHT * balance_penalty
                + rng.random_range(0.0..SCORE_NOISE);

            if score < best_score {
                best_score = score;
                best_nurse = nurse;
                best_time = projected_time;
                best_load = projected_load;
            }
        }

        routes[best_nurse].push(patient);
        route_time[best_nurse] = best_time;
        route_load[best_nurse] = best_load;
        route_location[best_nurse] = patient_node;
    }

    let mut sequence = Vec::with_capacity(n_patients);
    let mut lengths = vec![0usize; n_nurses];
    for nurse in 0..n_nurses {
        lengths[nurse] = routes[nurse].len();
        sequence.extend(routes[nurse].iter().copied());
    }

    let mut genome = Genome::new(sequence, lengths);
    genome.calculate_fitness(instance);
    genome
}

fn populate(population_size: usize, instance: &read_json::Instance) -> Vec<Genome> {
    const SEEDED_RATE: f64 = 0.85;

    let n_patients = instance.patients.len();
    let n_nurses = instance.nbr_nurses;
    let mut rng = rng();
    let mut population = Vec::with_capacity(population_size);

    for _ in 0..population_size {
        if rng.random_bool(SEEDED_RATE) {
            population.push(create_seeded_genome(instance));
        } else {
            let mut sequence: Vec<usize> = (0..n_patients).collect();
            sequence.shuffle(&mut rng);

            let mut lengths = vec![0usize; n_nurses];
            for _ in 0..n_patients {
                let k = rng.random_range(0..n_nurses);
                lengths[k] += 1;
            }

            let mut genome = Genome::new(sequence, lengths);
            genome.calculate_fitness(instance);
            population.push(genome);
        }
    }

    population
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

fn get_non_empty_routes(lengths: &[usize]) -> Vec<usize> {
    lengths
        .iter()
        .enumerate()
        .filter_map(|(idx, &len)| if len > 0 { Some(idx) } else { None })
        .collect()
}

fn get_feasible_genomes(population: &[Genome]) -> Vec<Genome> {
    population.iter().filter(|g| g.feasible).cloned().collect()
}

fn best_feasible_solution(population: &[Genome]) -> Option<&Genome> {
    population
        .iter()
        .filter(|g| g.feasible)
        .min_by(|a, b| a.travel_time.total_cmp(&b.travel_time))
}

fn is_better_solution(candidate: &Genome, current: &Genome) -> bool {
    solution_cmp(candidate, current) == std::cmp::Ordering::Less
}

fn local_search_reverse_route(genome: &Genome, instance: &read_json::Instance) -> Option<Genome> {
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

fn local_search_relocate_patient(
    genome: &Genome,
    instance: &read_json::Instance,
) -> Option<Genome> {
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

fn local_search_swap_patient(genome: &Genome, instance: &read_json::Instance) -> Option<Genome> {
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

fn local_search(mut genome: Genome, instance: &read_json::Instance, max_steps: usize) -> Genome {
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

fn swap_mutation(genome: &mut Genome, rng: &mut rand::rngs::ThreadRng) {
    let i = rng.random_range(0..genome.sequence.len());
    let j = rng.random_range(0..genome.sequence.len());
    genome.sequence.swap(i, j);
}

fn inversion_mutation(genome: &mut Genome, rng: &mut rand::rngs::ThreadRng) {
    if genome.sequence.len() > 1 {
        let start = rng.random_range(0..genome.sequence.len());
        let end = rng.random_range(start..genome.sequence.len());
        genome.sequence[start..=end].reverse();
    }
}

fn relocation_mutation(genome: &mut Genome, rng: &mut rand::rngs::ThreadRng) {
    let route_count = genome.lengths.len();
    if route_count < 2 {
        return;
    }

    let non_empty_routes = get_non_empty_routes(&genome.lengths);
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

fn route_length_mutation(genome: &mut Genome, rng: &mut rand::rngs::ThreadRng) {
    let route_count = genome.lengths.len();
    if route_count < 2 {
        return;
    }

    let non_empty_routes = get_non_empty_routes(&genome.lengths);
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

fn random_mutate(genome: &mut Genome, moves: usize, mutation_chance: f64) {
    if genome.sequence.is_empty() || genome.lengths.is_empty() {
        return;
    }

    let mut rng = rng();

    if rng.random_bool(mutation_chance) {
        for _ in 0..moves {
            match rng.random_range(0..4) {
                0 => swap_mutation(genome, &mut rng),
                1 => inversion_mutation(genome, &mut rng),
                2 => relocation_mutation(genome, &mut rng),
                _ => route_length_mutation(genome, &mut rng),
            }
        }
    }
}

fn iterated_local_search(
    seed: Genome,
    instance: &read_json::Instance,
    iterations: usize,
    local_search_steps: usize,
    max_shake_moves: usize,
) -> (Genome, Vec<f64>) {
    // Local search + mutate + local search, repeated until no improvement.
    let mut rng = rng();
    let mut current = local_search(seed, instance, local_search_steps);
    let mut best = current.clone();
    let mut best_travel_history = vec![best.travel_time];
    let mut iterations_since_improvement = 0usize;
    let mut iteration = 0;

    while iterations_since_improvement < iterations {
        let mut candidate = current.clone();
        let shake_moves = (2 + iterations_since_improvement / 40).min(max_shake_moves.max(2));
        random_mutate(&mut candidate, shake_moves, 1.0);
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

        if (iteration + 1) % 50 == 0 {
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

fn solve_until_target(
    instance: &read_json::Instance,
) -> (Genome, Vec<f64>, Vec<f64>, Vec<Option<f64>>) {
    const TARGET_GAP_PERCENT: f64 = 5.0;
    const MAX_RUNS: usize = 10;
    const POPULATION_SIZE: usize = 120;
    const GENERATIONS: usize = 3000;
    const REFINEMENT_CANDIDATES: usize = 8;
    const ILS_ITERATIONS: usize = 300;
    const ILS_LOCAL_SEARCH_STEPS: usize = 90;
    const ILS_MAX_SHAKE_MOVES: usize = 16;

    let target_travel_time = instance.benchmark * (1.0 + TARGET_GAP_PERCENT / 100.0);

    let mut best_overall: Option<Genome> = None;
    let mut best_fitness_history = Vec::new();
    let mut best_entropy_history = Vec::new();
    let mut best_feasible_history = Vec::new();

    for run_idx in 1..=MAX_RUNS {
        println!("\n=== Restart {}/{} ===", run_idx, MAX_RUNS);
        let (run_best, fitness_history, entropy_history, feasible_history, population) =
            genetic_algorithm(instance, POPULATION_SIZE, GENERATIONS);

        let mut refinement_pool = get_feasible_genomes(&population);
        refinement_pool.sort_by(solution_cmp);
        refinement_pool.truncate(REFINEMENT_CANDIDATES - 1);
        refinement_pool.push(run_best.clone());

        let mut refined_best = run_best;
        let refinement_total = refinement_pool.len();
        let mut refinement_histories = Vec::with_capacity(refinement_total);
        for (candidate_idx, candidate) in refinement_pool.into_iter().enumerate() {
            println!(
                "  Refining candidate {}/{}...",
                candidate_idx + 1,
                refinement_total
            );
            let (refined, history) = iterated_local_search(
                candidate,
                instance,
                ILS_ITERATIONS,
                ILS_LOCAL_SEARCH_STEPS,
                ILS_MAX_SHAKE_MOVES,
            );
            refinement_histories.push((candidate_idx + 1, history));
            if solution_cmp(&refined, &refined_best) == std::cmp::Ordering::Less {
                refined_best = refined;
            }
            if refined_best.feasible && refined_best.travel_time <= target_travel_time {
                break;
            }
        }

        let refinement_plot_path = format!("refinement_travel_time_sequential_restart_{}.png", run_idx);
        if let Err(err) = plot_refinement_travel_time(
            &refinement_histories,
            target_travel_time,
            &refinement_plot_path,
            &format!("Sequential Refinement Travel Time (Restart {})", run_idx),
        ) {
            eprintln!(
                "Failed to plot sequential refinement travel time for restart {}: {}",
                run_idx, err
            );
        } else {
            println!("Saved refinement plot: {}", refinement_plot_path);
        }

        if refined_best.feasible {
            let gap_pct =
                100.0 * (refined_best.travel_time - instance.benchmark) / instance.benchmark;
            println!(
                "Restart {} best travel time = {:.4} (gap {:.2}%)",
                run_idx, refined_best.travel_time, gap_pct
            );
        } else {
            println!(
                "Restart {} best solution is infeasible (fitness = {:.8})",
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

        if refined_best.feasible && refined_best.travel_time <= target_travel_time {
            println!(
                "Target reached on restart {} (<= {:.4})",
                run_idx, target_travel_time
            );
            break;
        }
    }

    (
        best_overall.expect("No solution produced across restarts"),
        best_fitness_history,
        best_entropy_history,
        best_feasible_history,
    )
}

fn multithreaded_solve_until_target(
    instance: &read_json::Instance,
) -> (Genome, Vec<f64>, Vec<f64>, Vec<Option<f64>>) {
    const TARGET_GAP_PERCENT: f64 = 5.0;
    const MAX_RUNS: usize = 10;
    const POPULATION_SIZE: usize = 120;
    const GENERATIONS: usize = 3000;
    const REFINEMENT_CANDIDATES: usize = 8;
    const ILS_ITERATIONS: usize = 300;
    const ILS_LOCAL_SEARCH_STEPS: usize = 90;
    const ILS_MAX_SHAKE_MOVES: usize = 16;

    let target_travel_time = instance.benchmark * (1.0 + TARGET_GAP_PERCENT / 100.0);

    let mut best_overall: Option<Genome> = None;
    let mut best_fitness_history = Vec::new();
    let mut best_entropy_history = Vec::new();
    let mut best_feasible_history = Vec::new();

    for run_idx in 1..=MAX_RUNS {
        println!("\n=== Restart {}/{} ===", run_idx, MAX_RUNS);
        let (run_best, fitness_history, entropy_history, feasible_history, population) =
            genetic_algorithm(instance, POPULATION_SIZE, GENERATIONS);

        let mut refinement_pool = get_feasible_genomes(&population);
        refinement_pool.sort_by(solution_cmp);
        refinement_pool.truncate(REFINEMENT_CANDIDATES - 1);
        refinement_pool.push(run_best.clone());

        let mut refined_best = run_best;
        let refinement_total = refinement_pool.len();
        let mut refined_candidates = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(refinement_total);

            for (candidate_idx, candidate) in refinement_pool.into_iter().enumerate() {
                let candidate_number = candidate_idx + 1;
                println!(
                    "  Refining candidate {}/{} (threaded)...",
                    candidate_number, refinement_total
                );
                handles.push(scope.spawn(move || {
                    let (refined, history) = iterated_local_search(
                        candidate,
                        instance,
                        ILS_ITERATIONS,
                        ILS_LOCAL_SEARCH_STEPS,
                        ILS_MAX_SHAKE_MOVES,
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
        let refinement_histories: Vec<(usize, Vec<f64>)> = refined_candidates
            .iter()
            .map(|(thread_number, _, history)| (*thread_number, history.clone()))
            .collect();
        let refinement_plot_path = format!("refinement_travel_time_threaded_restart_{}.png", run_idx);
        if let Err(err) = plot_refinement_travel_time(
            &refinement_histories,
            target_travel_time,
            &refinement_plot_path,
            &format!("Threaded Refinement Travel Time (Restart {})", run_idx),
        ) {
            eprintln!(
                "Failed to plot threaded refinement travel time for restart {}: {}",
                run_idx, err
            );
        } else {
            println!("Saved refinement plot: {}", refinement_plot_path);
        }

        for (_, refined, _) in refined_candidates {
            if solution_cmp(&refined, &refined_best) == std::cmp::Ordering::Less {
                refined_best = refined;
            }
        }

        if refined_best.feasible {
            let gap_pct =
                100.0 * (refined_best.travel_time - instance.benchmark) / instance.benchmark;
            println!(
                "Restart {} best travel time = {:.4} (gap {:.2}%)",
                run_idx, refined_best.travel_time, gap_pct
            );
        } else {
            println!(
                "Restart {} best solution is infeasible (fitness = {:.8})",
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

        if refined_best.feasible && refined_best.travel_time <= target_travel_time {
            println!(
                "Target reached on restart {} (<= {:.4})",
                run_idx, target_travel_time
            );
            break;
        }
    }

    (
        best_overall.expect("No solution produced across restarts"),
        best_fitness_history,
        best_entropy_history,
        best_feasible_history,
    )
}

fn elitism_scaling_probabilistic_crowding_repopulation(
    population: &mut Vec<Genome>,
    instance: &read_json::Instance,
    mutation_rate: f64,
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
        random_mutate(&mut child1, 1, mutation_rate);
        child1.calculate_fitness(instance);

        let mut child2 = edge_crossover(&parent2, &parent1, instance);
        random_mutate(&mut child2, 1, mutation_rate);
        child2.calculate_fitness(instance);

        if genome_difference(&child1, &parent1) + genome_difference(&child2, &parent2)
            < genome_difference(&child1, &parent2) + genome_difference(&child2, &parent1)
        {
            if child1.fitness > parent1.fitness {
                let child_prob =
                    child1.fitness / (child1.fitness + parent1.fitness * scaling_factor);
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
                let child_prob =
                    child2.fitness / (child2.fitness + parent2.fitness * scaling_factor);
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
                let child_prob =
                    child1.fitness / (child1.fitness + parent2.fitness * scaling_factor);
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
                let child_prob =
                    child2.fitness / (child2.fitness + parent1.fitness * scaling_factor);
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

fn best_fitness(population: &[Genome]) -> f64 {
    population
        .iter()
        .map(|g| g.fitness)
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
    let mut population = populate(population_size, instance);
    let mut fitness_history = Vec::with_capacity(generations + 1);
    let mut entropy_history = Vec::with_capacity(generations + 1);
    let mut feasible_travel_time_history = Vec::with_capacity(generations + 1);
    let mut running_lowest_feasible_travel_time: Option<f64> = None;

    let mut global_best_feasible = best_feasible_solution(&population).cloned();

    fitness_history.push(best_fitness(&population));
    entropy_history.push(calculate_entropy(&population));
    if let Some(lowest_feasible) = lowest_feasible_travel_time(&population) {
        running_lowest_feasible_travel_time = Some(lowest_feasible);
    }
    feasible_travel_time_history.push(running_lowest_feasible_travel_time);

    // Adaptive scaling factor
    let initial_scaling_factor = 1.0;
    let mut scaling_factor;

    // Adaptive mutation factors
    let initial_mutation_rate = 0.95;
    let mut mutation_rate;

    let initial_entropy = calculate_entropy(&population);
    let mut current_entropy;

    let mut generation = 0;
    let mut generations_since_improvement = 0;

    while generations_since_improvement < 1000 {
        current_entropy = calculate_entropy(&population);

        scaling_factor = initial_scaling_factor * current_entropy / initial_entropy.max(1e-12);
        mutation_rate =
            initial_mutation_rate * (1.0 - current_entropy / initial_entropy.max(1e-12));

        elitism_scaling_probabilistic_crowding_repopulation(
            &mut population,
            instance,
            mutation_rate,
            4,
            scaling_factor,
            population_size,
        );

        fitness_history.push(best_fitness(&population));
        entropy_history.push(calculate_entropy(&population));
        if let Some(lowest_feasible) = lowest_feasible_travel_time(&population) {
            running_lowest_feasible_travel_time = Some(
                running_lowest_feasible_travel_time.map_or(lowest_feasible, |best_so_far| {
                    best_so_far.min(lowest_feasible)
                }),
            );
        }
        feasible_travel_time_history.push(running_lowest_feasible_travel_time);
        if (generation + 1) % 500 == 0 || generation == 0 {
            match feasible_travel_time_history.last().unwrap() {
                Some(v) => println!(
                    "Generation {}: Best Fitness = {}, Entropy = {}, Lowest Feasible Travel Time = {}",
                    generation + 1,
                    fitness_history.last().unwrap(),
                    entropy_history.last().unwrap(),
                    v
                ),
                None => println!(
                    "Generation {}: Best Fitness = {}, Entropy = {}, Lowest Feasible Travel Time = None",
                    generation + 1,
                    fitness_history.last().unwrap(),
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
                generations_since_improvement = 0;
            }
        }
        generation += 1;
        generations_since_improvement += 1;
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
    let instance = read_json("src/train/train_9.json");
    let (best, fitness_history, entropy_history, feasible_travel_time_history) =
        multithreaded_solve_until_target(&instance);

    println!("\n=== Final Solution ===");
    println!("Nurse capacity: {}", instance.capacity_nurse);
    println!("Depot return time: {}", instance.depot.return_time);
    println!("\n----------------------");
    for nurse in 0..best.lengths.len() {
        let start = best.lengths[..nurse].iter().sum::<usize>();
        let end = start + best.lengths[nurse];
        let route = &best.sequence[start..end];
        let mut total_time = 0.0;
        let mut current_location = 0usize;
        let mut patient_sequence_with_times = Vec::with_capacity(route.len());

        for &patient_idx in route {
            let patient = &instance.patients[patient_idx];
            let patient_node = patient_idx + 1;
            let travel_time = get_travel_time(&instance.travel_times, current_location, patient_node);
            total_time += travel_time;

            let visit_time = total_time.max(patient.start_time as f64);
            let leave_time = visit_time + patient.care_time as f64;
            patient_sequence_with_times.push(format!(
                "P{} ({:.1} - {:.1}) [{:.1} - {:.1}]",
                patient_idx + 1,
                visit_time,
                leave_time,
                patient.start_time,
                patient.end_time,
            ));

            total_time = leave_time;
            current_location = patient_node;
        }

        let patient_sequence_text = if patient_sequence_with_times.is_empty() {
            String::from("None")
        } else {
            patient_sequence_with_times.join(" -> ")
        };

        println!(
            "Nurse: {} Route duration: {} Covered demand: {} Patient sequence: {}",
            nurse + 1,
            best.nurse_travel_times[nurse],
            best.nurse_covered_demands[nurse],
            patient_sequence_text
        );
    }
    println!("----------------------\n");
    println!("Total travel time: {}", best.travel_time);

    plot_metrics(
        &fitness_history,
        &entropy_history,
        &feasible_travel_time_history,
        "multi_start_ils_metrics.png",
    )
    .expect("Failed to plot metrics");
    plot_nurse_route_network(&instance, &best, "nurse_route_network.png")
        .expect("Failed to plot nurse route network");
}
