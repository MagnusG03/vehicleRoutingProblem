use vrp::{cli, parser, plotting::{self}, solver};

fn main() {
    let (dataset, config) = cli::parse_cli_args().unwrap_or_else(|err| {
        eprintln!("Error: {}", err);
        std::process::exit(2);
    });

    println!("Dataset: {}", dataset);
    println!(
        "Config: max_runs={}, population_size={}, generations={}, refinement_candidates={}, ils_iterations={}, ils_local_search_steps={}, elite_size={}, initial_mutation_rate={}, initial_scaling_factor={}, ls_every={}, ls_steps={}, stagnation_limit={}",
        config.max_runs,
        config.population_size,
        config.generations,
        config.refinement_candidates,
        config.ils_iterations,
        config.ils_local_search_steps,
        config.elite_size,
        config.initial_mutation_rate,
        config.initial_scaling_factor,
        config.ls_every,
        config.ls_steps,
        config.stagnation_limit
    );

    let instance = parser::read_json(&dataset);
    let output = solver::solve(&instance, &config, &solver::SolveOptions::default());

    let valid_runs: Vec<f64> = output
        .best_feasible_per_run
        .iter()
        .filter(|&&x| x > 0.0)
        .cloned()
        .collect();

    if !valid_runs.is_empty() {
        let sum: f64 = valid_runs.iter().sum();
        let count = valid_runs.len();
        let avg_per_run = sum / count as f64;

        println!(
            "Average travel time across {} successful runs: {:.2}",
            count, avg_per_run
        );
    } else {
        println!("No valid feasible solutions found in any run.");
    }

    plotting::print_solution(&instance, &output.best);

    plotting::plot_metrics(
        &output.fitness_history,
        &output.entropy_history,
        &output.feasible_travel_time_history,
        format!(
            "src/results/{}/multi_start_ils_metrics.png",
            config.dataset.as_str()
        )
        .as_str(),
    )
    .expect("Failed to plot metrics");

    plotting::plot_nurse_route_network(
        &instance,
        &output.best,
        format!(
            "src/results/{}/nurse_route_network.png",
            config.dataset.as_str()
        )
        .as_str(),
    )
    .expect("Failed to plot nurse route network");

    plotting::plot_nurse_route_network(
        &instance,
        &output.best,
        format!(
            "src/results/{}/nurse_route_network.png",
            config.dataset.as_str()
        )
        .as_str(),
    )
    .expect("Failed to plot nurse route network");

    plotting::write_solution_to_file(
        &instance,
        &output.best,
        format!("src/results/{}/output.txt", config.dataset.as_str()).as_str(),
    )
    .expect("Failed to write solution text");

    solver::write_run_metrics_csv(
        &output.run_metric_snapshots,
        format!("src/results/{}/run_metrics_sampled.csv", config.dataset.as_str()).as_str(),
    )
    .expect("Failed to write sampled run metrics CSV");
}
