use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use vrp::{cli, parser, solver};

const INFEASIBLE_GAP_PENALTY_PCT: f64 = 1000.0;

#[derive(Debug, Clone)]
struct TuningArgs {
    trials: usize,
    top_k: usize,
    stage1_max_runs: usize,
    stage1_generations: usize,
    stage1_ils_iterations: usize,
    stage1_ils_local_search_steps: usize,
    stage2_max_runs: usize,
    stage2_generations: usize,
    stage2_ils_iterations: usize,
    stage2_ils_local_search_steps: usize,
    seed: u64,
    datasets: Vec<String>,
    output: String,
}

impl Default for TuningArgs {
    fn default() -> Self {
        Self {
            trials: 40,
            top_k: 10,
            stage1_max_runs: 2,
            stage1_generations: 600,
            stage1_ils_iterations: 60,
            stage1_ils_local_search_steps: 20,
            stage2_max_runs: 8,
            stage2_generations: 3000,
            stage2_ils_iterations: 250,
            stage2_ils_local_search_steps: 70,
            seed: 42,
            datasets: Vec::new(),
            output: "src/results/hyperparameter_tuning.csv".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
struct TrialResult {
    stage: &'static str,
    trial_id: usize,
    score: f64,
    mean_gap_pct: f64,
    std_gap_pct: f64,
    infeasible_rate: f64,
    mean_runtime_ms: f64,
    config: cli::RunConfig,
}

fn parse_usize_arg(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("Invalid value for {}: '{}'", flag, value))
}

fn parse_u64_arg(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("Invalid value for {}: '{}'", flag, value))
}

fn parse_dataset_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(cli::resolve_dataset_arg)
        .collect()
}

fn print_usage(bin: &str) {
    println!(
        "Usage:
  {bin} [options]

Options:
  --trials <n>                Number of random configs in stage 1 (default: 40)
  --top-k <n>                 Number of stage 1 winners to reevaluate (default: 10)
  --stage1-max-runs <n>       Max runs during stage 1 (default: 2)
  --stage1-generations <n>    Generations during stage 1 (default: 600)
  --stage1-ils-iterations <n> ILS iterations during stage 1 (default: 60)
  --stage1-ils-ls-steps <n>   ILS local-search steps during stage 1 (default: 20)
  --stage2-max-runs <n>       Max runs during stage 2 (default: 8)
  --stage2-generations <n>    Generations during stage 2 (default: 3000)
  --stage2-ils-iterations <n> ILS iterations during stage 2 (default: 250)
  --stage2-ils-ls-steps <n>   ILS local-search steps during stage 2 (default: 70)
  --datasets <csv>            Comma-separated datasets, e.g. train_0,train_1
  --seed <n>                  RNG seed (default: 42)
  --output <path>             CSV output path (default: src/results/hyperparameter_tuning.csv)
  -h, --help                  Show this help"
    );
}

fn parse_args() -> Result<TuningArgs, String> {
    let mut args_cfg = TuningArgs::default();
    let mut args = env::args();
    let bin = args.next().unwrap_or_else(|| "tune".to_string());
    let args: Vec<String> = args.collect();

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage(&bin);
                std::process::exit(0);
            }
            "--trials" => {
                i += 1;
                if i >= args.len() {
                    return Err("--trials requires a value".to_string());
                }
                args_cfg.trials = parse_usize_arg("--trials", &args[i])?;
            }
            "--top-k" => {
                i += 1;
                if i >= args.len() {
                    return Err("--top-k requires a value".to_string());
                }
                args_cfg.top_k = parse_usize_arg("--top-k", &args[i])?;
            }
            "--stage1-max-runs" => {
                i += 1;
                if i >= args.len() {
                    return Err("--stage1-max-runs requires a value".to_string());
                }
                args_cfg.stage1_max_runs = parse_usize_arg("--stage1-max-runs", &args[i])?;
            }
            "--stage1-generations" => {
                i += 1;
                if i >= args.len() {
                    return Err("--stage1-generations requires a value".to_string());
                }
                args_cfg.stage1_generations = parse_usize_arg("--stage1-generations", &args[i])?;
            }
            "--stage1-ils-iterations" => {
                i += 1;
                if i >= args.len() {
                    return Err("--stage1-ils-iterations requires a value".to_string());
                }
                args_cfg.stage1_ils_iterations =
                    parse_usize_arg("--stage1-ils-iterations", &args[i])?;
            }
            "--stage1-ils-ls-steps" => {
                i += 1;
                if i >= args.len() {
                    return Err("--stage1-ils-ls-steps requires a value".to_string());
                }
                args_cfg.stage1_ils_local_search_steps =
                    parse_usize_arg("--stage1-ils-ls-steps", &args[i])?;
            }
            "--stage2-max-runs" => {
                i += 1;
                if i >= args.len() {
                    return Err("--stage2-max-runs requires a value".to_string());
                }
                args_cfg.stage2_max_runs = parse_usize_arg("--stage2-max-runs", &args[i])?;
            }
            "--stage2-generations" => {
                i += 1;
                if i >= args.len() {
                    return Err("--stage2-generations requires a value".to_string());
                }
                args_cfg.stage2_generations = parse_usize_arg("--stage2-generations", &args[i])?;
            }
            "--stage2-ils-iterations" => {
                i += 1;
                if i >= args.len() {
                    return Err("--stage2-ils-iterations requires a value".to_string());
                }
                args_cfg.stage2_ils_iterations =
                    parse_usize_arg("--stage2-ils-iterations", &args[i])?;
            }
            "--stage2-ils-ls-steps" => {
                i += 1;
                if i >= args.len() {
                    return Err("--stage2-ils-ls-steps requires a value".to_string());
                }
                args_cfg.stage2_ils_local_search_steps =
                    parse_usize_arg("--stage2-ils-ls-steps", &args[i])?;
            }
            "--datasets" => {
                i += 1;
                if i >= args.len() {
                    return Err("--datasets requires a comma-separated value".to_string());
                }
                args_cfg.datasets = parse_dataset_list(&args[i]);
            }
            "--seed" => {
                i += 1;
                if i >= args.len() {
                    return Err("--seed requires a value".to_string());
                }
                args_cfg.seed = parse_u64_arg("--seed", &args[i])?;
            }
            "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                args_cfg.output = args[i].clone();
            }
            other => {
                return Err(format!("Unknown argument: '{}'", other));
            }
        }
        i += 1;
    }

    if args_cfg.trials == 0 {
        return Err("--trials must be > 0".to_string());
    }
    if args_cfg.top_k == 0 {
        return Err("--top-k must be > 0".to_string());
    }
    if args_cfg.stage1_max_runs == 0 || args_cfg.stage2_max_runs == 0 {
        return Err("--stage1-max-runs and --stage2-max-runs must be > 0".to_string());
    }
    if args_cfg.stage1_generations == 0 || args_cfg.stage2_generations == 0 {
        return Err("--stage1-generations and --stage2-generations must be > 0".to_string());
    }

    Ok(args_cfg)
}

fn default_train_datasets() -> Result<Vec<String>, String> {
    let mut paths = Vec::new();
    let entries = fs::read_dir("src/data/train").map_err(|e| format!("read_dir failed: {}", e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("dir entry failed: {}", e))?;
        let path = entry.path();
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        {
            paths.push(path.to_string_lossy().to_string());
        }
    }

    paths.sort();
    if paths.is_empty() {
        return Err("No training datasets found under src/data/train".to_string());
    }
    Ok(paths)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn std_dev(values: &[f64], values_mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|v| {
            let d = v - values_mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn with_budget(mut config: cli::RunConfig, max_runs: usize, generations: usize) -> cli::RunConfig {
    config.max_runs = max_runs;
    config.generations = generations;
    if config.elite_size > config.population_size {
        config.elite_size = config.population_size;
    }
    config
}

fn sample_config(rng: &mut StdRng, base: &cli::RunConfig) -> cli::RunConfig {
    const POP_CHOICES: [usize; 4] = [80, 120, 160, 200];
    const ELITE_CHOICES: [usize; 5] = [2, 4, 6, 8, 10];
    const REFINEMENT_CHOICES: [usize; 5] = [4, 6, 8, 10, 12];
    const LS_EVERY_CHOICES: [usize; 5] = [100, 250, 500, 750, 1000];
    const ILS_ITER_CHOICES: [usize; 5] = [150, 250, 300, 400, 500];

    let mut c = base.clone();
    c.population_size = POP_CHOICES[rng.random_range(0..POP_CHOICES.len())];
    c.elite_size = ELITE_CHOICES[rng.random_range(0..ELITE_CHOICES.len())].min(c.population_size);
    c.refinement_candidates = REFINEMENT_CHOICES[rng.random_range(0..REFINEMENT_CHOICES.len())];

    c.initial_mutation_rate = rng.random_range(0.6..0.99);
    c.initial_scaling_factor = rng.random_range(0.3..3.0);
    c.ls_every = LS_EVERY_CHOICES[rng.random_range(0..LS_EVERY_CHOICES.len())];
    c.ls_steps = rng.random_range(5..41);
    c.stagnation_limit = rng.random_range(300..2501);
    c.ils_iterations = ILS_ITER_CHOICES[rng.random_range(0..ILS_ITER_CHOICES.len())];
    c.ils_local_search_steps = rng.random_range(40..121);

    c
}

fn evaluate_config(datasets: &[String], config: &cli::RunConfig) -> (f64, f64, f64, f64) {
    let mut gaps = Vec::with_capacity(datasets.len());
    let mut infeasible_rates = Vec::with_capacity(datasets.len());
    let mut runtimes_ms = Vec::with_capacity(datasets.len());

    let options = solver::SolveOptions {
        early_stop: false,
        verbose: false,
        save_refinement_plots: false,
        progress_log_interval: 100,
        metrics_sample_interval: 100,
    };

    for dataset_path in datasets {
        let instance = parser::read_json(dataset_path);
        let start = Instant::now();
        let output = solver::solve(&instance, config, &options);
        runtimes_ms.push(start.elapsed().as_secs_f64() * 1000.0);

        let feasible_runs = output
            .best_feasible_per_run
            .iter()
            .filter(|&&travel_time| travel_time > 0.0)
            .count();
        let infeasible_rate = 1.0 - feasible_runs as f64 / config.max_runs as f64;
        infeasible_rates.push(infeasible_rate);

        let gap_pct = if output.best.feasible {
            100.0 * (output.best.travel_time - instance.benchmark) / instance.benchmark
        } else {
            INFEASIBLE_GAP_PENALTY_PCT
        };
        gaps.push(gap_pct);
    }

    let mean_gap_pct = mean(&gaps);
    let std_gap_pct = std_dev(&gaps, mean_gap_pct);
    let infeasible_rate = mean(&infeasible_rates);
    let mean_runtime_ms = mean(&runtimes_ms);

    let score = 1000.0 * infeasible_rate + mean_gap_pct + 0.2 * std_gap_pct;
    (score, mean_gap_pct, std_gap_pct, mean_runtime_ms)
}

fn write_results_csv(
    output_path: &str,
    datasets: &[String],
    stage1: &[TrialResult],
    stage2: &[TrialResult],
) -> Result<(), String> {
    if let Some(parent) = Path::new(output_path).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    let mut file = fs::File::create(output_path)
        .map_err(|e| format!("Failed to create output CSV {}: {}", output_path, e))?;

    writeln!(
        file,
        "stage,trial_id,score,mean_gap_pct,std_gap_pct,infeasible_rate,mean_runtime_ms,max_runs,population_size,generations,refinement_candidates,ils_iterations,ils_local_search_steps,elite_size,initial_mutation_rate,initial_scaling_factor,ls_every,ls_steps,stagnation_limit,datasets"
    )
    .map_err(|e| format!("Failed to write CSV header: {}", e))?;

    let datasets_joined = datasets.join("|");
    for trial in stage1.iter().chain(stage2.iter()) {
        writeln!(
            file,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.3},{},{},{},{},{},{},{},{:.6},{:.6},{},{},{},{}",
            trial.stage,
            trial.trial_id,
            trial.score,
            trial.mean_gap_pct,
            trial.std_gap_pct,
            trial.infeasible_rate,
            trial.mean_runtime_ms,
            trial.config.max_runs,
            trial.config.population_size,
            trial.config.generations,
            trial.config.refinement_candidates,
            trial.config.ils_iterations,
            trial.config.ils_local_search_steps,
            trial.config.elite_size,
            trial.config.initial_mutation_rate,
            trial.config.initial_scaling_factor,
            trial.config.ls_every,
            trial.config.ls_steps,
            trial.config.stagnation_limit,
            datasets_joined,
        )
        .map_err(|e| format!("Failed to write CSV row: {}", e))?;
    }

    Ok(())
}

fn main() {
    let args = parse_args().unwrap_or_else(|err| {
        eprintln!("Error: {}", err);
        std::process::exit(2);
    });

    let datasets = if args.datasets.is_empty() {
        default_train_datasets().unwrap_or_else(|err| {
            eprintln!("Error: {}", err);
            std::process::exit(2);
        })
    } else {
        args.datasets.clone()
    };

    println!(
        "Tuning datasets ({}): {}",
        datasets.len(),
        datasets.join(", ")
    );
    println!(
        "Stage 1: trials={}, max_runs={}, generations={}, ils_iterations={}, ils_local_search_steps={}",
        args.trials,
        args.stage1_max_runs,
        args.stage1_generations,
        args.stage1_ils_iterations,
        args.stage1_ils_local_search_steps,
    );
    println!(
        "Stage 2: top_k={}, max_runs={}, generations={}, ils_iterations={}, ils_local_search_steps={}",
        args.top_k,
        args.stage2_max_runs,
        args.stage2_generations,
        args.stage2_ils_iterations,
        args.stage2_ils_local_search_steps,
    );

    let base_config = cli::RunConfig::default();
    let mut rng = StdRng::seed_from_u64(args.seed);

    let mut stage1_results = Vec::with_capacity(args.trials);
    for trial_id in 1..=args.trials {
        let sampled = sample_config(&mut rng, &base_config);
        let mut config = with_budget(sampled, args.stage1_max_runs, args.stage1_generations);
        config.ils_iterations = args.stage1_ils_iterations;
        config.ils_local_search_steps = args.stage1_ils_local_search_steps;
        config.refinement_candidates = config.refinement_candidates.min(5);
        let (score, mean_gap_pct, std_gap_pct, mean_runtime_ms) =
            evaluate_config(&datasets, &config);

        let trial = TrialResult {
            stage: "stage1",
            trial_id,
            score,
            mean_gap_pct,
            std_gap_pct,
            infeasible_rate: (score - mean_gap_pct - 0.2 * std_gap_pct) / 1000.0,
            mean_runtime_ms,
            config,
        };

        println!(
            "Stage 1 trial {:>3}/{:>3}: score={:.3}, mean_gap={:.3}%, infeasible_rate={:.3}",
            trial_id, args.trials, trial.score, trial.mean_gap_pct, trial.infeasible_rate
        );
        stage1_results.push(trial);
    }

    stage1_results.sort_by(|a, b| a.score.total_cmp(&b.score));
    let top_k = args.top_k.min(stage1_results.len());

    let mut stage2_results = Vec::with_capacity(top_k);
    for (idx, seed_trial) in stage1_results.iter().take(top_k).enumerate() {
        let config = with_budget(
            seed_trial.config.clone(),
            args.stage2_max_runs,
            args.stage2_generations,
        );
        let mut config = config;
        config.ils_iterations = args.stage2_ils_iterations;
        config.ils_local_search_steps = args.stage2_ils_local_search_steps;

        let (score, mean_gap_pct, std_gap_pct, mean_runtime_ms) =
            evaluate_config(&datasets, &config);
        let trial = TrialResult {
            stage: "stage2",
            trial_id: idx + 1,
            score,
            mean_gap_pct,
            std_gap_pct,
            infeasible_rate: (score - mean_gap_pct - 0.2 * std_gap_pct) / 1000.0,
            mean_runtime_ms,
            config,
        };

        println!(
            "Stage 2 candidate {:>2}/{:>2}: score={:.3}, mean_gap={:.3}%, infeasible_rate={:.3}",
            idx + 1,
            top_k,
            trial.score,
            trial.mean_gap_pct,
            trial.infeasible_rate
        );
        stage2_results.push(trial);
    }

    stage2_results.sort_by(|a, b| a.score.total_cmp(&b.score));
    let best = stage2_results
        .first()
        .unwrap_or_else(|| stage1_results.first().expect("No trials executed"));

    if let Err(err) = write_results_csv(&args.output, &datasets, &stage1_results, &stage2_results) {
        eprintln!("Error: {}", err);
        std::process::exit(2);
    }

    println!("\nBest config (score {:.4}):", best.score);
    println!(
        "--max-runs {} --population-size {} --generations {} --refinement-candidates {} --ils-iterations {} --ils-local-search-steps {} --elite-size {} --initial-mutation-rate {:.6} --initial-scaling-factor {:.6} --ls-every {} --ls-steps {} --stagnation-limit {}",
        best.config.max_runs,
        best.config.population_size,
        best.config.generations,
        best.config.refinement_candidates,
        best.config.ils_iterations,
        best.config.ils_local_search_steps,
        best.config.elite_size,
        best.config.initial_mutation_rate,
        best.config.initial_scaling_factor,
        best.config.ls_every,
        best.config.ls_steps,
        best.config.stagnation_limit
    );
    println!("Saved tuning results to {}", args.output);
}
