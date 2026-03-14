use std::{env, str::FromStr};

const MAX_RUNS: usize = 10;
const POPULATION_SIZE: usize = 120;
const GENERATIONS: usize = 3000;
const REFINEMENT_CANDIDATES: usize = 8;
const ILS_ITERATIONS: usize = 300;
const ILS_LOCAL_SEARCH_STEPS: usize = 90;
const ELITE_SIZE: usize = 4;
const INITIAL_MUTATION_RATE: f64 = 0.95;
const INITIAL_SCALING_FACTOR: f64 = 1.0;
const LS_EVERY: usize = 500;
const LS_STEPS: usize = 20;
const STAGNATION_LIMIT: usize = 1000;
const DEFAULT_DATASET: &str = "test_1";

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub max_runs: usize,
    pub population_size: usize,
    pub generations: usize,
    pub refinement_candidates: usize,
    pub ils_iterations: usize,
    pub ils_local_search_steps: usize,
    pub elite_size: usize,
    pub initial_mutation_rate: f64,
    pub initial_scaling_factor: f64,
    pub ls_every: usize,
    pub ls_steps: usize,
    pub stagnation_limit: usize,
    pub dataset: String,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            max_runs: MAX_RUNS,
            population_size: POPULATION_SIZE,
            generations: GENERATIONS,
            refinement_candidates: REFINEMENT_CANDIDATES,
            ils_iterations: ILS_ITERATIONS,
            ils_local_search_steps: ILS_LOCAL_SEARCH_STEPS,
            elite_size: ELITE_SIZE,
            initial_mutation_rate: INITIAL_MUTATION_RATE,
            initial_scaling_factor: INITIAL_SCALING_FACTOR,
            ls_every: LS_EVERY,
            ls_steps: LS_STEPS,
            stagnation_limit: STAGNATION_LIMIT,
            dataset: String::from_str(DEFAULT_DATASET).unwrap(),
        }
    }
}

fn parse_usize_arg(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("Invalid value for {}: '{}'", flag, value))
}

fn parse_f64_arg(flag: &str, value: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("Invalid value for {}: '{}'", flag, value))
}

pub fn resolve_dataset_arg(dataset: &str) -> String {
    if dataset.contains('/') || dataset.ends_with(".json") {
        return dataset.to_string();
    }

    if dataset.starts_with("test_") {
        return format!("src/data/test/{}.json", dataset);
    }
    if dataset.starts_with("train_") {
        return format!("src/data/train/{}.json", dataset);
    }

    dataset.to_string()
}

pub fn print_usage(bin: &str) {
    println!(
        "Usage:
  {bin} [dataset_name_or_path] [options]

Options:
  --dataset <name|path>               Dataset short name (e.g., test_1) or JSON path (default: {DEFAULT_DATASET})
  --max-runs <n>                      Max runs/restarts (default: {MAX_RUNS})
  --population-size <n>               Population size (default: {POPULATION_SIZE})
  --generations <n>                   Generation budget (default: {GENERATIONS})
  --refinement-candidates <n>         Number of refinement candidates (default: {REFINEMENT_CANDIDATES})
  --ils-iterations <n>                ILS no-improvement iteration cap (default: {ILS_ITERATIONS})
  --ils-local-search-steps <n>        Local-search steps per ILS call (default: {ILS_LOCAL_SEARCH_STEPS})
  --elite-size <n>                    Elites kept each generation (default: {ELITE_SIZE})
  --initial-mutation-rate <x>         Initial mutation rate (default: {INITIAL_MUTATION_RATE})
  --initial-scaling-factor <x>        Initial scaling factor (default: {INITIAL_SCALING_FACTOR})
  --ls-every <n>                      Run local search every N generations (default: {LS_EVERY})
  --ls-steps <n>                      Local-search steps during GA local-search phase (default: {LS_STEPS})
  --stagnation-limit <n>              Stop after N non-improving generations (default: {STAGNATION_LIMIT})
  -h, --help                          Show this help"
    );
}

pub fn parse_cli_args() -> Result<(String, RunConfig), String> {
    let mut dataset = DEFAULT_DATASET.to_string();
    let mut config = RunConfig::default();

    let mut args = env::args();
    let bin = args.next().unwrap_or_else(|| "vrp".to_string());
    let args: Vec<String> = args.collect();

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage(&bin);
                std::process::exit(0);
            }
            "--dataset" => {
                i += 1;
                if i >= args.len() {
                    return Err("--dataset requires a value".to_string());
                }
                dataset = args[i].clone();
                config.dataset = dataset.clone();
            }
            "--max-runs" => {
                i += 1;
                if i >= args.len() {
                    return Err("--max-runs requires a value".to_string());
                }
                config.max_runs = parse_usize_arg("--max-runs", &args[i])?;
            }
            "--population-size" => {
                i += 1;
                if i >= args.len() {
                    return Err("--population-size requires a value".to_string());
                }
                config.population_size = parse_usize_arg("--population-size", &args[i])?;
            }
            "--max-generations" => {
                i += 1;
                if i >= args.len() {
                    return Err("--max-generations requires a value".to_string());
                }
                config.generations = parse_usize_arg("--max-generations", &args[i])?;
            }
            "--refinement-candidates" => {
                i += 1;
                if i >= args.len() {
                    return Err("--refinement-candidates requires a value".to_string());
                }
                config.refinement_candidates =
                    parse_usize_arg("--refinement-candidates", &args[i])?;
            }
            "--ils-iterations" => {
                i += 1;
                if i >= args.len() {
                    return Err("--ils-iterations requires a value".to_string());
                }
                config.ils_iterations = parse_usize_arg("--ils-iterations", &args[i])?;
            }
            "--ils-local-search-steps" => {
                i += 1;
                if i >= args.len() {
                    return Err("--ils-local-search-steps requires a value".to_string());
                }
                config.ils_local_search_steps =
                    parse_usize_arg("--ils-local-search-steps", &args[i])?;
            }
            "--elite-size" => {
                i += 1;
                if i >= args.len() {
                    return Err("--elite-size requires a value".to_string());
                }
                config.elite_size = parse_usize_arg("--elite-size", &args[i])?;
            }
            "--initial-mutation-rate" => {
                i += 1;
                if i >= args.len() {
                    return Err("--initial-mutation-rate requires a value".to_string());
                }
                config.initial_mutation_rate = parse_f64_arg("--initial-mutation-rate", &args[i])?;
            }
            "--initial-scaling-factor" => {
                i += 1;
                if i >= args.len() {
                    return Err("--initial-scaling-factor requires a value".to_string());
                }
                config.initial_scaling_factor =
                    parse_f64_arg("--initial-scaling-factor", &args[i])?;
            }
            "--ls-every" => {
                i += 1;
                if i >= args.len() {
                    return Err("--ls-every requires a value".to_string());
                }
                config.ls_every = parse_usize_arg("--ls-every", &args[i])?;
            }
            "--ls-steps" => {
                i += 1;
                if i >= args.len() {
                    return Err("--ls-steps requires a value".to_string());
                }
                config.ls_steps = parse_usize_arg("--ls-steps", &args[i])?;
            }
            "--stagnation-limit" => {
                i += 1;
                if i >= args.len() {
                    return Err("--stagnation-limit requires a value".to_string());
                }
                config.stagnation_limit = parse_usize_arg("--stagnation-limit", &args[i])?;
            }
            maybe_dataset if !maybe_dataset.starts_with('-') => {
                dataset = maybe_dataset.to_string();
            }
            other => {
                return Err(format!("Unknown argument: '{}'", other));
            }
        }
        i += 1;
    }

    if config.max_runs == 0 {
        return Err("--max-runs must be > 0".to_string());
    }
    if config.population_size == 0 {
        return Err("--population-size must be > 0".to_string());
    }
    if config.refinement_candidates == 0 {
        return Err("--refinement-candidates must be > 0".to_string());
    }
    if config.elite_size > config.population_size {
        return Err("--elite-size must be <= --population-size".to_string());
    }
    if !(0.0..=1.0).contains(&config.initial_mutation_rate) {
        return Err("--initial-mutation-rate must be between 0 and 1".to_string());
    }
    if config.initial_scaling_factor <= 0.0 {
        return Err("--initial-scaling-factor must be > 0".to_string());
    }
    if config.ls_every == 0 {
        return Err("--ls-every must be > 0".to_string());
    }
    if config.stagnation_limit == 0 {
        return Err("--stagnation-limit must be > 0".to_string());
    }

    Ok((resolve_dataset_arg(&dataset), config))
}
