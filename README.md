# Vehicle Routing Problem (Nurse Scheduling)

Rust implementation of a genetic algorithm + iterated local search (ILS) solver for a nurse routing/scheduling style VRP.

## Requirements

- Rust (stable)
- Cargo

## Build

```bash
cargo build
```

## Run Solver

The main binary is `vehicleRoutingProblem`.

```bash
cargo run --release -- --help
```

You can pass datasets as shorthand names:

- `test_1` -> `src/data/test/test_1.json`
- `train_3` -> `src/data/train/train_3.json`

Example:

```bash
cargo run --release -- test_1 --max-runs 1 --population-size 120 --max-generations 3000
```

Params from hyperparameter search:
```bash
cargo run --release -- \
  --dataset test_1 \
  --max-generations 10000 \
  --population-size 200 \
  --max-runs 1 \
  --refinement-candidates 5 \
  --elite-size 10 \
  --initial-mutation-rate 0.852641 \
  --initial-scaling-factor 2.092114 \
  --ls-every 100 \
  --ls-steps 24 \
  --ils-iterations 250 \
  --ils-local-search-steps 70 \
  --stagnation-limit 2156
```

### Solver CLI Parameters

- `--dataset <name|path>`
- `--max-runs <n>`
- `--population-size <n>`
- `--max-generations <n>`
- `--refinement-candidates <n>`
- `--ils-iterations <n>`
- `--ils-local-search-steps <n>`
- `--elite-size <n>`
- `--initial-mutation-rate <x>`
- `--initial-scaling-factor <x>`
- `--ls-every <n>`
- `--ls-steps <n>`
- `--stagnation-limit <n>`

## Hyperparameter Tuning

A dedicated tuner binary is available:

```bash
cargo run --bin tune -- --help
```

Example (two-stage search):

```bash
cargo run --release --bin tune -- \
  --trials 40 \
  --top-k 10 \
  --stage1-max-runs 2 \
  --stage1-generations 600 \
  --stage1-ils-iterations 60 \
  --stage1-ils-ls-steps 20 \
  --stage2-max-runs 8 \
  --stage2-generations 3000 \
  --stage2-ils-iterations 250 \
  --stage2-ils-ls-steps 70
```

Optional datasets input:

```bash
--datasets train_0,train_1,train_2
```

If `--datasets` is omitted, all JSON files in `src/data/train/` are used.

Tuning results are written to:

- `src/results/hyperparameter_tuning.csv` (default)

You can override with:

```bash
--output /path/to/results.csv
```

## Project Structure

- `src/main.rs` - solver entrypoint
- `src/bin/tune.rs` - hyperparameter tuning entrypoint
- `src/solver.rs` - core GA + ILS solving pipeline
- `src/cli.rs` - CLI argument parsing and defaults
- `src/parser.rs` - dataset JSON parsing
- `src/plotting.rs` - all plotting utilities
