mod read_json;

use read_json::{get_travel_time, read_json};

struct Genome {
    sequence: Vec<usize>,
    lengths: Vec<usize>,
    fitness: f64,
}

impl Genome {
    fn new(sequence: Vec<usize>, lengths: Vec<usize>, fitness: f64) -> Self {
        Genome {
            sequence,
            lengths,
            fitness,
        }
    }

    fn calculate_fitness(&mut self, instance: &read_json::Instance) {
        // Check if all patients are visited within their time windows, check nurse load, calculate total travel time, and check if all patients have been visited

        let n_patients = instance.patients.len();
        let mut total_travel_time = 0;
        let mut nurse = 0;
        let mut seen = vec![false; n_patients];

        if self.sequence.len() != n_patients {
            self.fitness = 0.0;
            return;
        }

        if self.lengths.len() != instance.nbr_nurses {
            self.fitness = 0.0;
            return;
        }

        if self.lengths.iter().sum::<usize>() != n_patients {
            self.fitness = 0.0;
            return;
        }

        while nurse < instance.nbr_nurses {
            let mut total_time = 0;
            let mut load = 0;
            let mut current_location = 0;
            let mut total_nurse_travel_time = 0;

            let mut gene_index: usize = self.lengths[..nurse].iter().sum::<usize>();
            let gene_end_index = gene_index + self.lengths[nurse];
            while gene_index < gene_end_index {
                let patient = self.sequence[gene_index];
                let patient_node = patient + 1; // Because 0 is the depot

                // Check if patient has already been seen
                if patient >= n_patients || seen[patient] {
                    self.fitness = 0.0;
                    return;
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
                if total_time < patient_info.start_time {
                    total_time = patient_info.start_time;
                }

                // If we finish care after the patient end time.
                total_time += patient_info.care_time;
                if total_time > patient_info.end_time {
                    self.fitness = 0.0;
                    return;
                }

                // If we exceed nurse capacity.
                load += patient_info.demand;
                if load > instance.capacity_nurse {
                    self.fitness = 0.0;
                    return;
                }

                gene_index += 1;
            }

            total_travel_time += total_nurse_travel_time;

            // If we cant make it back to the depot on time.
            if total_time + get_travel_time(&instance.travel_times, current_location, 0)
                > instance.depot.return_time
            {
                self.fitness = 0.0;
                return;
            }

            total_travel_time += get_travel_time(&instance.travel_times, current_location, 0);

            nurse += 1;
        }

        // Check if all patients have been visited
        if !seen.iter().all(|&b| b) {
            self.fitness = 0.0;
            return;
        }

        // Calculate fitness.
        self.fitness = 1.0 / (1.0 + (total_travel_time as f64));
    }
}

fn main() {
    let instance = read_json("src/train/train_0.json");
    println!("Instance name: {}", instance.instance_name);
    println!("Number of nurses: {}", instance.nbr_nurses);
    println!("Nurse capacity: {}", instance.capacity_nurse);
    println!("Benchmark: {}", instance.benchmark);
    println!(
        "Depot: return_time={}, x_coord={}, y_coord={}",
        instance.depot.return_time, instance.depot.x_coord, instance.depot.y_coord
    );
    println!("Number of patients: {}", instance.patients.len());
    for (i, patient) in instance.patients.iter().enumerate() {
        println!(
            "Patient {}: x_coord={}, y_coord={}, demand={}, start_time={}, end_time={}, care_time={}",
            i,
            patient.x_coord,
            patient.y_coord,
            patient.demand,
            patient.start_time,
            patient.end_time,
            patient.care_time
        );
    }
    println!(
        "Travel time from depot to patient 0: {}",
        get_travel_time(&instance.travel_times, 0, 1)
    );
}
