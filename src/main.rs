mod read_json;

use bitvec::prelude::*;
use read_json::{read_json, get_travel_time};

fn main() {
    let instance = read_json("src/train/train_0.json");
    println!("Instance name: {}", instance.instance_name);
    println!("Number of nurses: {}", instance.nbr_nurses);
    println!("Nurse capacity: {}", instance.capacity_nurse);
    println!("Benchmark: {}", instance.benchmark);
    println!("Depot: return_time={}, x_coord={}, y_coord={}", instance.depot.return_time, instance.depot.x_coord, instance.depot.y_coord);
    println!("Number of patients: {}", instance.patients.len());
    for (i, patient) in instance.patients.iter().enumerate() {
        println!("Patient {}: x_coord={}, y_coord={}, demand={}, start_time={}, end_time={}, care_time={}", i, patient.x_coord, patient.y_coord, patient.demand, patient.start_time, patient.end_time, patient.care_time);
    }
    println!("Travel time from depot to patient 0: {}", get_travel_time(&instance.travel_times, 0, 1));
}