use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
pub struct Depot {
    pub return_time: usize,
    pub x_coord: usize,
    pub y_coord: usize,
}

#[derive(Deserialize)]
pub struct Patient {
    pub x_coord: usize,
    pub y_coord: usize,
    pub demand: usize,
    pub start_time: usize,
    pub end_time: usize,
    pub care_time: usize,
}

#[derive(Deserialize)]
struct JSONInstance {
    instance_name: String,
    nbr_nurses: usize,
    capacity_nurse: usize,
    benchmark: f64,
    depot: Depot,
    patients: BTreeMap<usize, Patient>,
    travel_times: Vec<Vec<f64>>,
}

pub struct TravelTimes {
    pub times: Vec<f64>,
    pub columns: usize,
}

pub struct Instance {
    pub instance_name: String,
    pub nbr_nurses: usize,
    pub capacity_nurse: usize,
    pub benchmark: f64,
    pub depot: Depot,
    pub patients: Vec<Patient>,
    pub travel_times: TravelTimes,
}

fn flatten_travel_times(travel_times: Vec<Vec<f64>>) -> TravelTimes {
    let columns = travel_times[0].len();
    let times = travel_times.into_iter().flatten().collect();
    TravelTimes { times, columns }
}

pub fn get_travel_time(travel_times: &TravelTimes, from: usize, to: usize) -> f64 {
    travel_times.times[from * travel_times.columns + to]
}

pub fn read_json(path: &str) -> Instance {
    let text = std::fs::read_to_string(path).expect("Failed to read file");
    let json_instance: JSONInstance = serde_json::from_str(&text).expect("Failed to parse JSON");
    Instance {
        instance_name: json_instance.instance_name,
        nbr_nurses: json_instance.nbr_nurses,
        capacity_nurse: json_instance.capacity_nurse,
        benchmark: json_instance.benchmark,
        depot: json_instance.depot,
        patients: json_instance.patients.into_values().collect(),
        travel_times: flatten_travel_times(json_instance.travel_times),
    }
}
