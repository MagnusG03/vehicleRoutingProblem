use serde::de;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
pub struct Depot {
    #[serde(deserialize_with = "deserialize_usize_from_number")]
    pub return_time: usize,
    #[serde(deserialize_with = "deserialize_usize_from_number")]
    pub x_coord: usize,
    #[serde(deserialize_with = "deserialize_usize_from_number")]
    pub y_coord: usize,
}

#[derive(Deserialize)]
pub struct Patient {
    #[serde(deserialize_with = "deserialize_usize_from_number")]
    pub x_coord: usize,
    #[serde(deserialize_with = "deserialize_usize_from_number")]
    pub y_coord: usize,
    #[serde(deserialize_with = "deserialize_usize_from_number")]
    pub demand: usize,
    #[serde(deserialize_with = "deserialize_usize_from_number")]
    pub start_time: usize,
    #[serde(deserialize_with = "deserialize_usize_from_number")]
    pub end_time: usize,
    #[serde(deserialize_with = "deserialize_usize_from_number")]
    pub care_time: usize,
}

#[derive(Deserialize)]
struct JSONInstance {
    instance_name: String,
    nbr_nurses: usize,
    capacity_nurse: usize,
    benchmark: f64,
    depot: Depot,
    #[serde(deserialize_with = "deserialize_patients_map")]
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

fn deserialize_usize_from_number<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                if u <= usize::MAX as u64 {
                    Ok(u as usize)
                } else {
                    Err(de::Error::custom("Integer too large for usize"))
                }
            } else if let Some(f) = n.as_f64() {
                if f.is_finite() && f >= 0.0 && f.fract() == 0.0 && f <= usize::MAX as f64 {
                    Ok(f as usize)
                } else {
                    Err(de::Error::custom("Expected non-negative integral number for usize"))
                }
            } else {
                Err(de::Error::custom("Invalid numeric value for usize"))
            }
        }
        _ => Err(de::Error::custom("Expected numeric value for usize")),
    }
}

fn parse_usize_key(key: &str) -> Result<usize, String> {
    if let Ok(u) = key.parse::<usize>() {
        return Ok(u);
    }

    if let Ok(f) = key.parse::<f64>() {
        if f.is_finite() && f >= 0.0 && f.fract() == 0.0 && f <= usize::MAX as f64 {
            return Ok(f as usize);
        }
    }

    Err(format!("Invalid patient id key `{key}`; expected integer or integral float"))
}

fn deserialize_patients_map<'de, D>(deserializer: D) -> Result<BTreeMap<usize, Patient>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = BTreeMap::<String, Patient>::deserialize(deserializer)?;
    raw.into_iter()
        .map(|(k, v)| parse_usize_key(&k).map(|parsed| (parsed, v)))
        .collect::<Result<BTreeMap<usize, Patient>, String>>()
        .map_err(de::Error::custom)
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
