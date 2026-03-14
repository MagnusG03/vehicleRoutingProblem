use crate::Genome;
use crate::parser;
use plotters::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::Write;

pub fn plot_refinement_travel_time(
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

pub fn plot_best_refinement_by_run(
    histories: &[(usize, Vec<f64>)],
    target_travel_time: f64,
    output_path: &str,
    dataset_name: &str
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
        .caption(format!("Best-Candidate Refinement by Run ({})", dataset_name), ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(65)
        .build_cartesian_2d(
            0usize..max_len,
            (travel_min - travel_padding)..(travel_max + travel_padding),
        )?;

    chart
        .configure_mesh()
        .x_desc("ILS Iteration")
        .y_desc("Best Travel Time")
        .draw()?;

    for (idx, (run_number, history)) in histories.iter().enumerate() {
        if history.is_empty() {
            continue;
        }
        chart
            .draw_series(LineSeries::new(
                history.iter().enumerate().map(|(i, v)| (i, *v)),
                &Palette99::pick(idx),
            ))?
            .label(format!("Run {}", run_number))
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

pub fn plot_nurse_route_network(
    instance: &parser::Instance,
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
        .build_cartesian_2d(
            (min_x - pad_x)..(max_x + pad_x),
            (min_y - pad_y)..(max_y + pad_y),
        )?;

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

        chart.draw_series(LineSeries::new(
            polyline.iter().copied(),
            ShapeStyle::from(&BLACK).stroke_width(4),
        ))?;

        chart
            .draw_series(LineSeries::new(
                polyline.iter().copied(),
                ShapeStyle::from(&color).stroke_width(2),
            ))?
            .label(format!("Nurse {}", nurse + 1))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 18, y)], Palette99::pick(nurse))
            });

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

pub fn plot_metrics(
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

pub fn plot_best_travel_times_by_run(
    best_feasible_per_run: &[f64],
    benchmark: f64,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    if best_feasible_per_run.is_empty() {
        return Ok(());
    }

    let points: Vec<(usize, f64)> = best_feasible_per_run
        .iter()
        .enumerate()
        .filter_map(|(run_idx, &travel_time)| {
            if travel_time > 0.0 {
                Some((run_idx + 1, travel_time))
            } else {
                None
            }
        })
        .collect();

    if points.is_empty() {
        return Ok(());
    }

    let travel_min = points
        .iter()
        .map(|(_, travel_time)| *travel_time)
        .fold(f64::INFINITY, f64::min)
        .min(benchmark);
    let travel_max = points
        .iter()
        .map(|(_, travel_time)| *travel_time)
        .fold(f64::NEG_INFINITY, f64::max)
        .max(benchmark);
    let travel_padding = if (travel_max - travel_min).abs() < f64::EPSILON {
        (travel_max.abs() * 0.01).max(1e-12)
    } else {
        0.0
    };

    let x_max = best_feasible_per_run.len() + 1;
    let root = BitMapBackend::new(output_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Best Feasible Travel Time by Run", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(
            1usize..x_max,
            (travel_min - travel_padding)..(travel_max + travel_padding),
        )?;

    chart
        .configure_mesh()
        .x_desc("Run")
        .y_desc("Best Feasible Travel Time")
        .draw()?;

    chart
        .draw_series(LineSeries::new(points.iter().copied(), &BLUE))?
        .label("Best feasible by run")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart.draw_series(
        points
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 4, BLUE.filled())),
    )?;

    chart
        .draw_series(LineSeries::new(
            (1..x_max).map(|run_idx| (run_idx, benchmark)),
            &BLACK.mix(0.5),
        ))?
        .label("Benchmark")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.mix(0.5)));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn print_solution(instance: &parser::Instance, best: &Genome) {
    println!("\n=== Final Solution ===");
    println!("Instance: {}", instance.instance_name);
    println!("Nurse capacity: {}", instance.capacity_nurse);
    println!("Depot return time: {}", instance.depot.return_time);
    println!("\n----------------------");
    for nurse in 0..best.lengths.len() {
        let start = best.lengths[..nurse].iter().sum::<usize>();
        let end = start + best.lengths[nurse];
        let route = &best.sequence[start..end];
        let mut total_time: f64 = 0.0;
        let mut current_location = 0usize;
        let mut patient_sequence_with_times = Vec::with_capacity(route.len());

        for &patient_idx in route {
            let patient = &instance.patients[patient_idx];
            let patient_node = patient_idx + 1;
            let travel_time = instance.get_travel_time(current_location, patient_node);
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
}

pub fn write_solution_to_file(
    instance: &parser::Instance,
    best: &Genome,
    file_path: &str,
) -> std::io::Result<()> {
    let mut file = File::create(file_path)?;

    writeln!(file, "\n=== Final Solution ===")?;
    writeln!(file, "Instance: {}", instance.instance_name)?;
    writeln!(file, "Nurse capacity: {}", instance.capacity_nurse)?;
    writeln!(file, "Depot return time: {}", instance.depot.return_time)?;
    writeln!(file, "\n----------------------")?;

    for nurse in 0..best.lengths.len() {
        let start = best.lengths[..nurse].iter().sum::<usize>();
        let end = start + best.lengths[nurse];
        let route = &best.sequence[start..end];

        let mut total_time: f64 = 0.0;
        let mut current_location = 0usize;
        let mut patient_sequence_with_times = Vec::with_capacity(route.len());

        for &patient_idx in route {
            let patient = &instance.patients[patient_idx];
            let patient_node = patient_idx + 1;

            let travel_time = instance.get_travel_time(current_location, patient_node);
            total_time += travel_time;

            let visit_time = total_time.max(patient.start_time as f64);
            let leave_time = visit_time + patient.care_time as f64;

            patient_sequence_with_times.push(format!(
                "{:02} ({:.1} - {:.1}) [{:.1} - {:.1}]",
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

        writeln!(
            file,
            "Nurse {:2}:  Route duration: {:>6.1}  Covered demand: {:3}  Patient sequence: {}",
            nurse + 1,
            best.nurse_travel_times[nurse],
            best.nurse_covered_demands[nurse],
            patient_sequence_text
        )?;
    }

    writeln!(file, "----------------------\n")?;
    writeln!(file, "Total travel time: {}", best.travel_time)?;

    Ok(())
}
