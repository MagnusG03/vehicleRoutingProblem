use rand::RngExt;
use rand::rngs::StdRng;

use crate::crossover;
use crate::mutation;
use crate::parser;
use crate::repr::{Genome, get_elites};

pub fn generalized_crowding_select(
    child: Genome,
    adult: Genome,
    scaling_factor: f64,
    rng: &mut impl rand::Rng,
) -> Genome {
    let child_prob = if child.fitness > adult.fitness {
        child.fitness / (child.fitness + adult.fitness * scaling_factor)
    } else {
        (scaling_factor * child.fitness) / (scaling_factor * child.fitness + adult.fitness)
    };

    if rng.random_bool(child_prob) {
        child
    } else {
        adult
    }
}

#[allow(dead_code)]
pub fn elitism_generalized_crowding_repopulation(
    population: &mut Vec<Genome>,
    instance: &parser::Instance,
    mutation_rate: f64,
    elite_size: usize,
    scaling_factor: f64,
    population_size: usize,
    rng: &mut StdRng,
) {
    let mut new_population = Vec::with_capacity(population_size);

    new_population.extend(get_elites(population, elite_size));

    while new_population.len() < population_size {
        let parent1 = population[rng.random_range(0..population.len())].clone();
        let parent2 = population[rng.random_range(0..population.len())].clone();

        let mut child1 = crossover::edge_recombination(&parent1, &parent2, instance, rng);
        mutation::random_mutate(&mut child1, 1, mutation_rate, rng);
        child1.calculate_fitness(instance);

        let mut child2 = crossover::edge_recombination(&parent2, &parent1, instance, rng);
        mutation::random_mutate(&mut child2, 1, mutation_rate, rng);
        child2.calculate_fitness(instance);

        if child1.genome_difference(&parent1) + child2.genome_difference(&parent2)
            < child1.genome_difference(&parent2) + child2.genome_difference(&parent1)
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
