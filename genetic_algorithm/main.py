from pathlib import Path
import random

import numpy as np

from genetic_algorithm.chromosome import (
    chromosome_to_gait,
    LOWER_LIMITS,
    UPPER_LIMITS,
)
from genetic_algorithm.population import (
    create_population,
    MATLAB_STANDING_POSE,
)
from genetic_algorithm.fitness_function import (
    fitness_function,
    get_ground_z,
)
from genetic_algorithm.updated_crossover import updated_crossover
from genetic_algorithm.updated_mutation import updated_mutate
from genetic_algorithm.selection import tournament_selection
from genetic_algorithm.crossover import crossover
from genetic_algorithm.mutation import mutate
from genetic_algorithm.visualisation import plot_fitness

POPULATION_SIZE = 20
GENERATIONS = 30
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.05


def main():
    random.seed(42)

    population = create_population(size=POPULATION_SIZE)

    # Keep the same ground plane throughout the search.
    ground_z = get_ground_z(MATLAB_STANDING_POSE)

    best_scores = []
    average_scores = []

    # One bound per gene, repeating the 24 joint bounds each frame.
    frames = chromosome_to_gait(population[0]).shape[0]
    lower_bounds = np.tile(LOWER_LIMITS, frames)
    upper_bounds = np.tile(UPPER_LIMITS, frames)

    for generation in range(GENERATIONS):
        scored_population = []

        for chromosome in population:
            try:
                score = float(fitness_function(chromosome, ground_z))
            except ValueError:
                # Undefined kinematic poses are invalid candidates.
                score = -np.inf

            if not np.isfinite(score):
                score = -np.inf

            scored_population.append((score, chromosome))

        # Highest fitness first.
        scored_population.sort(key=lambda item: item[0], reverse=True)

        best_score, best_chromosome = scored_population[0]
        scores = np.array([item[0] for item in scored_population])
        valid_scores = scores[np.isfinite(scores)]

        if valid_scores.size == 0:
            raise RuntimeError("No candidates have valid fitness scores.")

        best_scores.append(best_score)
        average_scores.append(float(valid_scores.mean()))

        print(
            f"Generation {generation + 1}/{GENERATIONS}: "
            f"best={best_score:.8f}, "
            f"average valid={average_scores[-1]:.8f}, "
            f"invalid={np.count_nonzero(~np.isfinite(scores))}"
        )

        # The final population has already been evaluated.
        if generation == GENERATIONS - 1:
            break

        # Elitism: preserve the best candidate unchanged.
        next_population = [np.array(best_chromosome, copy=True)]

        while len(next_population) < POPULATION_SIZE:
            parent1 = tournament_selection(scored_population, TOURNAMENT_SIZE)
            parent2 = tournament_selection(scored_population, TOURNAMENT_SIZE)

            child = updated_crossover(parent1, parent2)

            child = updated_mutate(
                child,
                mutation_rate=MUTATION_RATE,
                mutation_strength=MUTATION_STRENGTH,
            )

            # Enforce the per-joint bounds used during initialization.
            child = np.clip(
                np.asarray(child, dtype=float),
                lower_bounds,
                upper_bounds,
            )

            next_population.append(child)

        population = np.array(next_population)

    # Save results in the project's data folder.
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    gait_path = output_dir / "best_gait.csv"
    np.savetxt(
        gait_path,
        chromosome_to_gait(best_chromosome),
        delimiter=",",
        fmt="%.12f",
    )

    history = np.column_stack(
        (
            np.arange(1, len(best_scores) + 1),
            best_scores,
            average_scores,
        )
    )

    np.savetxt(
        output_dir / "fitness_history.csv",
        history,
        delimiter=",",
        header="generation,best_fitness,average_valid_fitness",
        comments="",
    )

    print(f"Saved best gait to: {gait_path}")
    plot_fitness(best_scores, average_scores)


if __name__ == "__main__":
    main()
